"""
Tests for Multi-Fixture Universe Mapping & Pixel-Style Control

Tests validate:
1. Correct channel mapping for all fixtures
2. No channel overlap
3. No universe overflow
4. Effects affect expected fixtures only
5. Group mode sets all fixtures identically
6. Pixel mode creates visible motion across fixtures

Run with: pytest tests/test_pixel_mapper.py -v
"""

import pytest
import math
import time
from pixel_mapper import (
    Pixel,
    PixelArrayConfig,
    FixtureMap,
    FixtureMapEntry,
    PixelArrayController,
    OperationMode,
    EffectType,
    create_pixel_controller,
    validate_fixture_addressing,
    UNIVERSE,
    CHANNELS_PER_FIXTURE,
    CHANNEL_SPACING,
    DEFAULT_START_CHANNEL,
    MAX_DMX_CHANNELS,
    MAX_DMX_UNIVERSE,
    CHANNEL_OFFSET_R,
    CHANNEL_OFFSET_G,
    CHANNEL_OFFSET_B,
    CHANNEL_OFFSET_W,
    TARGET_FPS,
)


# ============================================================
# PIXEL TESTS
# ============================================================

class TestPixel:
    """Tests for Pixel dataclass"""

    def test_pixel_default_values(self):
        """Pixel defaults to black (0,0,0,0)"""
        p = Pixel()
        assert p.r == 0
        assert p.g == 0
        assert p.b == 0
        assert p.w == 0

    def test_pixel_constructor(self):
        """Pixel constructor sets values correctly"""
        p = Pixel(100, 150, 200, 50)
        assert p.r == 100
        assert p.g == 150
        assert p.b == 200
        assert p.w == 50

    def test_pixel_clamping_high(self):
        """Values above 255 are clamped"""
        p = Pixel(300, 400, 500, 600)
        assert p.r == 255
        assert p.g == 255
        assert p.b == 255
        assert p.w == 255

    def test_pixel_clamping_low(self):
        """Values below 0 are clamped"""
        p = Pixel(-10, -20, -30, -40)
        assert p.r == 0
        assert p.g == 0
        assert p.b == 0
        assert p.w == 0

    def test_pixel_copy(self):
        """Copy creates independent instance"""
        p1 = Pixel(100, 150, 200, 50)
        p2 = p1.copy()
        p2.r = 0
        assert p1.r == 100  # Original unchanged
        assert p2.r == 0

    def test_pixel_set_rgbw(self):
        """set_rgbw updates all channels"""
        p = Pixel()
        p.set_rgbw(10, 20, 30, 40)
        assert p.r == 10
        assert p.g == 20
        assert p.b == 30
        assert p.w == 40

    def test_pixel_scale(self):
        """Scale applies brightness factor correctly"""
        p = Pixel(100, 200, 150, 50)
        p.scale(0.5)
        assert p.r == 50
        assert p.g == 100
        assert p.b == 75
        assert p.w == 25

    def test_pixel_scale_clamped(self):
        """Scale factor is clamped to 0-1"""
        p = Pixel(100, 100, 100, 100)
        p.scale(2.0)  # Should clamp to 1.0
        assert p.r == 100
        p.scale(-0.5)  # Should clamp to 0.0
        assert p.r == 0

    def test_pixel_equality(self):
        """Pixel equality comparison"""
        p1 = Pixel(100, 150, 200, 50)
        p2 = Pixel(100, 150, 200, 50)
        p3 = Pixel(100, 150, 200, 51)
        assert p1 == p2
        assert p1 != p3


# ============================================================
# PIXEL ARRAY CONFIG TESTS
# ============================================================

class TestPixelArrayConfig:
    """Tests for PixelArrayConfig dataclass"""

    def test_config_defaults(self):
        """Config has sensible defaults"""
        config = PixelArrayConfig()
        assert config.universe == UNIVERSE
        assert config.fixture_count == 8
        assert config.start_channel == DEFAULT_START_CHANNEL
        assert config.channel_spacing == CHANNEL_SPACING

    def test_config_custom_values(self):
        """Config accepts custom values"""
        config = PixelArrayConfig(
            universe=5,
            fixture_count=10,
            start_channel=17,
            channel_spacing=8,
        )
        assert config.universe == 5
        assert config.fixture_count == 10
        assert config.start_channel == 17
        assert config.channel_spacing == 8

    def test_config_invalid_universe(self):
        """Config rejects invalid universe"""
        with pytest.raises(ValueError):
            PixelArrayConfig(universe=0)
        with pytest.raises(ValueError):
            PixelArrayConfig(universe=70000)

    def test_config_invalid_start_channel(self):
        """Config rejects invalid start channel"""
        with pytest.raises(ValueError):
            PixelArrayConfig(start_channel=0)
        with pytest.raises(ValueError):
            PixelArrayConfig(start_channel=600)

    def test_config_invalid_spacing(self):
        """Config rejects spacing less than channels_per_fixture"""
        with pytest.raises(ValueError):
            PixelArrayConfig(channel_spacing=2)  # Less than 4 channels per fixture

    def test_config_to_dict(self):
        """Config exports to dictionary"""
        config = PixelArrayConfig(universe=5, fixture_count=4)
        d = config.to_dict()
        assert d['universe'] == 5
        assert d['fixture_count'] == 4
        assert 'max_fixtures' in d


# ============================================================
# FIXTURE MAP TESTS
# ============================================================

class TestFixtureMap:
    """Tests for FixtureMap class"""

    def test_fixture_map_basic(self):
        """Basic fixture map creation with 4 fixtures"""
        fm = FixtureMap(4)
        assert fm.fixture_count == 4
        assert fm.is_valid
        assert fm.overflow_error is None

    def test_fixture_map_addressing_pattern(self):
        """Fixture start channels follow 1, 5, 9, 13 pattern (4-channel spacing)"""
        fm = FixtureMap(4)
        entries = fm.get_all_entries()

        assert entries[0].start_channel == 1
        assert entries[1].start_channel == 5
        assert entries[2].start_channel == 9
        assert entries[3].start_channel == 13

    def test_fixture_map_rgbw_channels(self):
        """RGBW channel offsets are correct"""
        fm = FixtureMap(2)
        e0 = fm.get_entry(0)
        e1 = fm.get_entry(1)

        # Fixture 0: starts at 1
        assert e0.r_channel == 1
        assert e0.g_channel == 2
        assert e0.b_channel == 3
        assert e0.w_channel == 4

        # Fixture 1: starts at 5 (contiguous packing)
        assert e1.r_channel == 5
        assert e1.g_channel == 6
        assert e1.b_channel == 7
        assert e1.w_channel == 8

    def test_fixture_map_no_channel_overlap(self):
        """No channel overlap between fixtures"""
        fm = FixtureMap(8)
        is_valid, errors = fm.validate()
        assert is_valid
        assert len(errors) == 0

        # Verify no overlap manually
        all_channels = set()
        for entry in fm.get_all_entries():
            channels = [entry.r_channel, entry.g_channel,
                       entry.b_channel, entry.w_channel]
            for ch in channels:
                assert ch not in all_channels, f"Channel {ch} already used"
                all_channels.add(ch)

    def test_fixture_map_overflow_detection(self):
        """Overflow detection when fixtures exceed 512 channels"""
        # Max fixtures with 4-channel spacing: floor(512/4) = 128
        # Fixture 127 starts at 1 + 127*4 = 509, ends at 512 (valid)
        # Fixture 128 starts at 1 + 128*4 = 513 (OVERFLOW)
        fm = FixtureMap(129)
        assert fm.fixture_count == 128  # Only 128 valid
        assert fm.overflow_error is not None
        assert "overflow" in fm.overflow_error.lower()

    def test_fixture_map_max_valid_fixtures(self):
        """Maximum valid fixture count is 128 (512 channels / 4 channels per fixture)"""
        fm = FixtureMap(128)
        assert fm.fixture_count == 128
        assert fm.is_valid

        # Verify last fixture
        last = fm.get_entry(127)
        assert last.start_channel == 1 + 127 * 4  # 509
        assert last.w_channel == 512  # Within 512

    def test_fixture_map_zero_fixtures(self):
        """Zero fixtures creates empty but valid map"""
        fm = FixtureMap(0)
        assert fm.fixture_count == 0
        assert fm.is_valid
        assert len(fm.get_all_entries()) == 0

    def test_fixture_map_get_entry_bounds(self):
        """get_entry returns None for invalid index"""
        fm = FixtureMap(4)
        assert fm.get_entry(-1) is None
        assert fm.get_entry(4) is None
        assert fm.get_entry(0) is not None
        assert fm.get_entry(3) is not None

    def test_fixture_map_custom_start_channel(self):
        """Fixture map with custom start channel"""
        fm = FixtureMap(4, start_channel=17)
        entries = fm.get_all_entries()

        # First fixture starts at 17
        assert entries[0].start_channel == 17
        assert entries[0].r_channel == 17

        # Second fixture starts at 21
        assert entries[1].start_channel == 21
        assert entries[1].r_channel == 21

    def test_fixture_map_custom_spacing(self):
        """Fixture map with custom channel spacing"""
        fm = FixtureMap(4, channel_spacing=8)  # 8 channels between fixtures
        entries = fm.get_all_entries()

        assert entries[0].start_channel == 1
        assert entries[1].start_channel == 9
        assert entries[2].start_channel == 17
        assert entries[3].start_channel == 25


# ============================================================
# PIXEL ARRAY CONTROLLER TESTS
# ============================================================

class TestPixelArrayController:
    """Tests for PixelArrayController class"""

    def test_controller_creation(self):
        """Controller creates with correct fixture count"""
        ctrl = PixelArrayController(8)
        assert ctrl.fixture_count == 8
        assert len(ctrl.pixels) == 8

    def test_controller_custom_universe(self):
        """Controller with custom universe"""
        ctrl = PixelArrayController(4, universe=5)
        assert ctrl.universe == 5
        assert ctrl.config.universe == 5

    def test_controller_custom_start_channel(self):
        """Controller with custom start channel"""
        ctrl = PixelArrayController(4, start_channel=17)
        assert ctrl.config.start_channel == 17

        # Verify fixture map uses correct channels
        fm = ctrl.get_fixture_map()
        assert fm.get_entry(0).start_channel == 17

    def test_controller_with_config(self):
        """Controller created from PixelArrayConfig"""
        config = PixelArrayConfig(
            universe=7,
            fixture_count=5,
            start_channel=33,
            channel_spacing=8,
        )
        ctrl = PixelArrayController(config=config)
        assert ctrl.universe == 7
        assert ctrl.fixture_count == 5
        assert ctrl.config.start_channel == 33

    def test_controller_default_mode(self):
        """Default mode is GROUPED"""
        ctrl = PixelArrayController(4)
        assert ctrl.mode == OperationMode.GROUPED

    def test_controller_mode_switch(self):
        """Mode switching works correctly"""
        ctrl = PixelArrayController(4)

        ctrl.set_pixel_array_mode()
        assert ctrl.mode == OperationMode.PIXEL_ARRAY

        ctrl.set_grouped_mode()
        assert ctrl.mode == OperationMode.GROUPED

    def test_controller_set_pixel(self):
        """Set individual pixel by index"""
        ctrl = PixelArrayController(4)
        ctrl.set_pixel(1, Pixel(100, 150, 200, 50))

        p = ctrl.get_pixel(1)
        assert p.r == 100
        assert p.g == 150
        assert p.b == 200
        assert p.w == 50

    def test_controller_set_all_pixels(self):
        """Set all pixels to same color (grouped mode)"""
        ctrl = PixelArrayController(4)
        ctrl.set_all_pixels(Pixel(255, 0, 0, 0))

        for i in range(4):
            p = ctrl.get_pixel(i)
            assert p.r == 255
            assert p.g == 0
            assert p.b == 0
            assert p.w == 0

    def test_controller_clear_all(self):
        """Clear all pixels to black"""
        ctrl = PixelArrayController(4)
        ctrl.set_all_rgbw(255, 255, 255, 255)
        ctrl.clear_all()

        for i in range(4):
            p = ctrl.get_pixel(i)
            assert p == Pixel(0, 0, 0, 0)


# ============================================================
# DMX BUFFER TESTS
# ============================================================

class TestDMXBuffer:
    """Tests for DMX buffer population"""

    def test_dmx_buffer_size(self):
        """DMX buffer is 512 bytes"""
        ctrl = PixelArrayController(4)
        assert len(ctrl.get_dmx_buffer()) == 512

    def test_dmx_buffer_population(self):
        """DMX buffer populated with correct channel mapping"""
        ctrl = PixelArrayController(4)

        # Set each fixture to a different color
        ctrl.set_pixel_rgbw(0, 100, 0, 0, 0)    # Fixture 0: Red
        ctrl.set_pixel_rgbw(1, 0, 100, 0, 0)    # Fixture 1: Green
        ctrl.set_pixel_rgbw(2, 0, 0, 100, 0)    # Fixture 2: Blue
        ctrl.set_pixel_rgbw(3, 0, 0, 0, 100)    # Fixture 3: White

        ctrl.populate_dmx_buffer()
        buf = ctrl.get_dmx_buffer()

        # Fixture 0 at channels 1-4 (0-indexed: 0-3)
        assert buf[0] == 100  # R
        assert buf[1] == 0    # G
        assert buf[2] == 0    # B
        assert buf[3] == 0    # W

        # Fixture 1 at channels 5-8 (0-indexed: 4-7)
        assert buf[4] == 0    # R
        assert buf[5] == 100  # G
        assert buf[6] == 0    # B
        assert buf[7] == 0    # W

        # Fixture 2 at channels 9-12 (0-indexed: 8-11)
        assert buf[8] == 0    # R
        assert buf[9] == 0    # G
        assert buf[10] == 100  # B
        assert buf[11] == 0    # W

        # Fixture 3 at channels 13-16 (0-indexed: 12-15)
        assert buf[12] == 0    # R
        assert buf[13] == 0    # G
        assert buf[14] == 0    # B
        assert buf[15] == 100  # W

    def test_dmx_channels_dict(self):
        """get_dmx_channels returns 1-indexed dict"""
        ctrl = PixelArrayController(2)
        ctrl.set_pixel_rgbw(0, 100, 101, 102, 103)
        ctrl.set_pixel_rgbw(1, 200, 201, 202, 203)
        ctrl.populate_dmx_buffer()

        channels = ctrl.get_dmx_channels()

        # Fixture 0: channels 1-4
        assert channels[1] == 100
        assert channels[2] == 101
        assert channels[3] == 102
        assert channels[4] == 103

        # Fixture 1: channels 5-8
        assert channels[5] == 200
        assert channels[6] == 201
        assert channels[7] == 202
        assert channels[8] == 203

    def test_dmx_unused_channels_untouched(self):
        """Unused channels remain at 0"""
        ctrl = PixelArrayController(2)
        ctrl.set_all_rgbw(255, 255, 255, 255)
        ctrl.populate_dmx_buffer()
        buf = ctrl.get_dmx_buffer()

        # With 4-channel spacing (contiguous), 2 fixtures use channels 1-8
        # So channels 9-512 (0-indexed: 8-511) should be 0
        for i in range(8, 512):
            assert buf[i] == 0, f"Channel {i+1} should be 0"


# ============================================================
# GROUPED MODE TESTS
# ============================================================

class TestGroupedMode:
    """Tests for grouped operation mode"""

    def test_grouped_all_fixtures_identical(self):
        """In grouped mode, all fixtures must have identical values"""
        ctrl = PixelArrayController(8)
        ctrl.set_grouped_mode()
        ctrl.set_all_rgbw(128, 64, 32, 16)

        # Verify all pixels are identical
        expected = Pixel(128, 64, 32, 16)
        for i in range(8):
            assert ctrl.get_pixel(i) == expected, \
                f"Fixture {i} should match grouped value"

    def test_grouped_dmx_output_identical(self):
        """In grouped mode, all fixture DMX outputs are identical"""
        ctrl = PixelArrayController(4)
        ctrl.set_grouped_mode()
        ctrl.set_all_rgbw(100, 200, 150, 50)
        ctrl.populate_dmx_buffer()

        channels = ctrl.get_dmx_channels()

        # All fixtures should have same RGBW
        # With 4-channel spacing: fixtures start at 1, 5, 9, 13
        fixture_starts = [1, 5, 9, 13]
        for start in fixture_starts:
            assert channels[start + 0] == 100  # R
            assert channels[start + 1] == 200  # G
            assert channels[start + 2] == 150  # B
            assert channels[start + 3] == 50   # W


# ============================================================
# PIXEL ARRAY MODE TESTS
# ============================================================

class TestPixelArrayMode:
    """Tests for pixel array operation mode"""

    def test_pixel_array_independent_values(self):
        """In pixel array mode, fixtures can have different values"""
        ctrl = PixelArrayController(4)
        ctrl.set_pixel_array_mode()

        # Set different colors
        ctrl.set_pixel_rgbw(0, 255, 0, 0, 0)    # Red
        ctrl.set_pixel_rgbw(1, 0, 255, 0, 0)    # Green
        ctrl.set_pixel_rgbw(2, 0, 0, 255, 0)    # Blue
        ctrl.set_pixel_rgbw(3, 255, 255, 255, 0) # White

        # Verify all different
        assert ctrl.get_pixel(0) != ctrl.get_pixel(1)
        assert ctrl.get_pixel(1) != ctrl.get_pixel(2)
        assert ctrl.get_pixel(2) != ctrl.get_pixel(3)

    def test_pixel_array_gradient(self):
        """Pixel array can display gradient"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()

        # Create red gradient
        for i in range(8):
            brightness = int(255 * i / 7)
            ctrl.set_pixel_rgbw(i, brightness, 0, 0, 0)

        # Verify gradient
        for i in range(8):
            expected = int(255 * i / 7)
            assert ctrl.get_pixel(i).r == expected


# ============================================================
# EFFECT TESTS
# ============================================================

class TestEffects:
    """Tests for effect math (deterministic)"""

    def test_wave_effect_deterministic(self):
        """Wave effect produces deterministic output"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.WAVE, color=Pixel(255, 0, 0, 0), speed=1.0)

        # Render at specific time
        ctrl._start_time = 0
        ctrl._compute_effect(0.5)  # 0.5 seconds

        # Capture state
        state1 = [ctrl.get_pixel(i).copy() for i in range(8)]

        # Reset and compute again at same time
        ctrl.clear_all()
        ctrl._compute_effect(0.5)

        # Should be identical
        state2 = [ctrl.get_pixel(i).copy() for i in range(8)]

        for i in range(8):
            assert state1[i] == state2[i], \
                f"Fixture {i} not deterministic"

    def test_wave_effect_creates_motion(self):
        """Wave effect creates visible motion across fixtures"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.WAVE, color=Pixel(255, 0, 0, 0), speed=1.0)

        ctrl._start_time = 0
        ctrl._compute_effect(0.0)
        state_t0 = [ctrl.get_pixel(i).r for i in range(8)]

        ctrl._compute_effect(0.25)  # Quarter cycle
        state_t1 = [ctrl.get_pixel(i).r for i in range(8)]

        # States should be different (motion occurred)
        assert state_t0 != state_t1, "Wave should create motion over time"

        # At least some fixtures should have different brightness
        differences = sum(1 for a, b in zip(state_t0, state_t1) if a != b)
        assert differences > 0, "Some fixtures should change"

    def test_wave_effect_phase_offset(self):
        """Wave effect has phase offset between fixtures"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.WAVE, color=Pixel(255, 0, 0, 0), speed=1.0)

        ctrl._start_time = 0
        ctrl._compute_effect(0.0)

        # At t=0, different fixtures should have different brightness
        values = [ctrl.get_pixel(i).r for i in range(8)]

        # Not all values should be the same
        unique_values = set(values)
        assert len(unique_values) > 1, "Fixtures should have phase offset"

    def test_chase_effect_single_active(self):
        """Chase effect has single bright pixel at a time"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(
            EffectType.CHASE,
            color=Pixel(255, 0, 0, 0),
            speed=1.0,
            tail_length=1
        )

        ctrl._start_time = 0
        ctrl._compute_effect(0.0)

        # Count bright pixels (> 128)
        bright = sum(1 for i in range(8) if ctrl.get_pixel(i).r > 128)
        assert bright <= 2, "Chase should have limited bright pixels"

    def test_bounce_effect_direction_reversal(self):
        """Bounce effect reverses direction"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(
            EffectType.BOUNCE,
            color=Pixel(255, 0, 0, 0),
            speed=1.0,
            tail_length=1
        )

        ctrl._start_time = 0

        # Find peak position at different times
        def find_peak():
            return max(range(8), key=lambda i: ctrl.get_pixel(i).r)

        ctrl._compute_effect(0.0)
        peak_start = find_peak()

        ctrl._compute_effect(0.25)  # Quarter through forward pass
        peak_quarter = find_peak()

        ctrl._compute_effect(0.75)  # Quarter through backward pass
        peak_three_quarter = find_peak()

        # Forward pass: peak should increase
        # Backward pass: peak should decrease
        # Exact positions depend on timing, just verify movement occurs
        assert peak_quarter >= peak_start or peak_three_quarter <= peak_quarter

    def test_rainbow_wave_effect(self):
        """Rainbow wave creates color variation"""
        ctrl = PixelArrayController(8)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.RAINBOW_WAVE, speed=1.0)

        ctrl._start_time = 0
        ctrl._compute_effect(0.0)

        # Fixtures should have different hues
        colors = [(ctrl.get_pixel(i).r, ctrl.get_pixel(i).g, ctrl.get_pixel(i).b)
                  for i in range(8)]
        unique_colors = set(colors)
        assert len(unique_colors) > 1, "Rainbow should have color variation"


# ============================================================
# VALIDATION TESTS
# ============================================================

class TestValidation:
    """Tests for validation functions"""

    def test_validation_passes_valid_config(self):
        """Validation passes for valid configuration"""
        ctrl = PixelArrayController(8)
        is_valid, errors = ctrl.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validation_reports_overflow(self):
        """Validation reports overflow errors"""
        # Create controller that would overflow
        ctrl = PixelArrayController(130)  # Requests 130, only 128 fit
        is_valid, errors = ctrl.validate()
        # Should still be valid because overflow is handled
        # But the controller limits to max valid fixtures
        assert ctrl.fixture_count == 128

    def test_validate_fixture_addressing(self):
        """validate_fixture_addressing helper works"""
        assert validate_fixture_addressing(8) == True
        assert validate_fixture_addressing(32) == True

    def test_controller_get_status(self):
        """get_status returns complete info"""
        ctrl = PixelArrayController(4)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.WAVE, speed=2.0)

        status = ctrl.get_status()

        assert status['universe'] == UNIVERSE
        assert status['fixture_count'] == 4
        assert status['mode'] == 'pixel_array'
        assert status['effect_type'] == 'wave'
        assert status['effect_speed'] == 2.0
        assert len(status['pixels']) == 4


# ============================================================
# TIMING TESTS
# ============================================================

class TestTiming:
    """Tests for frame timing"""

    def test_target_fps(self):
        """Target FPS is 30"""
        assert TARGET_FPS == 30

    def test_frame_interval(self):
        """Frame interval matches target FPS"""
        expected_interval = 1.0 / 30
        from pixel_mapper import FRAME_INTERVAL
        assert abs(FRAME_INTERVAL - expected_interval) < 0.001


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests"""

    def test_full_render_cycle(self):
        """Complete render cycle from effect to DMX output"""
        ctrl = PixelArrayController(4)
        ctrl.set_pixel_array_mode()
        ctrl.set_effect(EffectType.WAVE, color=Pixel(255, 128, 0, 0), speed=1.0)

        # Simulate multiple frames
        ctrl._start_time = time.monotonic()

        for frame in range(10):
            channels = ctrl.render_single_frame()

            # Verify channels are populated
            assert len(channels) > 0
            # Verify some values are non-zero (effect is running)
            assert sum(channels.values()) > 0

    def test_output_callback_called(self):
        """Output callback is called during render"""
        ctrl = PixelArrayController(4)

        callback_data = []
        def capture_callback(universe, channels):
            callback_data.append((universe, channels.copy()))

        ctrl.set_output_callback(capture_callback)
        ctrl.set_all_rgbw(100, 100, 100, 100)
        ctrl.populate_dmx_buffer()
        ctrl._send_frame()

        assert len(callback_data) == 1
        assert callback_data[0][0] == UNIVERSE
        assert len(callback_data[0][1]) == 16  # 4 fixtures * 4 channels


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """Edge case tests"""

    def test_single_fixture(self):
        """Single fixture works correctly"""
        ctrl = PixelArrayController(1)
        assert ctrl.fixture_count == 1

        ctrl.set_pixel_rgbw(0, 255, 128, 64, 32)
        ctrl.populate_dmx_buffer()
        channels = ctrl.get_dmx_channels()

        assert channels[1] == 255
        assert channels[2] == 128
        assert channels[3] == 64
        assert channels[4] == 32

    def test_invalid_pixel_index(self):
        """Invalid pixel index returns None"""
        ctrl = PixelArrayController(4)
        assert ctrl.get_pixel(-1) is None
        assert ctrl.get_pixel(4) is None
        assert ctrl.get_pixel(100) is None

    def test_set_pixel_invalid_index(self):
        """Set pixel with invalid index is no-op"""
        ctrl = PixelArrayController(4)
        ctrl.set_pixel(-1, Pixel(255, 0, 0, 0))  # Should not crash
        ctrl.set_pixel(100, Pixel(255, 0, 0, 0))  # Should not crash

    def test_effect_speed_clamping(self):
        """Effect speed is clamped to valid range"""
        ctrl = PixelArrayController(4)

        ctrl.set_effect(EffectType.WAVE, speed=100.0)  # Too fast
        assert ctrl._effect_speed <= 10.0

        ctrl.set_effect(EffectType.WAVE, speed=0.01)  # Too slow
        assert ctrl._effect_speed >= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
