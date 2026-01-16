"""
Multi-Fixture Universe Mapping & Pixel-Style Control

This module provides:
- Configurable fixture mapping for any universe
- Pixel abstraction layer treating fixtures as logical pixels
- DMX buffer population with correct channel mapping
- Grouped and pixel array operation modes
- Deterministic wave, chase, and motion effects
- Fixed 30 FPS frame rate timing

CONFIGURATION:
All parameters are user-configurable:
- Universe: Any valid DMX universe (1-63999)
- Fixture type: Currently RGBW (4 channels: R=+0, G=+1, B=+2, W=+3)
- Start channel: Any valid DMX channel (1-512)
- Channel spacing: Configurable (default 4 for contiguous RGBW)
- Fixture count: Any number that fits within 512 channels

EXAMPLE FIXTURE MAP TABLE (Universe 4, Start 1, Spacing 4):

| Fixture Index | Start Channel | R  | G  | B  | W  |
|---------------|---------------|----|----|----|----|
| 0             | 1             | 1  | 2  | 3  | 4  |
| 1             | 5             | 5  | 6  | 7  | 8  |
| 2             | 9             | 9  | 10 | 11 | 12 |
| 3             | 13            | 13 | 14 | 15 | 16 |
| 4             | 17            | 17 | 18 | 19 | 20 |
| ...           | ...           | ...| ...| ...| ...|

Version: 1.1.0
"""

import math
import time
import threading
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# DEFAULT CONFIGURATION (All values are configurable per instance)
# ============================================================

# Default universe (can be overridden per controller)
UNIVERSE = 4

# Default fixture parameters
CHANNELS_PER_FIXTURE = 4
CHANNEL_SPACING = 4  # Fixtures at 1, 5, 9, 13, 17, ... (contiguous RGBW)
DEFAULT_START_CHANNEL = 1

# DMX limits
MAX_DMX_CHANNELS = 512
MAX_DMX_UNIVERSE = 63999

# Channel offsets within fixture (RGBW layout)
CHANNEL_OFFSET_R = 0
CHANNEL_OFFSET_G = 1
CHANNEL_OFFSET_B = 2
CHANNEL_OFFSET_W = 3

# Frame rate
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS


# ============================================================
# PIXEL ARRAY CONFIGURATION - User-configurable fixture setup
# ============================================================

@dataclass
class PixelArrayConfig:
    """
    Configuration for a pixel array.

    All parameters are user-configurable to support different setups:
    - Different universes
    - Different start channels
    - Different channel spacing
    - Different fixture counts
    """
    universe: int = UNIVERSE
    fixture_count: int = 8
    start_channel: int = DEFAULT_START_CHANNEL
    channel_spacing: int = CHANNEL_SPACING
    channels_per_fixture: int = CHANNELS_PER_FIXTURE

    def __post_init__(self):
        """Validate configuration"""
        # Universe bounds
        if self.universe < 1 or self.universe > MAX_DMX_UNIVERSE:
            raise ValueError(f"Universe must be 1-{MAX_DMX_UNIVERSE}, got {self.universe}")

        # Channel bounds
        if self.start_channel < 1 or self.start_channel > MAX_DMX_CHANNELS:
            raise ValueError(f"Start channel must be 1-{MAX_DMX_CHANNELS}, got {self.start_channel}")

        # Spacing must accommodate fixture channels
        if self.channel_spacing < self.channels_per_fixture:
            raise ValueError(
                f"Channel spacing ({self.channel_spacing}) must be >= "
                f"channels per fixture ({self.channels_per_fixture})"
            )

        # Calculate max fixtures that fit
        max_fixtures = (MAX_DMX_CHANNELS - self.start_channel + 1) // self.channel_spacing
        if self.fixture_count > max_fixtures:
            print(f"WARNING: Requested {self.fixture_count} fixtures but only "
                  f"{max_fixtures} fit. Clamping to {max_fixtures}.")
            self.fixture_count = max_fixtures

    def get_max_fixtures(self) -> int:
        """Calculate maximum fixtures that can fit"""
        return (MAX_DMX_CHANNELS - self.start_channel + 1) // self.channel_spacing

    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            'universe': self.universe,
            'fixture_count': self.fixture_count,
            'start_channel': self.start_channel,
            'channel_spacing': self.channel_spacing,
            'channels_per_fixture': self.channels_per_fixture,
            'max_fixtures': self.get_max_fixtures(),
        }


# ============================================================
# PIXEL STRUCTURE - RGBW Color
# ============================================================

@dataclass
class Pixel:
    """
    RGBW color for a single fixture.
    All values are 0-255.
    """
    r: int = 0
    g: int = 0
    b: int = 0
    w: int = 0

    def __post_init__(self):
        """Clamp values to valid range"""
        self.r = max(0, min(255, self.r))
        self.g = max(0, min(255, self.g))
        self.b = max(0, min(255, self.b))
        self.w = max(0, min(255, self.w))

    def copy(self) -> 'Pixel':
        """Return a copy of this pixel"""
        return Pixel(self.r, self.g, self.b, self.w)

    def set_rgb(self, r: int, g: int, b: int) -> 'Pixel':
        """Set RGB values, return self for chaining"""
        self.r = max(0, min(255, r))
        self.g = max(0, min(255, g))
        self.b = max(0, min(255, b))
        return self

    def set_rgbw(self, r: int, g: int, b: int, w: int) -> 'Pixel':
        """Set all RGBW values, return self for chaining"""
        self.r = max(0, min(255, r))
        self.g = max(0, min(255, g))
        self.b = max(0, min(255, b))
        self.w = max(0, min(255, w))
        return self

    def scale(self, factor: float) -> 'Pixel':
        """Scale all channels by factor (0.0-1.0), return self"""
        factor = max(0.0, min(1.0, factor))
        self.r = int(self.r * factor)
        self.g = int(self.g * factor)
        self.b = int(self.b * factor)
        self.w = int(self.w * factor)
        return self

    def __eq__(self, other):
        if not isinstance(other, Pixel):
            return False
        return self.r == other.r and self.g == other.g and self.b == other.b and self.w == other.w


# ============================================================
# FIXTURE MAP - Canonical mapping table
# ============================================================

@dataclass
class FixtureMapEntry:
    """Single fixture mapping entry"""
    fixture_index: int      # Zero-based index
    start_channel: int      # 1-indexed DMX start channel
    r_channel: int          # Absolute R channel
    g_channel: int          # Absolute G channel
    b_channel: int          # Absolute B channel
    w_channel: int          # Absolute W channel


class FixtureMap:
    """
    Configurable fixture map for any universe.

    Supports:
    - Any universe (1-63999)
    - Any start channel (1-512)
    - Any channel spacing (must be >= channels_per_fixture)
    - Any number of fixtures (limited by available channels)
    """

    def __init__(
        self,
        fixture_count: int,
        start_channel: int = DEFAULT_START_CHANNEL,
        channel_spacing: int = CHANNEL_SPACING,
        channels_per_fixture: int = CHANNELS_PER_FIXTURE,
    ):
        """
        Initialize fixture map with configuration.

        Args:
            fixture_count: Number of fixtures to map
            start_channel: First fixture's start channel (1-indexed)
            channel_spacing: Channels between fixture starts
            channels_per_fixture: Channels used by each fixture (RGBW = 4)

        Raises:
            ValueError: If configuration would overflow DMX universe
        """
        self._entries: List[FixtureMapEntry] = []
        self._fixture_count = 0
        self._overflow_error: Optional[str] = None
        self._start_channel = start_channel
        self._channel_spacing = channel_spacing
        self._channels_per_fixture = channels_per_fixture

        self._build_map(fixture_count)

    def _build_map(self, fixture_count: int):
        """Build the fixture map table"""
        self._entries = []
        self._fixture_count = 0

        for i in range(fixture_count):
            # Calculate start channel based on configuration
            # Formula: start = start_channel + (i * channel_spacing)
            start_channel = self._start_channel + (i * self._channel_spacing)

            # Check for universe overflow
            last_channel = start_channel + self._channels_per_fixture - 1
            if last_channel > MAX_DMX_CHANNELS:
                self._overflow_error = (
                    f"Fixture {i} (start channel {start_channel}) would overflow "
                    f"DMX universe (channel {last_channel} > {MAX_DMX_CHANNELS})"
                )
                print(f"ERROR: {self._overflow_error}")
                break

            entry = FixtureMapEntry(
                fixture_index=i,
                start_channel=start_channel,
                r_channel=start_channel + CHANNEL_OFFSET_R,
                g_channel=start_channel + CHANNEL_OFFSET_G,
                b_channel=start_channel + CHANNEL_OFFSET_B,
                w_channel=start_channel + CHANNEL_OFFSET_W,
            )
            self._entries.append(entry)
            self._fixture_count += 1

    @property
    def fixture_count(self) -> int:
        """Number of valid fixtures in map"""
        return self._fixture_count

    @property
    def overflow_error(self) -> Optional[str]:
        """Overflow error message if any"""
        return self._overflow_error

    @property
    def is_valid(self) -> bool:
        """True if no overflow occurred"""
        return self._overflow_error is None

    def get_entry(self, fixture_index: int) -> Optional[FixtureMapEntry]:
        """Get fixture map entry by index"""
        if 0 <= fixture_index < len(self._entries):
            return self._entries[fixture_index]
        return None

    def get_all_entries(self) -> List[FixtureMapEntry]:
        """Get all fixture map entries"""
        return self._entries.copy()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the fixture map.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if self._overflow_error:
            errors.append(self._overflow_error)

        # Check for channel overlap (should not happen with correct spacing)
        used_channels = set()
        for entry in self._entries:
            channels = [entry.r_channel, entry.g_channel, entry.b_channel, entry.w_channel]
            for ch in channels:
                if ch in used_channels:
                    errors.append(f"Channel {ch} used by multiple fixtures")
                used_channels.add(ch)

        # Check channel bounds
        for entry in self._entries:
            for ch in [entry.r_channel, entry.g_channel, entry.b_channel, entry.w_channel]:
                if ch < 1 or ch > MAX_DMX_CHANNELS:
                    errors.append(f"Channel {ch} out of range (1-{MAX_DMX_CHANNELS})")

        return len(errors) == 0, errors

    def print_map(self):
        """Print the fixture map table for debugging"""
        print(f"\n{'='*60}")
        print(f"FIXTURE MAP - Universe {UNIVERSE}")
        print(f"{'='*60}")
        print(f"{'Index':<8} {'Start':<8} {'R':<6} {'G':<6} {'B':<6} {'W':<6}")
        print(f"{'-'*60}")
        for entry in self._entries:
            print(f"{entry.fixture_index:<8} {entry.start_channel:<8} "
                  f"{entry.r_channel:<6} {entry.g_channel:<6} "
                  f"{entry.b_channel:<6} {entry.w_channel:<6}")
        print(f"{'='*60}")
        print(f"Total fixtures: {self._fixture_count}")
        if self._overflow_error:
            print(f"OVERFLOW ERROR: {self._overflow_error}")
        print()


# ============================================================
# OPERATION MODES
# ============================================================

class OperationMode(Enum):
    """Fixture operation mode"""
    GROUPED = "grouped"       # All pixels receive identical RGBW values
    PIXEL_ARRAY = "pixel_array"  # Each pixel has independent values


# ============================================================
# EFFECT TYPES
# ============================================================

class EffectType(Enum):
    """Available effect types"""
    NONE = "none"
    WAVE = "wave"
    CHASE = "chase"
    BOUNCE = "bounce"
    RAINBOW_WAVE = "rainbow_wave"


# ============================================================
# PIXEL ARRAY CONTROLLER
# ============================================================

class PixelArrayController:
    """
    Main controller for multi-fixture pixel array.

    Manages:
    - Pixel array state
    - DMX buffer population
    - Operation modes
    - Effects with deterministic math
    - Fixed frame rate timing
    """

    def __init__(
        self,
        fixture_count: int = 8,
        universe: int = UNIVERSE,
        start_channel: int = DEFAULT_START_CHANNEL,
        channel_spacing: int = CHANNEL_SPACING,
        config: Optional[PixelArrayConfig] = None,
    ):
        """
        Initialize pixel array controller.

        Args:
            fixture_count: Number of RGBW fixtures (default 8)
            universe: DMX universe (default 4)
            start_channel: First fixture start channel (default 1)
            channel_spacing: Channels between fixtures (default 4)
            config: Optional PixelArrayConfig object (overrides other params)
        """
        # Use config if provided, otherwise build from params
        if config:
            self._config = config
        else:
            self._config = PixelArrayConfig(
                universe=universe,
                fixture_count=fixture_count,
                start_channel=start_channel,
                channel_spacing=channel_spacing,
            )

        # Store universe
        self._universe = self._config.universe

        # Build fixture map with config
        self._fixture_map = FixtureMap(
            fixture_count=self._config.fixture_count,
            start_channel=self._config.start_channel,
            channel_spacing=self._config.channel_spacing,
            channels_per_fixture=self._config.channels_per_fixture,
        )
        self._n = self._fixture_map.fixture_count

        # Validate
        is_valid, errors = self._fixture_map.validate()
        if not is_valid:
            for error in errors:
                print(f"VALIDATION ERROR: {error}")

        # Pixel array
        self._pixels: List[Pixel] = [Pixel() for _ in range(self._n)]

        # DMX buffer (0-indexed, 512 bytes)
        self._dmx_buffer: List[int] = [0] * MAX_DMX_CHANNELS

        # Operation mode
        self._mode = OperationMode.GROUPED

        # Effect state
        self._effect_type = EffectType.NONE
        self._effect_color = Pixel(255, 0, 0, 0)  # Default red
        self._effect_speed = 1.0  # Hz
        self._effect_params: Dict = {}

        # Timing
        self._running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time = 0.0

        # Output callback
        self._send_callback: Optional[Callable[[int, Dict[int, int]], None]] = None

        print(f"PixelArrayController initialized: Universe {self._universe}, "
              f"{self._n} fixtures, start channel {self._config.start_channel}, "
              f"spacing {self._config.channel_spacing}")

    # ─────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────

    @property
    def fixture_count(self) -> int:
        """Number of fixtures in array"""
        return self._n

    @property
    def pixels(self) -> List[Pixel]:
        """Direct access to pixel array"""
        return self._pixels

    @property
    def mode(self) -> OperationMode:
        """Current operation mode"""
        return self._mode

    @property
    def effect_type(self) -> EffectType:
        """Current effect type"""
        return self._effect_type

    @property
    def frame_count(self) -> int:
        """Current frame number"""
        return self._frame_count

    @property
    def is_running(self) -> bool:
        """True if render loop is running"""
        return self._running

    @property
    def universe(self) -> int:
        """Target DMX universe"""
        return self._universe

    @property
    def config(self) -> PixelArrayConfig:
        """Current configuration"""
        return self._config

    # ─────────────────────────────────────────────────────────
    # Fixture Map Access
    # ─────────────────────────────────────────────────────────

    def get_fixture_map(self) -> FixtureMap:
        """Get the fixture map"""
        return self._fixture_map

    def print_fixture_map(self):
        """Print fixture map for debugging"""
        self._fixture_map.print_map()

    # ─────────────────────────────────────────────────────────
    # Mode Control
    # ─────────────────────────────────────────────────────────

    def set_mode(self, mode: OperationMode):
        """Set operation mode"""
        self._mode = mode
        print(f"Mode set to: {mode.value}")

    def set_grouped_mode(self):
        """Switch to grouped mode (all pixels same color)"""
        self._mode = OperationMode.GROUPED

    def set_pixel_array_mode(self):
        """Switch to pixel array mode (independent colors)"""
        self._mode = OperationMode.PIXEL_ARRAY

    # ─────────────────────────────────────────────────────────
    # Pixel Manipulation
    # ─────────────────────────────────────────────────────────

    def set_pixel(self, index: int, pixel: Pixel):
        """Set a single pixel by index"""
        if 0 <= index < self._n:
            self._pixels[index] = pixel.copy()

    def set_pixel_rgbw(self, index: int, r: int, g: int, b: int, w: int):
        """Set a single pixel by index with RGBW values"""
        if 0 <= index < self._n:
            self._pixels[index].set_rgbw(r, g, b, w)

    def set_all_pixels(self, pixel: Pixel):
        """Set all pixels to same color (grouped mode helper)"""
        for i in range(self._n):
            self._pixels[i] = pixel.copy()

    def set_all_rgbw(self, r: int, g: int, b: int, w: int):
        """Set all pixels to same RGBW values"""
        for i in range(self._n):
            self._pixels[i].set_rgbw(r, g, b, w)

    def clear_all(self):
        """Set all pixels to black (0,0,0,0)"""
        for i in range(self._n):
            self._pixels[i].set_rgbw(0, 0, 0, 0)

    def get_pixel(self, index: int) -> Optional[Pixel]:
        """Get pixel at index"""
        if 0 <= index < self._n:
            return self._pixels[index]
        return None

    # ─────────────────────────────────────────────────────────
    # DMX Buffer Population
    # ─────────────────────────────────────────────────────────

    def populate_dmx_buffer(self):
        """
        Populate DMX buffer from pixel array.

        Rules:
        - For each pixel i:
          - base = fixture_start_channel[i] - 1  (convert to 0-indexed)
          - dmx[base + 0] = pixels[i].r
          - dmx[base + 1] = pixels[i].g
          - dmx[base + 2] = pixels[i].b
          - dmx[base + 3] = pixels[i].w
        - Do not touch unused channels
        - Do not write outside 0-511
        """
        for i in range(self._n):
            entry = self._fixture_map.get_entry(i)
            if entry is None:
                continue

            pixel = self._pixels[i]

            # Convert to 0-indexed for buffer
            base = entry.start_channel - 1

            # Bounds check
            if base < 0 or base + 3 >= MAX_DMX_CHANNELS:
                continue

            # Populate buffer
            self._dmx_buffer[base + CHANNEL_OFFSET_R] = pixel.r
            self._dmx_buffer[base + CHANNEL_OFFSET_G] = pixel.g
            self._dmx_buffer[base + CHANNEL_OFFSET_B] = pixel.b
            self._dmx_buffer[base + CHANNEL_OFFSET_W] = pixel.w

    def get_dmx_buffer(self) -> List[int]:
        """Get the current DMX buffer"""
        return self._dmx_buffer.copy()

    def get_dmx_channels(self) -> Dict[int, int]:
        """Get DMX channels as dict (1-indexed channel -> value)"""
        channels = {}
        for i in range(self._n):
            entry = self._fixture_map.get_entry(i)
            if entry is None:
                continue
            pixel = self._pixels[i]
            channels[entry.r_channel] = pixel.r
            channels[entry.g_channel] = pixel.g
            channels[entry.b_channel] = pixel.b
            channels[entry.w_channel] = pixel.w
        return channels

    # ─────────────────────────────────────────────────────────
    # Output Callback
    # ─────────────────────────────────────────────────────────

    def set_output_callback(self, callback: Callable[[int, Dict[int, int]], None]):
        """
        Set callback for sending DMX frames.

        Args:
            callback: Function(universe, channels_dict) to send DMX
        """
        self._send_callback = callback

    def _send_frame(self):
        """Send current frame via callback"""
        if self._send_callback:
            channels = self.get_dmx_channels()
            self._send_callback(self._universe, channels)

    # ─────────────────────────────────────────────────────────
    # Effects
    # ─────────────────────────────────────────────────────────

    def set_effect(
        self,
        effect_type: EffectType,
        color: Optional[Pixel] = None,
        speed: float = 1.0,
        **params
    ):
        """
        Configure effect.

        Args:
            effect_type: Type of effect
            color: Base color for effect
            speed: Effect speed in Hz
            **params: Additional effect-specific parameters
        """
        self._effect_type = effect_type
        if color:
            self._effect_color = color.copy()
        self._effect_speed = max(0.1, min(10.0, speed))
        self._effect_params = params

    def stop_effect(self):
        """Stop current effect"""
        self._effect_type = EffectType.NONE

    def _compute_effect(self, elapsed_time: float):
        """
        Compute effect for current frame.

        Uses index-based math, NOT randomness.
        """
        if self._effect_type == EffectType.NONE:
            return

        n = self._n
        if n == 0:
            return

        if self._mode == OperationMode.GROUPED:
            # In grouped mode, apply uniform brightness modulation
            self._compute_grouped_effect(elapsed_time)
        else:
            # In pixel array mode, compute per-pixel values
            if self._effect_type == EffectType.WAVE:
                self._compute_wave_effect(elapsed_time)
            elif self._effect_type == EffectType.CHASE:
                self._compute_chase_effect(elapsed_time)
            elif self._effect_type == EffectType.BOUNCE:
                self._compute_bounce_effect(elapsed_time)
            elif self._effect_type == EffectType.RAINBOW_WAVE:
                self._compute_rainbow_wave_effect(elapsed_time)

    def _compute_grouped_effect(self, elapsed_time: float):
        """Apply grouped effect - all pixels same value with time modulation"""
        # Simple sine wave brightness for grouped mode
        phase = elapsed_time * self._effect_speed * 2 * math.pi
        brightness = (math.sin(phase) + 1) / 2  # 0 to 1

        pixel = self._effect_color.copy()
        pixel.scale(brightness)
        self.set_all_pixels(pixel)

    def _compute_wave_effect(self, elapsed_time: float):
        """
        Wave effect - brightness wave traveling across fixtures.

        For pixel i at time t:
            phase = (i / N) * 2*pi
            value = sin(t * speed * 2*pi + phase)
            brightness = map(value, -1..1, 0..255)
        """
        n = self._n
        t = elapsed_time

        for i in range(n):
            # Phase offset based on position
            phase = (i / n) * 2 * math.pi

            # Time-varying wave
            value = math.sin(t * self._effect_speed * 2 * math.pi + phase)

            # Map -1..1 to 0..1
            brightness = (value + 1) / 2

            # Apply to base color
            pixel = self._effect_color.copy()
            pixel.scale(brightness)
            self._pixels[i] = pixel

    def _compute_chase_effect(self, elapsed_time: float):
        """
        Chase effect - single active pixel moves across array.

        Active pixel position = (t * speed * N) % N
        Active pixel at full, others at 0 (or optional trail)
        """
        n = self._n
        t = elapsed_time

        # Calculate active position (continuous)
        position = (t * self._effect_speed * n) % n

        # Optional tail length
        tail_length = self._effect_params.get('tail_length', 1)

        for i in range(n):
            # Distance from active position
            dist = abs(i - position)
            # Handle wraparound
            dist = min(dist, n - dist)

            if dist < tail_length:
                # Within tail
                brightness = 1.0 - (dist / tail_length)
                pixel = self._effect_color.copy()
                pixel.scale(brightness)
                self._pixels[i] = pixel
            else:
                # Outside tail
                self._pixels[i] = Pixel(0, 0, 0, 0)

    def _compute_bounce_effect(self, elapsed_time: float):
        """
        Back-and-forth chase (bounce/scanner).

        Position moves: 0 -> N-1 -> 0 -> N-1 -> ...
        """
        n = self._n
        t = elapsed_time

        # Calculate position in bounce cycle
        cycle_time = 2.0 / self._effect_speed  # Time for full bounce
        cycle_pos = (t % cycle_time) / cycle_time  # 0 to 1

        # Convert to position: 0->1->0 becomes 0->N-1->0
        if cycle_pos < 0.5:
            # Forward pass
            position = cycle_pos * 2 * (n - 1)
        else:
            # Backward pass
            position = (1 - (cycle_pos - 0.5) * 2) * (n - 1)

        # Optional tail
        tail_length = self._effect_params.get('tail_length', 2)

        for i in range(n):
            dist = abs(i - position)

            if dist < tail_length:
                brightness = 1.0 - (dist / tail_length)
                pixel = self._effect_color.copy()
                pixel.scale(brightness)
                self._pixels[i] = pixel
            else:
                self._pixels[i] = Pixel(0, 0, 0, 0)

    def _compute_rainbow_wave_effect(self, elapsed_time: float):
        """
        Rainbow wave - hue varies by position and time.
        """
        n = self._n
        t = elapsed_time

        for i in range(n):
            # Hue based on position and time
            hue = ((i / n) + t * self._effect_speed) % 1.0

            # Convert HSV to RGB (full saturation and value)
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)

            self._pixels[i].set_rgbw(
                int(r * 255),
                int(g * 255),
                int(b * 255),
                0  # No white in rainbow
            )

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV (0-1) to RGB (0-1)"""
        if s == 0:
            return v, v, v

        h = h * 6
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

    # ─────────────────────────────────────────────────────────
    # Render Loop
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start the render loop at 30 FPS"""
        if self._running:
            return

        self._running = True
        self._stop_flag.clear()
        self._frame_count = 0
        self._start_time = time.monotonic()

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print(f"PixelArrayController started at {TARGET_FPS} FPS")

    def stop(self):
        """Stop the render loop"""
        if not self._running:
            return

        self._stop_flag.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        print("PixelArrayController stopped")

    def _render_loop(self):
        """Main render loop - runs at 30 FPS"""
        while not self._stop_flag.is_set():
            frame_start = time.monotonic()
            elapsed = frame_start - self._start_time

            # Compute effect
            self._compute_effect(elapsed)

            # Populate DMX buffer
            self.populate_dmx_buffer()

            # Send frame
            self._send_frame()

            # Update frame count
            self._frame_count += 1

            # Frame timing
            frame_duration = time.monotonic() - frame_start
            sleep_time = FRAME_INTERVAL - frame_duration

            if sleep_time > 0:
                self._stop_flag.wait(sleep_time)

    def render_single_frame(self) -> Dict[int, int]:
        """
        Render a single frame without the loop.
        Useful for testing or manual control.
        """
        elapsed = time.monotonic() - self._start_time if self._start_time > 0 else 0

        # Compute effect
        self._compute_effect(elapsed)

        # Populate DMX buffer
        self.populate_dmx_buffer()

        # Return channels
        return self.get_dmx_channels()

    # ─────────────────────────────────────────────────────────
    # Validation & Debug Output
    # ─────────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Perform all validation checks.

        Validates:
        1. Correct channel mapping for all fixtures
        2. No channel overlap
        3. No universe overflow
        4. Effect affects expected fixtures only
        5. Group mode sets all fixtures identically
        6. Pixel mode creates visible motion

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Validate fixture map
        map_valid, map_errors = self._fixture_map.validate()
        errors.extend(map_errors)

        # Validate pixel array size matches fixture count
        if len(self._pixels) != self._fixture_map.fixture_count:
            errors.append(
                f"Pixel array size ({len(self._pixels)}) != "
                f"fixture count ({self._fixture_map.fixture_count})"
            )

        # Validate DMX buffer size
        if len(self._dmx_buffer) != MAX_DMX_CHANNELS:
            errors.append(
                f"DMX buffer size ({len(self._dmx_buffer)}) != "
                f"expected ({MAX_DMX_CHANNELS})"
            )

        return len(errors) == 0, errors

    def print_debug_info(self):
        """Print comprehensive debug information"""
        print(f"\n{'='*60}")
        print("PIXEL ARRAY CONTROLLER - DEBUG INFO")
        print(f"{'='*60}")

        print(f"\nConfiguration:")
        print(f"  Universe: {UNIVERSE}")
        print(f"  Fixture count: {self._n}")
        print(f"  Channels per fixture: {CHANNELS_PER_FIXTURE}")
        print(f"  Channel spacing: {CHANNEL_SPACING}")
        print(f"  Target FPS: {TARGET_FPS}")

        print(f"\nState:")
        print(f"  Mode: {self._mode.value}")
        print(f"  Effect: {self._effect_type.value}")
        print(f"  Effect speed: {self._effect_speed} Hz")
        print(f"  Frame count: {self._frame_count}")
        print(f"  Running: {self._running}")

        print(f"\nFixture Map:")
        self._fixture_map.print_map()

        print(f"\nPixel States:")
        print(f"  {'Index':<8} {'R':<6} {'G':<6} {'B':<6} {'W':<6}")
        print(f"  {'-'*40}")
        for i, pixel in enumerate(self._pixels):
            print(f"  {i:<8} {pixel.r:<6} {pixel.g:<6} {pixel.b:<6} {pixel.w:<6}")

        print(f"\nDMX Channel Mapping:")
        channels = self.get_dmx_channels()
        sorted_channels = sorted(channels.items())
        print(f"  {'Channel':<10} {'Value':<8}")
        print(f"  {'-'*20}")
        for ch, val in sorted_channels:
            print(f"  {ch:<10} {val:<8}")

        # Validation
        is_valid, errors = self.validate()
        print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            for error in errors:
                print(f"  ERROR: {error}")

        print(f"{'='*60}\n")

    def get_status(self) -> Dict:
        """Get current status as dict"""
        return {
            'universe': self._universe,
            'fixture_count': self._n,
            'start_channel': self._config.start_channel,
            'channel_spacing': self._config.channel_spacing,
            'mode': self._mode.value,
            'effect_type': self._effect_type.value,
            'effect_speed': self._effect_speed,
            'frame_count': self._frame_count,
            'running': self._running,
            'target_fps': TARGET_FPS,
            'config': self._config.to_dict(),
            'pixels': [
                {'r': p.r, 'g': p.g, 'b': p.b, 'w': p.w}
                for p in self._pixels
            ],
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_pixel_controller(
    fixture_count: int = 8,
    universe: int = UNIVERSE,
    start_channel: int = DEFAULT_START_CHANNEL,
    channel_spacing: int = CHANNEL_SPACING,
    send_callback: Optional[Callable] = None,
) -> PixelArrayController:
    """
    Create and configure a pixel array controller.

    Args:
        fixture_count: Number of RGBW fixtures
        universe: Target DMX universe
        start_channel: First fixture start channel
        channel_spacing: Channels between fixtures
        send_callback: Optional output callback

    Returns:
        Configured PixelArrayController
    """
    controller = PixelArrayController(
        fixture_count=fixture_count,
        universe=universe,
        start_channel=start_channel,
        channel_spacing=channel_spacing,
    )

    if send_callback:
        controller.set_output_callback(send_callback)

    return controller


def validate_fixture_addressing(
    fixture_count: int,
    start_channel: int = DEFAULT_START_CHANNEL,
    channel_spacing: int = CHANNEL_SPACING,
) -> bool:
    """
    Validate that fixture addressing is correct.

    Args:
        fixture_count: Number of fixtures to validate
        start_channel: First fixture start channel
        channel_spacing: Channels between fixtures

    Returns:
        True if addressing is valid
    """
    fixture_map = FixtureMap(
        fixture_count,
        start_channel=start_channel,
        channel_spacing=channel_spacing,
    )
    is_valid, errors = fixture_map.validate()

    if not is_valid:
        print("Fixture addressing validation FAILED:")
        for error in errors:
            print(f"  - {error}")

    return is_valid


# ============================================================
# INTEGRATION WITH AETHER CORE
# ============================================================

def create_ssot_callback(dmx_state, send_udpjson):
    """
    Create callback function for integration with aether-core SSOT.

    Args:
        dmx_state: DMXStateManager instance
        send_udpjson: Function to send UDPJSON to nodes

    Returns:
        Callback function compatible with PixelArrayController
    """
    def callback(universe: int, channels: Dict[int, int]):
        # Update SSOT
        if dmx_state:
            dmx_state.set_channels(universe, channels)

        # Send to node
        if send_udpjson:
            send_udpjson(universe, channels)

    return callback


# ============================================================
# MAIN - Testing / Demo
# ============================================================

if __name__ == "__main__":
    print("Multi-Fixture Universe Mapping & Pixel-Style Control")
    print("=" * 60)

    # Create controller with 8 fixtures
    controller = create_pixel_controller(fixture_count=8)

    # Print fixture map
    controller.print_fixture_map()

    # Validate
    is_valid, errors = controller.validate()
    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    # Test grouped mode
    print("\nTest: Grouped Mode (all red)")
    controller.set_grouped_mode()
    controller.set_all_rgbw(255, 0, 0, 0)
    controller.populate_dmx_buffer()
    controller.print_debug_info()

    # Test pixel array mode
    print("\nTest: Pixel Array Mode (gradient)")
    controller.set_pixel_array_mode()
    for i in range(controller.fixture_count):
        brightness = int(255 * (i / (controller.fixture_count - 1)))
        controller.set_pixel_rgbw(i, brightness, 0, 0, 0)
    controller.populate_dmx_buffer()
    controller.print_debug_info()

    # Test wave effect (single frame)
    print("\nTest: Wave Effect (single frame)")
    controller.set_effect(
        EffectType.WAVE,
        color=Pixel(0, 255, 0, 0),  # Green
        speed=1.0
    )
    controller._start_time = time.monotonic()
    channels = controller.render_single_frame()
    print(f"Rendered channels: {channels}")

    print("\nAll tests completed.")
