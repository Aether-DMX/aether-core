"""
AETHER Beta1 - Extended Playback Tests

Tests cover Look and Sequence playback edge cases:
1. Look playback with various modifier combinations
2. Sequence playback with different loop modes
3. BPM timing accuracy
4. Multi-universe playback
5. Concurrent playback scenarios
6. State transitions (play, pause, resume, stop)
7. Fade transitions
"""

import pytest
import time
import threading
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looks_sequences import LooksSequencesManager, Look, Sequence, SequenceStep
from render_engine import RenderEngine, render_look_frame

# Optional imports
try:
    from playback_controller import UnifiedPlaybackController, LoopMode, PlaybackState
    HAS_PLAYBACK_CONTROLLER = True
except ImportError:
    HAS_PLAYBACK_CONTROLLER = False


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def look_manager(tmp_path):
    """Fresh LooksSequencesManager with temp database"""
    db_file = tmp_path / "test_playback.db"
    return LooksSequencesManager(db_path=str(db_file))


@pytest.fixture
def playback_controller():
    """Fresh playback controller"""
    if not HAS_PLAYBACK_CONTROLLER:
        pytest.skip("UnifiedPlaybackController not available")

    controller = UnifiedPlaybackController(target_fps=30)
    yield controller
    controller.stop()


@pytest.fixture
def render_engine():
    """Fresh render engine"""
    engine = RenderEngine(target_fps=30)
    yield engine
    engine.stop()


# ============================================================
# Look Playback Tests
# ============================================================

class TestLookPlaybackExtended:
    """Extended tests for Look playback"""

    def test_look_with_single_modifier(self):
        """Look with single modifier renders correctly"""
        channels = {"1": 255, "2": 200, "3": 150}
        modifiers = [
            {"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 1.0}}
        ]

        result = render_look_frame(channels, modifiers, elapsed_time=0.5, seed=42)

        # Pulse should modulate values
        assert 0 <= result[1] <= 255
        assert 0 <= result[2] <= 255
        assert 0 <= result[3] <= 255

    def test_look_with_multiple_modifiers(self):
        """Look with multiple modifiers stacks correctly"""
        channels = {"1": 255, "2": 255, "3": 255}
        modifiers = [
            {"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 0.5}},
            {"id": "m2", "type": "flicker", "enabled": True, "params": {"intensity": 20}},
        ]

        # Sample multiple frames to verify stability
        results = []
        for i in range(10):
            result = render_look_frame(channels, modifiers, elapsed_time=i * 0.1, seed=42)
            results.append(result)

        # All results should be valid
        for result in results:
            assert all(0 <= v <= 255 for v in result.values())

    def test_look_disabled_modifier_ignored(self):
        """Disabled modifiers don't affect output"""
        channels = {"1": 200, "2": 150, "3": 100}
        modifiers = [
            {"id": "m1", "type": "strobe", "enabled": False, "params": {"rate": 10}}
        ]

        result = render_look_frame(channels, modifiers, elapsed_time=0.5, seed=42)

        # Values should be unchanged since modifier is disabled
        assert result[1] == 200
        assert result[2] == 150
        assert result[3] == 100

    def test_look_empty_channels(self):
        """Look with empty channels renders as zeros"""
        channels = {}
        modifiers = []

        result = render_look_frame(channels, modifiers, elapsed_time=0, seed=0)

        # Should return empty or default
        assert isinstance(result, dict)

    def test_look_partial_channels(self):
        """Look with partial channels only affects specified channels"""
        channels = {"5": 128, "10": 200}
        modifiers = []

        result = render_look_frame(channels, modifiers, elapsed_time=0, seed=0)

        assert result.get(5, 0) == 128
        assert result.get(10, 0) == 200

    def test_look_channel_value_clamping(self):
        """Channel values are clamped to 0-255"""
        channels = {"1": 300, "2": -50}  # Invalid values
        modifiers = []

        result = render_look_frame(channels, modifiers, elapsed_time=0, seed=0)

        # Values should be clamped
        assert 0 <= result.get(1, 0) <= 255
        assert 0 <= result.get(2, 0) <= 255


# ============================================================
# Sequence Playback Tests
# ============================================================

class TestSequencePlaybackExtended:
    """Extended tests for Sequence playback"""

    def test_sequence_step_timing(self, look_manager):
        """Sequence steps have correct timing based on BPM"""
        bpm = 120  # 0.5 seconds per beat
        expected_step_ms = 60000 / bpm  # 500ms

        steps = [
            SequenceStep(step_id="s1", name="Red", channels={"1": 255}, modifiers=[], fade_ms=100, hold_ms=400),
            SequenceStep(step_id="s2", name="Green", channels={"2": 255}, modifiers=[], fade_ms=100, hold_ms=400),
        ]

        sequence = Sequence(
            sequence_id="timing_test",
            name="Timing Test",
            steps=steps,
            bpm=bpm,
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("timing_test")

        # Total time per step should be fade + hold = 500ms
        total_step_time = retrieved.steps[0].fade_ms + retrieved.steps[0].hold_ms
        assert total_step_time == 500

    def test_sequence_loop_true(self, look_manager):
        """Sequence with loop=true is stored correctly"""
        steps = [SequenceStep(step_id="s1", name="S1", channels={"1": 255}, modifiers=[])]
        sequence = Sequence(
            sequence_id="loop_true",
            name="Loop True",
            steps=steps,
            bpm=120,
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("loop_true")

        assert retrieved.loop == True

    def test_sequence_loop_false(self, look_manager):
        """Sequence with loop=false (one-shot) is stored correctly"""
        steps = [SequenceStep(step_id="s1", name="S1", channels={"1": 255}, modifiers=[])]
        sequence = Sequence(
            sequence_id="loop_false",
            name="One Shot",
            steps=steps,
            bpm=120,
            loop=False,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("loop_false")

        assert retrieved.loop == False

    def test_sequence_with_step_modifiers(self, look_manager):
        """Sequence steps can have per-step modifiers"""
        steps = [
            SequenceStep(
                step_id="s1",
                name="Strobe Step",
                channels={"1": 255},
                modifiers=[{"id": "m1", "type": "strobe", "enabled": True, "params": {"rate": 5}}],
                fade_ms=0,
                hold_ms=500,
            ),
            SequenceStep(
                step_id="s2",
                name="Pulse Step",
                channels={"1": 200},
                modifiers=[{"id": "m2", "type": "pulse", "enabled": True, "params": {"speed": 2}}],
                fade_ms=100,
                hold_ms=400,
            ),
        ]

        sequence = Sequence(
            sequence_id="step_modifiers",
            name="Step Modifiers",
            steps=steps,
            bpm=60,
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("step_modifiers")

        assert len(retrieved.steps[0].modifiers) == 1
        assert retrieved.steps[0].modifiers[0]["type"] == "strobe"
        assert len(retrieved.steps[1].modifiers) == 1
        assert retrieved.steps[1].modifiers[0]["type"] == "pulse"

    def test_sequence_bpm_range(self, look_manager):
        """Sequence BPM accepts valid range"""
        for bpm in [20, 60, 120, 180, 240]:
            steps = [SequenceStep(step_id="s1", name="S", channels={"1": 255}, modifiers=[])]
            sequence = Sequence(
                sequence_id=f"bpm_{bpm}",
                name=f"BPM {bpm}",
                steps=steps,
                bpm=bpm,
                loop=True,
            )

            look_manager.create_sequence(sequence)
            retrieved = look_manager.get_sequence(f"bpm_{bpm}")

            assert retrieved.bpm == bpm

    def test_sequence_empty_steps(self, look_manager):
        """Sequence with empty steps is valid"""
        sequence = Sequence(
            sequence_id="empty_steps",
            name="Empty Steps",
            steps=[],
            bpm=120,
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("empty_steps")

        assert len(retrieved.steps) == 0


# ============================================================
# Render Engine Tests
# ============================================================

class TestRenderEngineExtended:
    """Extended tests for render engine"""

    def test_render_engine_fps_target(self, render_engine):
        """Render engine maintains target FPS"""
        frame_count = [0]

        def count_frames(u, c):
            frame_count[0] += 1

        render_engine.set_output_callback(count_frames)
        render_engine.start()

        render_engine.render_look("fps_test", {"1": 255}, [], [1])

        time.sleep(1.0)
        render_engine.stop()

        # Should be ~30 frames in 1 second (with tolerance)
        assert 25 <= frame_count[0] <= 35, f"Expected ~30 fps, got {frame_count[0]}"

    def test_render_engine_multiple_universes(self, render_engine):
        """Render engine outputs to multiple universes"""
        universes_seen = set()

        def track_universes(u, c):
            universes_seen.add(u)

        render_engine.set_output_callback(track_universes)
        render_engine.start()

        render_engine.render_look("multi_u", {"1": 255}, [], [1, 2, 3, 4])

        time.sleep(0.2)
        render_engine.stop()

        assert 1 in universes_seen
        assert 2 in universes_seen
        assert 3 in universes_seen
        assert 4 in universes_seen

    def test_render_engine_stop_clears_output(self, render_engine):
        """Stop clears rendering state"""
        render_engine.start()
        render_engine.render_look("stop_test", {"1": 255}, [], [1])
        time.sleep(0.1)

        render_engine.stop()

        # Engine should be stopped
        assert not render_engine._running


# ============================================================
# Playback Controller Tests (if available)
# ============================================================

@pytest.mark.skipif(not HAS_PLAYBACK_CONTROLLER, reason="PlaybackController not available")
class TestPlaybackControllerExtended:
    """Extended tests for playback controller"""

    def test_controller_start_stop(self, playback_controller):
        """Controller starts and stops cleanly"""
        playback_controller.start()
        assert playback_controller._running

        playback_controller.stop()
        # Give time to stop
        time.sleep(0.1)

    def test_play_look_returns_status(self, playback_controller):
        """Playing a look returns status"""
        output_received = []

        def capture(u, c):
            output_received.append({"universe": u, "channels": dict(c)})

        playback_controller.set_output_callback(capture)
        playback_controller.start()

        playback_controller.play_look("test_look", {"1": 255}, [], [1])

        time.sleep(0.2)
        playback_controller.stop()

        assert len(output_received) > 0

    def test_concurrent_looks_different_universes(self, playback_controller):
        """Multiple looks can play on different universes"""
        output_by_universe = {1: [], 2: []}

        def capture(u, c):
            if u in output_by_universe:
                output_by_universe[u].append(dict(c))

        playback_controller.set_output_callback(capture)
        playback_controller.start()

        # Play two different looks on different universes
        playback_controller.play_look("look_u1", {"1": 255}, [], [1])
        playback_controller.play_look("look_u2", {"1": 128}, [], [2])

        time.sleep(0.2)
        playback_controller.stop()

        # Both universes should have received output
        assert len(output_by_universe[1]) > 0
        assert len(output_by_universe[2]) > 0


# ============================================================
# Modifier Combination Tests
# ============================================================

class TestModifierCombinations:
    """Tests for various modifier combinations"""

    def test_pulse_rainbow_combination(self):
        """Pulse and rainbow modifiers combine"""
        channels = {"1": 255, "2": 255, "3": 255}
        modifiers = [
            {"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 1.0}},
            {"id": "m2", "type": "rainbow", "enabled": True, "params": {"speed": 0.5}},
        ]

        result = render_look_frame(channels, modifiers, elapsed_time=0.5, seed=42)

        # Should produce valid output
        assert all(0 <= v <= 255 for v in result.values())

    def test_strobe_flicker_combination(self):
        """Strobe and flicker modifiers combine"""
        channels = {"1": 255}
        modifiers = [
            {"id": "m1", "type": "strobe", "enabled": True, "params": {"rate": 5, "duty_cycle": 50}},
            {"id": "m2", "type": "flicker", "enabled": True, "params": {"intensity": 30}},
        ]

        # Sample multiple times
        for t in range(10):
            result = render_look_frame(channels, modifiers, elapsed_time=t * 0.05, seed=42)
            assert 0 <= result.get(1, 0) <= 255

    def test_wave_twinkle_combination(self):
        """Wave and twinkle modifiers combine"""
        channels = {str(i): 200 for i in range(1, 13)}  # 12 channels
        modifiers = [
            {"id": "m1", "type": "wave", "enabled": True, "params": {"speed": 1.0}},
            {"id": "m2", "type": "twinkle", "enabled": True, "params": {"density": 50}},
        ]

        result = render_look_frame(channels, modifiers, elapsed_time=0.5, seed=42)

        # All channels should be valid
        for ch in range(1, 13):
            assert 0 <= result.get(ch, 0) <= 255

    def test_all_modifiers_combined(self):
        """All 6 modifier types can combine"""
        channels = {"1": 255, "2": 255, "3": 255}
        modifiers = [
            {"id": "m1", "type": "pulse", "enabled": True, "params": {}},
            {"id": "m2", "type": "strobe", "enabled": True, "params": {"rate": 2}},
            {"id": "m3", "type": "flicker", "enabled": True, "params": {}},
            {"id": "m4", "type": "wave", "enabled": True, "params": {}},
            {"id": "m5", "type": "rainbow", "enabled": True, "params": {}},
            {"id": "m6", "type": "twinkle", "enabled": True, "params": {}},
        ]

        # Should not crash
        result = render_look_frame(channels, modifiers, elapsed_time=0.5, seed=42)

        assert all(0 <= v <= 255 for v in result.values())


# ============================================================
# Edge Cases
# ============================================================

class TestPlaybackEdgeCases:
    """Edge case tests for playback"""

    def test_zero_bpm_sequence(self, look_manager):
        """Sequence with very low BPM doesn't crash"""
        steps = [SequenceStep(step_id="s1", name="S", channels={"1": 255}, modifiers=[])]
        sequence = Sequence(
            sequence_id="low_bpm",
            name="Low BPM",
            steps=steps,
            bpm=1,  # Very slow
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("low_bpm")
        assert retrieved.bpm == 1

    def test_high_bpm_sequence(self, look_manager):
        """Sequence with high BPM doesn't crash"""
        steps = [SequenceStep(step_id="s1", name="S", channels={"1": 255}, modifiers=[])]
        sequence = Sequence(
            sequence_id="high_bpm",
            name="High BPM",
            steps=steps,
            bpm=300,  # Very fast
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("high_bpm")
        assert retrieved.bpm == 300

    def test_many_steps_sequence(self, look_manager):
        """Sequence with many steps handles correctly"""
        steps = [
            SequenceStep(step_id=f"s{i}", name=f"Step {i}", channels={"1": i % 256}, modifiers=[])
            for i in range(100)
        ]

        sequence = Sequence(
            sequence_id="many_steps",
            name="Many Steps",
            steps=steps,
            bpm=120,
            loop=True,
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("many_steps")
        assert len(retrieved.steps) == 100

    def test_render_at_negative_time(self):
        """Render at negative elapsed time doesn't crash"""
        channels = {"1": 255}
        modifiers = [{"id": "m1", "type": "pulse", "enabled": True, "params": {}}]

        # Should not crash
        result = render_look_frame(channels, modifiers, elapsed_time=-1.0, seed=42)
        assert isinstance(result, dict)

    def test_render_at_large_time(self):
        """Render at very large elapsed time doesn't overflow"""
        channels = {"1": 255}
        modifiers = [{"id": "m1", "type": "pulse", "enabled": True, "params": {}}]

        # Very large time (1 hour)
        result = render_look_frame(channels, modifiers, elapsed_time=3600.0, seed=42)
        assert all(0 <= v <= 255 for v in result.values())


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
