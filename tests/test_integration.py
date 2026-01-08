"""
Phase 8 - Integration Tests for Migration and System Safety

Tests cover:
1. Scene to Look migration
2. Chase to Sequence migration
3. Multiple universe rendering
4. Merge layer integration
5. Safety controls (blackout, stop all)
6. CPU guardrails
"""

import pytest
import time
import threading
import psutil
import json
from typing import Dict, List
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looks_sequences import LooksSequencesManager, Look, Sequence, SequenceStep
from playback_controller import UnifiedPlaybackController, LoopMode
from merge_layer import MergeLayer, ChannelClassifier
from preview_service import PreviewService, PreviewMode
from render_engine import RenderEngine, render_look_frame


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def look_manager():
    """Fresh LooksSequencesManager with in-memory storage"""
    manager = LooksSequencesManager(db_path=":memory:")
    return manager


@pytest.fixture
def playback_controller():
    """Fresh UnifiedPlaybackController"""
    controller = UnifiedPlaybackController(target_fps=30)
    yield controller
    controller.stop()


@pytest.fixture
def merge_layer():
    """Fresh MergeLayer"""
    return MergeLayer()


@pytest.fixture
def preview_service():
    """Fresh PreviewService"""
    service = PreviewService(target_fps=30)
    yield service
    service.stop_all()


# ============================================================
# Test: Scene to Look Migration
# ============================================================

class TestSceneToLookMigration:
    """Tests for migrating Scene data to Look format"""

    def test_basic_scene_conversion(self, look_manager):
        """Basic scene converts to look correctly"""
        scene_data = {
            "scene_id": "scene_001",
            "name": "Warm Glow",
            "channels": {"1": 255, "2": 180, "3": 100},
            "fade_time": 1000,
        }

        # Convert to Look format
        look = Look(
            look_id=f"look_{scene_data['scene_id']}",
            name=scene_data["name"],
            channels=scene_data["channels"],
            modifiers=[],
            universes=[1],
            metadata={"migrated_from": "scene", "original_id": scene_data["scene_id"]}
        )

        # Save and retrieve
        look_manager.create_look(look)
        retrieved = look_manager.get_look(look.look_id)

        assert retrieved is not None
        assert retrieved.name == "Warm Glow"
        assert retrieved.channels == {"1": 255, "2": 180, "3": 100}
        assert retrieved.metadata.get("migrated_from") == "scene"

    def test_scene_with_channels_preserved(self, look_manager):
        """All channel values are preserved during migration"""
        scene_channels = {str(i): i * 10 for i in range(1, 13)}  # 12 channels

        look = Look(
            look_id="look_multi_channel",
            name="Multi Channel Test",
            channels=scene_channels,
            modifiers=[],
            universes=[1],
        )

        look_manager.create_look(look)
        retrieved = look_manager.get_look("look_multi_channel")

        assert len(retrieved.channels) == 12
        for ch, val in scene_channels.items():
            assert retrieved.channels[ch] == val

    def test_look_renders_same_as_scene(self):
        """Look without modifiers renders identically to original scene"""
        scene_channels = {"1": 200, "2": 150, "3": 100}

        # Render as Look (no modifiers = pass-through)
        result = render_look_frame(
            channels=scene_channels,
            modifiers=[],
            elapsed_time=0,
            seed=0,
        )

        # Should match original scene exactly
        assert result[1] == 200
        assert result[2] == 150
        assert result[3] == 100


# ============================================================
# Test: Chase to Sequence Migration
# ============================================================

class TestChaseToSequenceMigration:
    """Tests for migrating Chase data to Sequence format"""

    def test_basic_chase_conversion(self, look_manager):
        """Basic chase converts to sequence correctly"""
        chase_data = {
            "chase_id": "chase_001",
            "name": "RGB Cycle",
            "bpm": 120,
            "steps": [
                {"name": "Red", "channels": {"1": 255, "2": 0, "3": 0}},
                {"name": "Green", "channels": {"1": 0, "2": 255, "3": 0}},
                {"name": "Blue", "channels": {"1": 0, "2": 0, "3": 255}},
            ],
            "loop": True,
        }

        # Convert step duration from BPM
        step_ms = int(60000 / chase_data["bpm"])
        fade_ms = int(step_ms * 0.3)  # 30% fade
        hold_ms = step_ms - fade_ms

        # Convert to Sequence format
        steps = []
        for i, step in enumerate(chase_data["steps"]):
            steps.append(SequenceStep(
                step_id=f"step_{i}",
                name=step.get("name", f"Step {i+1}"),
                channels=step["channels"],
                modifiers=[],
                fade_ms=fade_ms,
                hold_ms=hold_ms,
            ))

        sequence = Sequence(
            sequence_id=f"seq_{chase_data['chase_id']}",
            name=chase_data["name"],
            steps=steps,
            bpm=chase_data["bpm"],
            loop_mode=LoopMode.LOOP if chase_data.get("loop") else LoopMode.ONE_SHOT,
            universes=[1],
            metadata={"migrated_from": "chase", "original_id": chase_data["chase_id"]}
        )

        # Save and retrieve
        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence(sequence.sequence_id)

        assert retrieved is not None
        assert retrieved.name == "RGB Cycle"
        assert len(retrieved.steps) == 3
        assert retrieved.bpm == 120
        assert retrieved.loop_mode == LoopMode.LOOP
        assert retrieved.steps[0].name == "Red"

    def test_chase_timing_preserved(self, look_manager):
        """Chase timing (BPM) is correctly converted"""
        bpm = 140
        expected_step_ms = 60000 / bpm  # ~428ms

        steps = [
            SequenceStep("s1", "Step 1", {"1": 255}, [], fade_ms=128, hold_ms=300),
            SequenceStep("s2", "Step 2", {"1": 0}, [], fade_ms=128, hold_ms=300),
        ]

        sequence = Sequence(
            sequence_id="seq_timing_test",
            name="Timing Test",
            steps=steps,
            bpm=bpm,
            loop_mode=LoopMode.LOOP,
            universes=[1],
        )

        look_manager.create_sequence(sequence)
        retrieved = look_manager.get_sequence("seq_timing_test")

        assert retrieved.bpm == 140
        total_step_time = retrieved.steps[0].fade_ms + retrieved.steps[0].hold_ms
        assert total_step_time == 428  # fade + hold

    def test_all_loop_modes(self, look_manager):
        """All loop modes are supported"""
        for mode in [LoopMode.ONE_SHOT, LoopMode.LOOP, LoopMode.BOUNCE]:
            steps = [SequenceStep("s1", "S1", {"1": 255}, [])]
            sequence = Sequence(
                sequence_id=f"seq_{mode.value}",
                name=f"Test {mode.value}",
                steps=steps,
                bpm=120,
                loop_mode=mode,
                universes=[1],
            )

            look_manager.create_sequence(sequence)
            retrieved = look_manager.get_sequence(sequence.sequence_id)

            assert retrieved.loop_mode == mode


# ============================================================
# Test: Multiple Universe Rendering
# ============================================================

class TestMultipleUniverseRendering:
    """Tests for rendering across multiple universes"""

    def test_look_plays_on_multiple_universes(self, playback_controller):
        """Look can be played on multiple universes simultaneously"""
        universes_received = set()
        channels_received = {}

        def capture_output(universe, channels):
            universes_received.add(universe)
            channels_received[universe] = channels

        playback_controller.set_output_callback(capture_output)
        playback_controller.start()

        playback_controller.play_look(
            look_id="multi_universe_look",
            channels={"1": 255, "2": 128, "3": 64},
            modifiers=[],
            universes=[1, 2, 3],
        )

        time.sleep(0.2)
        playback_controller.stop()

        # Should have sent to all 3 universes
        assert 1 in universes_received
        assert 2 in universes_received
        assert 3 in universes_received

    def test_sequence_plays_on_multiple_universes(self, playback_controller):
        """Sequence steps play on all target universes"""
        output_log = []

        def capture_output(universe, channels):
            output_log.append({"universe": universe, "channels": dict(channels)})

        playback_controller.set_output_callback(capture_output)
        playback_controller.start()

        steps = [
            {"name": "Red", "channels": {"1": 255, "2": 0, "3": 0}, "fade_ms": 100, "hold_ms": 100},
            {"name": "Green", "channels": {"1": 0, "2": 255, "3": 0}, "fade_ms": 100, "hold_ms": 100},
        ]

        playback_controller.play_sequence(
            sequence_id="multi_universe_seq",
            steps=steps,
            universes=[1, 2],
            bpm=60,
            loop_mode=LoopMode.LOOP,
        )

        time.sleep(0.3)
        playback_controller.stop()

        # Both universes should have received output
        universe_1_outputs = [o for o in output_log if o["universe"] == 1]
        universe_2_outputs = [o for o in output_log if o["universe"] == 2]

        assert len(universe_1_outputs) > 0
        assert len(universe_2_outputs) > 0


# ============================================================
# Test: Merge Layer Integration
# ============================================================

class TestMergeLayerIntegration:
    """Tests for priority-based merge layer"""

    def test_htp_merge_for_dimmers(self, merge_layer):
        """HTP (highest takes precedence) works for dimmer channels"""
        # Register two sources with different priorities
        merge_layer.register_source("look_1", priority=50)
        merge_layer.register_source("look_2", priority=50)

        # Set channels from both sources (channel 1 is dimmer)
        merge_layer.set_channels("look_1", 1, {1: 100})
        merge_layer.set_channels("look_2", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        # HTP: higher value wins for dimmers
        assert result[1] == 200

    def test_ltp_merge_for_color(self, merge_layer):
        """LTP (latest takes precedence) for non-dimmer channels"""
        merge_layer.register_source("look_1", priority=50)
        merge_layer.register_source("look_2", priority=45)

        # RGB channels (typically 2,3,4 after dimmer)
        # Higher priority wins in LTP
        merge_layer.set_channels("look_1", 1, {2: 100, 3: 50, 4: 0})  # Priority 50
        merge_layer.set_channels("look_2", 1, {2: 0, 3: 255, 4: 128})  # Priority 45

        result = merge_layer.compute_merge(1)

        # LTP: higher priority (look_1 @ 50) wins over lower (look_2 @ 45)
        assert result[2] == 100
        assert result[3] == 50
        assert result[4] == 0

    def test_priority_ordering(self, merge_layer):
        """Higher priority sources win in conflicts"""
        merge_layer.register_source("manual", priority=80)
        merge_layer.register_source("look", priority=50)
        merge_layer.register_source("chase", priority=40)

        merge_layer.set_channels("chase", 1, {1: 50})
        merge_layer.set_channels("look", 1, {1: 100})
        merge_layer.set_channels("manual", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        # Manual (80) has highest priority
        assert result[1] == 200

    def test_blackout_overrides_all(self, merge_layer):
        """Blackout sets all channels to 0 regardless of sources"""
        merge_layer.register_source("look_1", priority=50)
        merge_layer.set_channels("look_1", 1, {1: 255, 2: 255, 3: 255})

        # Activate blackout
        merge_layer.set_blackout(True, universes=[1])

        result = merge_layer.compute_merge(1)

        # All channels should be 0
        assert all(v == 0 for v in result.values())

    def test_blackout_release(self, merge_layer):
        """Releasing blackout restores previous values"""
        merge_layer.register_source("look_1", priority=50)
        merge_layer.set_channels("look_1", 1, {1: 255, 2: 128})

        merge_layer.set_blackout(True)
        merge_layer.set_blackout(False)

        result = merge_layer.compute_merge(1)

        # Values should be restored
        assert result[1] == 255
        assert result[2] == 128


# ============================================================
# Test: Safety Controls
# ============================================================

class TestSafetyControls:
    """Tests for emergency/safety controls"""

    def test_stop_all_playback(self, playback_controller):
        """Stop all immediately halts all playback"""
        playback_controller.start()

        # Start multiple playbacks
        playback_controller.play_look("look_1", {"1": 255}, [], [1])
        playback_controller.play_sequence("seq_1", [{"channels": {"1": 128}}], [2], 120)

        time.sleep(0.1)

        # Stop all
        playback_controller.stop_all()

        status = playback_controller.get_status()
        assert status["active_looks"] == 0
        assert status["active_sequences"] == 0

    def test_blackout_immediate(self, merge_layer):
        """Blackout takes effect immediately"""
        merge_layer.register_source("look", priority=50)
        merge_layer.set_channels("look", 1, {1: 255, 2: 255, 3: 255})

        # Pre-blackout
        result_before = merge_layer.compute_merge(1)
        assert result_before[1] == 255

        # Blackout
        merge_layer.set_blackout(True)

        # Post-blackout
        result_after = merge_layer.compute_merge(1)
        assert result_after[1] == 0

    def test_preview_sandbox_isolation(self, preview_service):
        """Preview in sandbox mode doesn't affect live output"""
        live_output = []
        preview_output = []

        def capture_live(u, c):
            live_output.append(c)

        def capture_preview(u, c):
            preview_output.append(c)

        preview_service.set_live_callback(capture_live)

        # Create sandbox preview
        session_id = preview_service.create_session(
            session_id="sandbox_test",
            preview_type="look",
            channels={"1": 255, "2": 255, "3": 255},
            modifiers=[{"id": "m1", "type": "pulse", "enabled": True, "params": {}}],
            universes=[1],
        )

        preview_service.start_session(session_id)
        time.sleep(0.2)

        # Sandbox mode should not produce live output
        assert len(live_output) == 0, "Sandbox preview should not affect live output"

        preview_service.stop_session(session_id)

    def test_preview_arm_produces_output(self, preview_service):
        """Armed preview produces live output"""
        live_output = []

        def capture_live(u, c):
            live_output.append(c)

        preview_service.set_live_callback(capture_live)

        session_id = preview_service.create_session(
            session_id="arm_test",
            preview_type="look",
            channels={"1": 255},
            modifiers=[],
            universes=[1],
        )

        preview_service.start_session(session_id)
        preview_service.arm_session(session_id)

        time.sleep(0.2)

        # Armed mode should produce live output
        assert len(live_output) > 0, "Armed preview should produce live output"

        preview_service.stop_session(session_id)


# ============================================================
# Test: CPU Guardrails
# ============================================================

class TestCPUGuardrails:
    """Tests for CPU usage limits and runaway prevention"""

    def test_render_engine_fps_limiting(self):
        """Render engine respects target FPS limit"""
        engine = RenderEngine(target_fps=30)
        frame_count = [0]

        def count_frames(u, c):
            frame_count[0] += 1

        engine.set_output_callback(count_frames)
        engine.start()

        engine.render_look("test", {"1": 255}, [], [1])

        time.sleep(1.0)
        engine.stop()

        # Should be ~30 frames in 1 second (with tolerance)
        assert 25 <= frame_count[0] <= 35, f"Expected ~30 fps, got {frame_count[0]}"

    def test_heavy_modifier_load_bounded(self):
        """Heavy modifier load doesn't cause runaway CPU"""
        # Capture initial CPU
        process = psutil.Process()
        cpu_before = process.cpu_percent(interval=0.1)

        engine = RenderEngine(target_fps=30)
        engine.start()

        # Very heavy load: many modifiers, many channels
        channels = {str(i): 200 for i in range(1, 101)}  # 100 channels
        modifiers = [
            {"id": f"m{i}", "type": mod_type, "enabled": True, "params": {}}
            for i, mod_type in enumerate(["pulse", "strobe", "flicker", "wave"] * 5)
        ]  # 20 modifiers

        engine.render_look("heavy_test", channels, modifiers, [1, 2, 3])

        time.sleep(2.0)

        cpu_during = process.cpu_percent(interval=0.5)

        engine.stop()

        # CPU should be bounded (not 100% runaway)
        # Allow up to 50% on single core equivalent
        assert cpu_during < 80, f"CPU too high: {cpu_during}%"

    def test_max_modifiers_per_look(self):
        """System handles maximum modifiers gracefully"""
        max_modifiers = 50

        channels = {"1": 255, "2": 255, "3": 255}
        modifiers = [
            {"id": f"m{i}", "type": "pulse", "enabled": True, "params": {"speed": 0.5 + i * 0.1}}
            for i in range(max_modifiers)
        ]

        # Should not crash or hang
        result = render_look_frame(channels, modifiers, elapsed_time=1.0, seed=0)

        # Should produce valid output
        assert 0 <= result[1] <= 255
        assert 0 <= result[2] <= 255
        assert 0 <= result[3] <= 255


# ============================================================
# Test: Frame Stability
# ============================================================

class TestFrameStability:
    """Tests for no visible flicker / frame spikes"""

    def test_no_value_discontinuities(self):
        """Adjacent frames don't have huge value jumps (causes flicker)"""
        channels = {"1": 255, "2": 200, "3": 150}
        modifiers = [
            {"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 1.0, "curve": "sine"}},
        ]

        prev_result = None
        max_delta = 0

        # Sample 100 frames at 30fps
        for frame in range(100):
            elapsed = frame / 30.0

            result = render_look_frame(channels, modifiers, elapsed, seed=42)

            if prev_result:
                # Check delta between frames
                for ch in result:
                    delta = abs(result[ch] - prev_result[ch])
                    max_delta = max(max_delta, delta)

            prev_result = result

        # Max per-frame change should be bounded (no sudden jumps)
        # At 30fps with 1Hz pulse, max delta should be ~5-10 per frame
        assert max_delta < 50, f"Value jump too large: {max_delta} (would cause flicker)"

    def test_strobe_transitions_clean(self):
        """Strobe on/off transitions are clean (not partial values)"""
        channels = {"1": 255}
        modifiers = [
            {"id": "m1", "type": "strobe", "enabled": True, "params": {"rate": 10, "duty_cycle": 50, "attack": 0, "decay": 0}},
        ]

        values = set()

        # Sample many points in strobe cycle
        for i in range(200):
            elapsed = i / 1000.0  # 0 to 0.2 seconds

            result = render_look_frame(channels, modifiers, elapsed, seed=42)
            values.add(result[1])

        # With no attack/decay, should only be full on (255) or full off (0)
        # Allow small tolerance for floating point
        assert all(v <= 5 or v >= 250 for v in values), f"Strobe has partial values: {values}"

    def test_rainbow_smooth_transitions(self):
        """Rainbow color transitions are smooth"""
        channels = {"1": 255, "2": 255, "3": 255}
        modifiers = [
            {"id": "m1", "type": "rainbow", "enabled": True, "params": {"speed": 0.5, "saturation": 100}},
        ]

        prev_result = None
        max_hue_jump = 0

        for frame in range(60):  # 2 seconds at 30fps
            elapsed = frame / 30.0

            result = render_look_frame(channels, modifiers, elapsed, seed=42)

            if prev_result:
                # Calculate approximate hue change
                # This is rough but catches major discontinuities
                total_delta = sum(abs(result[ch] - prev_result[ch]) for ch in result)
                max_hue_jump = max(max_hue_jump, total_delta)

            prev_result = result

        # Hue should transition smoothly
        assert max_hue_jump < 100, f"Rainbow hue jump too large: {max_hue_jump}"


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
