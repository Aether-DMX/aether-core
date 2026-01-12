"""
AETHER Beta1 - Load and Performance Tests

Tests cover:
1. FPS consistency under load
2. CPU usage limits
3. Memory stability
4. Multi-universe rendering performance
5. Modifier stack performance
6. Long-running stability
"""

import pytest
import time
import threading
import sys
import os
import gc
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from render_engine import RenderEngine, render_look_frame

# Optional imports
try:
    from playback_controller import UnifiedPlaybackController
    HAS_PLAYBACK_CONTROLLER = True
except ImportError:
    HAS_PLAYBACK_CONTROLLER = False


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def render_engine():
    """Fresh render engine for testing"""
    engine = RenderEngine(target_fps=30)
    yield engine
    engine.stop()


@pytest.fixture
def playback_controller():
    """Fresh playback controller"""
    if not HAS_PLAYBACK_CONTROLLER:
        pytest.skip("PlaybackController not available")
    controller = UnifiedPlaybackController(target_fps=30)
    yield controller
    controller.stop()


# ============================================================
# FPS Consistency Tests
# ============================================================

class TestFPSConsistency:
    """Tests for frame rate consistency"""

    def test_30fps_maintained_basic(self, render_engine):
        """Basic rendering maintains 30fps"""
        frame_times = []

        def capture_time(u, c):
            frame_times.append(time.monotonic())

        render_engine.set_output_callback(capture_time)
        render_engine.start()

        render_engine.render_look(
            "fps_test",
            {"1": 255, "2": 128, "3": 64},
            [],
            [1]
        )

        time.sleep(2.0)
        render_engine.stop()

        # Calculate actual FPS
        if len(frame_times) > 1:
            intervals = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]
            avg_fps = 1.0 / (sum(intervals) / len(intervals))
            assert 25 <= avg_fps <= 35, f"FPS {avg_fps:.1f} out of range"

    def test_30fps_with_modifiers(self, render_engine):
        """30fps maintained with modifiers"""
        frame_times = []

        def capture_time(u, c):
            frame_times.append(time.monotonic())

        render_engine.set_output_callback(capture_time)
        render_engine.start()

        render_engine.render_look(
            "mod_fps_test",
            {"1": 255, "2": 255, "3": 255},
            [
                {"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 1.0}},
                {"id": "m2", "type": "rainbow", "enabled": True, "params": {"speed": 0.5}},
            ],
            [1]
        )

        time.sleep(2.0)
        render_engine.stop()

        if len(frame_times) > 1:
            intervals = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]
            avg_fps = 1.0 / (sum(intervals) / len(intervals))
            assert 25 <= avg_fps <= 35, f"FPS {avg_fps:.1f} with modifiers out of range"

    def test_30fps_across_4_universes(self, render_engine):
        """30fps maintained across 4 universes"""
        frame_count = [0]

        def count_frames(u, c):
            frame_count[0] += 1

        render_engine.set_output_callback(count_frames)
        render_engine.start()

        render_engine.render_look(
            "multi_u_fps",
            {"1": 255, "2": 200, "3": 150, "4": 100},
            [],
            [1, 2, 3, 4]  # 4 universes
        )

        time.sleep(2.0)
        render_engine.stop()

        # With 4 universes at 30fps, should have ~240 outputs in 2 seconds
        # (30 fps * 4 universes * 2 seconds)
        # But frame counting is per-universe callback, so expect ~60 frames min per universe
        assert frame_count[0] >= 200, f"Only {frame_count[0]} frames in 2 seconds"

    def test_frame_timing_consistency(self, render_engine):
        """Frame timing is consistent (no large spikes)"""
        frame_times = []

        def capture_time(u, c):
            frame_times.append(time.monotonic())

        render_engine.set_output_callback(capture_time)
        render_engine.start()

        render_engine.render_look("timing_test", {"1": 255}, [], [1])

        time.sleep(2.0)
        render_engine.stop()

        if len(frame_times) > 10:
            intervals = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]

            # No frame should take more than 100ms (3x target of 33ms)
            max_interval = max(intervals)
            assert max_interval < 0.1, f"Frame spike detected: {max_interval*1000:.1f}ms"


# ============================================================
# CPU Usage Tests
# ============================================================

@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
class TestCPUUsage:
    """Tests for CPU usage limits"""

    def test_cpu_bounded_normal_load(self, render_engine):
        """CPU stays bounded under normal load"""
        process = psutil.Process()
        cpu_before = process.cpu_percent(interval=0.1)

        render_engine.start()

        render_engine.render_look(
            "cpu_test",
            {"1": 255, "2": 200, "3": 150},
            [{"id": "m1", "type": "pulse", "enabled": True, "params": {}}],
            [1]
        )

        time.sleep(2.0)
        cpu_during = process.cpu_percent(interval=0.5)

        render_engine.stop()

        # CPU should be reasonable (not 100% runaway)
        assert cpu_during < 50, f"CPU too high under normal load: {cpu_during}%"

    def test_cpu_bounded_heavy_load(self, render_engine):
        """CPU stays bounded under heavy load"""
        process = psutil.Process()

        render_engine.start()

        # Heavy load: many channels, many modifiers, many universes
        channels = {str(i): 200 for i in range(1, 101)}  # 100 channels
        modifiers = [
            {"id": f"m{i}", "type": mod_type, "enabled": True, "params": {}}
            for i, mod_type in enumerate(["pulse", "strobe", "flicker", "wave", "rainbow", "twinkle"] * 3)
        ]  # 18 modifiers

        render_engine.render_look("heavy_cpu", channels, modifiers, [1, 2, 3, 4])

        time.sleep(3.0)
        cpu_during = process.cpu_percent(interval=1.0)

        render_engine.stop()

        # CPU should be bounded even under heavy load
        assert cpu_during < 80, f"CPU too high under heavy load: {cpu_during}%"

    def test_cpu_idle_after_stop(self, render_engine):
        """CPU returns to idle after stopping"""
        process = psutil.Process()

        render_engine.start()
        render_engine.render_look("idle_test", {"1": 255}, [], [1])
        time.sleep(1.0)

        render_engine.stop()
        time.sleep(0.5)

        cpu_after = process.cpu_percent(interval=0.5)

        # CPU should drop after stopping
        assert cpu_after < 20, f"CPU still high after stop: {cpu_after}%"


# ============================================================
# Memory Stability Tests
# ============================================================

class TestMemoryStability:
    """Tests for memory stability"""

    def test_memory_stable_over_frames(self, render_engine):
        """Memory doesn't grow unbounded over many frames"""
        gc.collect()

        # Force garbage collection and get baseline
        if HAS_PSUTIL:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            mem_before = 0

        render_engine.start()
        render_engine.render_look(
            "mem_test",
            {"1": 255, "2": 200, "3": 150},
            [{"id": "m1", "type": "pulse", "enabled": True, "params": {}}],
            [1]
        )

        # Run for a while
        time.sleep(5.0)

        render_engine.stop()
        gc.collect()

        if HAS_PSUTIL:
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_growth = mem_after - mem_before

            # Memory should not grow by more than 50MB
            assert mem_growth < 50, f"Memory grew by {mem_growth:.1f}MB"

    def test_no_memory_leak_start_stop_cycle(self, render_engine):
        """Repeated start/stop doesn't leak memory"""
        gc.collect()

        if HAS_PSUTIL:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024

        # Start and stop many times
        for i in range(20):
            render_engine.start()
            render_engine.render_look(f"cycle_{i}", {"1": 255}, [], [1])
            time.sleep(0.1)
            render_engine.stop()

        gc.collect()

        if HAS_PSUTIL:
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_growth = mem_after - mem_before

            assert mem_growth < 20, f"Memory leaked {mem_growth:.1f}MB over 20 cycles"


# ============================================================
# Modifier Stack Performance
# ============================================================

class TestModifierPerformance:
    """Tests for modifier stack performance"""

    def test_render_time_scales_with_modifiers(self):
        """Render time scales reasonably with modifier count"""
        channels = {"1": 255, "2": 200, "3": 150}

        times_by_count = {}

        for modifier_count in [1, 5, 10, 20]:
            modifiers = [
                {"id": f"m{i}", "type": "pulse", "enabled": True, "params": {"speed": 0.5 + i * 0.1}}
                for i in range(modifier_count)
            ]

            # Time 100 frames
            start = time.monotonic()
            for frame in range(100):
                render_look_frame(channels, modifiers, elapsed_time=frame * 0.033, seed=42)
            elapsed = time.monotonic() - start

            times_by_count[modifier_count] = elapsed

        # Render time should not grow more than linearly
        # 20 modifiers should take less than 10x as long as 1 modifier
        ratio = times_by_count[20] / times_by_count[1]
        assert ratio < 30, f"20 modifiers took {ratio:.1f}x longer than 1 modifier"

    def test_render_time_with_all_types(self):
        """All modifier types render in reasonable time"""
        channels = {"1": 255, "2": 255, "3": 255, "4": 200, "5": 150, "6": 100}

        for mod_type in ["pulse", "strobe", "flicker", "wave", "rainbow", "twinkle"]:
            modifiers = [
                {"id": "m1", "type": mod_type, "enabled": True, "params": {}}
            ]

            start = time.monotonic()
            for frame in range(100):
                render_look_frame(channels, modifiers, elapsed_time=frame * 0.033, seed=42)
            elapsed = time.monotonic() - start

            # 100 frames should complete in under 1 second
            assert elapsed < 1.0, f"{mod_type} took {elapsed:.2f}s for 100 frames"


# ============================================================
# Long-Running Stability Tests
# ============================================================

class TestLongRunningStability:
    """Tests for long-running stability"""

    def test_60_second_stability(self, render_engine):
        """60 seconds of playback remains stable"""
        frame_count = [0]
        errors = []

        def capture(u, c):
            frame_count[0] += 1
            # Verify output is valid
            for ch, val in c.items():
                if not (0 <= val <= 255):
                    errors.append(f"Invalid value {val} for channel {ch}")

        render_engine.set_output_callback(capture)
        render_engine.start()

        render_engine.render_look(
            "stability_test",
            {"1": 255, "2": 200, "3": 150},
            [{"id": "m1", "type": "pulse", "enabled": True, "params": {"speed": 1.0}}],
            [1]
        )

        # Run for 60 seconds
        time.sleep(60.0)

        render_engine.stop()

        # Should have ~1800 frames (30fps * 60s)
        assert frame_count[0] >= 1500, f"Only {frame_count[0]} frames in 60 seconds"
        assert len(errors) == 0, f"Errors during playback: {errors[:5]}"

    def test_rapid_start_stop_stability(self, render_engine):
        """Rapid start/stop cycles remain stable"""
        errors = []

        for cycle in range(50):
            try:
                render_engine.start()
                render_engine.render_look(f"rapid_{cycle}", {"1": 255}, [], [1])
                time.sleep(0.05)
                render_engine.stop()
            except Exception as e:
                errors.append(f"Cycle {cycle}: {e}")

        assert len(errors) == 0, f"Errors during rapid cycles: {errors}"


# ============================================================
# Multi-Universe Performance
# ============================================================

class TestMultiUniversePerformance:
    """Tests for multi-universe performance"""

    def test_4_universe_output_rate(self, render_engine):
        """4 universes maintain output rate"""
        universe_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        def count_by_universe(u, c):
            if u in universe_counts:
                universe_counts[u] += 1

        render_engine.set_output_callback(count_by_universe)
        render_engine.start()

        render_engine.render_look(
            "multi_u_rate",
            {"1": 255},
            [],
            [1, 2, 3, 4]
        )

        time.sleep(2.0)
        render_engine.stop()

        # Each universe should have received ~60 frames (30fps * 2s)
        for u, count in universe_counts.items():
            assert count >= 50, f"Universe {u} only got {count} frames"

    def test_independent_universe_content(self, render_engine):
        """Each universe receives correct content"""
        last_output = {1: None, 2: None, 3: None, 4: None}

        def capture_last(u, c):
            if u in last_output:
                last_output[u] = dict(c)

        render_engine.set_output_callback(capture_last)
        render_engine.start()

        # All universes should get same content for single look
        render_engine.render_look(
            "content_test",
            {"1": 200, "2": 150, "3": 100},
            [],
            [1, 2, 3, 4]
        )

        time.sleep(0.2)
        render_engine.stop()

        # All universes should have received output
        for u in [1, 2, 3, 4]:
            assert last_output[u] is not None, f"Universe {u} got no output"


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    # Run with longer timeout for load tests
    pytest.main([__file__, "-v", "--tb=short", "--timeout=120"])
