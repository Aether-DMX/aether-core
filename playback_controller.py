"""
Unified Playback Controller - One playback system to rule them all

This module provides:
- UnifiedPlaybackController: Single controller for Look and Sequence playback
- Fade blending between steps (base channels interpolate, modifiers apply after)
- BPM-based timing with drift compensation
- Loop modes: loop, bounce, one-shot

Architecture:
- Look playback: Infinite hold with real-time modifier rendering
- Sequence playback: Steps with fade/hold timing, modifiers per step
- All output goes through RenderEngine for modifier composition
- SSOT integration via output callback

Version: 1.0.0
"""

import math
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Loop Modes
# ============================================================

class LoopMode(Enum):
    ONE_SHOT = "one_shot"   # Play once and stop
    LOOP = "loop"           # Loop from end to start
    BOUNCE = "bounce"       # Ping-pong: forward then backward


# ============================================================
# Playback State
# ============================================================

class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FADING = "fading"  # During crossfade between steps


# ============================================================
# Playback Job - Represents a single playback session
# ============================================================

@dataclass
class PlaybackJob:
    """A single playback job (Look or Sequence)"""
    job_id: str
    job_type: str  # "look" or "sequence"
    universes: List[int]
    seed: int

    # For Look playback
    look_id: Optional[str] = None
    channels: Optional[Dict[str, int]] = None
    modifiers: Optional[List[Dict]] = None

    # For Sequence playback
    sequence_id: Optional[str] = None
    steps: Optional[List[Dict]] = None
    bpm: int = 120
    loop_mode: LoopMode = LoopMode.LOOP

    # Runtime state
    state: PlaybackState = PlaybackState.STOPPED
    current_step: int = 0
    direction: int = 1  # 1 = forward, -1 = backward (for bounce)
    loop_count: int = 0

    # Timing (monotonic clock-based)
    start_time: float = 0.0
    step_start_time: float = 0.0
    pause_time: float = 0.0
    paused_duration: float = 0.0

    # Fade state
    fade_start_channels: Optional[Dict[int, int]] = None
    fade_end_channels: Optional[Dict[int, int]] = None
    fade_duration_ms: int = 0
    fade_progress: float = 0.0  # 0.0 to 1.0


# ============================================================
# Unified Playback Controller
# ============================================================

class UnifiedPlaybackController:
    """
    Single playback system for Looks and Sequences.

    Features:
    - Look playback with infinite hold and real-time modifiers
    - Sequence playback with step timing and crossfade
    - BPM-based timing with drift compensation
    - Monotonic clock for no drift/desync
    """

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Current playback job
        self._job: Optional[PlaybackJob] = None
        self._job_lock = threading.Lock()

        # Output callback
        self._output_callback: Optional[Callable] = None

        # Modifier renderer (will be set from render_engine)
        self._modifier_renderer = None
        self._modifier_states: Dict[str, Any] = {}

        # Look resolver (to fetch Look data by ID for Sequence steps)
        self._look_resolver: Optional[Callable] = None

        # Stats
        self._frame_count = 0
        self._actual_fps = 0.0
        self._last_frame_time = 0.0

    def set_output_callback(self, callback: Callable[[int, Dict[int, int]], None]):
        """Set callback for sending rendered frames: callback(universe, channels)"""
        self._output_callback = callback

    def set_modifier_renderer(self, renderer):
        """Set the modifier renderer for applying effects"""
        self._modifier_renderer = renderer

    def set_look_resolver(self, resolver: Callable[[str], Optional[Dict]]):
        """Set function to resolve Look ID to Look data"""
        self._look_resolver = resolver

    # ─────────────────────────────────────────────────────────
    # Engine Control
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start the playback engine loop"""
        if self._running:
            return

        self._running = True
        self._stop_flag.clear()
        self._frame_count = 0

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        print(f"🎬 UnifiedPlaybackController started at {self.target_fps} FPS")

    def stop_engine(self):
        """Stop the playback engine loop entirely"""
        if not self._running:
            return

        self._stop_flag.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._job = None
        self._modifier_states.clear()
        print("⏹️ UnifiedPlaybackController stopped")

    # ─────────────────────────────────────────────────────────
    # Look Playback
    # ─────────────────────────────────────────────────────────

    def play_look(
        self,
        look_id: str,
        channels: Dict[str, int],
        modifiers: List[Dict],
        universes: List[int],
        seed: Optional[int] = None,
        fade_from: Optional[Dict[int, int]] = None,
        fade_ms: int = 0,
    ) -> Dict:
        """
        Play a Look with infinite hold.

        Args:
            look_id: Unique ID for this Look
            channels: Base channel values {"1": 255, "2": 128}
            modifiers: List of modifier configs
            universes: Target universes
            seed: Random seed for deterministic output
            fade_from: Optional starting channels for fade-in
            fade_ms: Fade duration for transition
        """
        if seed is None:
            seed = hash(look_id) & 0xFFFFFFFF

        # Convert channel keys to int
        int_channels = {int(k): v for k, v in channels.items()}

        # Create playback job
        job = PlaybackJob(
            job_id=f"look_{look_id}_{int(time.time() * 1000)}",
            job_type="look",
            universes=universes,
            seed=seed,
            look_id=look_id,
            channels=int_channels,
            modifiers=modifiers,
            state=PlaybackState.PLAYING,
            start_time=time.monotonic(),
            step_start_time=time.monotonic(),
        )

        # Handle fade-in from previous state
        if fade_from and fade_ms > 0:
            job.state = PlaybackState.FADING
            job.fade_start_channels = fade_from
            job.fade_end_channels = int_channels
            job.fade_duration_ms = fade_ms
            job.fade_progress = 0.0

        # Initialize modifier states
        self._init_modifier_states(modifiers, seed)

        with self._job_lock:
            self._job = job
            self._frame_count = 0

        # Ensure engine is running
        if not self._running:
            self.start()

        return {
            "success": True,
            "job_id": job.job_id,
            "look_id": look_id,
            "universes": universes,
            "modifier_count": len(modifiers),
        }

    # ─────────────────────────────────────────────────────────
    # Sequence Playback
    # ─────────────────────────────────────────────────────────

    def play_sequence(
        self,
        sequence_id: str,
        steps: List[Dict],
        universes: List[int],
        bpm: int = 120,
        loop_mode: LoopMode = LoopMode.LOOP,
        seed: Optional[int] = None,
        start_step: int = 0,
    ) -> Dict:
        """
        Play a Sequence with step timing.

        Args:
            sequence_id: Unique ID for this Sequence
            steps: List of step configs with channels, modifiers, fade_ms, hold_ms
            universes: Target universes
            bpm: Beats per minute (affects default timing)
            loop_mode: How to handle end of sequence
            seed: Random seed for deterministic output
            start_step: Starting step index
        """
        if not steps:
            return {"success": False, "error": "Sequence has no steps"}

        if seed is None:
            seed = hash(sequence_id) & 0xFFFFFFFF

        # Resolve Look references in steps
        resolved_steps = self._resolve_sequence_steps(steps)

        # Get first step's modifiers for initialization
        first_step = resolved_steps[start_step]
        first_modifiers = first_step.get("modifiers", [])

        # Create playback job
        job = PlaybackJob(
            job_id=f"seq_{sequence_id}_{int(time.time() * 1000)}",
            job_type="sequence",
            universes=universes,
            seed=seed,
            sequence_id=sequence_id,
            steps=resolved_steps,
            bpm=bpm,
            loop_mode=loop_mode,
            state=PlaybackState.PLAYING,
            current_step=start_step,
            direction=1,
            start_time=time.monotonic(),
            step_start_time=time.monotonic(),
        )

        # Initialize modifier states for first step
        self._init_modifier_states(first_modifiers, seed)

        with self._job_lock:
            self._job = job
            self._frame_count = 0

        # Ensure engine is running
        if not self._running:
            self.start()

        return {
            "success": True,
            "job_id": job.job_id,
            "sequence_id": sequence_id,
            "universes": universes,
            "step_count": len(resolved_steps),
            "bpm": bpm,
            "loop_mode": loop_mode.value,
        }

    # ─────────────────────────────────────────────────────────
    # Playback Control
    # ─────────────────────────────────────────────────────────

    def stop(self) -> Dict:
        """Stop current playback"""
        with self._job_lock:
            if self._job:
                job_id = self._job.job_id
                self._job.state = PlaybackState.STOPPED
                self._job = None
                self._modifier_states.clear()
                return {"success": True, "stopped": job_id}
        return {"success": True, "stopped": None}

    def pause(self) -> Dict:
        """Pause current playback"""
        with self._job_lock:
            if self._job and self._job.state == PlaybackState.PLAYING:
                self._job.state = PlaybackState.PAUSED
                self._job.pause_time = time.monotonic()
                return {"success": True, "paused": self._job.job_id}
        return {"success": False, "error": "Nothing playing to pause"}

    def resume(self) -> Dict:
        """Resume paused playback"""
        with self._job_lock:
            if self._job and self._job.state == PlaybackState.PAUSED:
                # Account for paused duration
                pause_duration = time.monotonic() - self._job.pause_time
                self._job.paused_duration += pause_duration
                self._job.state = PlaybackState.PLAYING
                return {"success": True, "resumed": self._job.job_id}
        return {"success": False, "error": "Nothing paused to resume"}

    def get_status(self) -> Dict:
        """Get current playback status"""
        with self._job_lock:
            job = self._job

        if not job:
            return {
                "running": self._running,
                "playing": False,
                "job_id": None,
                "job_type": None,
            }

        status = {
            "running": self._running,
            "playing": job.state in (PlaybackState.PLAYING, PlaybackState.FADING),
            "job_id": job.job_id,
            "job_type": job.job_type,
            "state": job.state.value,
            "universes": job.universes,
            "frame_count": self._frame_count,
            "actual_fps": round(self._actual_fps, 1),
        }

        if job.job_type == "look":
            status["look_id"] = job.look_id
            status["modifier_count"] = len(job.modifiers) if job.modifiers else 0
        elif job.job_type == "sequence":
            status["sequence_id"] = job.sequence_id
            status["current_step"] = job.current_step
            status["step_count"] = len(job.steps) if job.steps else 0
            status["loop_count"] = job.loop_count
            status["loop_mode"] = job.loop_mode.value
            status["bpm"] = job.bpm

        if job.state == PlaybackState.FADING:
            status["fade_progress"] = round(job.fade_progress, 2)

        return status

    # ─────────────────────────────────────────────────────────
    # Internal: Main Playback Loop
    # ─────────────────────────────────────────────────────────

    def _playback_loop(self):
        """Main playback loop - runs at target FPS"""
        loop_start = time.monotonic()

        while not self._stop_flag.is_set():
            frame_start = time.monotonic()

            # Get current job
            with self._job_lock:
                job = self._job

            if job and job.state in (PlaybackState.PLAYING, PlaybackState.FADING):
                try:
                    self._render_frame(job, frame_start)
                except Exception as e:
                    print(f"❌ Playback error: {e}")
                    import traceback
                    traceback.print_exc()

            # Frame timing
            self._frame_count += 1
            frame_end = time.monotonic()
            frame_duration = frame_end - frame_start

            # Calculate actual FPS
            total_elapsed = frame_end - loop_start
            if total_elapsed > 0:
                self._actual_fps = self._frame_count / total_elapsed

            # Sleep to maintain target FPS
            sleep_time = self.frame_interval - frame_duration
            if sleep_time > 0:
                self._stop_flag.wait(sleep_time)

            self._last_frame_time = frame_end

    def _render_frame(self, job: PlaybackJob, frame_time: float):
        """Render a single frame based on job type"""
        if job.job_type == "look":
            self._render_look_frame(job, frame_time)
        elif job.job_type == "sequence":
            self._render_sequence_frame(job, frame_time)

    # ─────────────────────────────────────────────────────────
    # Internal: Look Rendering
    # ─────────────────────────────────────────────────────────

    def _render_look_frame(self, job: PlaybackJob, frame_time: float):
        """Render a Look frame with modifiers"""
        # Calculate elapsed time (accounting for pauses)
        elapsed = frame_time - job.start_time - job.paused_duration

        # Handle fade-in
        if job.state == PlaybackState.FADING:
            fade_elapsed_ms = elapsed * 1000
            job.fade_progress = min(1.0, fade_elapsed_ms / job.fade_duration_ms)

            # Interpolate base channels
            base_channels = self._interpolate_channels(
                job.fade_start_channels,
                job.fade_end_channels,
                job.fade_progress
            )

            if job.fade_progress >= 1.0:
                job.state = PlaybackState.PLAYING
                job.step_start_time = frame_time
        else:
            base_channels = job.channels

        # Apply modifiers
        final_channels = self._apply_modifiers(
            base_channels,
            job.modifiers or [],
            elapsed,
            job.seed
        )

        # Output to all universes
        self._send_output(job.universes, final_channels)

    # ─────────────────────────────────────────────────────────
    # Internal: Sequence Rendering
    # ─────────────────────────────────────────────────────────

    def _render_sequence_frame(self, job: PlaybackJob, frame_time: float):
        """Render a Sequence frame with step timing"""
        if not job.steps:
            return

        current_step = job.steps[job.current_step]

        # Get timing for this step
        step_fade_ms = current_step.get("fade_ms", 0)
        step_hold_ms = current_step.get("hold_ms", self._bpm_to_ms(job.bpm))
        total_step_ms = step_fade_ms + step_hold_ms

        # Calculate time within this step (accounting for pauses)
        step_elapsed = frame_time - job.step_start_time - job.paused_duration
        step_elapsed_ms = step_elapsed * 1000

        # Determine phase within step
        if step_elapsed_ms < step_fade_ms and step_fade_ms > 0:
            # FADE phase - interpolate from previous step
            job.state = PlaybackState.FADING
            fade_progress = step_elapsed_ms / step_fade_ms

            # Get previous step channels (or current if first step with no fade)
            if job.fade_start_channels:
                base_channels = self._interpolate_channels(
                    job.fade_start_channels,
                    self._step_channels(current_step),
                    fade_progress
                )
            else:
                base_channels = self._step_channels(current_step)
            job.fade_progress = fade_progress
        else:
            # HOLD phase - apply current step directly
            job.state = PlaybackState.PLAYING
            base_channels = self._step_channels(current_step)
            job.fade_progress = 1.0

        # Apply modifiers (step modifiers + sequence modifiers)
        step_modifiers = current_step.get("modifiers", [])
        elapsed = frame_time - job.start_time - job.paused_duration

        final_channels = self._apply_modifiers(
            base_channels,
            step_modifiers,
            elapsed,
            job.seed + job.current_step  # Different seed per step
        )

        # Check for step transition
        if step_elapsed_ms >= total_step_ms:
            self._advance_sequence_step(job, frame_time)

        # Output
        self._send_output(job.universes, final_channels)

    def _advance_sequence_step(self, job: PlaybackJob, frame_time: float):
        """Advance to next step in sequence"""
        # Store current channels as fade start for next step
        current_step = job.steps[job.current_step]
        job.fade_start_channels = self._step_channels(current_step)

        # Calculate next step index based on loop mode
        next_step = job.current_step + job.direction

        if next_step >= len(job.steps):
            # End of sequence
            if job.loop_mode == LoopMode.ONE_SHOT:
                job.state = PlaybackState.STOPPED
                with self._job_lock:
                    self._job = None
                print(f"⏹️ Sequence '{job.sequence_id}' completed (one-shot)")
                return
            elif job.loop_mode == LoopMode.LOOP:
                next_step = 0
                job.loop_count += 1
                print(f"🔁 Sequence '{job.sequence_id}' loop {job.loop_count}")
            elif job.loop_mode == LoopMode.BOUNCE:
                job.direction = -1
                next_step = len(job.steps) - 2  # Go to second-to-last
                if next_step < 0:
                    next_step = 0
        elif next_step < 0:
            # Start of sequence (during bounce)
            if job.loop_mode == LoopMode.BOUNCE:
                job.direction = 1
                next_step = 1 if len(job.steps) > 1 else 0
                job.loop_count += 1
                print(f"🔁 Sequence '{job.sequence_id}' bounce {job.loop_count}")

        # Update step
        job.current_step = next_step
        job.step_start_time = frame_time

        # Re-initialize modifier states for new step
        new_step = job.steps[job.current_step]
        new_modifiers = new_step.get("modifiers", [])
        self._init_modifier_states(new_modifiers, job.seed + job.current_step)

    # ─────────────────────────────────────────────────────────
    # Internal: Modifier Application
    # ─────────────────────────────────────────────────────────

    def _init_modifier_states(self, modifiers: List[Dict], seed: int):
        """Initialize modifier states for a new playback"""
        import random
        self._modifier_states.clear()

        for mod in modifiers:
            if not mod.get("enabled", True):
                continue
            mod_id = mod.get("id", f"mod_{id(mod)}")
            mod_type = mod.get("type", "")

            # Create seeded RNG for this modifier
            mod_seed = seed + hash(mod_id) & 0xFFFFFFFF

            self._modifier_states[mod_id] = {
                "modifier_id": mod_id,
                "modifier_type": mod_type,
                "random_state": random.Random(mod_seed),
                "phase": 0.0,
                "last_trigger": 0.0,
                "custom_data": {},
            }

    def _apply_modifiers(
        self,
        base_channels: Dict[int, int],
        modifiers: List[Dict],
        elapsed_time: float,
        seed: int,
    ) -> Dict[int, int]:
        """Apply modifiers to base channels"""
        if not self._modifier_renderer or not modifiers:
            return base_channels

        # Import here to avoid circular dependency
        from render_engine import TimeContext, ModifierState, MergeMode, DEFAULT_MERGE_MODES

        result = dict(base_channels)
        delta = self.frame_interval

        time_ctx = TimeContext(
            absolute_time=time.time(),
            delta_time=delta,
            elapsed_time=elapsed_time,
            frame_number=self._frame_count,
            seed=seed,
        )

        for mod in modifiers:
            if not mod.get("enabled", True):
                continue

            mod_id = mod.get("id", f"mod_{id(mod)}")
            mod_type = mod.get("type", "")
            params = mod.get("params", {})

            # Get or create state
            state_dict = self._modifier_states.get(mod_id)
            if not state_dict:
                import random
                mod_seed = seed + hash(mod_id) & 0xFFFFFFFF
                state_dict = {
                    "modifier_id": mod_id,
                    "modifier_type": mod_type,
                    "random_state": random.Random(mod_seed),
                    "phase": 0.0,
                    "last_trigger": 0.0,
                    "custom_data": {},
                }
                self._modifier_states[mod_id] = state_dict

            # Convert to ModifierState object
            state = ModifierState(
                modifier_id=state_dict["modifier_id"],
                modifier_type=state_dict["modifier_type"],
                random_state=state_dict["random_state"],
                phase=state_dict.get("phase", 0.0),
                last_trigger=state_dict.get("last_trigger", 0.0),
                custom_data=state_dict.get("custom_data", {}),
            )

            # Render modifier
            mod_result = self._modifier_renderer.render(
                modifier_type=mod_type,
                params=params,
                base_channels=result,
                time_ctx=time_ctx,
                state=state,
                fixture_index=0,
                total_fixtures=1,
            )

            # Update state dict with any changes
            state_dict["phase"] = state.phase
            state_dict["last_trigger"] = state.last_trigger
            state_dict["custom_data"] = state.custom_data

            # Apply merge mode
            merge_mode = DEFAULT_MERGE_MODES.get(mod_type, MergeMode.MULTIPLY)
            result = self._merge_modifier_result(result, mod_result, merge_mode)

        # Clamp all values to 0-255
        return {ch: max(0, min(255, int(val))) for ch, val in result.items()}

    def _merge_modifier_result(
        self,
        base: Dict[int, int],
        modifier_output: Dict[int, float],
        merge_mode,
    ) -> Dict[int, float]:
        """Apply merge mode to combine modifier output with base"""
        from render_engine import MergeMode

        result = {}
        for ch, base_val in base.items():
            mod_val = modifier_output.get(ch, 1.0 if merge_mode == MergeMode.MULTIPLY else base_val)

            if merge_mode == MergeMode.MULTIPLY:
                result[ch] = base_val * mod_val
            elif merge_mode == MergeMode.ADD:
                result[ch] = base_val + mod_val
            elif merge_mode == MergeMode.REPLACE:
                result[ch] = mod_val
            elif merge_mode == MergeMode.MAX:
                result[ch] = max(base_val, mod_val)
            elif merge_mode == MergeMode.MIN:
                result[ch] = min(base_val, mod_val)
            elif merge_mode == MergeMode.BLEND:
                blend_factor = 0.5
                result[ch] = base_val + (mod_val - base_val) * blend_factor
            else:
                result[ch] = base_val

        return result

    # ─────────────────────────────────────────────────────────
    # Internal: Channel Interpolation
    # ─────────────────────────────────────────────────────────

    def _interpolate_channels(
        self,
        start: Dict[int, int],
        end: Dict[int, int],
        progress: float,
    ) -> Dict[int, int]:
        """Linear interpolation between channel states"""
        result = {}
        all_channels = set(start.keys()) | set(end.keys())

        for ch in all_channels:
            start_val = start.get(ch, 0)
            end_val = end.get(ch, 0)
            result[ch] = int(start_val + (end_val - start_val) * progress)

        return result

    def _step_channels(self, step: Dict) -> Dict[int, int]:
        """Get channels from a step (resolve Look reference if needed)"""
        channels = step.get("channels", {})
        return {int(k): v for k, v in channels.items()}

    # ─────────────────────────────────────────────────────────
    # Internal: Helpers
    # ─────────────────────────────────────────────────────────

    def _resolve_sequence_steps(self, steps: List[Dict]) -> List[Dict]:
        """Resolve Look references in sequence steps"""
        resolved = []
        for step in steps:
            resolved_step = dict(step)

            # If step references a Look, fetch its data
            look_id = step.get("look_id")
            if look_id and self._look_resolver:
                look_data = self._look_resolver(look_id)
                if look_data:
                    # Use Look's channels if step doesn't have inline channels
                    if not resolved_step.get("channels"):
                        resolved_step["channels"] = look_data.get("channels", {})
                    # Merge Look's modifiers with step modifiers
                    look_modifiers = look_data.get("modifiers", [])
                    step_modifiers = resolved_step.get("modifiers", [])
                    resolved_step["modifiers"] = look_modifiers + step_modifiers

            resolved.append(resolved_step)

        return resolved

    def _bpm_to_ms(self, bpm: int) -> int:
        """Convert BPM to milliseconds per beat"""
        return int(60000 / bpm) if bpm > 0 else 500

    def _send_output(self, universes: List[int], channels: Dict[int, int]):
        """Send output to all universes via callback"""
        if not self._output_callback:
            return

        for universe in universes:
            self._output_callback(universe, channels)


# ============================================================
# Global Instance
# ============================================================

playback_controller = UnifiedPlaybackController(target_fps=30)
