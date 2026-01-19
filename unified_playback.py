"""
Unified Playback System - One Engine to Rule Them All

This module consolidates all playback types into a single, coherent system:
- Looks (static channels + modifiers)
- Sequences (multi-step with transitions)
- Chases (legacy BPM-based step sequences)
- Scenes (legacy static channel snapshots)
- Effects (dynamic lighting patterns)
- CueStacks (theatrical manual cueing)
- Shows (timeline-based event scheduling)

Architecture:
- PlaybackSession: Universal container for any playback type
- UnifiedPlaybackEngine: Single render loop for all playback
- SessionManager: Tracks all active sessions, handles priority
- SSOT Integration: All output through single dispatcher

Design Principles:
1. Single timing model: Frame-based rendering at 30 FPS
2. Composable modifiers: Any playback type can have modifiers
3. Consistent state: One source of truth for "what's playing"
4. Priority-based mixing: Higher priority wins, with optional layering
5. Graceful degradation: Legacy types work without modification

Version: 1.0.0
"""

import time
import threading
import random
import math
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ============================================================
# Enums and Constants
# ============================================================

class PlaybackType(Enum):
    """All supported playback content types"""
    LOOK = "look"
    SEQUENCE = "sequence"
    CHASE = "chase"           # Legacy, maps to sequence
    SCENE = "scene"           # Legacy, maps to look without modifiers
    EFFECT = "effect"
    CUE_STACK = "cue_stack"
    SHOW = "show"
    MANUAL = "manual"         # Direct fader/slider input
    BLACKOUT = "blackout"


class PlaybackState(Enum):
    """Runtime state of a playback session"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FADING_IN = "fading_in"   # Crossfading into this session
    FADING_OUT = "fading_out" # Crossfading out of this session
    ARMED = "armed"           # Ready but not outputting (preview)


class LoopMode(Enum):
    """How a sequence/chase loops"""
    ONE_SHOT = "one_shot"     # Play once and stop
    LOOP = "loop"             # Loop from end to start
    BOUNCE = "bounce"         # Ping-pong: forward then backward
    HOLD = "hold"             # Stop at last step, hold values


class Priority(Enum):
    """Playback priority levels (higher number = higher priority)"""
    IDLE = 0
    BACKGROUND = 10
    SCENE = 20
    CHASE = 40
    SEQUENCE = 45
    LOOK = 50
    EFFECT = 60
    CUE_STACK = 70
    MANUAL = 80
    SHOW = 85
    BLACKOUT = 100

    @classmethod
    def from_type(cls, playback_type: PlaybackType) -> 'Priority':
        """Get default priority for a playback type"""
        mapping = {
            PlaybackType.LOOK: cls.LOOK,
            PlaybackType.SEQUENCE: cls.SEQUENCE,
            PlaybackType.CHASE: cls.CHASE,
            PlaybackType.SCENE: cls.SCENE,
            PlaybackType.EFFECT: cls.EFFECT,
            PlaybackType.CUE_STACK: cls.CUE_STACK,
            PlaybackType.SHOW: cls.SHOW,
            PlaybackType.MANUAL: cls.MANUAL,
            PlaybackType.BLACKOUT: cls.BLACKOUT,
        }
        return mapping.get(playback_type, cls.IDLE)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Modifier:
    """A modifier that transforms base channel values"""
    id: str
    type: str  # pulse, strobe, flicker, wave, rainbow, twinkle, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    preset_id: Optional[str] = None


@dataclass
class Step:
    """A single step in a sequence or chase"""
    step_id: str
    name: str = ""
    channels: Dict[int, int] = field(default_factory=dict)
    look_id: Optional[str] = None  # Reference to existing Look
    modifiers: List[Modifier] = field(default_factory=list)
    fade_ms: int = 0           # Fade INTO this step
    hold_ms: int = 1000        # Hold AFTER fade completes

    @property
    def total_duration_ms(self) -> int:
        """Total time for this step"""
        return self.fade_ms + self.hold_ms


@dataclass
class TimelineEvent:
    """An event in a show timeline"""
    event_id: str
    time_ms: int              # Absolute time from show start
    event_type: str           # play_look, play_sequence, blackout, set_channels, etc.
    target_id: Optional[str] = None  # ID of content to play
    params: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False


@dataclass
class FadeState:
    """Tracks crossfade state during transitions"""
    start_channels: Dict[int, int]
    end_channels: Dict[int, int]
    duration_ms: int
    start_time: float  # Monotonic clock

    @property
    def progress(self) -> float:
        """Current fade progress 0.0 to 1.0"""
        elapsed = (time.monotonic() - self.start_time) * 1000
        return min(1.0, elapsed / self.duration_ms) if self.duration_ms > 0 else 1.0

    @property
    def is_complete(self) -> bool:
        return self.progress >= 1.0

    def interpolate(self) -> Dict[int, int]:
        """Get interpolated channel values at current progress"""
        p = self.progress
        result = {}
        all_channels = set(self.start_channels.keys()) | set(self.end_channels.keys())
        for ch in all_channels:
            start_val = self.start_channels.get(ch, 0)
            end_val = self.end_channels.get(ch, 0)
            result[ch] = int(start_val + (end_val - start_val) * p)
        return result


@dataclass
class PlaybackSession:
    """
    Universal container for any playback type.

    This is the core abstraction that unifies all playback systems.
    """
    session_id: str
    playback_type: PlaybackType
    name: str = ""

    # Universe targeting
    universes: List[int] = field(default_factory=lambda: [1])

    # Priority and state
    priority: Priority = Priority.IDLE
    state: PlaybackState = PlaybackState.STOPPED

    # Timing (monotonic clock based)
    start_time: float = 0.0
    paused_time: float = 0.0
    total_paused_duration: float = 0.0

    # Determinism
    seed: int = field(default_factory=lambda: int(time.time() * 1000) % 2**31)

    # Fade state
    fade_state: Optional[FadeState] = None

    # Base channel data (for LOOK, SCENE, MANUAL)
    channels: Dict[int, int] = field(default_factory=dict)

    # Modifiers (applicable to any type)
    modifiers: List[Modifier] = field(default_factory=list)

    # Sequence/Chase data
    steps: List[Step] = field(default_factory=list)
    current_step_index: int = 0
    step_start_time: float = 0.0
    loop_mode: LoopMode = LoopMode.LOOP
    direction: int = 1  # 1=forward, -1=backward (for bounce)
    bpm: int = 120  # For BPM-based timing

    # Effect data
    effect_type: str = ""
    effect_params: Dict[str, Any] = field(default_factory=dict)

    # CueStack data
    cue_stack_id: Optional[str] = None
    current_cue_index: int = -1  # -1 = not started

    # Show/Timeline data
    show_id: Optional[str] = None
    timeline: List[TimelineEvent] = field(default_factory=list)
    current_event_index: int = 0
    tempo_multiplier: float = 1.0

    # Reference resolution (for looks, cues that reference other content)
    look_id: Optional[str] = None  # Reference to existing Look

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_output_time: float = 0.0
    frame_count: int = 0

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since start, excluding paused duration"""
        if self.state == PlaybackState.STOPPED:
            return 0.0
        if self.state == PlaybackState.PAUSED:
            return self.paused_time - self.start_time - self.total_paused_duration
        return time.monotonic() - self.start_time - self.total_paused_duration

    @property
    def current_step(self) -> Optional[Step]:
        """Get current step for sequence/chase types"""
        if self.steps and 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_step_elapsed_time(self) -> float:
        """Time elapsed in current step (ms)"""
        if self.state in (PlaybackState.STOPPED, PlaybackState.PAUSED):
            return 0.0
        return (time.monotonic() - self.step_start_time) * 1000


# ============================================================
# Modifier State (per-session, per-modifier)
# ============================================================

@dataclass
class ModifierState:
    """Runtime state for a modifier instance"""
    modifier_id: str
    modifier_type: str
    rng: random.Random
    phase: float = 0.0
    last_trigger: float = 0.0
    custom_data: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Unified Playback Engine
# ============================================================

class UnifiedPlaybackEngine:
    """
    Single render engine for all playback types.

    Features:
    - 30 FPS frame-based rendering
    - Priority-based session management
    - Modifier composition for any playback type
    - Consistent timing with monotonic clock
    - SSOT-first output through callback
    """

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # Sessions
        self._sessions: Dict[str, PlaybackSession] = {}
        self._session_lock = threading.RLock()

        # Modifier states (per session)
        self._modifier_states: Dict[str, Dict[str, ModifierState]] = {}

        # Output
        self._output_callback: Optional[Callable[[int, Dict[int, int], int], None]] = None

        # Look/content resolver
        self._look_resolver: Optional[Callable[[str], Optional[Dict]]] = None
        self._cue_resolver: Optional[Callable[[str, int], Optional[Dict]]] = None

        # Modifier renderer (from render_engine.py)
        self._modifier_renderer = None

        # Engine state
        self._running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._frame_count = 0
        self._actual_fps = 0.0
        self._last_frame_time = 0.0

    # ─────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────

    def set_output_callback(self, callback: Callable[[int, Dict[int, int], int], None]):
        """Set callback for DMX output: callback(universe, channels, fade_ms)"""
        self._output_callback = callback

    def set_look_resolver(self, resolver: Callable[[str], Optional[Dict]]):
        """Set function to resolve Look ID to Look data"""
        self._look_resolver = resolver

    def set_cue_resolver(self, resolver: Callable[[str, int], Optional[Dict]]):
        """Set function to resolve Cue: resolver(stack_id, cue_index) -> cue_data"""
        self._cue_resolver = resolver

    def set_modifier_renderer(self, renderer):
        """Set the modifier renderer from render_engine.py"""
        self._modifier_renderer = renderer

    # ─────────────────────────────────────────────────────────
    # Engine Control
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start the unified playback engine"""
        if self._running:
            return

        self._running = True
        self._stop_flag.clear()
        self._frame_count = 0

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print(f"🎬 UnifiedPlaybackEngine started at {self.target_fps} FPS")

    def stop(self):
        """Stop the engine and all sessions"""
        if not self._running:
            return

        self._stop_flag.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._session_lock:
            self._sessions.clear()
            self._modifier_states.clear()

        print("⏹️ UnifiedPlaybackEngine stopped")

    def is_running(self) -> bool:
        return self._running

    # ─────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────

    def play(self, session: PlaybackSession, fade_from: Optional[Dict[int, int]] = None) -> str:
        """
        Start or resume a playback session.

        Args:
            session: The session to play
            fade_from: Optional starting channels for crossfade

        Returns:
            Session ID
        """
        with self._session_lock:
            # Initialize timing
            now = time.monotonic()
            session.start_time = now
            session.step_start_time = now
            session.state = PlaybackState.PLAYING
            session.frame_count = 0

            # Set up fade if requested
            if fade_from and session.channels:
                session.fade_state = FadeState(
                    start_channels=fade_from,
                    end_channels=session.channels.copy(),
                    duration_ms=session.effect_params.get('fade_ms', 0) or 500,
                    start_time=now
                )
                session.state = PlaybackState.FADING_IN

            # Initialize modifier states
            self._init_modifier_states(session)

            # Register session
            self._sessions[session.session_id] = session

            print(f"▶️ Playing {session.playback_type.value}: {session.name} (ID: {session.session_id})")
            return session.session_id

    def pause(self, session_id: str) -> bool:
        """Pause a session"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session or session.state != PlaybackState.PLAYING:
                return False

            session.paused_time = time.monotonic()
            session.state = PlaybackState.PAUSED
            print(f"⏸️ Paused: {session.name}")
            return True

    def resume(self, session_id: str) -> bool:
        """Resume a paused session"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session or session.state != PlaybackState.PAUSED:
                return False

            # Account for paused duration
            pause_duration = time.monotonic() - session.paused_time
            session.total_paused_duration += pause_duration
            session.state = PlaybackState.PLAYING
            print(f"▶️ Resumed: {session.name}")
            return True

    def stop_session(self, session_id: str, fade_ms: int = 0) -> bool:
        """Stop a specific session"""
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if fade_ms > 0:
                # Fade out to black
                current_channels = self._get_session_channels(session)
                session.fade_state = FadeState(
                    start_channels=current_channels,
                    end_channels={ch: 0 for ch in current_channels},
                    duration_ms=fade_ms,
                    start_time=time.monotonic()
                )
                session.state = PlaybackState.FADING_OUT
            else:
                session.state = PlaybackState.STOPPED
                del self._sessions[session_id]
                self._modifier_states.pop(session_id, None)
                print(f"⏹️ Stopped: {session.name}")

            return True

    def stop_all(self, fade_ms: int = 0):
        """Stop all sessions"""
        with self._session_lock:
            session_ids = list(self._sessions.keys())
            for sid in session_ids:
                self.stop_session(sid, fade_ms)

    def stop_type(self, playback_type: PlaybackType, fade_ms: int = 0):
        """Stop all sessions of a specific type"""
        with self._session_lock:
            to_stop = [sid for sid, s in self._sessions.items()
                      if s.playback_type == playback_type]
            for sid in to_stop:
                self.stop_session(sid, fade_ms)

    def get_session(self, session_id: str) -> Optional[PlaybackSession]:
        """Get a session by ID"""
        with self._session_lock:
            return self._sessions.get(session_id)

    def get_active_sessions(self) -> List[PlaybackSession]:
        """Get all active sessions"""
        with self._session_lock:
            return [s for s in self._sessions.values()
                   if s.state in (PlaybackState.PLAYING, PlaybackState.FADING_IN)]

    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        with self._session_lock:
            return {
                'running': self._running,
                'fps': self._actual_fps,
                'target_fps': self.target_fps,
                'frame_count': self._frame_count,
                'session_count': len(self._sessions),
                'sessions': [
                    {
                        'id': s.session_id,
                        'type': s.playback_type.value,
                        'name': s.name,
                        'state': s.state.value,
                        'priority': s.priority.value,
                        'universes': s.universes,
                    }
                    for s in self._sessions.values()
                ]
            }

    # ─────────────────────────────────────────────────────────
    # Render Loop
    # ─────────────────────────────────────────────────────────

    def _render_loop(self):
        """Main render loop - runs at target FPS"""
        last_time = time.monotonic()
        frame_times = []

        while not self._stop_flag.is_set():
            frame_start = time.monotonic()

            # Render all active sessions
            with self._session_lock:
                sessions_to_remove = []

                for session_id, session in self._sessions.items():
                    if session.state in (PlaybackState.PLAYING, PlaybackState.FADING_IN, PlaybackState.FADING_OUT):
                        try:
                            self._render_session(session)
                        except Exception as e:
                            print(f"❌ Render error for {session.name}: {e}")

                        # Check for fade-out completion
                        if session.state == PlaybackState.FADING_OUT:
                            if session.fade_state and session.fade_state.is_complete:
                                sessions_to_remove.append(session_id)

                # Clean up completed fade-outs
                for sid in sessions_to_remove:
                    session = self._sessions.pop(sid, None)
                    self._modifier_states.pop(sid, None)
                    if session:
                        print(f"⏹️ Stopped (fade complete): {session.name}")

            # Update stats
            self._frame_count += 1
            frame_times.append(time.monotonic() - frame_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
                self._actual_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            # Sleep to maintain target FPS
            elapsed = time.monotonic() - frame_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _render_session(self, session: PlaybackSession):
        """Render a single session frame"""
        # Handle fade transitions
        if session.fade_state:
            if session.fade_state.is_complete:
                if session.state == PlaybackState.FADING_IN:
                    session.state = PlaybackState.PLAYING
                session.fade_state = None

        # Get base channels based on playback type
        base_channels = self._get_session_channels(session)

        # Apply fade interpolation if active
        if session.fade_state:
            base_channels = session.fade_state.interpolate()

        # Apply modifiers
        final_channels = self._apply_modifiers(session, base_channels)

        # Advance playback state (for sequences, timelines, etc.)
        self._advance_session(session)

        # Output to each universe
        session.frame_count += 1
        session.last_output_time = time.monotonic()

        if self._output_callback:
            for universe in session.universes:
                self._output_callback(universe, final_channels, 0)

    def _get_session_channels(self, session: PlaybackSession) -> Dict[int, int]:
        """Get base channel values for a session based on its type"""

        if session.playback_type in (PlaybackType.LOOK, PlaybackType.SCENE, PlaybackType.MANUAL):
            # Direct channel data
            channels = session.channels.copy()

            # Resolve look reference if present
            if session.look_id and self._look_resolver:
                look_data = self._look_resolver(session.look_id)
                if look_data:
                    channels = {int(k): v for k, v in look_data.get('channels', {}).items()}
                    # Also get modifiers from look
                    if not session.modifiers and look_data.get('modifiers'):
                        session.modifiers = [
                            Modifier(**m) if isinstance(m, dict) else m
                            for m in look_data['modifiers']
                        ]
            return channels

        elif session.playback_type in (PlaybackType.SEQUENCE, PlaybackType.CHASE):
            # Get current step channels
            step = session.current_step
            if not step:
                return session.channels.copy()

            channels = step.channels.copy()

            # Resolve look reference in step
            if step.look_id and self._look_resolver:
                look_data = self._look_resolver(step.look_id)
                if look_data:
                    channels = {int(k): v for k, v in look_data.get('channels', {}).items()}

            # Handle step crossfade
            step_elapsed = session.get_step_elapsed_time()
            if step.fade_ms > 0 and step_elapsed < step.fade_ms:
                # We're in the fade portion of this step
                progress = step_elapsed / step.fade_ms
                prev_step = self._get_previous_step(session)
                if prev_step:
                    prev_channels = prev_step.channels.copy()
                    if prev_step.look_id and self._look_resolver:
                        look_data = self._look_resolver(prev_step.look_id)
                        if look_data:
                            prev_channels = {int(k): v for k, v in look_data.get('channels', {}).items()}

                    # Interpolate
                    all_chs = set(prev_channels.keys()) | set(channels.keys())
                    for ch in all_chs:
                        start = prev_channels.get(ch, 0)
                        end = channels.get(ch, 0)
                        channels[ch] = int(start + (end - start) * progress)

            return channels

        elif session.playback_type == PlaybackType.EFFECT:
            # Render effect (delegate to effect-specific logic)
            return self._render_effect(session)

        elif session.playback_type == PlaybackType.CUE_STACK:
            # Resolve current cue
            if session.cue_stack_id and session.current_cue_index >= 0 and self._cue_resolver:
                cue_data = self._cue_resolver(session.cue_stack_id, session.current_cue_index)
                if cue_data:
                    return {int(k): v for k, v in cue_data.get('channels', {}).items()}
            return session.channels.copy()

        elif session.playback_type == PlaybackType.SHOW:
            # Shows trigger other content, return current accumulated state
            return session.channels.copy()

        elif session.playback_type == PlaybackType.BLACKOUT:
            # All zeros
            return {ch: 0 for ch in range(1, 513)}

        return session.channels.copy()

    def _get_previous_step(self, session: PlaybackSession) -> Optional[Step]:
        """Get the previous step for crossfade calculation"""
        if not session.steps:
            return None

        prev_index = session.current_step_index - session.direction

        if session.loop_mode == LoopMode.LOOP:
            prev_index = prev_index % len(session.steps)
        elif session.loop_mode == LoopMode.BOUNCE:
            if prev_index < 0 or prev_index >= len(session.steps):
                # Direction was just reversed, previous was at boundary
                prev_index = session.current_step_index
        else:
            if prev_index < 0 or prev_index >= len(session.steps):
                return None

        return session.steps[prev_index] if 0 <= prev_index < len(session.steps) else None

    def _advance_session(self, session: PlaybackSession):
        """Advance playback state for sequences, timelines, etc."""

        if session.playback_type in (PlaybackType.SEQUENCE, PlaybackType.CHASE):
            self._advance_sequence(session)

        elif session.playback_type == PlaybackType.SHOW:
            self._advance_timeline(session)

    def _advance_sequence(self, session: PlaybackSession):
        """Advance to next step if current step is complete"""
        if not session.steps or session.state != PlaybackState.PLAYING:
            return

        step = session.current_step
        if not step:
            return

        step_elapsed = session.get_step_elapsed_time()

        # Check if step is complete
        if step_elapsed >= step.total_duration_ms:
            next_index = session.current_step_index + session.direction

            # Handle loop modes
            if session.loop_mode == LoopMode.LOOP:
                next_index = next_index % len(session.steps)

            elif session.loop_mode == LoopMode.BOUNCE:
                if next_index >= len(session.steps):
                    session.direction = -1
                    next_index = len(session.steps) - 2
                elif next_index < 0:
                    session.direction = 1
                    next_index = 1
                next_index = max(0, min(next_index, len(session.steps) - 1))

            elif session.loop_mode == LoopMode.ONE_SHOT:
                if next_index >= len(session.steps) or next_index < 0:
                    session.state = PlaybackState.STOPPED
                    return

            elif session.loop_mode == LoopMode.HOLD:
                if next_index >= len(session.steps) or next_index < 0:
                    # Stay on current step
                    return

            # Advance to next step
            session.current_step_index = next_index
            session.step_start_time = time.monotonic()

    def _advance_timeline(self, session: PlaybackSession):
        """Advance timeline events for shows"""
        if not session.timeline or session.state != PlaybackState.PLAYING:
            return

        elapsed_ms = session.elapsed_time * 1000 * session.tempo_multiplier

        # Execute any events that are due
        while session.current_event_index < len(session.timeline):
            event = session.timeline[session.current_event_index]

            if event.time_ms <= elapsed_ms and not event.executed:
                self._execute_timeline_event(session, event)
                event.executed = True
                session.current_event_index += 1
            else:
                break

    def _execute_timeline_event(self, session: PlaybackSession, event: TimelineEvent):
        """Execute a timeline event"""
        print(f"🎬 Show event: {event.event_type} at {event.time_ms}ms")

        # Timeline events typically trigger other playback
        # This is handled externally via callbacks
        if hasattr(self, '_timeline_callback') and self._timeline_callback:
            self._timeline_callback(session, event)

    # ─────────────────────────────────────────────────────────
    # Modifier System
    # ─────────────────────────────────────────────────────────

    def _init_modifier_states(self, session: PlaybackSession):
        """Initialize modifier states for a session"""
        states = {}
        rng = random.Random(session.seed)

        for modifier in session.modifiers:
            if modifier.enabled:
                mod_seed = rng.randint(0, 2**31)
                states[modifier.id] = ModifierState(
                    modifier_id=modifier.id,
                    modifier_type=modifier.type,
                    rng=random.Random(mod_seed)
                )

        # Also init modifiers from current step
        if session.current_step:
            for modifier in session.current_step.modifiers:
                if modifier.enabled and modifier.id not in states:
                    mod_seed = rng.randint(0, 2**31)
                    states[modifier.id] = ModifierState(
                        modifier_id=modifier.id,
                        modifier_type=modifier.type,
                        rng=random.Random(mod_seed)
                    )

        self._modifier_states[session.session_id] = states

    def _apply_modifiers(self, session: PlaybackSession, base_channels: Dict[int, int]) -> Dict[int, int]:
        """Apply all modifiers to base channels"""
        if not self._modifier_renderer:
            return base_channels

        result = base_channels.copy()
        mod_states = self._modifier_states.get(session.session_id, {})

        # Collect all active modifiers
        all_modifiers = list(session.modifiers)
        if session.current_step:
            all_modifiers.extend(session.current_step.modifiers)

        for modifier in all_modifiers:
            if not modifier.enabled:
                continue

            state = mod_states.get(modifier.id)
            if not state:
                continue

            try:
                # Create time context
                from render_engine import TimeContext
                time_ctx = TimeContext(
                    absolute_time=time.time(),
                    delta_time=self.frame_interval,
                    elapsed_time=session.elapsed_time,
                    frame_number=session.frame_count,
                    seed=session.seed
                )

                # Render modifier
                mod_output = self._modifier_renderer.render(
                    modifier_type=modifier.type,
                    params=modifier.params,
                    base_channels=result,
                    time_ctx=time_ctx,
                    state=state
                )

                # Apply modifier output based on merge mode
                merge_mode = modifier.params.get('merge_mode', 'multiply')
                for ch, mod_val in mod_output.items():
                    if ch in result:
                        if merge_mode == 'multiply':
                            result[ch] = int(result[ch] * mod_val)
                        elif merge_mode == 'add':
                            result[ch] = min(255, result[ch] + int(mod_val))
                        elif merge_mode == 'replace':
                            result[ch] = int(mod_val)
                        elif merge_mode == 'max':
                            result[ch] = max(result[ch], int(mod_val))
                        elif merge_mode == 'min':
                            result[ch] = min(result[ch], int(mod_val))

                        # Clamp to valid range
                        result[ch] = max(0, min(255, result[ch]))

            except Exception as e:
                print(f"⚠️ Modifier {modifier.type} error: {e}")

        return result

    # ─────────────────────────────────────────────────────────
    # Effect Rendering
    # ─────────────────────────────────────────────────────────

    def _render_effect(self, session: PlaybackSession) -> Dict[int, int]:
        """Render built-in effect to channels"""
        effect_type = session.effect_type
        params = session.effect_params
        elapsed = session.elapsed_time

        channels = {}

        # Effect implementations
        if effect_type == "pulse":
            # Breathing brightness effect
            speed = params.get('speed', 1.0)
            min_val = params.get('min_brightness', 20)
            max_val = params.get('max_brightness', 255)
            target_channels = params.get('channels', list(range(1, 5)))

            phase = (elapsed * speed) % 1.0
            brightness = min_val + (max_val - min_val) * (math.sin(phase * 2 * math.pi) + 1) / 2

            for ch in target_channels:
                channels[ch] = int(brightness)

        elif effect_type == "strobe":
            # On/off strobe
            rate = params.get('rate', 10.0)
            duty = params.get('duty_cycle', 50) / 100.0
            on_val = params.get('on_value', 255)
            off_val = params.get('off_value', 0)
            target_channels = params.get('channels', list(range(1, 5)))

            phase = (elapsed * rate) % 1.0
            value = on_val if phase < duty else off_val

            for ch in target_channels:
                channels[ch] = value

        elif effect_type == "rainbow":
            # Color cycling
            speed = params.get('speed', 0.5)
            target_channels = params.get('channels', [1, 2, 3])  # RGB

            hue = (elapsed * speed) % 1.0
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)

            if len(target_channels) >= 3:
                channels[target_channels[0]] = int(r * 255)
                channels[target_channels[1]] = int(g * 255)
                channels[target_channels[2]] = int(b * 255)

        elif effect_type == "wave":
            # Traveling brightness wave
            speed = params.get('speed', 1.0)
            width = params.get('width', 0.3)
            fixtures = params.get('fixtures', 4)
            channels_per_fixture = params.get('channels_per_fixture', 4)

            phase = (elapsed * speed) % 1.0

            for i in range(fixtures):
                fixture_pos = i / fixtures
                distance = abs(phase - fixture_pos)
                if distance > 0.5:
                    distance = 1.0 - distance

                brightness = max(0, 1.0 - distance / width)
                base_ch = i * channels_per_fixture + 1

                # Set all fixture channels to brightness
                for j in range(channels_per_fixture):
                    channels[base_ch + j] = int(brightness * 255)

        elif effect_type == "fire":
            # Fire flicker simulation
            mod_states = self._modifier_states.get(session.session_id, {})
            rng = mod_states.get('_fire_rng')
            if not rng:
                rng = random.Random(session.seed)
                if session.session_id not in self._modifier_states:
                    self._modifier_states[session.session_id] = {}
                self._modifier_states[session.session_id]['_fire_rng'] = rng

            fixtures = params.get('fixtures', 4)
            channels_per_fixture = params.get('channels_per_fixture', 4)

            for i in range(fixtures):
                # Random flicker
                brightness = 0.6 + rng.random() * 0.4
                red_shift = 0.9 + rng.random() * 0.1

                base_ch = i * channels_per_fixture + 1
                channels[base_ch] = int(255 * brightness * red_shift)  # R
                channels[base_ch + 1] = int(100 * brightness)  # G (less green for fire)
                channels[base_ch + 2] = int(20 * brightness)   # B (minimal blue)
                if channels_per_fixture > 3:
                    channels[base_ch + 3] = int(50 * brightness)  # W

        return channels

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        if s == 0.0:
            return v, v, v

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        return v, p, q


# ============================================================
# Session Factory - Create sessions from various sources
# ============================================================

class SessionFactory:
    """Factory for creating PlaybackSessions from various data sources"""

    @staticmethod
    def from_look(look_id: str, look_data: Dict, universes: List[int] = None,
                  fade_ms: int = 0) -> PlaybackSession:
        """Create session from Look data"""
        channels = {int(k): v for k, v in look_data.get('channels', {}).items()}
        modifiers = [
            Modifier(**m) if isinstance(m, dict) else m
            for m in look_data.get('modifiers', [])
        ]

        session = PlaybackSession(
            session_id=f"look_{look_id}_{int(time.time()*1000)}",
            playback_type=PlaybackType.LOOK,
            name=look_data.get('name', f'Look {look_id}'),
            universes=universes or [1],
            priority=Priority.LOOK,
            channels=channels,
            modifiers=modifiers,
            look_id=look_id,
        )

        if fade_ms > 0:
            session.effect_params['fade_ms'] = fade_ms

        return session

    @staticmethod
    def from_sequence(sequence_id: str, sequence_data: Dict,
                      universes: List[int] = None) -> PlaybackSession:
        """Create session from Sequence data"""
        steps = []
        for step_data in sequence_data.get('steps', []):
            step = Step(
                step_id=step_data.get('step_id', str(len(steps))),
                name=step_data.get('name', ''),
                channels={int(k): v for k, v in step_data.get('channels', {}).items()},
                look_id=step_data.get('look_id'),
                modifiers=[Modifier(**m) for m in step_data.get('modifiers', [])],
                fade_ms=step_data.get('fade_ms', 0),
                hold_ms=step_data.get('hold_ms', 1000),
            )
            steps.append(step)

        loop_mode_str = sequence_data.get('loop_mode', 'loop')
        loop_mode = LoopMode[loop_mode_str.upper()] if loop_mode_str else LoopMode.LOOP

        return PlaybackSession(
            session_id=f"seq_{sequence_id}_{int(time.time()*1000)}",
            playback_type=PlaybackType.SEQUENCE,
            name=sequence_data.get('name', f'Sequence {sequence_id}'),
            universes=universes or [1],
            priority=Priority.SEQUENCE,
            steps=steps,
            loop_mode=loop_mode,
            bpm=sequence_data.get('bpm', 120),
        )

    @staticmethod
    def from_chase(chase_id: str, chase_data: Dict,
                   universes: List[int] = None) -> PlaybackSession:
        """Create session from legacy Chase data"""
        steps = []
        default_fade = chase_data.get('fade_ms', 0)
        bpm = chase_data.get('bpm', 120)
        default_duration = int(60000 / bpm)  # ms per beat

        for i, step_data in enumerate(chase_data.get('steps', [])):
            # Handle legacy format
            if 'duration' in step_data:
                hold_ms = step_data['duration'] - step_data.get('fade_ms', default_fade)
            else:
                hold_ms = step_data.get('hold_ms', default_duration)

            step = Step(
                step_id=str(i),
                channels={int(k): v for k, v in step_data.get('channels', {}).items()},
                fade_ms=step_data.get('fade_ms', default_fade),
                hold_ms=hold_ms,
            )
            steps.append(step)

        return PlaybackSession(
            session_id=f"chase_{chase_id}_{int(time.time()*1000)}",
            playback_type=PlaybackType.CHASE,
            name=chase_data.get('name', f'Chase {chase_id}'),
            universes=universes or [1],
            priority=Priority.CHASE,
            steps=steps,
            loop_mode=LoopMode.LOOP if chase_data.get('loop', True) else LoopMode.ONE_SHOT,
            bpm=bpm,
        )

    @staticmethod
    def from_scene(scene_id: str, scene_data: Dict,
                   universes: List[int] = None, fade_ms: int = 0) -> PlaybackSession:
        """Create session from legacy Scene data"""
        channels = {int(k): v for k, v in scene_data.get('channels', {}).items()}

        session = PlaybackSession(
            session_id=f"scene_{scene_id}_{int(time.time()*1000)}",
            playback_type=PlaybackType.SCENE,
            name=scene_data.get('name', f'Scene {scene_id}'),
            universes=universes or [1],
            priority=Priority.SCENE,
            channels=channels,
        )

        if fade_ms > 0:
            session.effect_params['fade_ms'] = fade_ms

        return session

    @staticmethod
    def from_effect(effect_type: str, params: Dict = None,
                    universes: List[int] = None) -> PlaybackSession:
        """Create session for a built-in effect"""
        return PlaybackSession(
            session_id=f"effect_{effect_type}_{int(time.time()*1000)}",
            playback_type=PlaybackType.EFFECT,
            name=f'{effect_type.title()} Effect',
            universes=universes or [1],
            priority=Priority.EFFECT,
            effect_type=effect_type,
            effect_params=params or {},
        )

    @staticmethod
    def blackout(universes: List[int] = None, fade_ms: int = 0) -> PlaybackSession:
        """Create blackout session"""
        session = PlaybackSession(
            session_id=f"blackout_{int(time.time()*1000)}",
            playback_type=PlaybackType.BLACKOUT,
            name='Blackout',
            universes=universes or [1],
            priority=Priority.BLACKOUT,
            channels={ch: 0 for ch in range(1, 513)},
        )

        if fade_ms > 0:
            session.effect_params['fade_ms'] = fade_ms

        return session


# ============================================================
# Global Instance
# ============================================================

# Singleton instance
unified_engine = UnifiedPlaybackEngine()
session_factory = SessionFactory()


# ============================================================
# Convenience Functions
# ============================================================

def play_look(look_id: str, look_data: Dict, universes: List[int] = None,
              fade_ms: int = 0, fade_from: Dict[int, int] = None) -> str:
    """Play a Look"""
    session = session_factory.from_look(look_id, look_data, universes, fade_ms)
    return unified_engine.play(session, fade_from)


def play_sequence(sequence_id: str, sequence_data: Dict,
                  universes: List[int] = None) -> str:
    """Play a Sequence"""
    session = session_factory.from_sequence(sequence_id, sequence_data, universes)
    return unified_engine.play(session)


def play_chase(chase_id: str, chase_data: Dict,
               universes: List[int] = None) -> str:
    """Play a Chase (legacy support)"""
    session = session_factory.from_chase(chase_id, chase_data, universes)
    return unified_engine.play(session)


def play_scene(scene_id: str, scene_data: Dict, universes: List[int] = None,
               fade_ms: int = 0, fade_from: Dict[int, int] = None) -> str:
    """Play a Scene (legacy support)"""
    session = session_factory.from_scene(scene_id, scene_data, universes, fade_ms)
    return unified_engine.play(session, fade_from)


def play_effect(effect_type: str, params: Dict = None,
                universes: List[int] = None) -> str:
    """Play a built-in effect"""
    session = session_factory.from_effect(effect_type, params, universes)
    return unified_engine.play(session)


def blackout(universes: List[int] = None, fade_ms: int = 0,
             fade_from: Dict[int, int] = None) -> str:
    """Trigger blackout"""
    session = session_factory.blackout(universes, fade_ms)
    return unified_engine.play(session, fade_from)


def stop(session_id: str = None, fade_ms: int = 0):
    """Stop playback"""
    if session_id:
        unified_engine.stop_session(session_id, fade_ms)
    else:
        unified_engine.stop_all(fade_ms)


def get_status() -> Dict:
    """Get playback status"""
    return unified_engine.get_status()
