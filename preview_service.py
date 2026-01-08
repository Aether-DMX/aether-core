"""
Live Preview Service - Preview Look/Sequence edits without affecting live output

This module provides:
- PreviewSession: Isolated preview rendering for editing
- Sandbox output: Preview renders to virtual buffer, not live
- Arm Live: Optional toggle to push preview to actual universes
- Real-time streaming: WebSocket updates as params change

Architecture:
- Preview sessions are isolated from live playback
- Each session has its own render state and output buffer
- 'Armed' sessions output to actual universes via merge layer
- WebSocket streams preview frames for UI visualization

Safety:
- Preview is SAFE by default (sandbox mode)
- Explicit 'arm' action required to affect live output
- Clear visual indicators for armed vs sandbox state

Version: 1.0.0
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Preview Mode
# ============================================================

class PreviewMode(Enum):
    SANDBOX = "sandbox"      # Preview only, no live output
    ARMED = "armed"          # Preview + live output to targets


# ============================================================
# Preview Frame - Single rendered frame for streaming
# ============================================================

@dataclass
class PreviewFrame:
    """A single preview frame for streaming to UI"""
    timestamp: float
    frame_number: int
    channels: Dict[int, int]  # channel -> value
    universes: List[int]      # target universes
    elapsed_ms: int
    modifier_count: int


# ============================================================
# Preview Session
# ============================================================

@dataclass
class PreviewSession:
    """
    An isolated preview session for editing Look/Sequence content.

    Each session:
    - Has its own render loop
    - Outputs to a virtual buffer (sandbox)
    - Can be 'armed' to output to real universes
    - Streams frames via callback for UI visualization
    """
    session_id: str
    preview_type: str  # "look" or "sequence"

    # Content being previewed
    channels: Dict[str, int] = field(default_factory=dict)
    modifiers: List[Dict] = field(default_factory=list)

    # Target configuration
    universes: List[int] = field(default_factory=lambda: [1])
    fixture_filter: Optional[List[str]] = None  # Optional fixture IDs to target

    # Mode
    mode: PreviewMode = PreviewMode.SANDBOX

    # Runtime state
    running: bool = False
    frame_count: int = 0
    start_time: float = 0.0
    last_frame: Optional[PreviewFrame] = None

    # Seed for deterministic preview
    seed: int = 0


# ============================================================
# Preview Service
# ============================================================

class PreviewService:
    """
    Manages preview sessions for live editing.

    Features:
    - Multiple concurrent preview sessions
    - Sandbox mode by default (safe)
    - Arm/disarm for live output
    - Real-time streaming updates
    - Configurable targets (universes/fixtures)
    """

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._sessions: Dict[str, PreviewSession] = {}
        self._lock = threading.Lock()

        # Render thread
        self._running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._frame_callback: Optional[Callable] = None  # For streaming to UI
        self._live_output_callback: Optional[Callable] = None  # For armed output
        self._modifier_renderer = None

        # Stats
        self._total_frames = 0
        self._actual_fps = 0.0

    def set_frame_callback(self, callback: Callable[[str, PreviewFrame], None]):
        """Set callback for streaming frames: callback(session_id, frame)"""
        self._frame_callback = callback

    def set_live_output_callback(self, callback: Callable[[int, Dict[int, int]], None]):
        """Set callback for armed live output: callback(universe, channels)"""
        self._live_output_callback = callback

    def set_modifier_renderer(self, renderer):
        """Set the modifier renderer for preview rendering"""
        self._modifier_renderer = renderer

    # ─────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        preview_type: str,
        channels: Dict[str, int],
        modifiers: List[Dict],
        universes: List[int],
        fixture_filter: Optional[List[str]] = None,
    ) -> PreviewSession:
        """Create a new preview session"""
        session = PreviewSession(
            session_id=session_id,
            preview_type=preview_type,
            channels=channels,
            modifiers=modifiers,
            universes=universes,
            fixture_filter=fixture_filter,
            mode=PreviewMode.SANDBOX,
            seed=hash(session_id) & 0xFFFFFFFF,
        )

        with self._lock:
            # Stop existing session with same ID
            if session_id in self._sessions:
                self._sessions[session_id].running = False

            self._sessions[session_id] = session

        print(f"🔍 Preview: Created session '{session_id}' ({preview_type})")
        return session

    def get_session(self, session_id: str) -> Optional[PreviewSession]:
        """Get a preview session by ID"""
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        channels: Optional[Dict[str, int]] = None,
        modifiers: Optional[List[Dict]] = None,
        universes: Optional[List[int]] = None,
    ) -> bool:
        """Update preview session content (triggers immediate re-render)"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if channels is not None:
                session.channels = channels
            if modifiers is not None:
                session.modifiers = modifiers
            if universes is not None:
                session.universes = universes

            return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a preview session"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].running = False
                del self._sessions[session_id]
                print(f"🔍 Preview: Deleted session '{session_id}'")
                return True
            return False

    def list_sessions(self) -> List[Dict]:
        """List all active sessions"""
        with self._lock:
            return [
                {
                    'session_id': s.session_id,
                    'preview_type': s.preview_type,
                    'mode': s.mode.value,
                    'running': s.running,
                    'universes': s.universes,
                    'modifier_count': len(s.modifiers),
                    'frame_count': s.frame_count,
                }
                for s in self._sessions.values()
            ]

    # ─────────────────────────────────────────────────────────
    # Arm / Disarm (Live Output Control)
    # ─────────────────────────────────────────────────────────

    def arm_session(self, session_id: str) -> bool:
        """Arm a session for live output (preview affects real universes)"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.mode = PreviewMode.ARMED
            print(f"🔴 Preview: Session '{session_id}' ARMED for live output")
            return True

    def disarm_session(self, session_id: str) -> bool:
        """Disarm a session (back to sandbox mode)"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.mode = PreviewMode.SANDBOX
            print(f"🟢 Preview: Session '{session_id}' DISARMED (sandbox)")
            return True

    def is_armed(self, session_id: str) -> bool:
        """Check if a session is armed"""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.mode == PreviewMode.ARMED if session else False

    # ─────────────────────────────────────────────────────────
    # Preview Playback Control
    # ─────────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> bool:
        """Start preview playback for a session"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.running = True
            session.start_time = time.monotonic()
            session.frame_count = 0

        # Ensure render loop is running
        self._ensure_running()
        return True

    def stop_session(self, session_id: str) -> bool:
        """Stop preview playback for a session"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.running = False
            return True

    def _ensure_running(self):
        """Ensure the render loop is running"""
        if self._running:
            return

        self._running = True
        self._stop_flag.clear()

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print(f"🔍 Preview: Render loop started at {self.target_fps} FPS")

    def stop_all(self):
        """Stop all previews and the render loop"""
        self._stop_flag.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            for session in self._sessions.values():
                session.running = False

    # ─────────────────────────────────────────────────────────
    # Single Frame Preview (for instant feedback)
    # ─────────────────────────────────────────────────────────

    def render_preview_frame(
        self,
        channels: Dict[str, int],
        modifiers: List[Dict],
        elapsed_time: float = 0.0,
        seed: int = 0,
    ) -> Dict[int, int]:
        """
        Render a single preview frame without starting a session.
        Useful for instant feedback when editing params.
        """
        if not self._modifier_renderer:
            # No renderer, return base channels
            return {int(k): v for k, v in channels.items()}

        # Import render dependencies
        from render_engine import TimeContext, ModifierState, MergeMode, DEFAULT_MERGE_MODES
        import random

        int_channels = {int(k): v for k, v in channels.items()}
        result = dict(int_channels)

        time_ctx = TimeContext(
            absolute_time=time.time(),
            delta_time=self.frame_interval,
            elapsed_time=elapsed_time,
            frame_number=int(elapsed_time * self.target_fps),
            seed=seed,
        )

        for mod in modifiers:
            if not mod.get('enabled', True):
                continue

            mod_id = mod.get('id', f"mod_{id(mod)}")
            mod_type = mod.get('type', '')
            params = mod.get('params', {})

            # Create temporary state
            mod_seed = seed + hash(mod_id) & 0xFFFFFFFF
            state = ModifierState(
                modifier_id=mod_id,
                modifier_type=mod_type,
                random_state=random.Random(mod_seed),
            )

            merge_mode = DEFAULT_MERGE_MODES.get(mod_type, MergeMode.MULTIPLY)

            mod_result = self._modifier_renderer.render(
                modifier_type=mod_type,
                params=params,
                base_channels=result,
                time_ctx=time_ctx,
                state=state,
                fixture_index=0,
                total_fixtures=1,
            )

            # Apply merge
            for ch in result:
                mod_val = mod_result.get(ch, 1.0 if merge_mode == MergeMode.MULTIPLY else result[ch])

                if merge_mode == MergeMode.MULTIPLY:
                    result[ch] = result[ch] * mod_val
                elif merge_mode == MergeMode.REPLACE:
                    result[ch] = mod_val
                elif merge_mode == MergeMode.ADD:
                    result[ch] = result[ch] + mod_val

        # Clamp to 0-255
        return {ch: max(0, min(255, int(val))) for ch, val in result.items()}

    # ─────────────────────────────────────────────────────────
    # Internal: Render Loop
    # ─────────────────────────────────────────────────────────

    def _render_loop(self):
        """Main render loop for all active preview sessions"""
        loop_start = time.monotonic()
        frame_count = 0

        while not self._stop_flag.is_set():
            frame_start = time.monotonic()

            # Get running sessions
            with self._lock:
                running_sessions = [
                    (s.session_id, s) for s in self._sessions.values()
                    if s.running
                ]

            # Render each session
            for session_id, session in running_sessions:
                try:
                    self._render_session_frame(session, frame_start)
                except Exception as e:
                    print(f"❌ Preview render error ({session_id}): {e}")

            # Frame timing
            frame_count += 1
            self._total_frames = frame_count
            frame_end = time.monotonic()

            # Calculate actual FPS
            total_elapsed = frame_end - loop_start
            if total_elapsed > 0:
                self._actual_fps = frame_count / total_elapsed

            # Sleep to maintain target FPS
            sleep_time = self.frame_interval - (frame_end - frame_start)
            if sleep_time > 0:
                self._stop_flag.wait(sleep_time)

    def _render_session_frame(self, session: PreviewSession, frame_time: float):
        """Render a single frame for a preview session"""
        elapsed = frame_time - session.start_time
        elapsed_ms = int(elapsed * 1000)

        # Render with modifiers
        rendered_channels = self.render_preview_frame(
            channels=session.channels,
            modifiers=session.modifiers,
            elapsed_time=elapsed,
            seed=session.seed,
        )

        session.frame_count += 1

        # Create preview frame
        frame = PreviewFrame(
            timestamp=frame_time,
            frame_number=session.frame_count,
            channels=rendered_channels,
            universes=session.universes,
            elapsed_ms=elapsed_ms,
            modifier_count=len([m for m in session.modifiers if m.get('enabled', True)]),
        )

        session.last_frame = frame

        # Stream to UI callback
        if self._frame_callback:
            self._frame_callback(session.session_id, frame)

        # If armed, output to live universes
        if session.mode == PreviewMode.ARMED and self._live_output_callback:
            for universe in session.universes:
                self._live_output_callback(universe, rendered_channels)

    # ─────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get preview service status"""
        with self._lock:
            sessions = [
                {
                    'session_id': s.session_id,
                    'preview_type': s.preview_type,
                    'mode': s.mode.value,
                    'running': s.running,
                    'universes': s.universes,
                    'frame_count': s.frame_count,
                    'modifier_count': len(s.modifiers),
                }
                for s in self._sessions.values()
            ]

        return {
            'running': self._running,
            'target_fps': self.target_fps,
            'actual_fps': round(self._actual_fps, 1),
            'total_frames': self._total_frames,
            'session_count': len(sessions),
            'armed_count': sum(1 for s in sessions if s['mode'] == 'armed'),
            'sessions': sessions,
        }


# ============================================================
# Global Instance
# ============================================================

preview_service = PreviewService(target_fps=30)
