"""
AETHER DMX State Manager - Single Source of Truth for channel values

Extracted from aether-core.py for modularity.
Uses core_registry for cross-module references.
"""

import os
import json
import time
import threading
from datetime import datetime
import core_registry as reg


class DMXStateManager:
    """Manages DMX state for all universes - this is the SSOT for channel values

    FADE HANDLING [F07]:
    - ESP32 is the SOLE fade authority — handles real-time interpolation
    - SSOT stores TARGET values immediately (no Python-side output interpolation)
    - get_output_values() returns target values for hardware refresh loop
    - get_display_values() returns Hermite-interpolated values for UI only
    - Output goes via UDPJSON to ESP32 nodes with fade_ms parameter
    """
    def __init__(self):
        self.universes = {}  # {universe_num: [512 current values]}
        self.targets = {}    # {universe_num: [512 target values]}
        self.fade_info = {}  # {universe_num: {'start_time': float, 'duration': float, 'start_values': [512]}}
        self.master_level = 100  # 0-100 percent
        self.master_base = {}  # Captured state at 100%
        self.lock = threading.Lock()
        self._save_timer = None
        self._last_emit_time = 0.0  # Throttle socketio emit to ~10fps
        self._load_state()

    def _load_state(self):
        """[F09] On startup: channels start at 0, but remember what was playing.
        Loads previous session info for resume prompt. Multiple sessions supported.
        """
        # Don't restore channel values - start fresh (safe for DMX)
        # But save active playback info for resume prompt
        try:
            state_file = reg.DMX_STATE_FILE
            if state_file and os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    saved = json.load(f)
                    # [F09] Store all sessions for resume prompt
                    self.last_sessions = saved.get('active_sessions', [])
                    # Legacy compat: single session
                    self.last_session = saved.get('active_playback', None)
                    if not self.last_session and self.last_sessions:
                        self.last_session = self.last_sessions[0]
                    self._last_saved_at = saved.get('saved_at', None)
                    if self.last_session:
                        print(f"💾 Previous session had active playback: {self.last_session}")
                        if self._last_saved_at:
                            print(f"   Last saved: {self._last_saved_at}")
                    else:
                        print("✓ DMX starting fresh (no previous playback)")
            else:
                self.last_session = None
                self.last_sessions = []
                self._last_saved_at = None
        except Exception as e:
            print(f"⚠️ Could not check previous session: {e}")
            self.last_session = None
            self.last_sessions = []
            self._last_saved_at = None

    def _save_state(self):
        """[F09] Save DMX state and active playback info to disk.
        Called on debounce (1s) for channel changes, and immediately
        on playback transitions (play/stop/pause) via save_state_now().
        """
        try:
            with self.lock:
                # Get ALL active playback sessions for recovery
                active_sessions = []
                try:
                    if reg.playback_manager:
                        status = reg.playback_manager.get_status()
                        if status:
                            for univ, info in status.items():
                                if info and info.get('type'):
                                    active_sessions.append({
                                        'universe': univ,
                                        'type': info.get('type'),
                                        'id': info.get('id'),
                                        'name': info.get('name')
                                    })
                except Exception:
                    pass  # Playback status not critical for state save

                # [F09] Also capture arbitration state for recovery context
                arb_owner = None
                try:
                    if reg.arbitration:
                        arb_owner = reg.arbitration.current_owner
                except Exception:
                    pass

                # Legacy field: first active session (backward compat)
                active_playback = active_sessions[0] if active_sessions else None

                data = {
                    'universes': {str(k): v for k, v in self.universes.items()},
                    'active_playback': active_playback,
                    'active_sessions': active_sessions,
                    'arbitration_owner': arb_owner,
                    'saved_at': datetime.now().isoformat()
                }
            state_file = reg.DMX_STATE_FILE
            if state_file:
                with open(state_file, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save DMX state: {e}")

    def save_state_now(self):
        """[F09] Immediately persist state (called on playback transitions).
        Bypasses the 1-second debounce for critical state changes.
        """
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None
        self._save_state()

    def _schedule_save(self):
        """Debounce saves to avoid excessive disk writes"""
        if self._save_timer:
            self._save_timer.cancel()
        self._save_timer = threading.Timer(1.0, self._save_state)
        self._save_timer.daemon = True
        self._save_timer.start()

    def get_universe(self, universe):
        """Get or create universe state array"""
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            return self.universes[universe].copy()

    def get_channel(self, universe, channel):
        """Get single channel value (1-indexed)"""
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            if 1 <= channel <= 512:
                return self.universes[universe][channel - 1]
            return 0

    def set_channels(self, universe, channels_dict, fade_ms=0):
        """Update specific channels with optional fade

        [F07] ESP32 is sole fade authority. SSOT snaps to target values immediately.
        fade_info is stored for UI display interpolation only (WebSocket).
        The refresh loop sends target values; ESP32 handles real-time fading.

        If fade_ms > 0:
          - SSOT snapped to target immediately (ESP32 fades locally)
          - fade_info stored for UI display interpolation only
        If fade_ms == 0:
          - Immediate snap to new values

        [F12] Values clamped to 0-255, universe validated 1-64.
        """
        # [F12] Validate universe
        try:
            universe = int(universe)
        except (TypeError, ValueError):
            return
        if universe < 1 or universe > 64:
            return

        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            if universe not in self.targets:
                self.targets[universe] = [0] * 512

            if fade_ms > 0:
                # [F07] Capture start values for UI display interpolation
                start_snapshot = list(self.universes[universe])

                # [F07] Snap SSOT to target immediately — ESP32 handles the fade
                for ch_str, value in channels_dict.items():
                    ch = int(ch_str)
                    if 1 <= ch <= 512:
                        clamped = max(0, min(255, int(value)))  # [F12] clamp
                        self.universes[universe][ch - 1] = clamped
                        self.targets[universe][ch - 1] = clamped

                # [F07] Store fade_info for UI display ONLY (not for output)
                self.fade_info[universe] = {
                    'start_time': time.monotonic(),
                    'duration': fade_ms / 1000.0,
                    'start_values': start_snapshot,
                    'ui_only': True  # [F07] Flag: only used for WebSocket display
                }
            else:
                # Immediate snap — update both current and target
                for ch_str, value in channels_dict.items():
                    ch = int(ch_str)
                    if 1 <= ch <= 512:
                        clamped = max(0, min(255, int(value)))  # [F12] clamp
                        self.universes[universe][ch - 1] = clamped
                        self.targets[universe][ch - 1] = clamped
                # Clear any fade in progress
                self.fade_info.pop(universe, None)

        # Throttle socketio emit to ~10fps (avoid blocking render thread)
        now = time.monotonic()
        if now - self._last_emit_time > 0.1:
            self._last_emit_time = now
            if reg.socketio:
                reg.socketio.emit('dmx_state', {
                    'universe': universe,
                    'channels': self.get_display_values(universe)  # [F07] UI gets interpolated display
                })
            self._schedule_save()

    def get_output_values(self, universe):
        """[F07] Get current output values for DMX refresh loop (hardware output).

        Returns the actual SSOT values (target). ESP32 handles all fade interpolation.
        This is the authority for what gets sent over the wire.
        """
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
                return [0] * 512
            # [F07] Clean up expired fade_info (for UI tracking)
            fade = self.fade_info.get(universe)
            if fade:
                elapsed = time.monotonic() - fade['start_time']
                if elapsed >= fade['duration']:
                    self.fade_info.pop(universe, None)
            return list(self.universes[universe])

    def get_display_values(self, universe):
        """[F07] Get interpolated values for UI display (WebSocket).

        Uses Hermite smoothstep for smooth visual feedback in the frontend.
        This does NOT affect hardware output — purely for UI faders/meters.
        """
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
                return [0] * 512

            fade = self.fade_info.get(universe)
            if not fade:
                return list(self.universes[universe])

            elapsed = time.monotonic() - fade['start_time']
            duration = fade['duration']
            progress = min(1.0, elapsed / duration) if duration > 0 else 1.0

            if progress >= 1.0:
                # Fade complete — clean up
                self.fade_info.pop(universe, None)
                return list(self.universes[universe])

            # Hermite smoothstep for smooth UI animation: 3t² - 2t³
            smooth = progress * progress * (3.0 - 2.0 * progress)

            start = fade['start_values']
            target = list(self.universes[universe])  # Target = current SSOT
            interpolated = [
                int(start[i] + (target[i] - start[i]) * smooth + 0.5)
                for i in range(512)
            ]
            return interpolated

    def blackout(self, universe, fade_ms=0):
        """Set all channels to 0 with optional fade"""
        all_zeros = {str(ch): 0 for ch in range(1, 513)}
        self.set_channels(universe, all_zeros, fade_ms=fade_ms)

    def get_channels_for_esp(self, universe, up_to_channel):
        """Get channel array for sending to ESP32"""
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            return self.universes[universe][:up_to_channel]
