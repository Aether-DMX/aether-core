#!/usr/bin/env python3
"""
AETHER Core v5.0 - UDP JSON DMX Control System
Single source of truth for ALL system functionality

Transport:
- UDP JSON commands to ESP32 nodes on port 6455
- All 512 channels per universe
- Event-driven updates (no continuous refresh required)

Features:
- UDP JSON DMX output (set, fade, blackout)
- Scene/Chase/Effect management and playback
- Universe splitting (multiple nodes per universe via channel slices)
- Coordinated play/stop across all nodes
- RDM-ready architecture (future)
"""

import socket
import json
import sqlite3
import threading
import time
import os
import subprocess
import uuid
import platform
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
from croniter import croniter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ============================================================
# Speech-to-Text (Whisper) - Lazy loaded
# ============================================================
_whisper_model = None
_whisper_lock = threading.Lock()

def get_whisper_model():
    """Lazy-load faster-whisper model on first use"""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                    # Use tiny model for speed on Pi - still accurate for short commands
                    _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                    logging.info("✅ Whisper model loaded (tiny/int8)")
                except ImportError:
                    logging.warning("⚠️ faster-whisper not installed, STT disabled")
                    _whisper_model = False
                except Exception as e:
                    logging.error(f"❌ Failed to load Whisper: {e}")
                    _whisper_model = False
    return _whisper_model if _whisper_model else None
# import ai_ssot  # REMOVED - dead code, Node handles AI
# import ai_ops_registry  # REMOVED - dead code, Node handles AI
from effects_engine import DynamicEffectsEngine
from looks_sequences import (
    LooksSequencesManager, Look, Sequence, SequenceStep, Modifier,
    validate_look_data, validate_sequence_data, run_full_migration,
)
from modifier_registry import (
    modifier_registry, validate_modifier, normalize_modifier,
    get_modifier_presets,
)
from render_engine import (
    render_engine, render_look_frame, RenderEngine,
    ModifierRenderer, TimeContext, MergeMode,
)
# DELETED (Phase 3): playback_controller import removed
# UnifiedPlaybackEngine (unified_playback.py) is now the sole authority
# See TASK-0021 in TASK_LEDGER.md
from merge_layer import (
    merge_layer, channel_classifier, MergeLayer,
    ChannelClassifier, ChannelType, get_priority,
)
from fixture_library import (
    init_fixture_library, get_fixture_library, get_channel_mapper,
    FixtureLibrary, ChannelMapper, FixtureProfile, FixtureInstance,
    FixtureMode, ChannelCapability, BUILTIN_PROFILES,
)
from preview_service import (
    preview_service, PreviewService, PreviewSession,
    PreviewMode, PreviewFrame,
)
from cue_stacks import (
    CueStacksManager, CueStack, Cue,
    validate_cue_stack_data, validate_cue_data,
)
from pixel_mapper import (
    PixelArrayController, Pixel, OperationMode, EffectType,
    create_pixel_controller, UNIVERSE as PIXEL_UNIVERSE,
)
from unified_playback import (
    unified_engine, session_factory, PlaybackSession, PlaybackType,
    PlaybackState as UnifiedPlaybackState, Priority, LoopMode as UnifiedLoopMode,
    Modifier as UnifiedModifier, Step, play_look as unified_play_look,
    play_sequence as unified_play_sequence, play_chase as unified_play_chase,
    play_scene as unified_play_scene, play_effect as unified_play_effect,
    blackout as unified_blackout, stop as unified_stop, get_status as unified_get_status,
    pause as unified_pause, resume as unified_resume,
)

# Fixture-Centric Architecture (Phase 0-3)
from fixture_render import (
    RenderedFixtureFrame, RenderedFixtureState, FixtureFrameBuilder,
    create_frame_from_fixture_channels, AttributeType,
)
from distribution_modes import (
    DistributionMode, DistributionConfig, DistributionCalculator,
    DISTRIBUTION_PRESETS, get_distribution_preset, list_distribution_presets,
    get_supported_distributions, suggest_distribution_for_effect,
)
from ai_fixture_advisor import (
    get_ai_advisor, AIFixtureAdvisor, AISuggestion, SuggestionType,
    FixtureContext, get_distribution_suggestions,
    apply_ai_suggestion, dismiss_ai_suggestion,
)
from final_render_pipeline import (
    get_render_pipeline, init_render_pipeline, FinalRenderPipeline,
    RenderTimeContext, RenderJob, FeatureFlags,
)

# Operator Trust Enforcement (Phase 4 Lane 3)
from operator_trust import (
    trust_enforcer, report_node_heartbeat, report_backend_start,
    check_ui_sync, get_trust_status, get_trust_events,
    start_trust_monitoring, stop_trust_monitoring, clear_failure_halt,
)

# RDM Service — consolidated into RDMManager (rdm_service.py deleted)

# Supabase cloud sync (optional)
try:
    from services.supabase_service import get_supabase_service, sync_to_cloud
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    get_supabase_service = lambda: None
    sync_to_cloud = lambda x: lambda f: f  # No-op decorator
    print("⚠️ Supabase service not available - cloud sync disabled")

# Optional serial support for UART gateway
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ pyserial not installed - UART gateway disabled")

# UDPJSON DMX Transport - SSOT
# All DMX output goes directly to ESP32 nodes via UDP JSON on port 6455
# Event-driven: packets sent on value change, no continuous refresh required
AETHER_UDPJSON_PORT = 6455  # SSOT: Primary port for UDPJSON DMX commands

# ============================================================
# Configuration - Environment-based with sensible defaults
# ============================================================
API_PORT = int(os.environ.get('AETHER_API_PORT', 8891))
DISCOVERY_PORT = int(os.environ.get('AETHER_DISCOVERY_PORT', 9999))
WIFI_COMMAND_PORT = int(os.environ.get('AETHER_WIFI_PORT', 8888))

# UART Gateway configuration (Pi GPIO to ESP32)
UART_GATEWAY_PORT = "/dev/serial0"  # Pi GPIO 14/15
UART_GATEWAY_BAUD = 115200

# DMX Transport Mode - UDPJSON only
DMX_TRANSPORT_MODE = "udp_json"  # Only supported mode

# Dynamic paths - works for any user
HOME_DIR = os.path.expanduser("~")
DATABASE = os.path.join(HOME_DIR, "aether-core.db")
SETTINGS_FILE = os.path.join(HOME_DIR, "aether-settings.json")
DMX_STATE_FILE = os.path.join(HOME_DIR, "aether-dmx-state.json")

# ============================================================
# Version/Runtime Info - For SSOT verification
# ============================================================
AETHER_VERSION = "5.0.0"
AETHER_START_TIME = datetime.now()
AETHER_FILE_PATH = os.path.abspath(__file__)

def get_git_commit():
    """Get current git commit hash if available"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=2,
            cwd=os.path.dirname(AETHER_FILE_PATH)
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return "unknown"

AETHER_COMMIT = get_git_commit()

# Device identification for cloud sync
DEVICE_ID_FILE = os.path.join(HOME_DIR, ".aether-device-id")

def get_or_create_device_id():
    """Get or create persistent device identifier"""
    if os.path.exists(DEVICE_ID_FILE):
        with open(DEVICE_ID_FILE, 'r') as f:
            return f.read().strip()
    device_id = f"aether-{uuid.uuid4().hex[:8]}"
    with open(DEVICE_ID_FILE, 'w') as f:
        f.write(device_id)
    return device_id

# ============================================================
# SSOT Guardrail - Prevents future bypass regressions
# ============================================================
def ssot_integrity_check():
    """
    Startup check that scans for forbidden DMX send patterns outside the SSOT pipeline.

    The ONLY valid DMX output path is:
    ContentManager.set_channels() -> dmx_state -> UDPJSON to nodes

    All DMX goes through UDP JSON commands directly to ESP32 nodes on port 6455.
    """
    violations = []

    # Read our own source code
    try:
        with open(AETHER_FILE_PATH, 'r') as f:
            lines = f.readlines()
    except OSError as e:
        print(f"⚠️ SSOT check: Could not read source file: {e}")
        return []

    # Track what class/method we're in
    current_class = None
    current_method = None

    # Allowed locations for DMX send operations
    ALLOWED_CLASSES = {'NodeManager'}
    ALLOWED_METHODS = {
        'send_via_ola', 'send_to_node', 'send_command_to_wifi', 'send_blackout',
        'send_to_uart_gateway', 'send_config_to_gateway'
    }

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track class context
        if stripped.startswith('class ') and ':' in stripped:
            current_class = stripped.split('class ')[1].split('(')[0].split(':')[0]
            current_method = None
        elif stripped.startswith('def ') and current_class:
            current_method = stripped.split('def ')[1].split('(')[0]

        # Skip allowed locations
        if current_class in ALLOWED_CLASSES:
            continue
        if current_method in ALLOWED_METHODS:
            continue

        # Check for forbidden patterns
        # Check for forbidden direct socket operations outside allowed locations
        if 'udp_socket.sendto' in line:
            if current_method is None or ('send_udpjson' not in current_method and current_method not in ALLOWED_METHODS):
                violations.append(f"Line {i}: Direct UDP send outside SSOT pipeline")

    return violations


def ssot_startup_verify():
    """Run SSOT integrity check on startup and log results"""
    print("🔒 Running SSOT integrity check...", flush=True)
    violations = ssot_integrity_check()

    if violations:
        print(f"⚠️ SSOT VIOLATIONS DETECTED ({len(violations)}):", flush=True)
        for v in violations[:5]:  # Show first 5
            print(f"   ❌ {v}", flush=True)
        if len(violations) > 5:
            print(f"   ... and {len(violations) - 5} more", flush=True)
        print("   Fix: Route all DMX output through ContentManager.set_channels()", flush=True)
    else:
        print("✅ SSOT integrity verified - all DMX paths route through UDPJSON", flush=True)

    return len(violations) == 0


# Timing configuration
STALE_TIMEOUT = 60
DMX_OUTPUT_FPS = 40  # Max frames per second for DMX output

app = Flask(__name__)

# ============================================================
# CORS Configuration - Security hardened
# ============================================================
# Default allowed origins for local deployment (Pi + local network)
# Add custom origins via AETHER_CORS_ORIGINS environment variable (comma-separated)
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8891",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8891",
    "http://192.168.50.1:3000",
    "http://192.168.18.13:3000",
    "http://192.168.18.13:8891",   # Default Pi AP address
    "http://192.168.50.1:8891",
]

def get_allowed_origins():
    """Get list of allowed CORS origins from defaults + environment"""
    origins = DEFAULT_CORS_ORIGINS.copy()
    # Allow additional origins via environment variable
    env_origins = os.environ.get('AETHER_CORS_ORIGINS', '')
    if env_origins:
        for origin in env_origins.split(','):
            origin = origin.strip()
            if origin and origin not in origins:
                origins.append(origin)
    return origins

ALLOWED_ORIGINS = get_allowed_origins()
print(f"🔒 CORS allowed origins: {ALLOWED_ORIGINS}")

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS, async_mode='threading')

# ============================================================
# Thread Pool — bounded executor for async I/O (F13 fix)
# ============================================================
# Replaces 21+ raw threading.Thread spawns for Supabase sync,
# cloud logging, and node sync operations. Caps concurrency to
# prevent unbounded thread accumulation when Supabase is slow.
_cloud_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='cloud-sync')
_node_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='node-sync')

# Thread monitoring — track high-water mark for health endpoint
_thread_hwm = 0  # high-water mark
_thread_hwm_lock = threading.Lock()

def _update_thread_hwm():
    """Update thread high-water mark for monitoring."""
    global _thread_hwm
    count = threading.active_count()
    with _thread_hwm_lock:
        if count > _thread_hwm:
            _thread_hwm = count

def cloud_submit(fn, *args, **kwargs):
    """Submit a task to the cloud sync thread pool (bounded)."""
    _update_thread_hwm()
    try:
        _cloud_pool.submit(fn, *args, **kwargs)
    except RuntimeError:
        pass  # Pool shut down during graceful exit

def node_submit(fn, *args, **kwargs):
    """Submit a task to the node sync thread pool (bounded)."""
    _update_thread_hwm()
    try:
        _node_pool.submit(fn, *args, **kwargs)
    except RuntimeError:
        pass  # Pool shut down during graceful exit

# ============================================================
# Beta Debug Logging - Enable with AETHER_BETA_DEBUG=1
# ============================================================
BETA_DEBUG = os.environ.get('AETHER_BETA_DEBUG', '0') == '1'

def beta_log(action_name, data):
    """Structured debug logging for Beta 1 verification."""
    if not BETA_DEBUG:
        return
    timestamp = datetime.now().isoformat()
    log_entry = {"ts": timestamp, "action": action_name, **data}
    print(f"🔍 BETA_DEBUG: {json.dumps(log_entry)}", flush=True)


# ============================================================
# DMX State - THE SINGLE SOURCE OF TRUTH FOR CHANNEL VALUES
# ============================================================
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
            if os.path.exists(DMX_STATE_FILE):
                with open(DMX_STATE_FILE, 'r') as f:
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
                    status = playback_manager.get_status()
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
                    arb_owner = arbitration.current_owner
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
            with open(DMX_STATE_FILE, 'w') as f:
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
            socketio.emit('dmx_state', {
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

dmx_state = DMXStateManager()

# ============================================================
# Playback State Manager
# ============================================================
class PlaybackManager:
    """Tracks current playback state across all universes"""
    def __init__(self):
        self.lock = threading.Lock()
        self.current = {}  # {universe: {'type': 'scene'|'chase', 'id': '...', 'started': datetime}}
    
    def set_playing(self, universe, content_type, content_id):
        with self.lock:
            self.current[universe] = {
                'type': content_type,
                'id': content_id,
                'started': datetime.now().isoformat()
            }
        socketio.emit('playback_update', {'universe': universe, 'playback': self.current.get(universe)})
    
    def stop(self, universe=None):
        with self.lock:
            if universe:
                self.current.pop(universe, None)
            else:
                self.current.clear()
        socketio.emit('playback_update', {'universe': universe, 'playback': None})
    
    def get_status(self, universe=None):
        with self.lock:
            status = self.current.copy()
            # Include running effects
            if effects_engine and effects_engine.running:
                for effect_id in effects_engine.running.keys():
                    # Extract effect type from id (e.g., "strobe_1234" -> "strobe")
                    effect_type = effect_id.split('_')[0] if '_' in effect_id else effect_id
                    # Add to status as type 'effect'
                    status['effect'] = {
                        'type': 'effect',
                        'id': effect_id,
                        'name': effect_type.capitalize(),
                        'started': None
                    }
            if universe:
                return status.get(universe)
            return status

playback_manager = PlaybackManager()

# ============================================================
# Chase Playback Engine (DEPRECATED — F06 consolidation)
# ============================================================
# STATUS: ChaseEngine is no longer called from any API route.
# content_manager.play_chase() routes through UnifiedPlaybackEngine.
# This class is retained for backward compatibility with:
#   - show_engine._execute_event() 'chase' type → calls content_manager.play_chase()
#   - content_manager.stop_playback() → calls chase_engine.stop_all()
# Once ShowEngine is fully retired, this can be removed entirely.
# ============================================================================
class ChaseEngine:
    """Runs chases by streaming each step via UDPJSON to all universes.

    # ⚠️ AUTHORITY VIOLATION (TASK-0006) ⚠️
    # This engine MUST NOT own timing loops.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. Chase step computation
    # will be preserved as utilities called BY UnifiedPlaybackEngine.
    #
    # DO NOT START CHASES INDEPENDENTLY - See TASK_LEDGER.md

    RACE CONDITION FIX: Now routes all output through merge layer for proper
    priority-based merging with other playback sources (effects, scenes, etc.)
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.running_chases = {}  # {chase_id: thread}
        self.stop_flags = {}  # {chase_id: Event}
        # Health tracking for debugging
        self.chase_health = {}  # {chase_id: {"step": int, "last_time": float, "status": str}}
        # Merge layer source tracking - maps chase_id to source_id
        self._merge_sources = {}  # {chase_id: source_id}
        # Reference to merge layer (set after merge_layer is created)
        self._merge_layer = None

    def set_merge_layer(self, merge_layer_ref):
        """Set reference to merge layer for priority-based output"""
        self._merge_layer = merge_layer_ref

    def start_chase(self, chase, universes, fade_ms_override=None):
        """
        Start a chase on the given universes with optional fade override.

        # ⚠️ AUTHORITY VIOLATION WARNING ⚠️
        # This method spawns an independent chase thread, violating
        # AETHER Hard Rule 1.1. Only UnifiedPlaybackEngine should own timing.
        # See TASK-0006 in TASK_LEDGER.md
        """
        # PHASE 1 GUARD: Log violation when chase spawns independent thread
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: ChaseEngine.start_chase() spawning "
            "independent timing thread. This violates AETHER Hard Rule 1.1 - "
            "Only UnifiedPlaybackEngine should own playback timing. See TASK-0006"
        )

        chase_id = chase['chase_id']

        # ARBITRATION: Acquire chase ownership — returns token for TOCTOU safety [F08]
        arb_token = arbitration.acquire('chase', chase_id)
        if not arb_token:
            print(f"⚠️ Cannot start chase - arbitration denied (owner: {arbitration.current_owner})", flush=True)
            return False

        # Stop any other running chases first
        self.stop_chase(chase_id)

        # MERGE LAYER: Register as a merge source for proper priority handling
        if self._merge_layer:
            source_id = f"chase_{chase_id}"
            self._merge_layer.register_source(source_id, 'chase', universes)
            with self.lock:
                self._merge_sources[chase_id] = source_id
            print(f"📥 Chase '{chase['name']}' registered as merge source (priority=40)", flush=True)

        # Create stop flag
        stop_flag = threading.Event()
        self.stop_flags[chase_id] = stop_flag

        # Start chase thread with fade override and arbitration token [F08]
        thread = threading.Thread(
            target=self._run_chase,
            args=(chase, universes, stop_flag, fade_ms_override, arb_token),
            daemon=True
        )
        self.running_chases[chase_id] = thread
        thread.start()
        print(f"🏃 Chase engine started: {chase['name']} (fade_override={fade_ms_override})", flush=True)
        return True

    def stop_chase(self, chase_id=None, wait=True):
        """Stop a chase or all chases, optionally waiting for thread to finish"""
        threads_to_join = []
        sources_to_unregister = []
        with self.lock:
            if chase_id:
                if chase_id in self.stop_flags:
                    self.stop_flags[chase_id].set()
                    self.stop_flags.pop(chase_id, None)
                    thread = self.running_chases.pop(chase_id, None)
                    if thread and wait:
                        threads_to_join.append(thread)
                # Track merge source to unregister
                source_id = self._merge_sources.pop(chase_id, None)
                if source_id:
                    sources_to_unregister.append(source_id)
            else:
                # Stop all
                for flag in self.stop_flags.values():
                    flag.set()
                if wait:
                    threads_to_join = list(self.running_chases.values())
                self.stop_flags.clear()
                self.running_chases.clear()
                # Track all merge sources to unregister
                sources_to_unregister = list(self._merge_sources.values())
                self._merge_sources.clear()

            # ARBITRATION: Release chase ownership if no more chases running
            if not self.running_chases:
                arbitration.release('chase')

        # Wait for threads outside of lock to avoid deadlock
        if wait:
            for thread in threads_to_join:
                thread.join(timeout=0.5)  # Max 500ms wait per thread

        # MERGE LAYER: Unregister sources after threads have stopped
        if self._merge_layer and sources_to_unregister:
            for source_id in sources_to_unregister:
                self._merge_layer.unregister_source(source_id)
                print(f"📤 Chase unregistered from merge layer: {source_id}", flush=True)

    def stop_all(self):
        """Stop all running chases"""
        self.stop_chase(None)

    def _run_chase(self, chase, universes, stop_flag, fade_ms_override=None, arb_token=None):
        """Chase playback loop - runs in background thread"""
        chase_id = chase['chase_id']
        steps = chase.get('steps', [])
        bpm = chase.get('bpm', 120)
        loop = chase.get('loop', True)
        distribution_mode = chase.get('distribution_mode', 'unified')
        # Apply-time fade override > chase default > 0
        chase_fade_ms = fade_ms_override if fade_ms_override is not None else chase.get('fade_ms', 0)

        if not steps:
            print(f"⚠️ Chase {chase['name']} has no steps", flush=True)
            self.chase_health[chase_id] = {"step": -1, "last_time": time.time(), "status": "no_steps"}
            return

        # Default step interval from BPM (used if step doesn't have duration)
        default_interval = 60.0 / bpm
        step_index = 0
        loop_count = 0

        print(f"🎬 Chase '{chase['name']}': {len(steps)} steps, fade={chase_fade_ms}ms, universes={universes}", flush=True)
        self.chase_health[chase_id] = {"step": 0, "last_time": time.time(), "status": "running", "loop": 0}

        try:
            while not stop_flag.is_set():
                step = steps[step_index]
                channels = step.get('channels', {})
                # Use step fade or chase fade
                fade_ms = step.get('fade_ms', chase_fade_ms)
                # Calculate step duration: support both 'duration' and 'hold_ms' formats
                # hold_ms = time to hold AFTER fade completes
                # duration = total step time (legacy format)
                if 'hold_ms' in step:
                    # New format: fade_ms + hold_ms = total step time
                    step_duration_ms = fade_ms + step['hold_ms']
                elif 'duration' in step:
                    # Legacy format: duration is total step time
                    step_duration_ms = step['duration']
                else:
                    # Fallback to BPM timing
                    step_duration_ms = int(default_interval * 1000)

                # Update health heartbeat
                self.chase_health[chase_id] = {
                    "step": step_index,
                    "last_time": time.time(),
                    "status": "running",
                    "loop": loop_count,
                    "duration_ms": step_duration_ms,
                    "fade_ms": fade_ms,
                    "channel_count": len(channels)
                }

                # Log step transition (SSOT tracing)
                print(f"🔄 Chase '{chase['name']}' step {step_index}/{len(steps)-1} (loop {loop_count}): "
                      f"{len(channels)} channels, duration={step_duration_ms}ms, fade={fade_ms}ms", flush=True)

                # Send step to all universes in parallel for synchronized playback
                # TODO: Future improvement - store chase data on ESP nodes and trigger playback
                # locally for perfect sync. See: sync_chase_to_node() infrastructure already exists.
                # Would need ESP firmware to handle local chase playback with 'play_chase' command.
                def send_to_universe(univ, cid, sflag, token):
                    try:
                        # Pass chase_id, stop_flag, and arb_token for TOCTOU safety [F08]
                        self._send_step(univ, channels, fade_ms, distribution_mode, chase_id=cid, stop_flag=sflag, arb_token=token)
                    except Exception as e:
                        print(f"❌ Chase step send error (U{univ}): {e}", flush=True)

                # Check stop flag before spawning threads (race condition fix)
                if stop_flag.is_set():
                    break

                threads = [threading.Thread(target=send_to_universe, args=(univ, chase_id, stop_flag, arb_token)) for univ in universes]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()  # Wait for all sends to complete before timing next step

                # Wait for step duration (in seconds)
                stop_flag.wait(step_duration_ms / 1000.0)

                # Advance step
                step_index += 1
                if step_index >= len(steps):
                    if loop:
                        step_index = 0
                        loop_count += 1
                        print(f"🔁 Chase '{chase['name']}' loop {loop_count}", flush=True)
                    else:
                        break

        except Exception as e:
            print(f"❌ Chase '{chase['name']}' crashed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.chase_health[chase_id] = {"step": step_index, "last_time": time.time(), "status": f"error: {e}"}
        finally:
            self.chase_health[chase_id] = {"step": step_index, "last_time": time.time(), "status": "stopped"}
            print(f"⏹️ Chase '{chase['name']}' stopped after {loop_count} loops", flush=True)

    def _send_step(self, universe, channels, fade_ms=0, distribution_mode='unified', chase_id=None, stop_flag=None, arb_token=None):
        """Send chase step with intelligent distribution.

        [F08] TOCTOU FIX: Validates arb_token before writing to ensure another
        engine hasn't acquired arbitration since the chase started.

        distribution_mode: 'unified' = replicate to all, 'pixel' = unique per fixture"""
        # Check stop flag BEFORE writing (race condition fix)
        if stop_flag and stop_flag.is_set():
            return
        # [F08] Validate arbitration token — reject stale writes
        if arb_token is not None and not arbitration.validate_token(arb_token):
            return

        if not channels:
            return
        parsed = {}
        for key, value in channels.items():
            key_str = str(key)
            if ':' in key_str:
                parts = key_str.split(':')
                if int(parts[0]) == universe:
                    parsed[int(parts[1])] = value
            else:
                parsed[int(key_str)] = value
        if not parsed:
            return
        # Get fixtures and apply distribution mode
        fixtures = content_manager.get_fixtures(universe)
        if fixtures:
            fixtures = sorted(fixtures, key=lambda f: f.get('start_channel', 1))
            pattern_vals = list(parsed.values())
            expanded = {}
            if distribution_mode == 'pixel':
                # PIXEL: Each fixture gets unique sequential value
                for idx, fix in enumerate(fixtures):
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', 1)
                    val = pattern_vals[idx % len(pattern_vals)] if pattern_vals else 0
                    for ch in range(count):
                        expanded[start + ch] = val
            else:
                # UNIFIED: Replicate pattern to all fixtures
                for fix in fixtures:
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', len(pattern_vals))
                    for i in range(min(count, len(pattern_vals))):
                        expanded[start + i] = pattern_vals[i]
            if expanded:
                parsed = expanded

        # Check stop flag again before writing (double-check for race condition)
        if stop_flag and stop_flag.is_set():
            return

        # MERGE LAYER: Route through merge layer for proper priority handling
        if self._merge_layer and chase_id:
            source_id = self._merge_sources.get(chase_id)
            if source_id:
                # Update merge layer source channels
                self._merge_layer.set_source_channels(source_id, universe, parsed)
                # Compute merged output and send
                merged = self._merge_layer.compute_merge(universe)
                if merged:
                    # Send merged result to SSOT (content_manager handles node dispatch)
                    content_manager.set_channels(universe, {str(k): v for k, v in merged.items()}, fade_ms=fade_ms)
                return

        # Fallback: direct write if merge layer not available (legacy behavior)
        # [F08] Token-validated arbitration guard
        if arbitration and not arbitration.can_write('chase', token=arb_token):
            return
        content_manager.set_channels(universe, parsed, fade_ms=fade_ms)


chase_engine = ChaseEngine()
effects_engine = DynamicEffectsEngine()

# ============================================================
# Pixel Array Manager - Multi-fixture pixel-style control
# ============================================================
# Store active pixel array controllers by ID
_pixel_arrays: Dict[str, PixelArrayController] = {}

# ============================================================
# Arbitration Manager - Single source of truth for "who owns output"
# ============================================================
class ArbitrationManager:
    """
    Priority-based arbitration for DMX output control.
    Priority: BLACKOUT(100) > MANUAL(80) > EFFECT(60) > LOOK(50) > SEQUENCE(45) > CHASE(40) > SCENE(20) > IDLE(0)

    SSOT ENFORCEMENT: All DMX write attempts must check arbitration first.
    Rejected writes are tracked for diagnostics.

    [F08] TOKEN-BASED TOCTOU FIX:
    acquire() returns a monotonic token (int). Each new acquire increments the
    token. can_write() and validate_token() check that the caller's token matches
    the current token — if another engine acquired in between, the token is stale
    and the write is rejected. This prevents the race where Engine A acquires,
    Engine B force-acquires, and Engine A's subsequent write goes through unchecked.
    """
    PRIORITY = {'blackout': 100, 'manual': 80, 'effect': 60, 'look': 50, 'sequence': 45, 'chase': 40, 'scene': 20, 'idle': 0}

    def __init__(self):
        self.current_owner = 'idle'
        self.current_id = None
        self.blackout_active = False
        self.last_change = None
        self.lock = threading.Lock()
        self.history = []
        # [F08] Monotonic token — incremented on every successful acquire
        self._token = 0
        # SSOT diagnostics tracking
        self.rejected_writes = []  # Track rejected acquire attempts
        self.stale_writes = 0  # [F08] Count of writes rejected due to stale token
        self.last_writer = None  # Last service that successfully wrote
        self.last_scene_id = None  # Last scene played
        self.last_scene_time = None  # When last scene was played
        self.writes_per_service = {}  # Count writes per service type

    def acquire(self, owner_type, owner_id=None, force=False):
        """Acquire arbitration. Returns token (int > 0) on success, 0 on failure.

        [F08] The returned token must be passed to can_write() or validate_token()
        before writing DMX. A stale token (from a previous acquire) will be rejected.
        For backward compatibility, the token is also truthy (non-zero = success).
        """
        with self.lock:
            now = datetime.now().isoformat()
            if self.blackout_active and owner_type != 'blackout':
                self._track_rejection(owner_type, owner_id, 'blackout_active', now)
                return 0
            new_pri = self.PRIORITY.get(owner_type, 0)
            cur_pri = self.PRIORITY.get(self.current_owner, 0)
            if force or new_pri >= cur_pri:
                old = self.current_owner
                self.current_owner = owner_type
                self.current_id = owner_id
                self.last_change = now
                self.last_writer = owner_type
                # [F08] Increment token — invalidates all previous tokens
                self._token += 1
                token = self._token
                # Track scene plays specifically
                if owner_type == 'scene':
                    self.last_scene_id = owner_id
                    self.last_scene_time = now
                # Track writes per service
                self.writes_per_service[owner_type] = self.writes_per_service.get(owner_type, 0) + 1
                self.history.append({'time': now, 'from': old, 'to': owner_type, 'id': owner_id, 'action': 'acquire', 'token': token})
                if len(self.history) > 50: self.history = self.history[-50:]
                print(f"🎯 Arbitration: {old} → {owner_type} (token={token})", flush=True)
                return token
            self._track_rejection(owner_type, owner_id, f'priority_too_low (current: {self.current_owner})', now)
            return 0

    def validate_token(self, token):
        """[F08] Check if a token is still valid (i.e., no one acquired since).

        Returns True if the token matches the current token, False if stale.
        """
        with self.lock:
            if token == self._token:
                return True
            self.stale_writes += 1
            return False

    def _track_rejection(self, owner_type, owner_id, reason, timestamp):
        """Track rejected write attempts for diagnostics"""
        self.rejected_writes.append({
            'time': timestamp,
            'requester': owner_type,
            'requester_id': owner_id,
            'reason': reason,
            'current_owner': self.current_owner
        })
        if len(self.rejected_writes) > 20:
            self.rejected_writes = self.rejected_writes[-20:]
        print(f"⚠️ Arbitration REJECTED: {owner_type} (reason: {reason})", flush=True)

    def release(self, owner_type=None):
        with self.lock:
            if owner_type is None or self.current_owner == owner_type:
                old = self.current_owner
                self.current_owner = 'idle'
                self.current_id = None
                self._token += 1  # [F08] Invalidate tokens on release too
                self.last_change = datetime.now().isoformat()
                self.history.append({'time': self.last_change, 'from': old, 'to': 'idle', 'action': 'release', 'token': self._token})
                if len(self.history) > 50: self.history = self.history[-50:]

    def set_blackout(self, active):
        with self.lock:
            self.blackout_active = active
            self.current_owner = 'blackout' if active else 'idle'
            self._token += 1  # [F08] Blackout invalidates all tokens
            self.last_change = datetime.now().isoformat()
            print(f"{'⬛ BLACKOUT ACTIVE' if active else '🔓 Blackout released'} (token={self._token})", flush=True)

    def get_status(self):
        with self.lock:
            return {
                'current_owner': self.current_owner,
                'current_id': self.current_id,
                'blackout_active': self.blackout_active,
                'last_change': self.last_change,
                'token': self._token,
                'stale_writes': self.stale_writes,
                'last_writer': self.last_writer,
                'last_scene_id': self.last_scene_id,
                'last_scene_time': self.last_scene_time,
                'writes_per_service': dict(self.writes_per_service),
                'rejected_writes': self.rejected_writes[-5:],  # Last 5 rejections
                'history': self.history[-10:]
            }

    def can_write(self, owner_type, token=None):
        """Check if owner_type can currently write.

        [F08] If token is provided, also validates it hasn't been superseded.
        This prevents TOCTOU races where another engine acquired between
        the caller's acquire() and this can_write() check.
        """
        with self.lock:
            if self.blackout_active and owner_type != 'blackout': return False
            owner_ok = self.current_owner == owner_type or self.current_owner == 'idle'
            if not owner_ok:
                return False
            # [F08] Token validation — reject stale tokens
            if token is not None and token != self._token:
                self.stale_writes += 1
                return False
            return True

arbitration = ArbitrationManager()

# ============================================================
# Schedule Runner
# ============================================================
class ScheduleRunner:
    """Background scheduler that runs cron-based lighting events"""
    
    def __init__(self):
        self.schedules = []
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the schedule runner background thread"""
        if self.running:
            return
        self.running = True
        self.update_schedules()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("📅 Schedule runner started")
    
    def stop(self):
        """Stop the schedule runner"""
        self.running = False
        print("📅 Schedule runner stopped")
    
    def update_schedules(self):
        """Reload schedules from database"""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM schedules WHERE enabled = 1')
        rows = c.fetchall()
        conn.close()
        self.schedules = []
        for row in rows:
            self.schedules.append({
                'schedule_id': row[0], 'name': row[1], 'cron': row[2],
                'action_type': row[3], 'action_id': row[4]
            })
        print(f"📅 Loaded {len(self.schedules)} active schedules")
    
    def _parse_cron(self, cron_str):
        """Check if cron expression matches the current minute.

        [F14] Uses croniter library instead of fragile hand-rolled parser.
        Supports full cron syntax: ranges (1-5), lists (1,3,5), steps (*/5),
        combined expressions (1-5,10,15), day-of-week names, and more.
        """
        try:
            now = datetime.now()
            # croniter.match() returns True if 'now' matches the cron schedule
            return croniter.match(cron_str, now)
        except (ValueError, KeyError, TypeError) as e:
            print(f"⚠️ Cron parse error for '{cron_str}': {e}")
            return False
    
    def _run_loop(self):
        """Background loop checking schedules every minute"""
        last_check_minute = -1
        while self.running:
            now = datetime.now()
            # Only check once per minute
            if now.minute != last_check_minute:
                last_check_minute = now.minute
                for sched in self.schedules:
                    if self._parse_cron(sched['cron']):
                        print(f"⏰ Triggering schedule: {sched['name']}")
                        self.run_schedule(sched['schedule_id'])
            time.sleep(5)  # Check every 5 seconds for minute change
    
    def run_schedule(self, schedule_id):
        """Execute a schedule's action.

        RACE CONDITION NOTE: Schedules respect the arbitration system.
        If a higher priority source (effect, manual, etc.) is active,
        the schedule's action may be denied by arbitration. This is correct
        behavior - schedules shouldn't override live performances.
        """
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM schedules WHERE schedule_id = ?', (schedule_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {'success': False, 'error': 'Schedule not found'}

        action_type = row[3]
        action_id = row[4]
        schedule_name = row[1]

        # Check arbitration status before executing (for logging)
        current_owner = arbitration.current_owner
        if current_owner not in ('idle', 'scene', 'chase'):
            print(f"📅 Schedule '{schedule_name}' triggered but {current_owner} has control", flush=True)

        # Update last_run
        c.execute('UPDATE schedules SET last_run = ? WHERE schedule_id = ?',
                  (datetime.now().isoformat(), schedule_id))
        conn.commit()
        conn.close()

        # Execute action (underlying methods handle arbitration)
        try:
            if action_type == 'scene':
                result = content_manager.play_scene(action_id)
            elif action_type == 'chase':
                result = content_manager.play_chase(action_id)
            elif action_type == 'blackout':
                # Blackout is highest priority - always executes
                result = content_manager.blackout()
            else:
                result = {'success': False, 'error': f'Unknown action type: {action_type}'}

            success = result.get('success', False)
            if success:
                print(f"📅 Schedule '{schedule_name}' executed successfully", flush=True)
            else:
                print(f"📅 Schedule '{schedule_name}' denied: {result.get('error', 'unknown')}", flush=True)
            return result
        except Exception as e:
            print(f"❌ Schedule error: {e}")
            return {'success': False, 'error': str(e)}

schedule_runner = ScheduleRunner()

# ============================================================
# Timer Runner (Countdown Timers)
# ============================================================

class TimerRunner:
    """Background runner for countdown timers that can trigger actions"""

    def __init__(self):
        self.active_timers = {}  # timer_id -> threading.Timer
        self.lock = threading.Lock()

    def start_timer(self, timer_id, remaining_ms):
        """Start a countdown timer in the background"""
        with self.lock:
            # Cancel existing timer if any
            if timer_id in self.active_timers:
                self.active_timers[timer_id].cancel()

            # Schedule the completion callback
            delay_seconds = remaining_ms / 1000.0
            timer = threading.Timer(delay_seconds, self._on_timer_complete, args=[timer_id])
            timer.daemon = True
            timer.start()
            self.active_timers[timer_id] = timer
            print(f"⏱️ Timer '{timer_id}' started: {remaining_ms}ms")

    def stop_timer(self, timer_id):
        """Stop/cancel a running timer"""
        with self.lock:
            if timer_id in self.active_timers:
                self.active_timers[timer_id].cancel()
                del self.active_timers[timer_id]
                print(f"⏱️ Timer '{timer_id}' stopped")

    def _on_timer_complete(self, timer_id):
        """Called when a timer completes"""
        print(f"⏱️ Timer '{timer_id}' completed!")

        # Update database status
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT action_type, action_id, name FROM timers WHERE timer_id = ?', (timer_id,))
        row = c.fetchone()

        if row:
            action_type, action_id, name = row
            c.execute('''UPDATE timers SET status='completed', remaining_ms=0
                WHERE timer_id=?''', (timer_id,))
            conn.commit()

            # Execute action if defined
            if action_type and action_id:
                try:
                    if action_type == 'look':
                        looks_sequences_manager.get_look(action_id)
                        # Trigger look playback (simplified - would need playback integration)
                        print(f"⏱️ Timer '{name}' triggering look: {action_id}")
                    elif action_type == 'sequence':
                        print(f"⏱️ Timer '{name}' triggering sequence: {action_id}")
                    elif action_type == 'blackout':
                        print(f"⏱️ Timer '{name}' triggering blackout")
                except Exception as e:
                    print(f"❌ Timer action error: {e}")

            # Broadcast completion via WebSocket
            broadcast_ws({
                'type': 'timer_complete',
                'timer_id': timer_id,
                'name': name
            })

        conn.close()

        # Remove from active timers
        with self.lock:
            if timer_id in self.active_timers:
                del self.active_timers[timer_id]

timer_runner = TimerRunner()

# ============================================================
# Show Engine (Timeline Playback)
# ============================================================
# ============================================================
# Show Timeline Engine (F06 — last remaining timing violator)
# ============================================================
# STATUS: ShowEngine still owns its timeline thread (TASK-0007).
# Unlike ChaseEngine/RenderEngine/EffectsEngine which now route through
# UnifiedPlaybackEngine, shows have no unified equivalent yet.
# The _execute_event() dispatcher delegates to unified_play_look(),
# unified_play_sequence(), and content_manager.play_scene/chase()
# so individual event playback IS consolidated — only the meta-timeline
# scheduling thread remains as a violation.
# TODO: Port timeline scheduling into UnifiedPlaybackEngine session type.
# ============================================================================
class ShowEngine:
    """Plays back timeline-based shows with timed events.

    # ⚠️ AUTHORITY VIOLATION (TASK-0007) ⚠️
    # This engine MUST NOT own timing loops.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. Timeline event scheduling
    # will be preserved as utilities called BY UnifiedPlaybackEngine.
    #
    # DO NOT START SHOWS INDEPENDENTLY - See TASK_LEDGER.md

    RACE CONDITION FIX: Now integrates with merge layer for proper
    priority-based merging. Direct channel writes route through merge layer.
    """

    def __init__(self):
        self.current_show = None
        self.running = False
        self.thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.paused = False
        self.tempo = 1.0
        # MERGE LAYER: Reference and source tracking
        self._merge_layer = None
        self._merge_source_id = None
        self._last_look_session = None

    def set_merge_layer(self, merge_layer_ref):
        """Set reference to merge layer for priority-based output"""
        self._merge_layer = merge_layer_ref
    
    def play_show(self, show_id, universe=1):
        """
        Play a show timeline.

        # ⚠️ AUTHORITY VIOLATION WARNING ⚠️
        # This method spawns an independent timeline thread, violating
        # AETHER Hard Rule 1.1. Only UnifiedPlaybackEngine should own timing.
        # See TASK-0007 in TASK_LEDGER.md
        """
        # PHASE 1 GUARD: Log violation when show spawns independent thread
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: ShowEngine.play_show() spawning "
            "independent timing thread. This violates AETHER Hard Rule 1.1 - "
            "Only UnifiedPlaybackEngine should own playback timing. See TASK-0007"
        )

        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM shows WHERE show_id = ?', (show_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return {'success': False, 'error': 'Show not found'}

        timeline = json.loads(row[3]) if row[3] else []
        if not timeline:
            return {'success': False, 'error': 'Show has no timeline events'}

        # Stop any current show
        self.stop()

        # MERGE LAYER: Register as a show source for priority handling
        # Shows get sequence priority (45) for timeline-based playback
        if self._merge_layer:
            self._merge_source_id = f"show_{show_id}"
            # Determine all universes that might be affected by this show
            show_universes = [universe]
            if row[6] if len(row) > 6 else False:  # distributed mode
                # In distributed mode, show might affect multiple universes
                show_universes = list(range(1, 65))  # Support all possible universes
            self._merge_layer.register_source(self._merge_source_id, 'sequence', show_universes)
            print(f"📥 Show '{row[1]}' registered as merge source (priority=45)", flush=True)

        self.current_show = {
            'show_id': show_id,
            'name': row[1],
            'timeline': timeline,
            'distributed': row[6] if len(row) > 6 else False,
            'universe': universe
        }
        self.running = True
        self.stop_flag.clear()

        self.thread = threading.Thread(
            target=self._run_timeline,
            args=(timeline, universe, True),
            daemon=True
        )
        self.thread.start()

        print(f"🎬 Playing show '{row[1]}' on universe {universe}")
        return {'success': True, 'show_id': show_id, 'name': row[1]}
    
    def stop(self):
        """Stop current show"""
        if self.running:
            self.stop_flag.set()
            self.pause_flag.clear()
            self.running = False
            self.paused = False
            if self.current_show:
                print(f"⏹️ Show '{self.current_show['name']}' stopped")
            self.current_show = None
        # MERGE LAYER: Unregister source
        self._unregister_merge_source()

    def stop_silent(self):
        """Stop without blackout (for SSOT transitions)"""
        print(f"🛑 stop_silent called, running={self.running}", flush=True)
        if self.running:
            self.stop_flag.set()
            self.pause_flag.clear()
            self.running = False
            self.paused = False
            self.current_show = None
        # MERGE LAYER: Unregister source
        self._unregister_merge_source()

    def _unregister_merge_source(self):
        """Unregister from merge layer when stopping"""
        if self._merge_layer and self._merge_source_id:
            self._merge_layer.unregister_source(self._merge_source_id)
            print(f"📤 Show unregistered from merge layer: {self._merge_source_id}", flush=True)
            self._merge_source_id = None

    def pause(self):
        """Pause current show"""
        if self.running and not self.paused:
            self.pause_flag.set()
            self.paused = True
            print(f"⏸️ Show paused")

    def resume(self):
        """Resume paused show"""
        if self.running and self.paused:
            self.pause_flag.clear()
            self.paused = False
            print(f"▶️ Show resumed")

    def set_tempo(self, tempo):
        """Set playback tempo (0.25 to 100.0)"""
        self.tempo = max(0.25, min(100.0, tempo))
        print(f"⏩ Tempo set to {self.tempo}x")
    
    def _run_timeline(self, timeline, universe, loop=True):
        """Execute timeline events in sequence, with optional looping.

        [F15] Uses time.monotonic() to prevent NTP-induced timing jumps.
        Tempo scaling is applied to logical time tracking, not just sleep duration.
        """
        sorted_events = sorted(timeline, key=lambda x: x.get('time_ms', 0))

        while self.running and not self.stop_flag.is_set():
            start_time = time.monotonic()  # [F15] monotonic prevents NTP drift

            for event in sorted_events:
                if self.stop_flag.is_set():
                    break
                # Wait until event time (scaled by tempo)
                event_time_s = event.get('time_ms', 0) / 1000.0  # Convert ms to seconds

                while not self.stop_flag.is_set():
                    # Check pause — paused time doesn't count toward elapsed
                    while self.pause_flag.is_set() and not self.stop_flag.is_set():
                        time.sleep(0.1)
                        start_time += 0.1  # Shift origin so pause doesn't eat timeline
                    if self.stop_flag.is_set():
                        break

                    # [F15] Elapsed logical time = wall time * tempo
                    elapsed_wall = time.monotonic() - start_time
                    elapsed_logical = elapsed_wall * self.tempo
                    remaining = event_time_s - elapsed_logical

                    if remaining <= 0:
                        break  # Event time reached

                    # Sleep in small chunks, converting logical remaining to wall time
                    wall_remaining = remaining / self.tempo
                    sleep_chunk = min(wall_remaining, 0.1)
                    time.sleep(sleep_chunk)
                if self.stop_flag.is_set():
                    break
                # Execute the event
                # Check if distributed mode - extract all scene IDs from timeline
                distributed = self.current_show.get('distributed', False) if self.current_show else False
                all_scenes = None
                event_index = 0
                if distributed:
                    all_scenes = [e.get('scene_id') for e in sorted_events if e.get('type') == 'scene' and e.get('scene_id')]
                    event_index = [i for i, e in enumerate(sorted_events) if e.get('type') == 'scene'].index(
                        sorted_events.index(event)) if event.get('type') == 'scene' else 0
                self._execute_event(event, universe, distributed, all_scenes, event_index)
            
            if not loop or self.stop_flag.is_set():
                break
            print("🔁 Show looping...")
        
        self.running = False
        self.current_show = None
        print("🎬 Show playback stopped")
    def _execute_event(self, event, universe, distributed=False, all_scenes=None, event_index=0):
        """Execute a single timeline event.

        RACE CONDITION FIX: Check stop flag before each operation.
        Direct channel writes now route through merge layer for proper priority handling.
        """
        # Check stop flag before executing (race condition fix)
        if self.stop_flag.is_set():
            return

        # Support both old format (type) and new format (action_type)
        event_type = event.get('action_type') or event.get('type', 'scene')

        # Get universes from event if specified, otherwise use default
        event_universes = event.get('universes', [universe])
        if isinstance(event_universes, int):
            event_universes = [event_universes]

        try:
            if event_type == 'scene':
                scene_id = event.get('scene_id') or event.get('action_id')
                fade_ms = event.get('fade_ms', 500)

                if distributed and all_scenes:
                    # Get all online universes
                    all_nodes = node_manager.get_all_nodes(include_offline=False)
                    universes = sorted(set(node.get('universe', 1) for node in all_nodes))

                    # Send offset scenes to each universe
                    for i, univ in enumerate(universes):
                        if self.stop_flag.is_set():
                            return
                        offset_index = (event_index + i) % len(all_scenes)
                        offset_scene_id = all_scenes[offset_index]
                        content_manager.play_scene(offset_scene_id, fade_ms=fade_ms, universe=univ, skip_ssot=True)
                    print(f"  🌈 Distributed at {event.get('time_ms')}ms -> {len(universes)} universes")
                else:
                    if self.stop_flag.is_set():
                        return
                    content_manager.play_scene(scene_id, fade_ms=fade_ms, universe=universe, skip_ssot=True)
                    print(f"  ▶️ Scene '{scene_id}' at {event.get('time_ms')}ms")

            elif event_type == 'chase':
                if self.stop_flag.is_set():
                    return
                chase_id = event.get('chase_id') or event.get('action_id')
                content_manager.play_chase(chase_id, universe=universe)
                print(f"  ▶️ Chase '{chase_id}' at {event.get('time_ms')}ms")

            elif event_type == 'sequence':
                if self.stop_flag.is_set():
                    return
                sequence_id = event.get('action_id') or event.get('sequence_id')
                fade_ms = event.get('fade_ms', 0)

                # Load the sequence from database
                try:
                    sequence = looks_sequences_manager.get_sequence(sequence_id)
                    if not sequence or not sequence.steps:
                        print(f"  ❌ Sequence '{sequence_id}' not found or empty")
                    else:
                        # Build sequence_data dict for unified playback
                        steps = []
                        for step in sequence.steps:
                            step_data = {
                                'step_id': step.step_id,
                                'name': step.name,
                                'channels': step.channels or {},
                                'modifiers': step.modifiers or [],
                                'fade_ms': step.fade_ms,
                                'hold_ms': step.hold_ms,
                            }
                            if step.look_id:
                                step_data['look_id'] = step.look_id
                            steps.append(step_data)

                        sequence_data = {
                            'name': sequence.name,
                            'steps': steps,
                            'loop_mode': 'one_shot',  # Shows control timing, not sequence
                            'bpm': sequence.bpm,
                        }

                        # Play on specified universes
                        unified_play_sequence(
                            sequence_id,
                            sequence_data,
                            universes=event_universes
                        )
                        print(f"  ▶️ Sequence '{sequence.name}' at {event.get('time_ms')}ms on universes {event_universes}")
                except Exception as seq_err:
                    print(f"  ❌ Sequence play error: {seq_err}")

            elif event_type == 'look':
                if self.stop_flag.is_set():
                    return
                # Stop previous look session from this show
                if hasattr(self, '_last_look_session') and self._last_look_session:
                    from unified_playback import stop as unified_stop
                    unified_stop(self._last_look_session)
                look_id = event.get('look_id') or event.get('action_id')
                fade_ms = event.get('fade_ms', 500)
                try:
                    look = looks_sequences_manager.get_look(look_id)
                    if not look:
                        print(f"  ❌ Look '{look_id}' not found")
                    else:
                        look_data = look.to_dict()
                        # Extract universes from channel keys (e.g. "4:1")
                        lu = set()
                        for k in look.channels:
                            if ':' in str(k):
                                lu.add(int(str(k).split(':')[0]))
                        tgt = sorted(lu) if lu else event_universes
                        sid = unified_play_look(
                            look_id,
                            look_data,
                            universes=tgt,
                            fade_ms=fade_ms
                        )
                        self._last_look_session = sid
                        print(f"  ▶️ Look '{look.name}' at {event.get('time_ms')}ms on universes {tgt} (lu={lu})")
                except Exception as look_err:
                    print(f"  ❌ Look play error: {look_err}")

            elif event_type == 'blackout':
                if self.stop_flag.is_set():
                    return
                fade_ms = event.get('fade_ms', 1000)
                content_manager.blackout(universe=universe, fade_ms=fade_ms)
                print(f"  ⬛ Blackout at {event.get('time_ms')}ms")

            elif event_type == 'channels':
                if self.stop_flag.is_set():
                    return
                channels = event.get('channels', {})
                fade_ms = event.get('fade_ms', 0)

                # MERGE LAYER: Route direct channel writes through merge layer
                if self._merge_layer and self._merge_source_id:
                    # Convert channels to int keys for merge layer
                    parsed_channels = {int(k): int(v) for k, v in channels.items()}
                    self._merge_layer.set_source_channels(self._merge_source_id, universe, parsed_channels)
                    # Compute merged output
                    merged = self._merge_layer.compute_merge(universe)
                    if merged:
                        content_manager.set_channels(universe, {str(k): v for k, v in merged.items()}, fade_ms)
                else:
                    # Fallback: direct write
                    content_manager.set_channels(universe, channels, fade_ms)
                print(f"  🎛️ Channels at {event.get('time_ms')}ms")

        except Exception as e:
            print(f"  ❌ Event error: {e}")

show_engine = ShowEngine()

# ============================================================
# Settings Management
# ============================================================
DEFAULT_SETTINGS = {
    "theme": {"mode": "dark", "accentColor": "#3b82f6", "fontSize": "medium"},
    "background": {"type": "gradient", "gradient": "purple-blue", "bubbles": True, "bubbleCount": 15, "bubbleSpeed": 1.0},
    "ai": {"enabled": True, "model": "claude-3-sonnet", "contextLength": 4096, "temperature": 0.7},
    "dmx": {"defaultFadeMs": 500, "refreshRate": 40, "maxUniverse": 64},
    "security": {"pinEnabled": False, "sessionTimeout": 3600},
    "setup": {"complete": False, "mode": None, "userProfile": {}}
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                settings = DEFAULT_SETTINGS.copy()
                for key in saved:
                    if key in settings and isinstance(settings[key], dict):
                        settings[key].update(saved[key])
                    elif key in settings:
                        settings[key] = saved[key]
                return settings
    except Exception as e:
        print(f"⚠️ Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving settings: {e}")
        return False

app_settings = load_settings()

# ============================================================
# Database Setup
# [F04] Thread-local connection pool with WAL mode.
# Each thread gets ONE reusable connection instead of creating
# a new connection on every call (was ~87 connect/close cycles).
# WAL mode enables concurrent readers + single writer without
# "database is locked" errors from 35+ threads.
# ============================================================
_db_local = threading.local()

class _ThreadLocalConnection:
    """[F04] Wrapper that makes .close() a no-op so existing call sites
    don't kill the thread-local cached connection. The real connection
    is only closed by close_db() during thread/request teardown.
    All other sqlite3.Connection methods are proxied transparently.
    """
    __slots__ = ('_conn',)

    def __init__(self, conn):
        object.__setattr__(self, '_conn', conn)

    def close(self):
        pass  # No-op — lifecycle managed by close_db()

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._conn.executemany(*args, **kwargs)

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn.row_factory = value

    @property
    def total_changes(self):
        return self._conn.total_changes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # No-op — don't close on context manager exit

def get_db():
    """Get a thread-local SQLite connection (reused within same thread).
    [F04] Connections are cached per-thread to avoid the overhead and
    leak risk of creating a new connection on every call.
    Returns a wrapper that makes .close() a no-op for backward compat.
    """
    conn = getattr(_db_local, 'connection', None)
    if conn is not None:
        try:
            conn.execute('SELECT 1')  # Verify connection is still alive
            return _ThreadLocalConnection(conn)
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            # Connection was closed or broken — create a new one
            _db_local.connection = None
    conn = sqlite3.connect(DATABASE, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')       # [F04] WAL for concurrent access
    conn.execute('PRAGMA busy_timeout=5000')       # [F04] Wait up to 5s instead of failing
    conn.execute('PRAGMA synchronous=NORMAL')      # [F04] Safe with WAL, faster than FULL
    conn.execute('PRAGMA cache_size=-8000')         # [F04] 8MB cache per connection
    _db_local.connection = conn
    return _ThreadLocalConnection(conn)

def close_db(e=None):
    """Close the thread-local connection (for real).
    [F04] Called by Flask teardown_appcontext and can be called
    manually for non-Flask threads. This is the ONLY place
    connections actually close.
    """
    conn = getattr(_db_local, 'connection', None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _db_local.connection = None

def init_database():
    conn = get_db()
    c = conn.cursor()

    # [F04] Verify WAL mode is active
    wal_mode = c.execute('PRAGMA journal_mode').fetchone()[0]
    print(f"🗄️  SQLite journal mode: {wal_mode.upper()}")

    c.execute('''CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY, name TEXT, hostname TEXT, mac TEXT, ip TEXT,
        universe INTEGER DEFAULT 1, channel_start INTEGER DEFAULT 1, channel_end INTEGER DEFAULT 512,
        slice_mode TEXT DEFAULT 'zero_outside',
        mode TEXT DEFAULT 'output', type TEXT DEFAULT 'wifi', connection TEXT, firmware TEXT,
        status TEXT DEFAULT 'offline', is_builtin BOOLEAN DEFAULT 0, is_paired BOOLEAN DEFAULT 0,
        can_delete BOOLEAN DEFAULT 1, uptime INTEGER DEFAULT 0, rssi INTEGER DEFAULT 0, fps REAL DEFAULT 0,
        last_seen TIMESTAMP, first_seen TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        via_seance TEXT, seance_ip TEXT
    )''')

    # Add via_seance columns if they don't exist (for existing databases)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN via_seance TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN seance_ip TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists

    c.execute('''CREATE TABLE IF NOT EXISTS scenes (
        scene_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, universe INTEGER DEFAULT 1,
        channels TEXT, fade_ms INTEGER DEFAULT 500, curve TEXT DEFAULT 'linear', color TEXT DEFAULT '#3b82f6',
        icon TEXT DEFAULT 'lightbulb', is_favorite BOOLEAN DEFAULT 0, play_count INTEGER DEFAULT 0,
        synced_to_nodes BOOLEAN DEFAULT 0,
        distribution_mode TEXT DEFAULT 'unified',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS chases (
        chase_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, universe INTEGER DEFAULT 1,
        bpm INTEGER DEFAULT 120, loop BOOLEAN DEFAULT 1, steps TEXT, color TEXT DEFAULT '#10b981',
        fade_ms INTEGER DEFAULT 0,
        synced_to_nodes BOOLEAN DEFAULT 0,
        distribution_mode TEXT DEFAULT 'unified',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS groups (
        group_id TEXT PRIMARY KEY, name TEXT NOT NULL, universe INTEGER DEFAULT 1,
        channels TEXT, color TEXT DEFAULT '#8b5cf6', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT DEFAULT 'generic',
        manufacturer TEXT, model TEXT, universe INTEGER DEFAULT 1,
        start_channel INTEGER NOT NULL, channel_count INTEGER DEFAULT 1,
        channel_map TEXT, color TEXT DEFAULT '#8b5cf6', notes TEXT,
        rdm_uid TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS schedules (
        schedule_id TEXT PRIMARY KEY, name TEXT NOT NULL, cron TEXT NOT NULL,
        action_type TEXT NOT NULL, action_id TEXT,
        enabled BOOLEAN DEFAULT 1, last_run TIMESTAMP, next_run TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Timers table (countdown timers that trigger actions)
    c.execute('''CREATE TABLE IF NOT EXISTS timers (
        timer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        duration_ms INTEGER NOT NULL,
        remaining_ms INTEGER,
        action_type TEXT,
        action_id TEXT,
        status TEXT DEFAULT 'stopped',
        started_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # RDM (Remote Device Management) tables
    c.execute('''CREATE TABLE IF NOT EXISTS rdm_devices (
        uid TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        universe INTEGER,
        manufacturer_id INTEGER,
        device_model_id INTEGER,
        device_label TEXT,
        dmx_address INTEGER,
        dmx_footprint INTEGER,
        personality_id INTEGER,
        personality_count INTEGER,
        software_version TEXT,
        sensor_count INTEGER DEFAULT 0,
        last_seen TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (node_id) REFERENCES nodes(node_id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS rdm_personalities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_uid TEXT,
        personality_id INTEGER,
        slot_count INTEGER,
        description TEXT,
        FOREIGN KEY (device_uid) REFERENCES rdm_devices(uid)
    )''')

    # Node Groups - logical groupings of physical nodes acting as one universe
    c.execute('''CREATE TABLE IF NOT EXISTS node_groups (
        group_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        channel_mode TEXT DEFAULT 'auto',
        manual_channel_count INTEGER DEFAULT 512,
        node_order TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()

    # Add synced_to_nodes column if missing (migration)
    try:
        c.execute('ALTER TABLE scenes ADD COLUMN synced_to_nodes BOOLEAN DEFAULT 0')
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        c.execute('ALTER TABLE chases ADD COLUMN synced_to_nodes BOOLEAN DEFAULT 0')
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add fade_ms column to chases table for smooth transitions
    try:
        c.execute('ALTER TABLE chases ADD COLUMN fade_ms INTEGER DEFAULT 0')
        conn.commit()
        print("✓ Added fade_ms column to chases table")
    except sqlite3.OperationalError:
        pass  # Column already exists


    # Add distribution_mode to chases (unified/pixel)
    try:
        c.execute('ALTER TABLE chases ADD COLUMN distribution_mode TEXT DEFAULT \'unified\'')
        conn.commit()
        print("Added distribution_mode column to chases table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add node_id to fixtures (which Pulse node outputs this fixture)
    try:
        c.execute('ALTER TABLE fixtures ADD COLUMN node_id TEXT')
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    # Add fixture_ids to groups
    try:
        c.execute('ALTER TABLE groups ADD COLUMN fixture_ids TEXT')
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add slice_mode column to nodes table (migration for existing databases)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN slice_mode TEXT DEFAULT \'zero_outside\'')
        conn.commit()
        print("✓ Added slice_mode column to nodes table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add group_id to nodes table (for Node Groups feature)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN group_id TEXT REFERENCES node_groups(group_id)')
        conn.commit()
        print("✓ Added group_id column to nodes table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add channel_offset to nodes table (position within node group)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN channel_offset INTEGER DEFAULT 0')
        conn.commit()
        print("✓ Added channel_offset column to nodes table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add channel_ceiling to nodes table (calculated max channel needed)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN channel_ceiling INTEGER DEFAULT 512')
        conn.commit()
        print("✓ Added channel_ceiling column to nodes table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add hidden_from_dashboard to nodes table (hide built-in node from dashboard)
    try:
        c.execute('ALTER TABLE nodes ADD COLUMN hidden_from_dashboard BOOLEAN DEFAULT 0')
        conn.commit()
        print("✓ Added hidden_from_dashboard column to nodes table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # NOTE: Universe 1 built-in node removed - all nodes are WiFi ESP32 via UDPJSON

    print("✓ Database initialized [F04: thread-local pool + WAL mode]")
    conn.close()

# ============================================================
# Node Manager
# ============================================================
class NodeManager:
    """Node management with UDPJSON DMX output.

    DMX data is sent via UDP JSON commands to ESP32 nodes on port 6455.
    Protocol v2: {"v":2,"type":"set","u":N,"seq":M,"ch":[[ch,val],...]}
    Legacy v1:   {"type":"set","universe":N,"channels":{...},"ts":...}

    Event-driven: packets sent on value change, no continuous refresh required.
    Nodes hold last values until next update.
    """

    # Protocol version
    PROTOCOL_VERSION = 2  # v2: Compact ch/fill/frame encodings
    MAX_PAYLOAD_SIZE = 1200  # MTU-safe payload limit

    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.lock = threading.Lock()

        # Sequence number for duplicate detection
        self._seq = 0
        self._seq_lock = threading.Lock()

        # Diagnostics tracking - UDPJSON DMX output
        self._last_udpjson_send = None
        self._udpjson_send_count = 0
        self._udpjson_errors = 0
        self._udpjson_per_universe = {}  # {universe: send_count}

        # Diagnostics tracking - UDP config commands
        self._last_udp_send = None
        self._udp_send_count = 0

        # DMX refresh loop control
        self._refresh_running = False
        self._refresh_thread = None
        self._refresh_rate = 40  # Hz (frames per second)

        print(f"✅ DMX Transport: UDPJSON v{self.PROTOCOL_VERSION} (port {AETHER_UDPJSON_PORT})")

    def _next_seq(self):
        """Get next sequence number (thread-safe, wraps at 2^32)"""
        with self._seq_lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            return self._seq

    def _channels_to_compact(self, channels_dict):
        """Convert {channel: value} dict to compact [[ch,val],...] array.

        This is the v2 protocol format - much more compact than object keys.
        """
        return [[int(ch), int(val)] for ch, val in channels_dict.items()]

    def _estimate_payload_size(self, ch_pairs):
        """Estimate JSON payload size for channel pairs."""
        # Rough estimate: "[[1,255]," = 9 chars per pair average
        return 50 + len(ch_pairs) * 9  # 50 for header overhead

    # ─────────────────────────────────────────────────────────
    # UDPJSON DMX Protocol - Primary output method
    # ─────────────────────────────────────────────────────────
    def send_udpjson(self, ip, port, payload_dict):
        """Send a UDPJSON command to a node.

        Args:
            ip: Node IP address
            port: UDP port (typically AETHER_UDPJSON_PORT=6455)
            payload_dict: Dictionary to send as JSON

        Returns:
            True on success, False on error
        """
        try:
            json_data = json.dumps(payload_dict, separators=(',', ':'))

            # Debug: log what we're sending (format depends on v1 or v2)
            msg_type = payload_dict.get('type', 'unknown')
            version = payload_dict.get('v', 1)
            seq = payload_dict.get('seq', 0)
            universe = payload_dict.get('u') or payload_dict.get('universe', 0)

            # Count channels from either format
            ch_count = len(payload_dict.get('ch', [])) or len(payload_dict.get('channels', {}))

            # Rate-limited logging (only log every 100th packet or first few)
            if self._udpjson_send_count < 10 or self._udpjson_send_count % 100 == 0:
                print(f"📡 UDP v{version}: {ip}:{port} type={msg_type} u={universe} ch={ch_count} seq={seq} bytes={len(json_data)}", flush=True)

            self.udp_socket.sendto(json_data.encode(), (ip, port))

            # Track diagnostics
            self._udpjson_send_count += 1
            self._last_udpjson_send = time.time()

            if universe:
                self._udpjson_per_universe[universe] = self._udpjson_per_universe.get(universe, 0) + 1

            return True
        except Exception as e:
            self._udpjson_errors += 1
            print(f"❌ UDPJSON send error to {ip}:{port}: {e}")
            return False

    def send_udpjson_set(self, node_ip, universe, channels_dict, source="backend", fade_ms=0):
        """Send a 'set' command to a node via UDPJSON v2 protocol.

        Uses compact [[ch,val],...] format instead of {"ch":val,...} for smaller payloads.
        Automatically splits large payloads into multiple packets to stay under MTU.

        Args:
            node_ip: Node IP address
            universe: Universe number
            channels_dict: {channel: value, ...} dictionary
            source: Source identifier (e.g., "frontend", "chase", "scene")
            fade_ms: Optional fade duration in milliseconds
        """
        ch_pairs = self._channels_to_compact(channels_dict)

        # Check if we need to split the payload
        estimated_size = self._estimate_payload_size(ch_pairs)
        if estimated_size > self.MAX_PAYLOAD_SIZE and len(ch_pairs) > 50:
            # Split into multiple packets
            return self._send_chunked(node_ip, universe, ch_pairs, fade_ms, source)

        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "set",
            "u": universe,
            "seq": self._next_seq(),
            "ch": ch_pairs
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def _send_chunked(self, node_ip, universe, ch_pairs, fade_ms, source):
        """Send channel updates in multiple packets to stay under MTU."""
        chunk_size = 100  # ~900 bytes per chunk
        success = True
        for i in range(0, len(ch_pairs), chunk_size):
            chunk = ch_pairs[i:i + chunk_size]
            payload = {
                "v": self.PROTOCOL_VERSION,
                "type": "set",
                "u": universe,
                "seq": self._next_seq(),
                "ch": chunk
            }
            if fade_ms > 0:
                payload["fade"] = fade_ms
            if not self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload):
                success = False
        return success

    def send_udpjson_fade(self, node_ip, universe, channels_dict, duration_ms, easing="linear", source="backend"):
        """Send a 'set' command with fade to a node via UDPJSON v2 protocol.

        Note: v2 protocol uses 'fade' field on 'set' type, not separate 'fade' type.
        Nodes handle fading locally based on the fade duration.

        Args:
            node_ip: Node IP address
            universe: Universe number
            channels_dict: {channel: target_value, ...}
            duration_ms: Fade duration in milliseconds
            easing: Easing function (ignored in v2 - linear only for now)
            source: Source identifier
        """
        # v2 protocol: use set with fade parameter
        return self.send_udpjson_set(node_ip, universe, channels_dict, source, fade_ms=duration_ms)

    def send_udpjson_fill(self, node_ip, universe, ranges, fade_ms=0):
        """Send a 'fill' command for efficient range fills.

        Efficiently sets contiguous channel ranges to the same value.
        Ideal for blackouts, full-on, or wipes.

        Args:
            node_ip: Node IP address
            universe: Universe number
            ranges: List of [start, end, value] tuples
            fade_ms: Optional fade duration in milliseconds
        """
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "fill",
            "u": universe,
            "seq": self._next_seq(),
            "ranges": ranges
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def send_udpjson_blackout(self, node_ip, universe, fade_ms=0, source="backend"):
        """Send a 'blackout' command to a node via UDPJSON v2 protocol.

        Uses efficient fill command: ranges=[[1,512,0]] instead of sending 512 zeros.
        """
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "blackout",
            "u": universe,
            "seq": self._next_seq()
        }
        if fade_ms > 0:
            payload["fade"] = fade_ms

        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def send_udpjson_panic(self, node_ip, universe):
        """Send a 'panic' command - immediate blackout with no fade.

        SAFETY ACTION: This bypasses all playback/effects and commands
        immediate zero output on the target universe.
        """
        print(f"🚨 PANIC: Sending to {node_ip} universe {universe}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "panic",
            "u": universe,
            "seq": self._next_seq()
        }
        result = self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)
        if result:
            print(f"   ✓ PANIC sent successfully to {node_ip}:{universe}", flush=True)
        else:
            print(f"   ✗ PANIC FAILED to {node_ip}:{universe}", flush=True)
        return result

    def send_udpjson_ping(self, node_ip):
        """Send a 'ping' command to a node and expect a 'pong' response.

        SAFETY ACTION: Health check for node connectivity.
        """
        print(f"🏓 PING: Sending to {node_ip}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "ping",
            "seq": self._next_seq()
        }
        result = self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)
        if result:
            print(f"   ✓ PING sent to {node_ip}", flush=True)
        else:
            print(f"   ✗ PING FAILED to {node_ip}", flush=True)
        return result

    def send_udpjson_reset(self, node_ip):
        """Send a 'reset' command to a node.

        SAFETY ACTION: Commands node to reset its internal state.
        This clears any stuck effects, resets DMX output, and reinitializes.
        """
        print(f"🔄 RESET: Sending to {node_ip}", flush=True)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "reset",
            "seq": self._next_seq()
        }
        result = self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)
        if result:
            print(f"   ✓ RESET sent to {node_ip}", flush=True)
        else:
            print(f"   ✗ RESET FAILED to {node_ip}", flush=True)
        return result

    def start_dmx_refresh(self):
        """Start the continuous DMX refresh loop.

        Sends DMX data at 40fps to all active nodes via UDPJSON.
        """
        if self._refresh_running:
            print("⚠️ DMX Refresh: Already running")
            return

        self._refresh_running = True
        self._refresh_thread = threading.Thread(target=self._dmx_refresh_loop, daemon=True)
        self._refresh_thread.start()
        print(f"✅ DMX Refresh: Started at {self._refresh_rate} fps (UDPJSON on port {AETHER_UDPJSON_PORT})")
        print(f"   Thread alive: {self._refresh_thread.is_alive()}")

    def stop_dmx_refresh(self):
        """Stop the DMX refresh loop"""
        self._refresh_running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=1.0)
        print("⏹️ DMX Refresh: Stopped")

    def _refresh_node_cache(self):
        """Background refresh of node-to-universe mapping (runs off render thread).

        [F04] Optimized: single UPDATE subquery instead of row-by-row loop.
        Retry logic prevents 'database is locked' from crashing the cache refresh.
        """
        while self._refresh_running:
            try:
                conn = get_db()
                c = conn.cursor()

                # [F04] Single atomic UPDATE — replaces row-by-row loop that held write lock
                # Calculates channel_ceiling = min(512, max(1, max_fixture_channel + 16))
                for attempt in range(5):
                    try:
                        c.execute("""
                            UPDATE nodes SET channel_ceiling = MIN(512, MAX(1,
                                COALESCE((
                                    SELECT MAX(f.start_channel + f.channel_count - 1) + 16
                                    FROM fixtures f WHERE f.universe = nodes.universe
                                ), 1)
                            ))
                            WHERE is_paired = 1 AND ip IS NOT NULL
                        """)
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if 'locked' in str(e) and attempt < 4:
                            time.sleep(0.2 * (attempt + 1))  # 200ms, 400ms, 600ms, 800ms backoff
                        else:
                            raise  # Give up after 5 attempts (2s total backoff)

                # Now fetch node cache with channel_ceiling (read-only, no lock contention)
                c.execute("""
                    SELECT universe, ip, channel_start, channel_end, via_seance, seance_ip, channel_ceiling
                    FROM nodes
                    WHERE is_paired = 1 AND ip IS NOT NULL AND type = 'wifi'
                """)
                new_cache = {}
                for row in c.fetchall():
                    u, ip, ch_start, ch_end, via_seance, seance_ip, ceiling = row
                    if u == 1:
                        continue
                    if u not in new_cache:
                        new_cache[u] = []
                    target_ip = seance_ip if via_seance and seance_ip else ip
                    s = ch_start or 1
                    # Use channel_ceiling if set and in auto mode, otherwise use channel_end
                    e = min(ch_end or 512, ceiling or 512)
                    new_cache[u].append((target_ip, s, e, ip, via_seance, ceiling or 512))
                conn.close()
                self._node_cache = new_cache
            except Exception as ex:
                print(f"⚠️ Node cache refresh error: {ex}")
            time.sleep(5.0)  # [F04] Reduced from 2s — less write contention

    def _dmx_refresh_loop(self):
        """Background thread that sends DMX data to nodes via UDPJSON.

        All output uses the new UDPJSON protocol on port 6455.
        Optimized for smooth fading: monotonic timing, no blocking I/O in hot path.
        """
        frame_interval = 1.0 / self._refresh_rate
        frame_count = 0
        self._node_cache = {}
        self._last_sent = {}  # {universe:ip -> {ch_str: val}} delta tracking
        print(f"🔄 DMX Refresh loop starting (interval={frame_interval:.3f}s) - UDPJSON on port {AETHER_UDPJSON_PORT}")

        # Start background node cache refresh thread
        import threading
        node_thread = threading.Thread(target=self._refresh_node_cache, daemon=True)
        node_thread.start()

        while self._refresh_running:
            try:
                loop_start = time.monotonic()

                # Get active universes from dmx_state + cached nodes
                active_universes = set(dmx_state.universes.keys())
                universe_to_nodes = self._node_cache
                active_universes.update(universe_to_nodes.keys())

                # Log active universes periodically (every 5 seconds)
                frame_count += 1
                if frame_count == 1 or frame_count % (self._refresh_rate * 5) == 0:
                    print(f"🔄 DMX Refresh: universes={sorted(active_universes)}, udpjson sends={self._udpjson_send_count}")

                # Send DMX data for each active universe (skip universe 1)
                for universe in active_universes:
                    if not self._refresh_running:
                        break
                    if universe == 1:
                        continue

                    # Get output values (handles fade interpolation internally)
                    dmx_values = dmx_state.get_output_values(universe)

                    # Send to each node in this universe via UDPJSON
                    nodes = universe_to_nodes.get(universe, [])
                    for node_data in nodes:
                        # Unpack node data (now includes ceiling)
                        if len(node_data) == 6:
                            target_ip, slice_start, slice_end, original_ip, via_seance, ceiling = node_data
                        else:
                            # Backwards compatibility
                            target_ip, slice_start, slice_end, original_ip, via_seance = node_data
                            ceiling = slice_end

                        node_key = f"{universe}:{target_ip}"
                        prev_sent = self._last_sent.get(node_key, {})

                        # Smart channel optimization: only send up to ceiling
                        # This reduces network traffic significantly when only a few fixtures are used
                        effective_end = min(slice_end, ceiling)

                        # Build channels dict: include non-zero AND channels that changed to zero
                        node_channels = {}
                        for ch in range(slice_start, effective_end + 1):
                            val = dmx_values[ch - 1]
                            ch_str = str(ch)
                            if val > 0:
                                node_channels[ch_str] = val
                            elif ch_str in prev_sent:
                                # Was non-zero last frame, now zero — must send the zero
                                node_channels[ch_str] = 0

                        # Send either when there's data, or once per second for keepalive
                        if node_channels or frame_count % self._refresh_rate == 0:
                            self.send_udpjson_set(target_ip, universe, node_channels, source="refresh")

                        # Track what we sent for next frame's delta
                        self._last_sent[node_key] = {k: v for k, v in node_channels.items() if v > 0}

                # Maintain consistent frame rate using monotonic clock
                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"❌ DMX Refresh error: {e}")
                time.sleep(0.1)

    def _uart_listener(self):
        """Background thread to listen for heartbeats from gateway node"""
        buffer = ""
        while True:
            try:
                if self._uart and self._uart.is_open and self._uart.in_waiting:
                    data = self._uart.read(self._uart.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    # Process complete JSON lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._handle_gateway_message(line)
                time.sleep(0.1)
            except Exception as e:
                # Silent fail - gateway may not be connected
                time.sleep(1)

    def _handle_gateway_message(self, json_str):
        """Handle incoming message from gateway node"""
        try:
            data = json.loads(json_str)
            msg_type = data.get('type')

            if msg_type == 'heartbeat':
                # Register/update gateway node
                node_data = {
                    'node_id': data.get('node_id'),
                    'hostname': data.get('name', 'Gateway'),
                    'universe': data.get('universe', 1),
                    'slice_start': data.get('slice_start', 1),
                    'slice_end': data.get('slice_end', 512),
                    'slice_mode': data.get('slice_mode', 'zero_outside'),
                    'firmware': data.get('firmware', 'pulse-gateway'),
                    'transport': 'uart',
                    'uptime': data.get('uptime', 0)
                }
                self._register_gateway_node(node_data)
            elif msg_type == 'status':
                print(f"📡 Gateway status: {json_str}")
        except json.JSONDecodeError:
            pass  # Ignore malformed messages
        except Exception as e:
            print(f"⚠️ Gateway message error: {e}")

    def _register_gateway_node(self, data):
        """Register or update gateway node in database"""
        node_id = str(data.get('node_id'))
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
        existing = c.fetchone()
        now = datetime.now().isoformat()

        firmware_str = data.get('firmware', 'pulse-gateway')
        if 'uart' not in firmware_str.lower():
            firmware_str = f"{firmware_str} (UART)"

        if existing:
            c.execute('''UPDATE nodes SET
                hostname = COALESCE(?, hostname), uptime = COALESCE(?, uptime),
                firmware = COALESCE(?, firmware), status = 'online', last_seen = ?
                WHERE node_id = ?''',
                (data.get('hostname'), data.get('uptime'), firmware_str, now, node_id))
        else:
            c.execute('''INSERT INTO nodes (node_id, name, hostname, universe, channel_start,
                channel_end, slice_mode, type, firmware, status, is_paired, is_builtin,
                can_delete, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'gateway', ?, 'online', 1, 0, 0, ?, ?)''',
                (node_id, data.get('hostname', 'Gateway'), data.get('hostname'),
                 data.get('universe', 1), data.get('slice_start', 1), data.get('slice_end', 512),
                 data.get('slice_mode', 'zero_outside'), firmware_str, now, now))
        conn.commit()
        conn.close()

    def get_gateway_nodes_in_universe(self, universe):
        """Get gateway nodes in a universe"""
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND type = 'gateway'
                     AND status = 'online' ORDER BY channel_start''', (universe,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def send_to_uart_gateway(self, universe, channels_list):
        """Send DMX data to gateway node via UART

        Args:
            universe: DMX universe number
            channels_list: List of 512 channel values
        """
        if not self._uart or not self._uart.is_open:
            return False

        try:
            with self._uart_lock:
                command = {
                    'cmd': 'dmx',
                    'universe': universe,
                    'data': channels_list
                }
                json_data = json.dumps(command) + '\n'
                self._uart.write(json_data.encode('utf-8'))
                self._uart.flush()

                self._last_uart_send = {
                    'time': datetime.now().isoformat(),
                    'universe': universe
                }
                self._uart_send_count += 1
                return True
        except Exception as e:
            print(f"❌ UART send error: {e}")
            self._uart_errors += 1
            return False

    def send_config_to_gateway(self, config):
        """Send configuration to gateway node via UART"""
        if not self._uart or not self._uart.is_open:
            return False

        try:
            with self._uart_lock:
                command = {
                    'cmd': 'config',
                    'name': config.get('name', 'Gateway'),
                    'universe': config.get('universe', 1),
                    'channel_start': config.get('channel_start', 1),
                    'channel_end': config.get('channel_end', 512),
                    'slice_mode': config.get('slice_mode', 'zero_outside')
                }
                json_data = json.dumps(command) + '\n'
                self._uart.write(json_data.encode('utf-8'))
                self._uart.flush()
                print(f"📤 UART Gateway config: U{command['universe']} ch{command['channel_start']}-{command['channel_end']}")
                return True
        except Exception as e:
            print(f"❌ UART config error: {e}")
            return False

    def get_all_nodes(self, include_offline=True):
        conn = get_db()
        c = conn.cursor()
        if include_offline:
            c.execute('SELECT * FROM nodes ORDER BY universe, channel_start')
        else:
            c.execute('SELECT * FROM nodes WHERE status = "online" ORDER BY universe, channel_start')
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_node(self, node_id):
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE node_id = ?', (str(node_id),))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_nodes_in_universe(self, universe):
        """Get all paired/builtin nodes in a universe

        All nodes receive DMX via UDPJSON on port 6455.
        """
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND (is_paired = 1 OR is_builtin = 1)
                     ORDER BY channel_start''', (universe,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_wifi_nodes_in_universe(self, universe):
        """Get only WiFi nodes in a universe (for syncing content)"""
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND type = "wifi" 
                     AND (is_paired = 1) AND status = "online" ORDER BY channel_start''', (universe,))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def register_node(self, data):
        node_id = str(data.get('node_id'))
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM nodes WHERE node_id = ?', (node_id,))
        existing = c.fetchone()
        now = datetime.now().isoformat()

        was_offline = False
        # Build firmware string: prefer 'firmware' field, fall back to 'version'
        firmware_str = data.get('firmware') or data.get('version')
        transport_str = data.get('transport', '')
        if transport_str and firmware_str:
            firmware_str = f"{firmware_str} ({transport_str})"

        # Extract Seance routing info (if node is connected via Seance bridge)
        via_seance = data.get('via_seance')  # Seance node ID or SSID
        seance_ip = data.get('seance_ip')    # IP of Seance on Pi's network (192.168.50.x)
        original_ip = data.get('original_ip') or data.get('ip')  # Node's IP on Seance's AP network

        if existing:
            # Check if node was offline before updating, and current paired state
            c.execute('SELECT status, is_paired FROM nodes WHERE node_id = ?', (node_id,))
            row = c.fetchone()
            was_offline = row and row[0] == 'offline'
            pi_thinks_paired = row and row[1] == 1

            # Check if node reports unpaired status (intentional unpair or NVS wipe)
            node_reports_unpaired = data.get('is_paired') == False or data.get('waiting_for_config') == True

            # If Pi thinks node is paired but node reports unpaired, sync Pi's state
            if pi_thinks_paired and node_reports_unpaired:
                print(f"Node {node_id} reports unpaired - syncing Pi database")
                c.execute('UPDATE nodes SET is_paired = 0 WHERE node_id = ?', (node_id,))

            # Update basic fields that always come from heartbeats/registrations
            c.execute('''UPDATE nodes SET hostname = COALESCE(?, hostname), mac = COALESCE(?, mac),
                ip = COALESCE(?, ip), uptime = COALESCE(?, uptime), rssi = COALESCE(?, rssi),
                fps = COALESCE(?, fps), firmware = COALESCE(?, firmware), status = 'online', last_seen = ?,
                via_seance = ?, seance_ip = ?
                WHERE node_id = ?''',
                (data.get('hostname'), data.get('mac'), original_ip, data.get('uptime'),
                 data.get('rssi'), data.get('fps'), firmware_str, now, via_seance, seance_ip, node_id))

            # Also update slice config if provided in heartbeat (nodes now send full config)
            # This ensures Pi always has accurate slice info even for Seance-bridged nodes
            slice_start = data.get('slice_start')
            slice_end = data.get('slice_end')
            slice_mode = data.get('slice_mode')
            universe = data.get('u') or data.get('universe')

            if slice_start is not None and slice_end is not None:
                c.execute('''UPDATE nodes SET channel_start = ?, channel_end = ?,
                    slice_mode = COALESCE(?, slice_mode), universe = COALESCE(?, universe)
                    WHERE node_id = ?''',
                    (slice_start, slice_end, slice_mode, universe, node_id))

            # Log Seance routing changes
            if via_seance:
                print(f"📡 Node {node_id} via Seance: {via_seance} @ {seance_ip}")
        else:
            # Support both legacy (startChannel/channelCount) and new (slice_start/slice_end/slice_mode) fields
            slice_start = data.get('slice_start') or data.get('startChannel', 1)
            slice_end = data.get('slice_end') or data.get('channelCount', 512)
            slice_mode = data.get('slice_mode', 'zero_outside')
            c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start, type,
                channel_end, slice_mode, firmware, status, is_paired, first_seen, last_seen, via_seance, seance_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'wifi', ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (node_id, data.get('hostname', f'Node-{node_id[-4:]}'), data.get('hostname'),
                 data.get('mac'), original_ip, data.get('universe', 1), slice_start,
                 slice_end, slice_mode, firmware_str, 'online', False, now, now, via_seance, seance_ip))
            if via_seance:
                print(f"📡 New node {node_id} via Seance: {via_seance} @ {seance_ip}")
        conn.commit()
        conn.close()
        # Re-send config to paired WiFi nodes ONLY on reconnect (was offline, now online)
        node = self.get_node(node_id)
        if node and node.get('is_paired') and node.get('type') == 'wifi' and existing and was_offline:
            print(f"🔄 Re-sending config to reconnected node {node_id}")
            self.send_config_to_node(node, {
                'name': node.get('name'),
                'universe': node.get('universe', 1),
                'channel_start': node.get('channel_start', 1),
                'channel_end': node.get('channel_end', 512),
                'slice_mode': node.get('slice_mode', 'zero_outside')
            })
        self.broadcast_status()

        # Async sync node to Supabase (non-blocking)
        if SUPABASE_AVAILABLE and node:
            supabase = get_supabase_service()
            if supabase and supabase.is_enabled():
                cloud_submit(supabase.sync_node, node)

        return node

    def pair_node(self, node_id, config):
        conn = get_db()
        c = conn.cursor()

        # Support both legacy and new slice field names
        channel_start = config.get('channel_start') or config.get('channelStart', 1)
        channel_end = config.get('channel_end') or config.get('channelEnd', 512)
        slice_mode = config.get('slice_mode', 'zero_outside')

        c.execute('''UPDATE nodes SET name = COALESCE(?, name), universe = ?, channel_start = ?,
            channel_end = ?, slice_mode = ?, mode = COALESCE(?, 'output'), is_paired = 1 WHERE node_id = ?''',
            (config.get('name'), config.get('universe', 1), channel_start,
             channel_end, slice_mode, config.get('mode'), str(node_id)))
        conn.commit()
        conn.close()

        # Send config to node via UDP
        node = self.get_node(node_id)
        if node:
            self.send_config_to_node(node, config)
            self.sync_content_to_node(node)
            print(f"✅ Node paired: {node.get('name')} on U{config.get('universe', 1)} ch{channel_start}-{channel_end} ({slice_mode})")

        self.broadcast_status()

        # Async sync node to Supabase (non-blocking)
        if SUPABASE_AVAILABLE and node:
            supabase = get_supabase_service()
            if supabase and supabase.is_enabled():
                cloud_submit(supabase.sync_node, node)

        return node

    def unpair_node(self, node_id):
        # Get node info before updating DB
        node = self.get_node(node_id)

        # Send unpair command to WiFi node to clear its config
        if node and node.get('type') == 'wifi' and node.get('ip'):
            # Route through Seance if node is connected via Seance bridge
            target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            self.send_command_to_wifi(target_ip, {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({target_ip})")

        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE nodes SET is_paired = 0 WHERE node_id = ? AND can_delete = 1', (str(node_id),))
        conn.commit()
        conn.close()
        self.broadcast_status()

    def delete_node(self, node_id):
        # Get node info before deleting
        node = self.get_node(node_id)

        # Send unpair command to WiFi node to clear its config
        if node and node.get('type') == 'wifi' and node.get('ip'):
            # Route through Seance if node is connected via Seance bridge
            target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            self.send_command_to_wifi(target_ip, {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({target_ip})")

        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM nodes WHERE node_id = ? AND can_delete = 1', (str(node_id),))
        conn.commit()
        conn.close()
        self.broadcast_status()

    def check_stale_nodes(self):
        conn = get_db()
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(seconds=STALE_TIMEOUT)).isoformat()
        # Find which nodes are going offline BEFORE updating
        c.execute('SELECT node_id FROM nodes WHERE last_seen < ? AND status = "online" AND is_builtin = 0', (cutoff,))
        stale_node_ids = [row['node_id'] for row in c.fetchall()]
        c.execute('UPDATE nodes SET status = "offline" WHERE last_seen < ? AND status = "online" AND is_builtin = 0', (cutoff,))
        if c.rowcount > 0:
            conn.commit()
            self.broadcast_status()
            # Mark all RDM devices on stale nodes as offline in live_inventory
            for node_id in stale_node_ids:
                try:
                    rdm_manager.mark_node_devices_offline(node_id)
                except Exception as e:
                    print(f"⚠️ RDM inventory update failed for stale node {node_id}: {e}", flush=True)
        conn.close()

    def broadcast_status(self):
        nodes = self.get_all_nodes()
        socketio.emit('nodes_update', {'nodes': nodes, 'timestamp': datetime.now().isoformat()})

    # ─────────────────────────────────────────────────────────
    # Channel Translation for Universe Splitting
    # ─────────────────────────────────────────────────────────
    def translate_channels_for_node(self, node, channels):
        """Translate universe channels to channels within node's range"""
        node_start = node.get('channel_start', 1)
        node_end = node.get('channel_end', 512)
        translated = {}
        for ch_str, value in channels.items():
            ch = int(ch_str)
            if node_start <= ch <= node_end:
                # Keep original channel number - node knows its range
                translated[str(ch)] = value
        return translated

    # ─────────────────────────────────────────────────────────
    # Send Commands to Nodes - UDPJSON
    # ─────────────────────────────────────────────────────────
    def send_to_node(self, node, channels_dict, fade_ms=0):
        """Send DMX values to a node via UDPJSON

        All DMX output goes through UDPJSON to ESP32 nodes on port 6455.
        """
        universe = node.get("universe", 1)

        # Universe 1 is offline
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - skipping send to node", flush=True)
            return False

        non_zero = sum(1 for v in channels_dict.values() if v > 0) if channels_dict else 0
        print(f"📡 UDPJSON: U{universe} -> {len(channels_dict) if channels_dict else 0} ch ({non_zero} non-zero), fade={fade_ms}ms", flush=True)

        return self.update_dmx_state(universe, channels_dict, fade_ms)

    def update_dmx_state(self, universe, channels_dict, fade_ms=0):
        """Update DMX state for a universe - the refresh loop handles UDPJSON output

        SSOT COMPLIANCE: This method ONLY updates dmx_state.
        The continuous refresh loop (_dmx_refresh_loop) handles:
        - Sending UDPJSON commands at consistent 40fps
        - [F07] ESP32 handles fades; SSOT returns target values via get_output_values()
        """
        # Universe 1 is offline
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - not updating state", flush=True)
            return False

        try:
            non_zero = sum(1 for v in channels_dict.values() if v > 0) if channels_dict else 0

            if fade_ms > 0:
                print(f"📤 SSOT U{universe} -> {len(channels_dict)} ch ({non_zero} non-zero), fade={fade_ms}ms", flush=True)
            else:
                print(f"📤 SSOT U{universe} -> {len(channels_dict)} ch ({non_zero} non-zero), snap", flush=True)

            # Update SSOT state with fade info - refresh loop handles UDPJSON output
            dmx_state.set_channels(universe, channels_dict, fade_ms=fade_ms)

            return True

        except Exception as e:
            print(f"❌ SSOT update error: {e}")
            return False

    def send_command_to_wifi(self, ip, command):
        """Send config command to WiFi node (not DMX data)"""
        try:
            json_data = json.dumps(command)
            self.udp_socket.sendto(json_data.encode(), (ip, WIFI_COMMAND_PORT))
            self._last_udp_send = datetime.now().isoformat()
            self._udp_send_count += 1
            return True
        except Exception as e:
            print(f"❌ UDP command error to {ip}: {e}")
            return False

    def send_blackout(self, node, fade_ms=1000):
        """Send blackout to a node via UDPJSON with fade"""
        universe = node.get('universe', 1)
        if universe == 1:
            print(f"⚠️ Universe 1 is offline - skipping blackout", flush=True)
            return False
        all_zeros = {str(ch): 0 for ch in range(1, 513)}
        return self.update_dmx_state(universe, all_zeros, fade_ms=fade_ms)

    def send_config_to_node(self, node, config):
        """Send configuration update to a WiFi or gateway node"""
        node_type = node.get('type')

        universe = config.get('universe', node.get('universe', 1))

        # Support both legacy and new slice field names
        channel_start = config.get('channel_start') or config.get('channelStart') or node.get('channel_start', 1)
        channel_end = config.get('channel_end') or config.get('channelEnd') or node.get('channel_end', 512)
        slice_mode = config.get('slice_mode', node.get('slice_mode', 'zero_outside'))

        if node_type == 'gateway':
            # Send config to gateway via UART
            result = self.send_config_to_gateway({
                'name': config.get('name', node.get('name')),
                'universe': universe,
                'channel_start': channel_start,
                'channel_end': channel_end,
                'slice_mode': slice_mode
            })
        elif node_type == 'wifi':
            # Send config to ESP32 via UDP - include both new and legacy field names
            command = {
                'cmd': 'config',
                'name': config.get('name', node.get('name')),
                'universe': universe,
                'channel_start': channel_start,
                'channel_end': channel_end,
                'slice_mode': slice_mode,
                # Legacy field names for backward compatibility with older firmware
                'startChannel': channel_start,
                'channelCount': channel_end
            }
            # Route through Seance if node is connected via Seance bridge
            if node.get('via_seance') and node.get('seance_ip'):
                target_ip = node.get('seance_ip')
                # Add routing info for Seance to forward to correct node
                command['node_id'] = node.get('node_id')
                command['_route_to'] = node.get('ip')  # Node's IP on Seance's AP network
                print(f"📡 Config via Seance: {node.get('node_id')} -> {target_ip}:8888 (route to {node.get('ip')})", flush=True)
            else:
                target_ip = node.get('ip')
                print(f"📡 Config direct: {node.get('node_id')} -> {target_ip}:8888", flush=True)
            result = self.send_command_to_wifi(target_ip, command)
            print(f"📡 Config result: {result}", flush=True)
        else:
            return False

        return result

    # ─────────────────────────────────────────────────────────
    # Sync Content to Nodes (Scenes/Chases)
    # ─────────────────────────────────────────────────────────
    def sync_scene_to_node(self, node, scene):
        """Send a scene to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False

        # Route through Seance if node is connected via Seance bridge
        target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')

        # Filter channels for this node's range
        node_channels = self.translate_channels_for_node(node, scene.get('channels', {}))
        
        if not node_channels:
            print(f"  ⚠️ Scene '{scene['name']}' has no channels for {node['name']}")
            return True  # Not an error, just nothing to sync
        
        command = {
            'cmd': 'store_scene',
            'id': scene['scene_id'],
            'name': scene['name'],
            'channels': node_channels,
            'fade_ms': scene.get('fade_ms', 500)
        }
        
        # Send in chunks if needed (large scenes)
        json_data = json.dumps(command)
        if len(json_data) > 1400:  # Near MTU limit
            # Send scene metadata first
            meta_cmd = {
                'cmd': 'store_scene',
                'id': scene['scene_id'],
                'name': scene['name'],
                'channels': {},
                'fade_ms': scene.get('fade_ms', 500)
            }
            self.send_command_to_wifi(target_ip, meta_cmd)
            time.sleep(CHUNK_DELAY)

            # Then send channels in chunks
            channel_items = list(node_channels.items())
            for i in range(0, len(channel_items), CHUNK_SIZE * 2):
                chunk = dict(channel_items[i:i + CHUNK_SIZE * 2])
                chunk_cmd = {
                    'cmd': 'set_channels',
                    'channels': chunk,
                    'fade_ms': 0
                }
                self.send_command_to_wifi(target_ip, chunk_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']} (chunked)")
        else:
            self.send_command_to_wifi(target_ip, command)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']}")
        
        return True

    def sync_chase_to_node(self, node, chase):
        """Send a chase to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False

        # Route through Seance if node is connected via Seance bridge
        target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')

        # Filter each step's channels for this node's range
        filtered_steps = []
        for step in chase.get('steps', []):
            step_channels = step.get('channels', {})
            node_channels = self.translate_channels_for_node(node, step_channels)
            if node_channels:
                filtered_steps.append({'channels': node_channels})
        
        if not filtered_steps:
            print(f"  ⚠️ Chase '{chase['name']}' has no channels for {node['name']}")
            return True
        
        command = {
            'cmd': 'store_chase',
            'id': chase['chase_id'],
            'name': chase['name'],
            'bpm': chase.get('bpm', 120),
            'loop': chase.get('loop', True),
            'steps': filtered_steps
        }
        
        # Check size and send
        json_data = json.dumps(command)
        if len(json_data) > 1400:
            # Large chase - need to send in parts
            # First clear and send metadata
            meta_cmd = {
                'cmd': 'store_chase',
                'id': chase['chase_id'],
                'name': chase['name'],
                'bpm': chase.get('bpm', 120),
                'loop': chase.get('loop', True),
                'steps': []
            }
            self.send_command_to_wifi(target_ip, meta_cmd)
            time.sleep(CHUNK_DELAY)

            # Send steps in batches
            for i in range(0, len(filtered_steps), 5):
                batch_steps = filtered_steps[i:i+5]
                batch_cmd = {
                    'cmd': 'append_chase_steps',
                    'id': chase['chase_id'],
                    'steps': batch_steps
                }
                self.send_command_to_wifi(target_ip, batch_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Chase '{chase['name']}' -> {node['name']} (chunked, {len(filtered_steps)} steps)")
        else:
            self.send_command_to_wifi(target_ip, command)
            print(f"  📤 Chase '{chase['name']}' -> {node['name']} ({len(filtered_steps)} steps)")
        
        return True

    def sync_content_to_node(self, node):
        """Sync all scenes and chases to a single node"""
        if node.get('type') != 'wifi':
            return
        
        universe = node.get('universe', 1)
        node_name = node.get('name') or node.get('node_id', 'unknown')
        print(f"🔄 Syncing content to {node_name} (U{universe})")
        
        # Get all scenes for this universe
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes WHERE universe = ?', (universe,))
        scenes = [dict(row) for row in c.fetchall()]
        
        c.execute('SELECT * FROM chases WHERE universe = ?', (universe,))
        chases = [dict(row) for row in c.fetchall()]
        conn.close()
        
        # Sync scenes
        for scene in scenes:
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            self.sync_scene_to_node(node, scene)
            time.sleep(CHUNK_DELAY)
        
        # Sync chases
        for chase in chases:
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            self.sync_chase_to_node(node, chase)
            time.sleep(CHUNK_DELAY)
        
        print(f"✓ Synced {len(scenes)} scenes, {len(chases)} chases to {node_name}")

    def sync_all_content(self):
        """Sync all content to all paired WiFi nodes"""
        print("🔄 Starting full content sync to all nodes...")
        nodes = self.get_all_nodes(include_offline=False)
        for node in nodes:
            if node.get('type') == 'wifi' and node.get('is_paired'):
                self.sync_content_to_node(node)
        print("✓ Full sync complete")

    # ─────────────────────────────────────────────────────────
    # Playback Commands
    # ─────────────────────────────────────────────────────────
    def play_scene_on_nodes(self, universe, scene_id, fade_ms=None):
        """Tell all nodes in universe to play a stored scene"""
        nodes = self.get_nodes_in_universe(universe)
        results = []

        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                command = {'cmd': 'play_scene', 'id': scene_id}
                if fade_ms is not None:
                    command['fade_ms'] = fade_ms
                success = self.send_command_to_wifi(target_ip, command)
                results.append({'node': node['name'], 'success': success})

        return results

    def play_chase_on_nodes(self, universe, chase_id):
        """Tell all nodes in universe to play a stored chase"""
        nodes = self.get_nodes_in_universe(universe)
        results = []

        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                command = {'cmd': 'play_chase', 'id': chase_id}
                success = self.send_command_to_wifi(target_ip, command)
                results.append({'node': node['name'], 'success': success})

        return results

    def stop_playback_on_nodes(self, universe=None):
        """Tell nodes to stop playback"""
        if universe:
            nodes = self.get_nodes_in_universe(universe)
        else:
            nodes = self.get_all_nodes(include_offline=False)

        results = []
        for node in nodes:
            if node.get('type') == 'wifi':
                # Route through Seance if node is connected via Seance bridge
                target_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
                success = self.send_command_to_wifi(target_ip, {'cmd': 'stop'})
                results.append({'node': node['name'], 'success': success})

        return results

node_manager = NodeManager()

# ============================================================
# RDM Manager - Remote Device Management
# ============================================================
@dataclass
class RDMDevice:
    """Represents an RDM-capable device discovered on the bus."""
    uid: str                           # Unique ID: "XXXX:XXXXXXXX" (manufacturer:device)
    manufacturer_id: int = 0
    device_model_id: int = 0
    dmx_address: int = 0
    dmx_footprint: int = 0
    personality_id: int = 1
    personality_count: int = 1
    software_version: int = 0
    sensor_count: int = 0
    label: str = ""                    # User-assigned label
    discovered_via: str = ""           # Node ID that discovered this device
    discovered_at: str = ""            # ISO timestamp
    last_seen: str = ""               # ISO timestamp
    manufacturer_name: str = ""        # Resolved from manufacturer ID
    model_name: str = ""              # Resolved from device info

    def to_dict(self):
        return {
            'uid': self.uid, 'manufacturer_id': self.manufacturer_id,
            'device_model_id': self.device_model_id, 'dmx_address': self.dmx_address,
            'dmx_footprint': self.dmx_footprint, 'personality_id': self.personality_id,
            'personality_count': self.personality_count, 'software_version': self.software_version,
            'sensor_count': self.sensor_count, 'label': self.label,
            'discovered_via': self.discovered_via, 'discovered_at': self.discovered_at,
            'last_seen': self.last_seen, 'manufacturer_name': self.manufacturer_name,
            'model_name': self.model_name,
        }


KNOWN_MANUFACTURERS = {
    0x0000: "PLASA (Development)", 0x0001: "ESTA (Standards)",
    0x414C: "Avolites", 0x4144: "ADJ", 0x4348: "Chauvet",
    0x434D: "City Theatrical", 0x454C: "ETC", 0x4D41: "Martin",
    0x5052: "PR Lighting", 0x524F: "Robe", 0x534C: "Signify (Philips)",
    0x5354: "Strong Entertainment", 0x5641: "Varilite",
}


class RDMManager:
    """Consolidated RDM (Remote Device Management) — Single Source of Truth.

    Sends RDM commands to ESP32 nodes via UDPJSON and processes responses.
    Maintains authoritative live_inventory of all known RDM devices.
    Emits rdm_update via SocketIO when device status changes.
    """

    RDM_TIMEOUT_MS = 5000
    DISCOVERY_TIMEOUT_MS = 30000
    UDP_PORT = 6455

    def __init__(self):
        self.discovery_tasks = {}  # node_id -> discovery status
        self.response_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.response_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_socket.settimeout(10.0)
        self.pending_requests = {}  # request_id -> callback

        # Authoritative live inventory
        # { "uid": { "status": "online"/"offline"/"stale", "temp": 0.0, "is_patched": bool } }
        self.live_inventory = {}

        # In-memory device cache
        self._devices: Dict[str, RDMDevice] = {}
        self._lock = threading.RLock()
        self._last_discovery = None
        self._discovery_in_progress = False
        self._last_emit_time = 0.0

        # External references (set after init)
        self._node_manager = None
        self._socketio = None
        self._playback_engine = None

        # Hydrate in-memory cache from database (devices from previous sessions)
        self._hydrate_from_db()

        print("✓ RDMManager initialized (consolidated)")

    def _hydrate_from_db(self):
        """Load existing RDM devices from database into in-memory cache on startup."""
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute('SELECT * FROM rdm_devices ORDER BY node_id, dmx_address')
            columns = [d[0] for d in c.description]
            rows = c.fetchall()
            with self._lock:
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    uid = row_dict['uid']
                    self._devices[uid] = RDMDevice(
                        uid=uid,
                        manufacturer_id=row_dict.get('manufacturer_id', 0) or 0,
                        device_model_id=row_dict.get('device_model_id', 0) or 0,
                        dmx_address=row_dict.get('dmx_address', 0) or 0,
                        dmx_footprint=row_dict.get('dmx_footprint', 0) or 0,
                        personality_id=row_dict.get('personality_id', 1) or 1,
                        personality_count=row_dict.get('personality_count', 1) or 1,
                        software_version=int(row_dict.get('software_version', 0) or 0),
                        sensor_count=row_dict.get('sensor_count', 0) or 0,
                        label=row_dict.get('device_label', '') or '',
                        discovered_via=row_dict.get('node_id', '') or '',
                        discovered_at=str(row_dict.get('created_at', '')) or '',
                        last_seen=str(row_dict.get('last_seen', '')) or '',
                        manufacturer_name=KNOWN_MANUFACTURERS.get(row_dict.get('manufacturer_id', 0), 'Unknown'),
                        model_name=''
                    )
                    # Also populate live_inventory with "offline" status (will be updated by heartbeats)
                    if uid not in self.live_inventory:
                        self.live_inventory[uid] = {
                            'status': 'offline',
                            'temp': 0.0,
                            'is_patched': False
                        }
            if rows:
                print(f"  ↳ Hydrated {len(rows)} RDM devices from database")
        except Exception as e:
            print(f"  ↳ Warning: Could not hydrate RDM cache from DB: {e}")

    # ─────────────────────────────────────────────────────────
    # External Wiring
    # ─────────────────────────────────────────────────────────

    def set_node_manager(self, nm):
        """Set reference to NodeManager."""
        self._node_manager = nm

    def set_socketio(self, sio):
        """Set reference to Flask-SocketIO for real-time updates."""
        self._socketio = sio

    def set_playback_engine(self, engine):
        """Set reference to UnifiedPlaybackEngine for offset auto-cleanup."""
        self._playback_engine = engine

    # ─────────────────────────────────────────────────────────
    # Live Inventory & SocketIO
    # ─────────────────────────────────────────────────────────

    def _emit_rdm_update(self):
        """Emit live_inventory to all connected clients via SocketIO (throttled)."""
        if not self._socketio:
            return
        now = time.monotonic()
        if now - self._last_emit_time < 0.5:  # Max 2 emits/sec
            return
        self._last_emit_time = now
        self._socketio.emit('rdm_update', {
            'inventory': self.live_inventory,
            'device_count': len(self._devices),
            'timestamp': datetime.now().isoformat()
        })

    def update_inventory(self, uid, status, temp=0.0, is_patched=None):
        """Update a device's live inventory entry. Emits on status change."""
        old_entry = self.live_inventory.get(uid, {})
        old_status = old_entry.get('status', 'unknown')

        # Check is_patched from fixtures table if not provided
        if is_patched is None:
            is_patched = old_entry.get('is_patched', False)

        self.live_inventory[uid] = {
            'status': status,
            'temp': temp,
            'is_patched': is_patched,
        }

        # Auto-cleanup AI offsets when device returns to healthy
        if old_status != 'online' and status == 'online' and self._playback_engine:
            fixture_id = self._resolve_fixture_for_rdm_uid(uid)
            if fixture_id:
                self._playback_engine.clear_offsets_for_fixture(fixture_id)

        if old_status != status:
            self._emit_rdm_update()

    def mark_node_devices_offline(self, node_id):
        """Mark all devices discovered via a node as offline."""
        with self._lock:
            for uid, dev in self._devices.items():
                if dev.discovered_via == node_id:
                    self.update_inventory(uid, 'offline')

    def _resolve_fixture_for_rdm_uid(self, uid):
        """Look up fixture_id in the fixtures table for an RDM UID."""
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute('SELECT fixture_id FROM fixtures WHERE rdm_uid = ?', (uid,))
            row = c.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def _send_rdm_command(self, node_ip, action, params=None):
        """Send RDM command to a node and wait for response.

        Args:
            node_ip: IP address of the ESP32 node
            action: RDM action (discover, get_info, identify, set_address, etc.)
            params: Additional parameters for the action

        Returns:
            dict with response data or error
        """
        try:
            # Build the RDM command (v:2 required for V2 protocol parser)
            payload = {"v": 2, "type": "rdm", "action": action}
            if params:
                payload.update(params)

            # Create a socket for sending and receiving
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(15.0 if action == 'discover' else 5.0)  # Discovery takes longer

            # Send the command
            json_data = json.dumps(payload, separators=(',', ':'))
            sock.sendto(json_data.encode(), (node_ip, AETHER_UDPJSON_PORT))
            print(f"📡 RDM: {action} -> {node_ip}")

            # Wait for response
            try:
                data, addr = sock.recvfrom(4096)
                response = json.loads(data.decode())
                sock.close()
                return response
            except socket.timeout:
                sock.close()
                return {"success": False, "error": "Response timeout"}

        except Exception as e:
            print(f"❌ RDM error: {e}")
            return {"success": False, "error": str(e)}

    def discover_devices(self, node_id):
        """Start RDM discovery on a node.

        Args:
            node_id: The node to scan for RDM devices

        Returns:
            dict with discovery results
        """
        node = node_manager.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        self.discovery_tasks[node_id] = {"status": "scanning", "started_at": datetime.now().isoformat()}

        # Send discover command
        result = self._send_rdm_command(node['ip'], 'discover')

        # Update status
        self.discovery_tasks[node_id] = {"status": "complete", "result": result}

        # Save discovered devices to database and purge stale entries
        if result.get('success'):
            universe = node.get('universe', 1)
            found_uids = []
            for d in result.get('devices', []):
                found_uids.append(d if isinstance(d, str) else d.get('uid'))

            if result.get('devices'):
                self._save_devices(node_id, universe, result['devices'])

            # Purge stale devices: remove DB entries for this node that were NOT found
            self._purge_stale_devices(node_id, found_uids)

            # Fetch detailed info for each device and populate cache + inventory
            for device in result['devices']:
                uid = device if isinstance(device, str) else device.get('uid')
                if uid:
                    now_iso = datetime.now().isoformat()
                    # Populate in-memory cache
                    with self._lock:
                        if uid not in self._devices:
                            self._devices[uid] = RDMDevice(
                                uid=uid, discovered_via=node_id,
                                discovered_at=now_iso, last_seen=now_iso
                            )
                        else:
                            self._devices[uid].last_seen = now_iso

                    try:
                        info = self._send_rdm_command(node['ip'], 'get_info', {"uid": uid})
                        if info.get('success'):
                            self._update_device_info(uid, info)
                            # Update cache with device info
                            with self._lock:
                                if uid in self._devices:
                                    dev = self._devices[uid]
                                    dev.manufacturer_id = info.get('manufacturer_id', 0)
                                    dev.device_model_id = info.get('device_model_id', 0)
                                    dev.dmx_address = info.get('dmx_address', 0)
                                    dev.dmx_footprint = info.get('dmx_footprint', info.get('footprint', 0))
                                    dev.personality_id = info.get('personality_id', 1)
                                    dev.personality_count = info.get('personality_count', 1)
                                    dev.software_version = info.get('software_version', 0)
                                    dev.sensor_count = info.get('sensor_count', 0)
                                    mid = dev.manufacturer_id
                                    if mid in KNOWN_MANUFACTURERS:
                                        dev.manufacturer_name = KNOWN_MANUFACTURERS[mid]
                            print(f"  📋 Got info for {uid}: Ch{info.get('dmx_address', '?')}, {info.get('dmx_footprint', info.get('footprint', '?'))}ch")
                    except Exception as e:
                        print(f"  ⚠️ Failed to get info for {uid}: {e}")

                    # Check if fixture is patched
                    is_patched = self._resolve_fixture_for_rdm_uid(uid) is not None
                    self.update_inventory(uid, 'online', is_patched=is_patched)

            self._last_discovery = datetime.now()
            self._emit_rdm_update()

        return result

    def _save_devices(self, node_id, universe, devices):
        """Save discovered devices to database.

        Devices can be either:
        - List of UID strings: ["02CA:C207DFA1", ...]
        - List of dicts: [{"uid": "...", "manufacturer": ...}, ...]
        """
        conn = get_db()
        c = conn.cursor()

        for device in devices:
            # Handle both string UIDs and dict format
            if isinstance(device, str):
                uid = device
                manufacturer_id = 0
                device_model_id = 0
            else:
                uid = device.get('uid')
                manufacturer_id = device.get('manufacturer', 0)
                device_model_id = device.get('device_id', 0)

            if not uid:
                continue

            c.execute('''INSERT OR REPLACE INTO rdm_devices
                (uid, node_id, universe, manufacturer_id, device_model_id, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (uid, node_id, universe, manufacturer_id, device_model_id,
                 datetime.now().isoformat()))

        conn.commit()
        print(f"✓ Saved {len(devices)} RDM devices to database")

    def _purge_stale_devices(self, node_id, found_uids):
        """Remove devices from DB and caches that belong to this node but were NOT found in latest discovery."""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT uid FROM rdm_devices WHERE node_id = ?', (node_id,))
        db_uids = [row[0] for row in c.fetchall()]

        stale_uids = [uid for uid in db_uids if uid not in found_uids]
        if not stale_uids:
            return

        for uid in stale_uids:
            c.execute('DELETE FROM rdm_devices WHERE uid = ?', (uid,))
            c.execute('DELETE FROM rdm_personalities WHERE device_uid = ?', (uid,))
            with self._lock:
                self._devices.pop(uid, None)
                self.live_inventory.pop(uid, None)

        conn.commit()
        print(f"🗑️ Purged {len(stale_uids)} stale RDM devices from {node_id}: {stale_uids}")

    def get_device_info(self, node_id, uid):
        """Get detailed info for a specific RDM device."""
        node = node_manager.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        result = self._send_rdm_command(node['ip'], 'get_info', {"uid": uid})

        # Update database with new info
        if result.get('success'):
            self._update_device_info(uid, result)

        return result

    def _update_device_info(self, uid, info):
        """Update device info in database."""
        conn = get_db()
        c = conn.cursor()

        c.execute('''UPDATE rdm_devices SET
            dmx_address = ?, dmx_footprint = ?, personality_id = ?, personality_count = ?,
            software_version = ?, sensor_count = ?, last_seen = ?
            WHERE uid = ?''',
            (info.get('dmx_address', 0), info.get('dmx_footprint', info.get('footprint', 0)),
             info.get('personality_id', info.get('personality_current', 0)), info.get('personality_count', 0),
             info.get('software_version', ''), info.get('sensor_count', 0),
             datetime.now().isoformat(), uid))

        conn.commit()

    def identify_device(self, node_id, uid, state):
        """Set identify mode on a device (flashes LED)."""
        node = node_manager.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        return self._send_rdm_command(node['ip'], 'identify', {"uid": uid, "state": state})

    def set_dmx_address(self, node_id, uid, address):
        """Set DMX start address for a device."""
        node = node_manager.get_node(node_id)
        if not node or node.get('status') != 'online':
            return {"success": False, "error": "Node not found or offline"}

        result = self._send_rdm_command(node['ip'], 'set_address', {"uid": uid, "address": address})

        # Update database
        if result.get('success'):
            conn = get_db()
            c = conn.cursor()
            c.execute('UPDATE rdm_devices SET dmx_address = ?, last_seen = ? WHERE uid = ?',
                     (address, datetime.now().isoformat(), uid))
            conn.commit()

        return result

    def get_devices_for_node(self, node_id):
        """Get all RDM devices for a node from database."""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices WHERE node_id = ? ORDER BY dmx_address', (node_id,))
        columns = [d[0] for d in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]

    def get_all_devices(self):
        """Get all RDM devices from database."""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices ORDER BY node_id, dmx_address')
        columns = [d[0] for d in c.description]
        return [dict(zip(columns, row)) for row in c.fetchall()]

    def delete_device(self, uid):
        """Remove a device from the database and cache."""
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM rdm_devices WHERE uid = ?', (uid,))
        c.execute('DELETE FROM rdm_personalities WHERE device_uid = ?', (uid,))
        conn.commit()
        with self._lock:
            self._devices.pop(uid, None)
        self.live_inventory.pop(uid, None)
        self._emit_rdm_update()
        return {"success": True}

    # ─────────────────────────────────────────────────────────
    # Cross-Node Discovery (absorbed from rdm_service.py)
    # ─────────────────────────────────────────────────────────

    def discover_all(self):
        """Run RDM discovery on ALL online nodes."""
        if self._discovery_in_progress:
            return {'success': False, 'error': 'Discovery already in progress', 'devices': []}

        self._discovery_in_progress = True
        print("🔍 RDM: Starting discovery on all nodes...", flush=True)

        try:
            nodes = self._get_rdm_capable_nodes()
            if not nodes:
                return {'success': False, 'error': 'No RDM-capable nodes online', 'devices': []}

            all_device_uids = []
            for node in nodes:
                node_id = node.get('node_id')
                if not node_id:
                    continue
                print(f"🔍 RDM: Discovering on {node_id}...", flush=True)
                result = self.discover_devices(node_id)
                if result.get('success'):
                    devices = result.get('devices', [])
                    for dev in devices:
                        uid = dev if isinstance(dev, str) else dev.get('uid')
                        if uid:
                            all_device_uids.append(uid)

            self._last_discovery = datetime.now()
            with self._lock:
                device_list = [self._devices[uid].to_dict() for uid in all_device_uids if uid in self._devices]

            return {
                'success': True,
                'devices': device_list,
                'count': len(all_device_uids),
                'timestamp': self._last_discovery.isoformat()
            }
        finally:
            self._discovery_in_progress = False

    def get_cached_devices(self):
        """Get list of all known RDM devices (from database)."""
        return self.get_all_devices()

    # ─────────────────────────────────────────────────────────
    # UID-Based Operations (resolve uid → node_id internally)
    # ─────────────────────────────────────────────────────────

    def _get_node_for_device(self, uid):
        """Get the node that can communicate with a device."""
        dev = self._devices.get(uid)
        if dev and dev.discovered_via and self._node_manager:
            return self._node_manager.get_node(dev.discovered_via)
        # Fallback: try any RDM-capable node
        nodes = self._get_rdm_capable_nodes()
        return nodes[0] if nodes else None

    def identify_by_uid(self, uid, state=True):
        """Identify a device by UID (resolves node automatically)."""
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        return self._send_rdm_command(node.get('ip'), 'identify', {"uid": uid, "state": state})

    def get_address_by_uid(self, uid):
        """Get DMX address for a device by UID."""
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        result = self._send_rdm_command(node.get('ip'), 'get_address', {"uid": uid})
        if result and result.get('success'):
            address = result.get('address', 0)
            with self._lock:
                if uid in self._devices:
                    self._devices[uid].dmx_address = address
            return {'success': True, 'uid': uid, 'address': address}
        return {'success': False, 'error': result.get('error', 'Unknown') if result else 'No response'}

    def set_address_by_uid(self, uid, address):
        """Set DMX address for a device by UID (with conflict check)."""
        if address < 1 or address > 512:
            return {'success': False, 'error': 'Invalid address (must be 1-512)'}
        dev = self._devices.get(uid)
        if not dev:
            return {'success': False, 'error': 'Device not found in cache'}
        # Check for conflicts
        conflicts = self._check_address_conflict(uid, address, dev.dmx_footprint)
        if conflicts:
            return {'success': False, 'error': 'Address conflict', 'conflicts': conflicts}
        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}
        result = self._send_rdm_command(node.get('ip'), 'set_address', {"uid": uid, "address": address})
        if result and result.get('success'):
            with self._lock:
                if uid in self._devices:
                    self._devices[uid].dmx_address = address
            # Also update DB
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('UPDATE rdm_devices SET dmx_address = ?, last_seen = ? WHERE uid = ?',
                         (address, datetime.now().isoformat(), uid))
                conn.commit()
            except Exception:
                pass
            return {'success': True, 'uid': uid, 'address': address}
        return {'success': False, 'error': result.get('error', 'Unknown') if result else 'No response'}

    def get_cached_device_info(self, uid):
        """Get info for a specific device (from database)."""
        # Try in-memory cache first
        dev = self._devices.get(uid)
        if dev:
            return {'success': True, 'device': dev.to_dict()}
        # Fallback to database
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM rdm_devices WHERE uid = ?', (uid,))
        row = c.fetchone()
        if not row:
            return {'success': False, 'error': 'Device not found'}
        columns = [d[0] for d in c.description]
        return {'success': True, 'device': dict(zip(columns, row))}

    # ─────────────────────────────────────────────────────────
    # Address Conflict Detection (absorbed from rdm_service.py)
    # ─────────────────────────────────────────────────────────

    def _check_address_conflict(self, uid, new_address, footprint):
        """Check if setting an address would cause a conflict."""
        conflicts = []
        new_end = new_address + max(footprint, 1) - 1
        with self._lock:
            for other_uid, other in self._devices.items():
                if other_uid == uid or other.dmx_address == 0:
                    continue
                other_end = other.dmx_address + max(other.dmx_footprint, 1) - 1
                if not (new_end < other.dmx_address or new_address > other_end):
                    conflicts.append({'uid': other_uid, 'address': other.dmx_address, 'footprint': other.dmx_footprint})
        return conflicts

    def _find_all_conflicts(self):
        """Find all address conflicts among cached devices."""
        conflicts = []
        devices = list(self._devices.values())
        for i, dev1 in enumerate(devices):
            if dev1.dmx_address == 0:
                continue
            end1 = dev1.dmx_address + max(dev1.dmx_footprint, 1) - 1
            for dev2 in devices[i+1:]:
                if dev2.dmx_address == 0:
                    continue
                end2 = dev2.dmx_address + max(dev2.dmx_footprint, 1) - 1
                if not (end1 < dev2.dmx_address or dev1.dmx_address > end2):
                    conflicts.append({'device1': dev1.uid, 'device2': dev2.uid, 'overlap': True})
        return conflicts

    def suggest_addresses(self):
        """Analyze current addressing and suggest optimal assignments."""
        with self._lock:
            devices = list(self._devices.values())
        if not devices:
            return {'success': True, 'suggestions': [], 'conflicts': []}

        devices.sort(key=lambda d: d.dmx_address)
        conflicts = []
        used_ranges = []

        for dev in devices:
            if dev.dmx_address == 0:
                continue
            start = dev.dmx_address
            end = start + max(dev.dmx_footprint, 1) - 1
            for other_start, other_end, other_uid in used_ranges:
                if not (end < other_start or start > other_end):
                    conflicts.append({'device1': dev.uid, 'device2': other_uid,
                                     'range1': [start, end], 'range2': [other_start, other_end]})
            used_ranges.append((start, end, dev.uid))

        suggestions = []
        next_available = 1
        for dev in devices:
            footprint = max(dev.dmx_footprint, 1)
            has_conflict = any(c['device1'] == dev.uid or c['device2'] == dev.uid for c in conflicts)
            if has_conflict or dev.dmx_address == 0:
                while True:
                    end = next_available + footprint - 1
                    if end > 512:
                        suggestions.append({'uid': dev.uid, 'current_address': dev.dmx_address,
                                           'suggested_address': 0, 'footprint': footprint, 'reason': 'no_space'})
                        break
                    slot_free = all(
                        end < o.dmx_address or next_available > o.dmx_address + max(o.dmx_footprint, 1) - 1
                        for o in devices if o.uid != dev.uid and o.dmx_address > 0
                    )
                    if slot_free:
                        suggestions.append({'uid': dev.uid, 'current_address': dev.dmx_address,
                                           'suggested_address': next_available, 'footprint': footprint,
                                           'reason': 'conflict' if has_conflict else 'unaddressed'})
                        next_available = end + 1
                        break
                    next_available += 1
            else:
                next_available = max(next_available, dev.dmx_address + footprint)

        return {
            'success': True, 'suggestions': suggestions, 'conflicts': conflicts,
            'total_devices': len(devices),
            'conflicting_devices': len(set(c['device1'] for c in conflicts).union(c['device2'] for c in conflicts))
        }

    def auto_fix_addresses(self):
        """Automatically fix all address conflicts."""
        analysis = self.suggest_addresses()
        if not analysis.get('success'):
            return analysis
        results = []
        for suggestion in analysis.get('suggestions', []):
            uid = suggestion['uid']
            new_address = suggestion['suggested_address']
            if new_address == 0:
                results.append({'uid': uid, 'success': False, 'error': 'No space available'})
                continue
            result = self.set_address_by_uid(uid, new_address)
            results.append({'uid': uid, 'success': result.get('success', False),
                           'address': new_address if result.get('success') else None,
                           'error': result.get('error')})
        return {'success': all(r['success'] for r in results), 'results': results}

    def verify_cue_readiness(self, cue_data):
        """Verify all fixtures required for a cue are ready."""
        issues = []
        warnings = []
        required_fixtures = cue_data.get('fixtures', [])
        if not self._devices:
            warnings.append({'type': 'no_rdm_devices',
                            'message': 'No RDM devices discovered - manual verification recommended'})
        for fixture in required_fixtures:
            fixture_uid = fixture.get('rdm_uid')
            expected_address = fixture.get('dmx_address')
            expected_footprint = fixture.get('footprint')
            if not fixture_uid:
                continue
            device = self._devices.get(fixture_uid)
            if not device:
                issues.append({'type': 'device_not_found', 'uid': fixture_uid,
                              'message': f'Device {fixture_uid} not found in RDM cache'})
                continue
            if expected_address and device.dmx_address != expected_address:
                issues.append({'type': 'address_mismatch', 'uid': fixture_uid,
                              'expected': expected_address, 'actual': device.dmx_address,
                              'message': f'Wrong address: expected {expected_address}, found {device.dmx_address}'})
            if expected_footprint and device.dmx_footprint != expected_footprint:
                warnings.append({'type': 'footprint_mismatch', 'uid': fixture_uid,
                                'expected': expected_footprint, 'actual': device.dmx_footprint,
                                'message': f'Footprint mismatch: expected {expected_footprint}, device reports {device.dmx_footprint}'})
        conflicts = self._find_all_conflicts()
        if conflicts:
            issues.append({'type': 'address_conflicts', 'conflicts': conflicts,
                          'message': f'Found {len(conflicts)} address conflicts'})
        ready = len(issues) == 0
        return {'ready': ready, 'issues': issues, 'warnings': warnings,
                'can_proceed': ready, 'recommendation': 'OK to proceed' if ready else 'Review issues before proceeding'}

    # ─────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────

    def _get_rdm_capable_nodes(self):
        """Get list of online nodes that support RDM."""
        if not self._node_manager:
            return []
        all_nodes = self._node_manager.get_all_nodes(include_offline=False)
        return [n for n in all_nodes if 'rdm' in n.get('caps', [])]

    def get_status(self):
        """Get consolidated RDM status."""
        all_devices = self.get_all_devices()
        with self._lock:
            return {
                'enabled': True,
                'device_count': len(all_devices),
                'last_discovery': self._last_discovery.isoformat() if self._last_discovery else None,
                'discovery_in_progress': self._discovery_in_progress,
                'live_inventory': self.live_inventory,
                'devices': all_devices
            }

rdm_manager = RDMManager()

# ============================================================
# Content Manager
# ============================================================
CHUNK_SIZE = 50
CHUNK_DELAY = 0.05

class ContentManager:
    def __init__(self):
        """Initialize ContentManager with SSOT lock for thread-safe playback transitions"""
        self.ssot_lock = threading.Lock()
        self.current_playback = {"type": None, "id": None, "universe": None}
        print("✓ ContentManager initialized with SSOT lock")

    def set_channels(self, universe, channels, fade_ms=0):
        """Set DMX channels - builds full 512-channel frame and sends via UDPJSON

        SSOT COMPLIANCE:
        1. Updates dmx_state with the requested channel changes
        2. Builds full 512-channel frame from SSOT
        3. Sends to each node via UDPJSON (filtered by node's slice)

        All nodes receive complete universe data for their slice.
        Missing channels default to their SSOT value (or 0 if never set).
        """
        # Update SSOT with the channel changes
        dmx_state.set_channels(universe, channels, fade_ms=fade_ms)

        # Build full 512-channel frame from SSOT
        full_frame = dmx_state.get_output_values(universe)

        nodes = node_manager.get_nodes_in_universe(universe)

        if not nodes:
            return {'success': True, 'results': []}

        # Send full frame to each node via UDPJSON
        results = []
        for node in nodes:
            # Route through Seance if node is connected via Seance bridge
            node_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            if not node_ip or node_ip == 'localhost':
                continue

            # Get node's channel slice - use fixture-aware calculation if available
            # Priority: 1) Explicit channel_start/end, 2) Calculate from fixtures, 3) Default 1-4 for RGBW
            slice_start = node.get('channel_start') or 1
            slice_end = node.get('channel_end')

            # If no explicit slice_end, default to 4 channels (single RGBW fixture)
            # This is a safe default since most Pulse nodes have 1 fixture
            if slice_end is None:
                slice_end = 4  # Default: single RGBW fixture

            # Build channels dict for this node's slice from the full frame
            node_channels = {}
            for ch in range(slice_start, slice_end + 1):
                value = full_frame[ch - 1] if ch <= 512 else 0
                node_channels[str(ch)] = value

            node_non_zero = sum(1 for v in node_channels.values() if v > 0)

            if fade_ms > 0:
                success = node_manager.send_udpjson_fade(node_ip, universe, node_channels, fade_ms)
            else:
                success = node_manager.send_udpjson_set(node_ip, universe, node_channels)

            results.append({'node': node['name'], 'success': success, 'channels': len(node_channels)})

        return {'success': True, 'results': results}

    def blackout(self, universe=None, fade_ms=1000):
        """Blackout all channels - if universe is None, blackout ALL universes"""
        # ARBITRATION: Blackout is highest priority - stop everything first
        arbitration.set_blackout(True)
        effects_engine.stop_effect()  # Stop all dynamic effects
        chase_engine.stop_all()  # Stop all chases

        all_nodes = node_manager.get_all_nodes(include_offline=False)
        all_universes_online = list(set(node.get('universe', 1) for node in all_nodes))
        playback_before = dict(self.current_playback) if hasattr(self, 'current_playback') else {}

        if universe is not None:
            universes_to_blackout = [universe]
        else:
            universes_to_blackout = all_universes_online

        beta_log("blackout", {
            "requested_universe": universe,
            "selected_universes_at_action_time": sorted(all_universes_online),
            "expanded_target_universes": sorted(universes_to_blackout),
            "dispatch_targets_final": sorted(universes_to_blackout),
            "playback_state_before": playback_before,
            "fade_ms": fade_ms
        })

        print(f"⬛ Blackout on universes: {sorted(universes_to_blackout)}", flush=True)

        results = []
        for univ in universes_to_blackout:
            dmx_state.blackout(univ, fade_ms=fade_ms)
            playback_manager.stop(univ)
            nodes = node_manager.get_nodes_in_universe(univ)
            for node in nodes:
                node_ip = node.get('ip')
                if node_ip and node_ip != 'localhost':
                    success = node_manager.send_udpjson_blackout(node_ip, univ, fade_ms=fade_ms)
                    results.append({'node': node['name'], 'success': success})

        if hasattr(self, 'current_playback'):
            self.current_playback = {"type": None, "id": None, "universe": None}

        beta_log("blackout_complete", {
            "dispatch_targets_final": sorted(universes_to_blackout),
            "playback_state_after": {"type": None, "id": None, "universe": None}
        })

        # Release blackout after sending zeros (allow future commands)
        arbitration.set_blackout(False)

        return {'success': True, 'results': results}


    # ─────────────────────────────────────────────────────────
    # Scenes
    # ─────────────────────────────────────────────────────────
    def create_scene(self, data):
        """Create/update scene and sync to nodes"""
        scene_id = data.get('scene_id', f"scene_{int(time.time())}")
        universe = data.get('universe', 1)
        channels = data.get('channels', {})
        
        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO scenes (scene_id, name, description, universe, channels,
            fade_ms, curve, color, icon, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (scene_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, json.dumps(channels), data.get('fade_ms', 500), data.get('curve', 'linear'),
             data.get('color', '#3b82f6'), data.get('icon', 'lightbulb'), False, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        # Sync to all nodes in this universe
        scene = self.get_scene(scene_id)
        if scene:
            nodes = node_manager.get_wifi_nodes_in_universe(universe)
            for node in nodes:
                node_manager.sync_scene_to_node(node, scene)
                time.sleep(CHUNK_DELAY)
            
            # Mark as synced
            conn = get_db()
            c = conn.cursor()
            c.execute('UPDATE scenes SET synced_to_nodes = 1 WHERE scene_id = ?', (scene_id,))
            conn.commit()
            conn.close()
        
        socketio.emit('scenes_update', {'scenes': self.get_scenes()})

        # Async sync to Supabase (non-blocking)
        if SUPABASE_AVAILABLE and scene:
            supabase = get_supabase_service()
            if supabase and supabase.is_enabled():
                cloud_submit(supabase.sync_scene, scene)

        return {'success': True, 'scene_id': scene_id}

    def get_scenes(self):
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        scenes = []
        for row in rows:
            scene = dict(row)
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            scenes.append(scene)
        return scenes

    def get_scene(self, scene_id):
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,))
        row = c.fetchone()
        conn.close()
        if row:
            scene = dict(row)
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            return scene
        return None

    def replicate_scene_to_fixtures(self, channels, fixture_size=4, max_fixtures=128, distribution_mode='unified', universe=None):
        """Replicate a scene pattern across all fixtures in a universe.

        If scene has channels 1-4, replicate to 5-8, 9-12, etc.
        distribution_mode: 'unified' = replicate same pattern, 'pixel' = unique per fixture
        """
        if not channels:
            return channels

        # PIXEL MODE: distribute unique values to each fixture
        if distribution_mode == 'pixel' and universe is not None:
            fixtures = self.get_fixtures(universe)
            if fixtures:
                fixtures = sorted(fixtures, key=lambda f: f.get('start_channel', 1))
                pattern_vals = list(channels.values())
                distributed = {}
                for idx, fix in enumerate(fixtures):
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', 1)
                    val = pattern_vals[idx % len(pattern_vals)] if pattern_vals else 0
                    for ch in range(count):
                        distributed[str(start + ch)] = val
                return distributed

        # Find the base pattern - channels that define one fixture
        ch_nums = sorted(int(k) for k in channels.keys())
        if not ch_nums:
            return channels

        # Detect fixture size from the scene (could be 3 for RGB, 4 for RGBW, etc.)
        min_ch = min(ch_nums)
        max_ch = max(ch_nums)
        pattern_size = max_ch - min_ch + 1

        # If pattern is larger than typical fixture, don't replicate
        if pattern_size > 8:
            return channels

        # Use the larger of detected pattern or standard fixture size
        fixture_size = max(fixture_size, pattern_size)

        # Build the base pattern (normalized to start at channel 1)
        base_pattern = {}
        for ch_str, value in channels.items():
            ch = int(ch_str)
            offset = (ch - 1) % fixture_size  # 0-indexed offset within fixture
            base_pattern[offset] = value

        # Replicate across all fixtures
        replicated = {}
        for fixture_num in range(max_fixtures):
            fixture_start = fixture_num * fixture_size + 1
            if fixture_start > 512:
                break
            for offset, value in base_pattern.items():
                ch = fixture_start + offset
                if ch <= 512:
                    replicated[str(ch)] = value

        print(f"🔄 Replicated {len(channels)} channels -> {len(replicated)} channels (pattern size: {fixture_size})")
        return replicated

    def play_scene(self, scene_id, fade_ms=None, use_local=True, target_channels=None, universe=None, universes=None, skip_ssot=False, replicate=True):
        """Play a scene via unified engine - supports modifier stacking

        Args:
            scene_id: ID of the scene to play
            fade_ms: Fade time override
            use_local: Use local playback
            target_channels: Optional list of specific channels
            universe: Single universe (legacy, use universes instead)
            universes: List of universes to target (preferred)
            skip_ssot: Skip SSOT lock (internal use)
            replicate: Replicate scene across fixtures
        """
        print(f"▶️ play_scene called: scene_id={scene_id}", flush=True)
        scene = self.get_scene(scene_id)
        if not scene:
            return {'success': False, 'error': 'Scene not found'}

        # ARBITRATION: Acquire scene ownership
        if not arbitration.acquire('scene', scene_id):
            print(f"⚠️ Cannot play scene - arbitration denied (owner: {arbitration.current_owner})", flush=True)
            return {'success': False, 'error': f'Arbitration denied: {arbitration.current_owner} has control'}

        # Get target universes - priority: universes array > single universe > all online paired nodes
        all_nodes = node_manager.get_all_nodes(include_offline=False)
        if universes is not None and len(universes) > 0:
            universes_with_nodes = list(universes)
        elif universe is not None:
            universes_with_nodes = [universe]
        else:
            # Default: all online PAIRED universes only
            universes_with_nodes = list(set(node.get('universe', 1) for node in all_nodes if node.get('is_paired')))
            if not universes_with_nodes:
                universes_with_nodes = [1]

        fade = fade_ms if fade_ms is not None else scene.get('fade_ms', 500)
        channels_to_apply = scene['channels']

        # Replicate scene pattern across all fixtures (unless targeting specific channels)
        if replicate and not target_channels:
            channels_to_apply = self.replicate_scene_to_fixtures(channels_to_apply)

        if target_channels:
            target_set = set(target_channels)
            channels_to_apply = {k: v for k, v in channels_to_apply.items() if int(k) in target_set}

        print(f"🎬 Playing scene '{scene['name']}' on universes: {sorted(universes_with_nodes)}, fade={fade}ms", flush=True)

        # SSOT: Acquire lock and stop conflicting playback
        if not skip_ssot:
            with self.ssot_lock:
                print(f"🔒 SSOT Lock - stopping conflicting playback", flush=True)
                try:
                    show_engine.stop_silent() if hasattr(show_engine, "stop_silent") else show_engine.stop()
                except Exception as e:
                    print(f"⚠️ Show stop error: {e}", flush=True)
                chase_engine.stop_all()
                unified_engine.stop_type(PlaybackType.SCENE)
                unified_engine.stop_type(PlaybackType.CHASE)
                effects_engine.stop_effect()
                self.current_playback = {'type': 'scene', 'id': scene_id, 'universe': universe}

        # Set playback state for all universes
        for univ in universes_with_nodes:
            playback_manager.set_playing(univ, 'scene', scene_id)

        # Create scene data with expanded channels for unified engine
        scene_data = {
            'name': scene.get('name', f'Scene {scene_id}'),
            'channels': {int(k): v for k, v in channels_to_apply.items()},
            'fade_ms': fade
        }

        # Create unified engine session from scene data
        from unified_playback import SessionFactory
        session = SessionFactory.from_scene(scene_id, scene_data, universes_with_nodes, fade)

        # Get current DMX state for fade-from values
        fade_from = None
        if fade > 0 and universes_with_nodes:
            current_universe_values = dmx_state.universes.get(universes_with_nodes[0], [0] * 512)
            fade_from = {i+1: v for i, v in enumerate(current_universe_values) if v > 0}

        # Start via unified engine - this allows modifier stacking!
        unified_engine.play(session, fade_from)
        print(f"▶️ Scene '{scene['name']}' started via unified engine: {session.session_id}", flush=True)

        # Update play count
        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE scenes SET play_count = play_count + 1 WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()

        dmx_state.save_state_now()  # [F09] Persist immediately on playback start
        return {'success': True, 'universes': universes_with_nodes, 'fade_ms': fade, 'session_id': session.session_id}

    def delete_scene(self, scene_id):
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM scenes WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()
        socketio.emit('scenes_update', {'scenes': self.get_scenes()})
        return {'success': True}

    # ─────────────────────────────────────────────────────────
    # Chases
    # ─────────────────────────────────────────────────────────
    def create_chase(self, data):
        """Create/update chase and sync to nodes"""
        chase_id = data.get('chase_id', f"chase_{int(time.time())}")
        universe = data.get('universe', 1)
        steps = data.get('steps', [])

        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO chases (chase_id, name, description, universe, bpm, loop,
            steps, color, fade_ms, distribution_mode, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (chase_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, data.get('bpm', 120), data.get('loop', True),
             json.dumps(steps), data.get('color', '#10b981'), data.get('fade_ms', 0), data.get('distribution_mode', 'unified'), False, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Sync to nodes (non-blocking)
        try:
            chase = self.get_chase(chase_id)
            if chase:
                nodes = node_manager.get_wifi_nodes_in_universe(universe)
                for node in nodes:
                    node_manager.sync_chase_to_node(node, chase)
                    time.sleep(CHUNK_DELAY)
                conn = get_db()
                c = conn.cursor()
                c.execute('UPDATE chases SET synced_to_nodes = 1 WHERE chase_id = ?', (chase_id,))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Chase sync error: {e}", flush=True)

        socketio.emit('chases_update', {'chases': self.get_chases()})

        # Async sync to Supabase
        try:
            if SUPABASE_AVAILABLE and chase:
                supabase = get_supabase_service()
                if supabase and supabase.is_enabled():
                    cloud_submit(supabase.sync_chase, chase)
        except Exception as e:
            print(f"Supabase error: {e}", flush=True)

        return {'success': True, 'chase_id': chase_id}

    def get_chases(self):
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM chases ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        chases = []
        for row in rows:
            chase = dict(row)
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            chases.append(chase)
        return chases

    def get_chase(self, chase_id):
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM chases WHERE chase_id = ?', (chase_id,))
        row = c.fetchone()
        conn.close()
        if row:
            chase = dict(row)
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            return chase
        return None

    def play_chase(self, chase_id, target_channels=None, universe=None, universes=None, fade_ms=None):
        """Start chase playback via unified engine - supports modifier stacking

        Args:
            chase_id: ID of the chase to play
            target_channels: Optional list of specific channels to target
            universe: Single universe (legacy, use universes instead)
            universes: List of universes to target (preferred)
            fade_ms: Fade time override
        """
        chase = self.get_chase(chase_id)
        if not chase:
            return {'success': False, 'error': 'Chase not found'}

        # Apply-time fade override: fade_ms param > chase default > 0
        effective_fade_ms = fade_ms if fade_ms is not None else chase.get('fade_ms', 0)
        print(f"🎚️ Chase fade: requested={fade_ms}, chase_default={chase.get('fade_ms')}, effective={effective_fade_ms}", flush=True)

        # Get target universes - priority: universes array > single universe > all online paired nodes
        if universes is not None and len(universes) > 0:
            universes_with_nodes = list(universes)
        elif universe is not None:
            universes_with_nodes = [universe]
        else:
            # Default: all online PAIRED universes only
            all_nodes = node_manager.get_all_nodes(include_offline=False)
            universes_with_nodes = list(set(node.get('universe', 1) for node in all_nodes if node.get('is_paired')))
        print(f"🎬 Playing chase '{chase['name']}' on universes: {sorted(universes_with_nodes)}, fade={effective_fade_ms}ms", flush=True)

        # SSOT: Acquire lock and stop only conflicting playback on target universes
        with self.ssot_lock:
            print(f"🔒 SSOT Lock - stopping playback on target universes: {universes_with_nodes}", flush=True)
            try:
                show_engine.stop()
            except Exception as e:
                print(f"⚠️ Show stop: {e}", flush=True)
            # Stop old chase engine (legacy)
            chase_engine.stop_all()
            # Stop existing chase sessions in unified engine for these universes
            unified_engine.stop_type(PlaybackType.CHASE)
            effects_engine.stop_effect()
            for univ in universes_with_nodes:
                playback_manager.stop(univ)
            self.current_playback = {'type': 'chase', 'id': chase_id, 'universe': universe}
            print(f"✓ SSOT: Now playing chase '{chase_id}'", flush=True)

        # Set playback state for all universes
        for univ in universes_with_nodes:
            playback_manager.set_playing(univ, 'chase', chase_id)

        # Create unified engine session from chase data
        from unified_playback import SessionFactory
        session = SessionFactory.from_chase(chase_id, chase, universes_with_nodes)

        # Replicate chase step channels across all fixtures in the universe
        # Without this, chase steps only affect one fixture (scenes do this in play_scene)
        if universes_with_nodes:
            fixtures_in_universe = self.get_fixtures(universes_with_nodes[0])
            if fixtures_in_universe and len(fixtures_in_universe) > 1:
                sorted_fx = sorted(fixtures_in_universe, key=lambda f: f.get('start_channel', 1))
                fixture_ch_count = sorted_fx[0].get('channel_count', 4)
                for step in session.steps:
                    # Extract base pattern using modulo (ch 1→offset 0, ch 6→offset 1, etc.)
                    base_pattern = {}
                    for ch, val in step.channels.items():
                        offset = (ch - 1) % fixture_ch_count
                        if val > 0 or offset not in base_pattern:
                            base_pattern[offset] = val
                    # Expand to all fixtures
                    expanded = {}
                    for fix in sorted_fx:
                        start = fix.get('start_channel', 1)
                        count = fix.get('channel_count', fixture_ch_count)
                        for offset, value in base_pattern.items():
                            if offset < count:
                                expanded[start + offset] = value
                    step.channels = expanded
                print(f"🔄 Replicated chase steps across {len(sorted_fx)} fixtures (ch_count={fixture_ch_count})", flush=True)

        # Apply fade override if specified
        if effective_fade_ms > 0:
            for step in session.steps:
                step.fade_ms = effective_fade_ms

        # Start via unified engine - this allows modifier stacking!
        unified_engine.play(session)
        print(f"▶️ Chase '{chase['name']}' started via unified engine: {session.session_id}", flush=True)

        dmx_state.save_state_now()  # [F09] Persist immediately on playback start
        return {'success': True, 'universes': universes_with_nodes, 'fade_ms': effective_fade_ms, 'session_id': session.session_id}

    def stop_playback(self, universe=None):
        """Stop all playback"""
        chase_engine.stop_all()  # Stop chase engine
        playback_manager.stop(universe)
        node_results = node_manager.stop_playback_on_nodes(universe)
        dmx_state.save_state_now()  # [F09] Persist immediately on playback stop
        return {'success': True, 'results': node_results}

    def delete_chase(self, chase_id):
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM chases WHERE chase_id = ?', (chase_id,))
        conn.commit()
        conn.close()
        socketio.emit('chases_update', {'chases': self.get_chases()})
        return {'success': True}

    # ─────────────────────────────────────────────────────────
    # Fixtures
    # ─────────────────────────────────────────────────────────
    def create_fixture(self, data):
        """Create or update a fixture definition"""
        fixture_id = data.get('fixture_id', f"fixture_{int(time.time())}")

        # Default channel map based on type
        default_map = self._get_default_channel_map(data.get('type', 'generic'), data.get('channel_count', 1))
        channel_map = data.get('channel_map', default_map)

        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO fixtures (fixture_id, name, type, manufacturer, model,
            universe, start_channel, channel_count, channel_map, color, notes, rdm_uid, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (fixture_id, data.get('name', 'Untitled Fixture'), data.get('type', 'generic'),
             data.get('manufacturer', ''), data.get('model', ''),
             data.get('universe', 1), data.get('start_channel', 1), data.get('channel_count', 1),
             json.dumps(channel_map), data.get('color', '#8b5cf6'),
             data.get('notes', ''), data.get('rdm_uid'), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        socketio.emit('fixtures_update', {'fixtures': self.get_fixtures()})
        return {'success': True, 'fixture_id': fixture_id}

    def _get_default_channel_map(self, fixture_type, channel_count):
        """Generate default channel names based on fixture type"""
        maps = {
            'rgb': ['Red', 'Green', 'Blue'],
            'rgbw': ['Red', 'Green', 'Blue', 'White'],
            'rgba': ['Red', 'Green', 'Blue', 'Amber'],
            'rgbwa': ['Red', 'Green', 'Blue', 'White', 'Amber'],
            'dimmer': ['Intensity'],
            'moving_head': ['Pan', 'Pan Fine', 'Tilt', 'Tilt Fine', 'Speed', 'Dimmer', 'Strobe', 'Color', 'Gobo', 'Prism'],
            'par': ['Red', 'Green', 'Blue', 'White', 'Dimmer', 'Strobe'],
            'wash': ['Red', 'Green', 'Blue', 'White', 'Dimmer', 'Pan', 'Tilt'],
        }
        default = maps.get(fixture_type.lower(), [])
        # Pad with generic channel names if needed
        while len(default) < channel_count:
            default.append(f'Channel {len(default) + 1}')
        return default[:channel_count]

    def get_fixtures(self, universe=None):
        """Get all fixtures, optionally filtered by universe"""
        conn = get_db()
        c = conn.cursor()
        if universe:
            c.execute('SELECT * FROM fixtures WHERE universe = ? ORDER BY start_channel', (universe,))
        else:
            c.execute('SELECT * FROM fixtures ORDER BY universe, start_channel')
        rows = c.fetchall()
        conn.close()
        fixtures = []
        for row in rows:
            fixture = dict(row)
            fixture['channel_map'] = json.loads(fixture['channel_map']) if fixture['channel_map'] else []
            # Calculate end channel
            fixture['end_channel'] = fixture['start_channel'] + fixture['channel_count'] - 1
            fixtures.append(fixture)
        return fixtures

    def get_fixture(self, fixture_id):
        """Get single fixture by ID"""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM fixtures WHERE fixture_id = ?', (fixture_id,))
        row = c.fetchone()
        conn.close()
        if row:
            fixture = dict(row)
            fixture['channel_map'] = json.loads(fixture['channel_map']) if fixture['channel_map'] else []
            fixture['end_channel'] = fixture['start_channel'] + fixture['channel_count'] - 1
            return fixture
        return None

    def update_fixture(self, fixture_id, data):
        """Update an existing fixture"""
        existing = self.get_fixture(fixture_id)
        if not existing:
            return {'success': False, 'error': 'Fixture not found'}

        # Merge with existing data
        merged = {**existing, **data}
        merged['fixture_id'] = fixture_id
        return self.create_fixture(merged)

    def delete_fixture(self, fixture_id):
        """Delete a fixture"""
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM fixtures WHERE fixture_id = ?', (fixture_id,))
        conn.commit()
        conn.close()
        socketio.emit('fixtures_update', {'fixtures': self.get_fixtures()})
        return {'success': True}

    def get_fixtures_for_channels(self, universe, channels):
        """Find which fixtures cover the given channels"""
        fixtures = self.get_fixtures(universe)
        affected = []
        channel_nums = [int(c) for c in channels.keys()]

        for fixture in fixtures:
            start = fixture['start_channel']
            end = fixture['end_channel']
            for ch in channel_nums:
                if start <= ch <= end:
                    affected.append(fixture)
                    break
        return affected

content_manager = ContentManager()

# ============================================================
# Looks & Sequences Manager (new unified architecture)
# ============================================================
looks_sequences_manager = LooksSequencesManager(DATABASE)

# ============================================================
# Cue Stacks Manager (theatrical manual cueing)
# ============================================================
cue_stacks_manager = CueStacksManager(DATABASE)

# ============================================================
# Fixture Library (profiles, instances, OFL integration)
# ============================================================
fixture_library = init_fixture_library(DATABASE)
channel_mapper = get_channel_mapper()

# ============================================================
# SSOT Output Router - Unified output path for all DMX writes
# ============================================================
# Track active effect sources for merge layer
_effect_merge_sources = {}  # effect_id -> source_id

def ssot_send_frame(universe, channels_array, owner_type='effect'):
    """
    Unified output function for all DMX writes.
    Routes through merge layer for proper priority-based merging.

    RACE CONDITION FIX: Now routes effects through merge layer to allow
    concurrent playback with proper HTP/LTP merging.
    """
    try:
        # Check arbitration - if we can't write, skip silently
        if not arbitration.can_write(owner_type):
            return

        # Convert array to dict format
        channels_dict = {}
        for i, value in enumerate(channels_array):
            if value > 0:  # Only include non-zero values
                channels_dict[i + 1] = value  # int keys for merge layer

        if not channels_dict:
            return

        # MERGE LAYER: Route effects through merge layer for proper priority
        if owner_type == 'effect' and effects_engine.current_effect:
            effect_id = effects_engine.current_effect
            source_id = _effect_merge_sources.get(effect_id)

            # Auto-register if not registered
            if not source_id:
                source_id = f"effect_{effect_id}"
                merge_layer.register_source(source_id, 'effect', [universe])
                _effect_merge_sources[effect_id] = source_id

            # Update merge layer with new channel values
            merge_layer.set_source_channels(source_id, universe, channels_dict)

            # Compute merged output
            merged = merge_layer.compute_merge(universe)
            if merged:
                content_manager.set_channels(universe, {str(k): v for k, v in merged.items()}, fade_ms=0)
            return

        # Fallback: direct write for non-effect sources
        content_manager.set_channels(universe, {str(k): v for k, v in channels_dict.items()}, fade_ms=0)

    except Exception as e:
        print(f"❌ SSOT output error U{universe}: {e}")

def cleanup_effect_merge_source(effect_id):
    """Clean up merge source when effect stops"""
    source_id = _effect_merge_sources.pop(effect_id, None)
    if source_id:
        merge_layer.unregister_source(source_id)
        print(f"📤 Effect unregistered from merge layer: {source_id}", flush=True)

# Hook up effects engine to SSOT with arbitration
effects_engine.set_ssot_hooks(dmx_state, ssot_send_frame, arbitration)

# Wrap stop_effect to also clean up merge sources
_original_stop_effect = effects_engine.stop_effect
def _wrapped_stop_effect(effect_id=None):
    """Stop effect and clean up merge layer source"""
    # Get effect IDs to clean up before stopping
    if effect_id:
        effects_to_clean = [effect_id] if effect_id in _effect_merge_sources else []
    else:
        effects_to_clean = list(_effect_merge_sources.keys())

    # Call original stop
    _original_stop_effect(effect_id)

    # Clean up merge sources
    for eid in effects_to_clean:
        cleanup_effect_merge_source(eid)

effects_engine.stop_effect = _wrapped_stop_effect

# Hook up render engine to SSOT for Look playback
def render_engine_output(universe: int, channels: dict):
    """Callback for render engine to send frames through SSOT"""
    # Check arbitration
    if not arbitration.can_write('look'):
        return
    # Convert dict to format expected by set_channels
    content_manager.set_channels(universe, {str(k): v for k, v in channels.items()}, fade_ms=0)

render_engine.set_output_callback(render_engine_output)
render_engine.set_arbitration(arbitration)

# ============================================================
# Merge Layer Integration (Phase 5)
# ============================================================
# The merge layer allows multiple playback sources to run simultaneously
# with HTP (Highest Takes Precedence) for dimmer channels and
# LTP (Latest Takes Precedence) for color/other channels.

# Load fixture profiles into channel classifier for HTP/LTP determination
def load_fixtures_into_classifier():
    """Load fixture definitions to classify channels as dimmer/color/etc"""
    try:
        fixtures = content_manager.get_fixtures()
        channel_classifier.load_fixtures(fixtures)
        dimmer_count = sum(len(channel_classifier.get_dimmer_channels(f.get('universe', 1)))
                          for f in fixtures)
        print(f"📊 MergeLayer: Loaded {len(fixtures)} fixtures, {dimmer_count} dimmer channels classified")
    except Exception as e:
        print(f"⚠️ MergeLayer: Could not load fixtures: {e}")

# Merge layer output callback -> SSOT
def merge_layer_output(universe: int, channels: dict):
    """Final output from merge layer to SSOT"""
    if not channels:
        return
    # Convert dict to format expected by set_channels
    content_manager.set_channels(universe, {str(k): v for k, v in channels.items()}, fade_ms=0)

merge_layer.set_output_callback(merge_layer_output)

# RACE CONDITION FIX: Connect ChaseEngine and ShowEngine to merge layer
# This ensures chases and shows route through priority-based merging
chase_engine.set_merge_layer(merge_layer)
show_engine.set_merge_layer(merge_layer)
print("✓ ChaseEngine and ShowEngine connected to merge layer for SSOT compliance")

# ============================================================
# Unified Stop All Playback - SSOT Control Point
# ============================================================
# This function is the SINGLE SOURCE OF TRUTH for stopping all playback.
# All frontend stop buttons should call this function to ensure consistent behavior.

def stop_all_playback(blackout=False, fade_ms=1000, universe=None):
    """
    SSOT: Stop all active playback sources.

    This is the unified stop function that should be called from ALL frontend
    stop controls to ensure consistent behavior. It stops:
    - Shows (timeline playback)
    - Chases
    - Effects
    - Unified playback controller (looks/sequences)
    - Optionally performs a blackout

    Args:
        blackout: If True, also send blackout command after stopping
        fade_ms: Fade time for blackout (only used if blackout=True)
        universe: Specific universe to stop (None = all)

    Returns:
        Dict with status of each stopped system
    """
    results = {
        'show': False,
        'chase': False,
        'effect': False,
        'playback': False,
        'blackout': False
    }

    print(f"🛑 SSOT: stop_all_playback called (blackout={blackout}, universe={universe})", flush=True)

    # Stop show engine
    try:
        if show_engine.running:
            show_engine.stop_silent()
            results['show'] = True
            print("  ✓ Show stopped", flush=True)
    except Exception as e:
        print(f"  ❌ Show stop error: {e}", flush=True)

    # Stop all chases
    try:
        if chase_engine.running_chases:
            chase_engine.stop_all()
            results['chase'] = True
            print("  ✓ Chases stopped", flush=True)
    except Exception as e:
        print(f"  ❌ Chase stop error: {e}", flush=True)

    # Stop all effects
    try:
        if effects_engine.running:
            effects_engine.stop_effect()
            results['effect'] = True
            print("  ✓ Effects stopped", flush=True)
    except Exception as e:
        print(f"  ❌ Effect stop error: {e}", flush=True)

    # Stop UnifiedPlaybackEngine (canonical authority)
    try:
        status = unified_get_status()
        if status.get('sessions'):
            unified_stop()
            results['playback'] = True
            print("  ✓ UnifiedPlaybackEngine stopped", flush=True)
    except Exception as e:
        print(f"  ❌ Unified playback stop error: {e}", flush=True)

    # Unregister all merge sources to clear the merge layer
    try:
        for source_id in list(_active_merge_sources.values()):
            merge_layer.unregister_source(source_id)
        _active_merge_sources.clear()
        print("  ✓ Merge sources cleared", flush=True)
    except Exception as e:
        print(f"  ❌ Merge source cleanup error: {e}", flush=True)

    # Release arbitration to idle
    try:
        arbitration.release()
        print("  ✓ Arbitration released to idle", flush=True)
    except Exception as e:
        print(f"  ❌ Arbitration release error: {e}", flush=True)

    # Optionally blackout
    if blackout:
        try:
            content_manager.blackout(universe=universe, fade_ms=fade_ms)
            results['blackout'] = True
            print("  ✓ Blackout sent", flush=True)
        except Exception as e:
            print(f"  ❌ Blackout error: {e}", flush=True)

    # Broadcast stop event to all connected clients
    try:
        broadcast_ws({
            'type': 'playback_stopped',
            'all_stopped': True,
            'blackout': blackout
        })
    except Exception as e:
        print(f"  ⚠️ WebSocket broadcast error: {e}", flush=True)

    dmx_state.save_state_now()  # [F09] Persist immediately on stop-all
    print(f"🛑 SSOT: stop_all_playback complete: {results}", flush=True)
    return {'success': True, 'results': results}

# Active merge sources tracking
_active_merge_sources = {}  # job_id -> source_id

def register_playback_source(job_id, source_type, universes):
    """Register a playback job as a merge source"""
    source_id = f"{source_type}_{job_id}"
    merge_layer.register_source(source_id, source_type, universes)
    _active_merge_sources[job_id] = source_id
    return source_id

def unregister_playback_source(job_id):
    """Unregister a playback job from merge layer"""
    source_id = _active_merge_sources.pop(job_id, None)
    if source_id:
        merge_layer.unregister_source(source_id)

def get_active_source_id(job_id):
    """Get the merge source ID for a playback job"""
    return _active_merge_sources.get(job_id)

# Hook up UnifiedPlaybackEngine to merge layer
def playback_output(universe: int, channels: dict):
    """Callback for UnifiedPlaybackEngine - routes through merge layer"""
    # Get the current session from unified engine
    status = unified_get_status()
    sessions = status.get('sessions', [])

    # Use first active session as source identifier
    job_id = None
    source_type = 'look'
    universes = [universe]
    for session in sessions:
        if session.get('state') in ('playing', 'fading_in'):
            job_id = session.get('session_id')
            source_type = session.get('playback_type', 'look')
            universes = session.get('universes', [universe])
            break

    if not job_id:
        # No active session, just output directly
        merge_layer_output(universe, channels)
        return

    source_id = get_active_source_id(job_id)
    if not source_id:
        # Auto-register if not registered
        source_id = register_playback_source(job_id, source_type, universes)

    # Update merge layer with new channel values
    merge_layer.set_source_channels(source_id, universe, channels)

    # Output merged result
    merged = merge_layer.compute_merge(universe)
    if merged:
        merge_layer_output(universe, merged)

# DELETED (Phase 3): playback_controller configuration removed
# playback_controller.set_output_callback(playback_output)
# playback_controller.set_modifier_renderer(ModifierRenderer())

# Set look resolver for Sequence playback (to resolve Look references in steps)
def resolve_look(look_id: str):
    """Resolve a Look ID to Look data"""
    look = looks_sequences_manager.get_look(look_id)
    if look:
        return look.to_dict()
    return None

# DELETED (Phase 3): playback_controller.set_look_resolver removed

# ============================================================
# UnifiedPlaybackEngine Configuration (Canonical Authority)
# ============================================================
# Configure the canonical UnifiedPlaybackEngine with proper callbacks.
#
# NOTE: The actual startup happens in __main__ block (line ~8690) which
# re-configures with the proper SSOT callbacks. This pre-configuration
# ensures unified_engine is ready if anything tries to use it before __main__.

# Output callback - routes DMX through MergeLayer (fallback, overridden in __main__)
unified_engine.set_output_callback(playback_output)

# Look resolver - resolves Look IDs to Look data for sequence steps
unified_engine.set_look_resolver(resolve_look)

# NOTE: Engine startup is handled in __main__ block to ensure proper initialization order
print("[MIGRATION] UnifiedPlaybackEngine pre-configured (will start in __main__)", flush=True)

# Load fixtures on startup (deferred until content_manager is ready)
threading.Timer(2.0, load_fixtures_into_classifier).start()

# ============================================================
# Preview Service Integration (Phase 6)
# ============================================================
# Preview service allows live editing preview without affecting output

# Set modifier renderer for preview
preview_service.set_modifier_renderer(ModifierRenderer())

# Preview live output callback (only used when session is ARMED)
def preview_live_output(universe, channels):
    """Callback for armed preview sessions to output to live universes"""
    # Register as a preview source in merge layer with high priority
    source_id = f"preview_armed_{universe}"
    source = merge_layer.get_source(source_id)
    if not source:
        merge_layer.register_source(source_id, 'manual', [universe])  # manual = priority 80
    merge_layer.set_source_channels(source_id, universe, channels)
    # Output merged result
    merged = merge_layer.compute_merge(universe)
    if merged:
        merge_layer_output(universe, merged)

preview_service.set_live_output_callback(preview_live_output)

# WebSocket streaming for preview frames
def stream_preview_frame(session_id, frame):
    """Stream preview frame to UI via WebSocket"""
    socketio.emit('preview_frame', {
        'session_id': session_id,
        'frame_number': frame.frame_number,
        'channels': frame.channels,
        'universes': frame.universes,
        'elapsed_ms': frame.elapsed_ms,
        'modifier_count': frame.modifier_count,
    })

preview_service.set_frame_callback(stream_preview_frame)

# ============================================================
# Blueprint Registration (F02 — god file decomposition)
# ============================================================
# Phase 1: Extract most isolated route groups into blueprints.
# Each blueprint imports only what it needs — no circular deps.

from blueprints.trust_bp import trust_bp
from blueprints.modifiers_bp import modifiers_bp
from blueprints.migrate_bp import migrate_bp, init_app as migrate_init
from blueprints.settings_bp import settings_bp, init_app as settings_init
from blueprints.merge_bp import merge_bp, init_app as merge_init
from blueprints.scenes_bp import scenes_bp, init_app as scenes_init
from blueprints.chases_bp import chases_bp, init_app as chases_init
from blueprints.pixel_bp import pixel_bp, init_app as pixel_init
from blueprints.shows_bp import shows_bp, init_app as shows_init
from blueprints.schedules_bp import schedules_bp, init_app as schedules_init
from blueprints.cloud_bp import cloud_bp, init_app as cloud_init
from blueprints.session_bp import session_bp, init_app as session_init
from blueprints.system_bp import system_bp, init_app as system_init
from blueprints.groups_bp import groups_bp, init_app as groups_init
from blueprints.ai_bp import ai_bp, init_app as ai_init
from blueprints.fixtures_bp import fixtures_bp, init_app as fixtures_init
from blueprints.fixture_library_bp import fixture_library_bp, init_app as fixture_library_init
from blueprints.node_groups_bp import node_groups_bp, init_app as node_groups_init
from blueprints.rdm_bp import rdm_bp, init_app as rdm_init
from blueprints.preview_bp import preview_bp, init_app as preview_init
from blueprints.effects_bp import effects_bp, init_app as effects_init

# Wire dependencies into blueprints that need shared state
migrate_init(DATABASE, looks_sequences_manager)
settings_init(app_settings, save_settings, socketio)
merge_init(merge_layer, content_manager, load_fixtures_into_classifier)
scenes_init(content_manager)
chases_init(content_manager, chase_engine, unified_engine)
pixel_init(_pixel_arrays, content_manager)
shows_init(get_db, show_engine, cloud_submit, SUPABASE_AVAILABLE, get_supabase_service)
schedules_init(get_db, schedule_runner, timer_runner, cloud_submit,
               SUPABASE_AVAILABLE, get_supabase_service, socketio)
cloud_init(SUPABASE_AVAILABLE, get_supabase_service, get_db, cloud_submit, looks_sequences_manager)
session_init(dmx_state, content_manager, get_whisper_model)
system_init(app_settings, save_settings, AETHER_VERSION, AETHER_COMMIT,
            AETHER_FILE_PATH, AETHER_START_TIME, get_or_create_device_id, app)
groups_init(get_db, SUPABASE_AVAILABLE, get_supabase_service, cloud_submit)
ai_init(get_ai_advisor, get_render_pipeline)
fixtures_init(content_manager, get_db, rdm_manager)
fixture_library_init(fixture_library, content_manager, channel_mapper, looks_sequences_manager, get_db)
node_groups_init(get_db)
rdm_init(rdm_manager, get_db)
preview_init(preview_service)
effects_init(effects_engine, content_manager, unified_engine)

# Register all blueprints with the Flask app
app.register_blueprint(trust_bp)
app.register_blueprint(modifiers_bp)
app.register_blueprint(migrate_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(merge_bp)
app.register_blueprint(scenes_bp)
app.register_blueprint(chases_bp)
app.register_blueprint(pixel_bp)
app.register_blueprint(shows_bp)
app.register_blueprint(schedules_bp)
app.register_blueprint(cloud_bp)
app.register_blueprint(session_bp)
app.register_blueprint(system_bp)
app.register_blueprint(groups_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(fixtures_bp)
app.register_blueprint(fixture_library_bp)
app.register_blueprint(node_groups_bp)
app.register_blueprint(rdm_bp)
app.register_blueprint(preview_bp)
app.register_blueprint(effects_bp)

print(f"[F02] 21 blueprints registered: trust, modifiers, migrate, settings, merge, scenes, chases, pixel, shows, schedules, cloud, session, system, groups, ai, fixtures, fixture_library, node_groups, rdm, preview, effects")

# ============================================================
# Background Services
# ============================================================
def discovery_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', DISCOVERY_PORT))
    sock.settimeout(1.0)
    print(f"✓ Discovery listening on UDP {DISCOVERY_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            msg = json.loads(data.decode())
            msg['ip'] = addr[0]
            msg_type = msg.get('type', 'unknown')
            if msg_type in ('register', 'heartbeat'):
                node_manager.register_node(msg)
                # Report heartbeat to Trust Enforcer (Phase 4 Lane 3)
                node_id = msg.get('node_id', msg.get('hostname', 'unknown'))
                report_node_heartbeat(node_id, {
                    'ip': addr[0],
                    'rssi': msg.get('rssi'),
                    'uptime': msg.get('uptime'),
                    'stale': msg.get('stale', False),
                    'type': msg_type
                })
                if msg_type == 'register':
                    print(f"📥 Node registered: {msg.get('hostname', 'Unknown')} @ {addr[0]}")
                    # Auto-sync content to newly registered node if paired
                    node = node_manager.get_node(msg.get('node_id'))
                    if node and node.get('is_paired'):
                        node_submit(node_manager.sync_content_to_node, node)
        except socket.timeout:
            pass
        except Exception as e:
            if "timed out" not in str(e):
                print(f"Discovery error: {e}")

def stale_checker():
    while True:
        time.sleep(30)
        node_manager.check_stale_nodes()

# ============================================================
# API Routes
# ============================================================
_startup_time = time.monotonic()  # [F01] Track uptime

def check_database_health():
    """Check SQLite database health: exists, not corrupt, disk space OK."""
    result = {'healthy': False, 'file_exists': False, 'size_bytes': 0,
              'integrity': 'unknown', 'disk_free_mb': 0, 'journal_mode': 'unknown',
              'error': None}
    try:
        # Check file exists
        if not os.path.exists(DATABASE):
            result['error'] = 'Database file not found'
            return result
        result['file_exists'] = True
        result['size_bytes'] = os.path.getsize(DATABASE)

        # Check disk space
        stat = os.statvfs(os.path.dirname(DATABASE))
        result['disk_free_mb'] = round((stat.f_bavail * stat.f_frsize) / (1024 * 1024), 1)

        # Quick integrity check using thread-local connection [F04]
        conn = get_db()
        cursor = conn.execute("PRAGMA quick_check(1)")
        check_result = cursor.fetchone()[0]
        result['integrity'] = check_result  # 'ok' if healthy

        # [F04] Report journal mode
        jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
        result['journal_mode'] = jm

        result['healthy'] = (check_result == 'ok' and result['disk_free_mb'] > 50)
    except Exception as e:
        result['error'] = str(e)
    return result


@app.route('/api/flow-map', methods=['GET'])
def flow_map():
    """Serve the interactive AETHER architecture flow map dashboard."""
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AETHER_FLOW_MAP.html')
    if os.path.exists(map_path):
        return send_file(map_path, mimetype='text/html')
    return jsonify({'error': 'Flow map not found'}), 404


@app.route('/api/flow-status', methods=['GET'])
def flow_status():
    """Serve the aether-status.json for flow map live status overlay."""
    status_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aether-status.json')
    if os.path.exists(status_path):
        return send_file(status_path, mimetype='application/json')
    return jsonify({'error': 'Status file not found'}), 404


@app.route('/api/health', methods=['GET'])
def health():
    db_health = check_database_health()
    overall_healthy = db_health['healthy']
    uptime_s = time.monotonic() - _startup_time
    thread_count = threading.active_count()
    _update_thread_hwm()

    # Memory stats via /proc/self/status (Linux) or resource module
    mem_info = {}
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        mem_info['rss_mb'] = round(rusage.ru_maxrss / 1024, 1)  # Linux: kB → MB
    except Exception:
        pass
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    mem_info['rss_mb'] = round(int(line.split()[1]) / 1024, 1)
                elif line.startswith('VmSize:'):
                    mem_info['vms_mb'] = round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass

    # Thread pool stats
    pool_stats = {
        'cloud_pool': {
            'max_workers': _cloud_pool._max_workers,
            'pending': _cloud_pool._work_queue.qsize(),
        },
        'node_pool': {
            'max_workers': _node_pool._max_workers,
            'pending': _node_pool._work_queue.qsize(),
        },
    }

    # Alert thresholds
    thread_alert = thread_count > 30
    if mem_info.get('rss_mb') and mem_info['rss_mb'] > 512:
        overall_healthy = False

    return jsonify({
        'status': 'healthy' if overall_healthy else 'degraded',
        'version': AETHER_VERSION,
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': round(uptime_s, 1),
        'uptime_human': f"{int(uptime_s//3600)}h {int((uptime_s%3600)//60)}m {int(uptime_s%60)}s",
        'thread_count': thread_count,
        'thread_hwm': _thread_hwm,
        'thread_alert': thread_alert,
        'memory': mem_info,
        'pools': pool_stats,
        'services': {
            'database': db_health['healthy'],
            'discovery': True,
            'udpjson': True,
        },
        'database': db_health,
    })

@app.route('/api/arbitration', methods=['GET'])
def get_arbitration():
    """Get arbitration status - who currently owns DMX output"""
    return jsonify({
        'arbitration': arbitration.get_status(),
        'effects': effects_engine.get_status(),
        'chases': {
            'running': list(chase_engine.running_chases.keys()),
            'health': chase_engine.chase_health
        },
        'playback': playback_manager.get_status()
    })

@app.route('/api/version', methods=['GET'])
def version():
    """Get version info for SSOT verification - confirms which backend file is running"""
    uptime_seconds = (datetime.now() - AETHER_START_TIME).total_seconds()
    return jsonify({
        'version': AETHER_VERSION,
        'commit': AETHER_COMMIT,
        'git_commit': AETHER_COMMIT,  # Alias for deploy.sh verification
        'file_path': AETHER_FILE_PATH,
        'cwd': os.getcwd(),
        'started_at': AETHER_START_TIME.isoformat(),
        'uptime_seconds': int(uptime_seconds),
        'python': subprocess.run(['python3', '--version'], capture_output=True, text=True).stdout.strip() if os.name != 'nt' else 'N/A'
    })

# [F02] Session + STT routes moved to blueprints/session_bp.py (4 routes)

# [F02] System routes moved to blueprints/system_bp.py (7 routes + autosync helper)

@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    return jsonify(node_manager.get_all_nodes())

@app.route('/api/nodes/online', methods=['GET'])
def get_online_nodes():
    return jsonify(node_manager.get_all_nodes(include_offline=False))

@app.route('/api/nodes/<node_id>', methods=['GET'])
def get_node(node_id):
    node = node_manager.get_node(node_id)
    return jsonify(node) if node else (jsonify({'error': 'Node not found'}), 404)

@app.route('/api/nodes/<node_id>/pair', methods=['POST'])
def pair_node(node_id):
    try:
        node = node_manager.pair_node(node_id, request.get_json() or {})
        if node:
            return jsonify(node)
        else:
            return jsonify({'error': 'Node not found - it may not have registered yet'}), 404
    except Exception as e:
        print(f"Error pairing node {node_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/nodes/<node_id>/configure', methods=['POST'])
def configure_node(node_id):
    """Update configuration for an already-paired node"""
    config = request.get_json() or {}
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404

    # Support both legacy and new slice field names
    channel_start = config.get('channelStart') or config.get('channel_start')
    channel_end = config.get('channelEnd') or config.get('channel_end')
    slice_mode = config.get('slice_mode')

    # Update database (including type and slice_mode if specified)
    conn = get_db()
    c = conn.cursor()
    new_type = config.get('type')
    if new_type:
        c.execute('''UPDATE nodes SET
            name = COALESCE(?, name),
            universe = COALESCE(?, universe),
            channel_start = COALESCE(?, channel_start),
            channel_end = COALESCE(?, channel_end),
            slice_mode = COALESCE(?, slice_mode),
            type = ?
            WHERE node_id = ?''',
            (config.get('name'), config.get('universe'),
             channel_start, channel_end, slice_mode,
             new_type, str(node_id)))
    else:
        c.execute('''UPDATE nodes SET
            name = COALESCE(?, name),
            universe = COALESCE(?, universe),
            channel_start = COALESCE(?, channel_start),
            channel_end = COALESCE(?, channel_end),
            slice_mode = COALESCE(?, slice_mode)
            WHERE node_id = ?''',
            (config.get('name'), config.get('universe'),
             channel_start, channel_end, slice_mode, str(node_id)))
    conn.commit()
    conn.close()

    # Refresh node from DB
    node = node_manager.get_node(node_id)

    # Send config to node via UDP command port
    if node and node.get('type') == 'wifi':
        node_manager.send_config_to_node(node, {
            'name': node.get('name'),
            'universe': node.get('universe'),
            'channel_start': node.get('channel_start'),
            'channel_end': node.get('channel_end'),
            'slice_mode': node.get('slice_mode', 'zero_outside')
        })

    node_manager.broadcast_status()
    return jsonify({'success': True, 'node': node})

@app.route('/api/nodes/<node_id>/unpair', methods=['POST'])
def unpair_node(node_id):
    node_manager.unpair_node(node_id)
    return jsonify({'success': True})

@app.route('/api/nodes/<node_id>', methods=['DELETE'])
def delete_node(node_id):
    node_manager.delete_node(node_id)
    return jsonify({'success': True})

@app.route('/api/nodes/<node_id>/toggle-visibility', methods=['POST'])
def toggle_node_visibility(node_id):
    """Toggle whether a node is hidden from the dashboard"""
    data = request.get_json() or {}
    hidden = data.get('hidden', False)

    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE nodes SET hidden_from_dashboard = ? WHERE node_id = ?', (1 if hidden else 0, node_id))
    conn.commit()
    conn.close()

    node_manager.broadcast_status()
    return jsonify({'success': True, 'hidden_from_dashboard': hidden})

@app.route('/api/nodes/<node_id>/sync', methods=['POST'])
def sync_node(node_id):
    """Force sync content to a specific node"""
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    node_submit(node_manager.sync_content_to_node, node)
    return jsonify({'success': True, 'message': 'Sync started'})

@app.route('/api/nodes/sync', methods=['POST'])
def sync_all_nodes():
    """Force sync content to all nodes"""
    node_submit(node_manager.sync_all_content)
    return jsonify({'success': True, 'message': 'Full sync started'})

# [F02] Node groups routes moved to blueprints/node_groups_bp.py (10 routes)

# [F02] RDM node-scoped routes moved to blueprints/rdm_bp.py (3 routes)

# ─────────────────────────────────────────────────────────
# DMX Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/dmx/set', methods=['POST'])
def dmx_set():
    """
    Set DMX channel values.

    Supports both legacy channel-based and new fixture-centric modes.

    Legacy (channel-based):
    {
        "universe": 2,
        "channels": {"1": 255, "2": 128},
        "fade_ms": 0
    }

    Fixture-Centric (Phase 0+):
    {
        "universe": 2,
        "fixture_id": "par_1",
        "attributes": {"intensity": 255, "color": [255, 0, 0]},
        "fade_ms": 0
    }

    Or multiple fixtures:
    {
        "universe": 2,
        "fixture_channels": {
            "par_1": {"intensity": 255, "color": [255, 0, 0]},
            "par_2": {"intensity": 200, "color": [0, 255, 0]}
        },
        "fade_ms": 0
    }
    """
    data = request.get_json()
    universe = data.get('universe', 1)

    # Universe 1 is offline - reject
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400

    fade_ms = data.get('fade_ms', 0)

    # Check for fixture-centric mode
    fixture_id = data.get('fixture_id')
    attributes = data.get('attributes')
    fixture_channels = data.get('fixture_channels')

    pipeline = get_render_pipeline()

    # Single fixture mode
    if fixture_id and attributes and pipeline.features.FIXTURE_CENTRIC_ENABLED:
        try:
            channels = pipeline.render_fixture_values(fixture_id, attributes, universe)
            if channels:
                # Convert to string keys for content_manager
                str_channels = {str(k): v for k, v in channels.items()}
                return jsonify(content_manager.set_channels(universe, str_channels, fade_ms))
            else:
                # Fixture not found or not registered
                return jsonify({
                    'error': f'Fixture {fixture_id} not registered in render pipeline',
                    'success': False
                }), 404
        except Exception as e:
            # Fall back to legacy mode on error
            print(f"Fixture-centric render failed, falling back: {e}")

    # Multiple fixtures mode
    if fixture_channels and pipeline.features.FIXTURE_CENTRIC_ENABLED:
        try:
            all_channels = {}
            for fid, attrs in fixture_channels.items():
                channels = pipeline.render_fixture_values(fid, attrs, universe)
                all_channels.update({str(k): v for k, v in channels.items()})
            if all_channels:
                return jsonify(content_manager.set_channels(universe, all_channels, fade_ms))
        except Exception as e:
            print(f"Multi-fixture render failed, falling back: {e}")

    # Legacy channel-based mode (or fallback)
    return jsonify(content_manager.set_channels(
        universe, data.get('channels', {}), fade_ms))

@app.route('/api/dmx/fade', methods=['POST'])
def dmx_fade():
    """Fade channels over duration - sends UDPJSON fade command"""
    data = request.get_json()
    universe = data.get('universe', 2)  # Default to 2 (not 1)
    # Universe 1 is offline - reject
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400
    channels = data.get('channels', {})
    duration_ms = data.get('duration_ms', 1000)
    return jsonify(content_manager.set_channels(universe, channels, duration_ms))

@app.route('/api/dmx/blackout', methods=['POST'])
def dmx_blackout():
    data = request.get_json() or {}
    universe = data.get('universe')
    # Universe 1 is offline - reject if explicitly requested
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400
    # If no universe specified, blackout all online universes (2-5)
    return jsonify(content_manager.blackout(universe, data.get('fade_ms', 1000)))


# ============================================================
# SAFETY ACTIONS - Phase 4 Hardening
# ============================================================
# These endpoints MUST work regardless of:
# - UI state
# - Playback state
# - AI layer availability
# - Backend degradation
#
# Each action logs success/failure explicitly.
# ============================================================

@app.route('/api/dmx/panic', methods=['POST'])
def dmx_panic():
    """SAFETY ACTION: Immediate blackout with no fade.

    Bypasses all playback/effects and commands immediate zero output.
    Use this when something is wrong and you need lights OFF NOW.

    Request body:
        universe (optional): Target universe. If not specified, panics all universes.
    """
    print("🚨 SAFETY ACTION: /api/dmx/panic called", flush=True)
    data = request.get_json() or {}
    universe = data.get('universe')

    results = {'success': True, 'action': 'panic', 'universes': []}

    # Stop all playback first (bypasses normal paths)
    try:
        stop_all_playback(blackout=False)
        print("   ✓ All playback stopped", flush=True)
    except Exception as e:
        print(f"   ⚠️ Failed to stop playback: {e}", flush=True)

    # Get all online nodes
    all_nodes = node_manager.get_all_nodes(include_offline=False)

    if universe is not None:
        # Panic specific universe
        target_nodes = [n for n in all_nodes if n.get('universe') == universe]
        universes_to_panic = [universe]
    else:
        # Panic all universes
        target_nodes = all_nodes
        universes_to_panic = list(set(n.get('universe', 1) for n in all_nodes))

    for univ in universes_to_panic:
        univ_nodes = [n for n in target_nodes if n.get('universe') == univ]
        for node in univ_nodes:
            node_ip = node.get('ip')
            if node_ip:
                try:
                    ssot.send_udpjson_panic(node_ip, univ)
                    results['universes'].append({'universe': univ, 'node': node_ip, 'success': True})
                except Exception as e:
                    results['universes'].append({'universe': univ, 'node': node_ip, 'success': False, 'error': str(e)})
                    results['success'] = False

    # Also clear SSOT state
    for univ in universes_to_panic:
        try:
            dmx_state.universes[univ] = [0] * 512
            print(f"   ✓ SSOT cleared for universe {univ}", flush=True)
        except Exception as e:
            print(f"   ⚠️ Failed to clear SSOT for universe {univ}: {e}", flush=True)

    print(f"🚨 PANIC complete: {len(results['universes'])} nodes commanded", flush=True)
    return jsonify(results)


@app.route('/api/nodes/<node_id>/ping', methods=['POST'])
def ping_node(node_id):
    """SAFETY ACTION: Health check for a specific node.

    Sends a ping command to verify node connectivity.
    Returns success/failure status.
    """
    print(f"🏓 SAFETY ACTION: /api/nodes/{node_id}/ping called", flush=True)

    node = node_manager.get_node(node_id)
    if not node:
        print(f"   ✗ Node {node_id} not found", flush=True)
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        print(f"   ✗ Node {node_id} has no IP address", flush=True)
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = ssot.send_udpjson_ping(node_ip)
        success = result is not None
        print(f"   {'✓' if success else '✗'} Ping to {node_id} ({node_ip}): {'success' if success else 'failed'}", flush=True)
        return jsonify({
            'success': success,
            'node_id': node_id,
            'ip': node_ip,
            'action': 'ping'
        })
    except Exception as e:
        print(f"   ✗ Ping to {node_id} failed: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nodes/<node_id>/reset', methods=['POST'])
def reset_node(node_id):
    """SAFETY ACTION: Reset a specific node.

    Commands the node to reset its internal state:
    - Clears any stuck effects
    - Resets DMX output to zero
    - Reinitializes the node

    Use this when a node is misbehaving or stuck.
    """
    print(f"🔄 SAFETY ACTION: /api/nodes/{node_id}/reset called", flush=True)

    node = node_manager.get_node(node_id)
    if not node:
        print(f"   ✗ Node {node_id} not found", flush=True)
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        print(f"   ✗ Node {node_id} has no IP address", flush=True)
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = ssot.send_udpjson_reset(node_ip)
        success = result is not None
        print(f"   {'✓' if success else '✗'} Reset {node_id} ({node_ip}): {'success' if success else 'failed'}", flush=True)
        return jsonify({
            'success': success,
            'node_id': node_id,
            'ip': node_ip,
            'action': 'reset'
        })
    except Exception as e:
        print(f"   ✗ Reset {node_id} failed: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nodes/<node_id>/stats', methods=['GET'])
def node_stats(node_id):
    """Get real-time stats for a node from stored heartbeat data."""
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404

    return jsonify({
        'rssi': node.get('rssi', 0),
        'wifi_rssi': node.get('rssi', 0),
        'dmx_fps': node.get('fps', 0),
        'fps': node.get('fps', 0),
        'uptime': node.get('uptime', 0),
        'free_heap': node.get('free_heap', 0),
        'firmware': node.get('firmware', 'Unknown'),
        'hardware': 'ESP32',
        'status': node.get('status', 'offline'),
        'rx_packets': node.get('rx_total', 0),
        'tx_packets': node.get('tx_dmx_frames', 0),
        'rx_errors': node.get('rx_bad', 0),
        'tx_errors': 0,
        'node_id': node_id,
        'ip': node.get('ip'),
    })


@app.route('/api/nodes/<node_id>/identify', methods=['POST'])
def identify_node(node_id):
    """Send identify command to flash the node's LED."""
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = node_manager.send_command_to_wifi(node_ip, {"cmd": "identify"})
        return jsonify({'success': bool(result), 'node_id': node_id, 'action': 'identify'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nodes/<node_id>/reboot', methods=['POST'])
def reboot_node(node_id):
    """Send reboot command to restart the node."""
    print(f"🔄 REBOOT: /api/nodes/{node_id}/reboot called", flush=True)
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = node_manager.send_command_to_wifi(node_ip, {"cmd": "reboot"})
        print(f"   {'✓' if result else '✗'} Reboot sent to {node_id} ({node_ip})", flush=True)
        return jsonify({'success': bool(result), 'node_id': node_id, 'action': 'reboot'})
    except Exception as e:
        print(f"   ✗ Reboot {node_id} failed: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nodes/<node_id>/update', methods=['POST'])
def update_node_firmware(node_id):
    """Firmware OTA update — not yet supported over network."""
    return jsonify({
        'success': False,
        'error': 'OTA firmware update not yet supported. Flash via USB.',
        'node_id': node_id,
        'action': 'update'
    }), 501


@app.route('/api/nodes/ping', methods=['POST'])
def ping_all_nodes():
    """SAFETY ACTION: Health check for all online nodes.

    Sends ping commands to all known online nodes.
    Returns a summary of which nodes responded.
    """
    print("🏓 SAFETY ACTION: /api/nodes/ping (all) called", flush=True)

    all_nodes = node_manager.get_all_nodes(include_offline=False)
    results = {'success': True, 'nodes': [], 'total': len(all_nodes), 'responded': 0}

    for node in all_nodes:
        node_id = node.get('node_id')
        node_ip = node.get('ip')

        if not node_ip:
            results['nodes'].append({'node_id': node_id, 'success': False, 'error': 'No IP'})
            continue

        try:
            result = ssot.send_udpjson_ping(node_ip)
            success = result is not None
            results['nodes'].append({'node_id': node_id, 'ip': node_ip, 'success': success})
            if success:
                results['responded'] += 1
        except Exception as e:
            results['nodes'].append({'node_id': node_id, 'ip': node_ip, 'success': False, 'error': str(e)})

    # Overall success only if all nodes responded
    results['success'] = results['responded'] == results['total']
    print(f"🏓 Ping all complete: {results['responded']}/{results['total']} nodes responded", flush=True)
    return jsonify(results)


# ============================================================
# Trust Enforcement API (Phase 4 Lane 3)
# ============================================================

# [F02] Trust routes moved to blueprints/trust_bp.py (4 routes)

# [F02] RDM consolidated routes moved to blueprints/rdm_bp.py (12 routes)

@app.route('/api/dmx/master', methods=['POST'])
def dmx_master():
    """Master dimmer - scales all output proportionally

    SSOT COMPLIANCE: Routes through ContentManager.set_channels for unified dispatch.
    """
    data = request.get_json() or {}
    level = data.get('level', 100)
    capture = data.get('capture', False)

    print(f"🎚️ Master dimmer: level={level}%, capture={capture}", flush=True)

    # Capture current state if requested or if we don't have a base yet
    if capture or not dmx_state.master_base:
        dmx_state.master_base = {}
        captured_any = False

        for univ, channels in dmx_state.universes.items():
            if any(v > 0 for v in channels):
                dmx_state.master_base[univ] = list(channels)
                total_val = sum(channels)
                print(f"   📸 Captured universe {univ}: {total_val} total brightness", flush=True)
                captured_any = True

        if not captured_any:
            print("   ⚠️ No active channels to capture", flush=True)
            return jsonify({'success': False, 'error': 'No active lighting to dim'})

    dmx_state.master_level = level
    scale = level / 100.0

    all_results = []
    for univ, base in dmx_state.master_base.items():
        scaled = {}
        for ch_idx, base_val in enumerate(base):
            if base_val > 0:
                scaled[str(ch_idx + 1)] = int(base_val * scale)

        if scaled:
            print(f"   🔧 Scaling universe {univ}: {len(scaled)} channels at {level}%", flush=True)
            # SSOT FIX: Route through ContentManager.set_channels (was direct node send)
            result = content_manager.set_channels(univ, scaled, fade_ms=0)
            all_results.append({'universe': univ, 'result': result})

    return jsonify({'success': True, 'level': level, 'results': all_results})

@app.route('/api/dmx/master/reset', methods=['POST'])
def dmx_master_reset():
    dmx_state.master_base = {}
    dmx_state.master_level = 100
    return jsonify({'success': True})


@app.route('/api/dmx/universe/<int:universe>', methods=['GET'])
def dmx_get_universe(universe):
    return jsonify({'universe': universe, 'channels': dmx_state.get_universe(universe)})

@app.route('/api/dmx/status', methods=['GET'])
def dmx_status():
    """Get DMX system status with online nodes and universe info"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT node_id, name, universe, ip, status, channel_start, channel_end, slice_mode, last_seen
        FROM nodes
        WHERE is_paired = 1 AND type = 'wifi'
    """)
    nodes = []
    for row in c.fetchall():
        nodes.append({
            'node_id': row[0],
            'name': row[1],
            'universe': row[2],
            'ip': row[3],
            'status': row[4],
            'slice_start': row[5] or 1,
            'slice_end': row[6] or 512,
            'slice_mode': row[7] or 'zero_outside',
            'last_seen': row[8]
        })
    conn.close()

    # Group by universe
    universes = {}
    for node in nodes:
        u = node['universe']
        if u not in universes:
            universes[u] = []
        universes[u].append(node)

    return jsonify({
        'transport': 'udpjson',
        'port': AETHER_UDPJSON_PORT,
        'online_nodes': [n for n in nodes if n['status'] == 'online'],
        'all_nodes': nodes,
        'universes': universes,
        'universe_1_note': 'Universe 1 is OFFLINE - use universes 2-5',
        'stats': {
            'total_sends': node_manager._udpjson_send_count,
            'errors': node_manager._udpjson_errors,
            'per_universe': node_manager._udpjson_per_universe
        }
    })

@app.route('/api/dmx/diagnostics', methods=['GET'])
def dmx_diagnostics():
    """Diagnostics endpoint for debugging DMX output issues

    SSOT COMPLIANCE: This endpoint shows complete state of the SSOT system,
    including ownership, routing, and any rejected writes.
    """
    arb_status = arbitration.get_status()

    # Calculate active channels per universe
    universe_stats = {}
    for univ, channels in dmx_state.universes.items():
        non_zero = sum(1 for v in channels if v > 0)
        total_brightness = sum(channels)
        universe_stats[univ] = {
            'active_channels': non_zero,
            'total_brightness': total_brightness,
            'max_value': max(channels) if channels else 0
        }

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'packet_version': NodeManager.PACKET_VERSION,
        'packet_version_info': 'v3: full 512-ch frames, ESP32 firmware v1.1 has 2500-byte buffer',

        # Transport diagnostics (UDPJSON only)
        'transport': {
            'udpjson': {
                'port': AETHER_UDPJSON_PORT,
                'last_send': node_manager._last_udpjson_send,
                'total_sends': node_manager._udpjson_send_count,
                'errors': node_manager._udpjson_errors,
                'per_universe': node_manager._udpjson_per_universe,
                'output': 'Direct UDP JSON to ESP32 nodes'
            },
            'config_udp': {
                'last_send': node_manager._last_udp_send,
                'total_sends': node_manager._udp_send_count,
                'port': WIFI_COMMAND_PORT
            }
        },

        # SSOT ownership and control
        'ownership': {
            'current_owner': arb_status.get('current_owner'),
            'current_id': arb_status.get('current_id'),
            'priority': ArbitrationManager.PRIORITY.get(arb_status.get('current_owner'), 0),
            'blackout_active': arb_status.get('blackout_active'),
            'last_change': arb_status.get('last_change'),
            'last_writer': arb_status.get('last_writer'),
            'last_scene_id': arb_status.get('last_scene_id'),
            'last_scene_time': arb_status.get('last_scene_time')
        },

        # Write statistics for detecting spammers
        'writes_per_service': arb_status.get('writes_per_service', {}),

        # Rejected writes (potential conflicts)
        'rejected_writes': arb_status.get('rejected_writes', []),

        # Arbitration history
        'arbitration_history': arb_status.get('history', []),

        # Running engines
        'engines': {
            'effects': effects_engine.get_status(),
            'chase': {
                'running_chases': list(chase_engine.running_chases.keys()),
                'health': chase_engine.chase_health
            },
            'show': {
                'running': show_engine.running,
                'current_show': show_engine.current_show.get('name') if show_engine.current_show else None,
                'paused': show_engine.paused,
                'tempo': show_engine.tempo
            },
            'schedule': {
                'running': schedule_runner.running,
                'schedule_count': len(schedule_runner.schedules)
            }
        },

        # Playback state per universe
        'playback': playback_manager.get_status(),

        # SSOT state
        'ssot': {
            'universes_active': list(dmx_state.universes.keys()),
            'universe_stats': universe_stats,
            'master_level': dmx_state.master_level,
            'has_master_base': bool(dmx_state.master_base)
        },

        # System info
        'system': {
            'version': AETHER_VERSION,
            'commit': AETHER_COMMIT,
            'uptime_seconds': int((datetime.now() - AETHER_START_TIME).total_seconds()),
            'file_path': AETHER_FILE_PATH
        },

        # SSOT compliance summary
        'ssot_compliance': {
            'all_services_routed': True,  # After fixes, all services route through SSOT
            'arbitration_enforced': True,
            'dispatcher_unified': True,
            'notes': [
                'Manual faders: /api/dmx/set -> ContentManager.set_channels',
                'Scenes: /api/scenes/{id}/play -> ContentManager.play_scene -> set_channels',
                'Chases: ChaseEngine._send_step -> ContentManager.set_channels',
                'Effects: DynamicEffectsEngine._send_frame -> ssot_send_frame -> set_channels',
                'Blackout: ContentManager.blackout',
                'Master dimmer: /api/dmx/master -> ContentManager.set_channels',
                'Output: NodeManager UDPJSON to ESP32 nodes on port 6455'
            ]
        }
    })

# [F02] Pixel Array routes moved to blueprints/pixel_bp.py (11 routes)


# [F02] Pixel Array route handlers removed — see blueprints/pixel_bp.py

# [F02] Scene routes moved to blueprints/scenes_bp.py (6 routes)

# [F02] Chase routes moved to blueprints/chases_bp.py (8 routes)

# ─────────────────────────────────────────────────────────
# Looks Routes (New unified architecture - replaces Scenes)
# ─────────────────────────────────────────────────────────
@app.route('/api/looks', methods=['GET'])
def get_looks():
    """Get all Looks"""
    looks = looks_sequences_manager.get_all_looks()
    return jsonify([l.to_dict() for l in looks])

@app.route('/api/looks', methods=['POST'])
def create_look():
    """Create a new Look"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_look_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create Look object
    look = Look(
        look_id=data.get('look_id', f"look_{int(time.time() * 1000)}"),
        name=data['name'],
        channels=data['channels'],
        modifiers=[Modifier.from_dict(m) for m in data.get('modifiers', [])],
        fade_ms=data.get('fade_ms', 0),
        color=data.get('color', 'blue'),
        icon=data.get('icon', 'lightbulb'),
        description=data.get('description', ''),
    )

    result = looks_sequences_manager.create_look(look)

    # Async sync to Supabase (non-blocking)
    if SUPABASE_AVAILABLE:
        supabase = get_supabase_service()
        if supabase and supabase.is_enabled():
            cloud_submit(supabase.sync_look, result.to_dict())

    return jsonify({'success': True, 'look': result.to_dict()})

@app.route('/api/looks/<look_id>', methods=['GET'])
def get_look(look_id):
    """Get a Look by ID"""
    look = looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'error': 'Look not found'}), 404
    return jsonify(look.to_dict())

@app.route('/api/looks/<look_id>', methods=['PUT'])
def update_look(look_id):
    """Update an existing Look"""
    data = request.get_json() or {}

    # Validate if full replacement
    if 'channels' in data:
        valid, error = validate_look_data(data)
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = looks_sequences_manager.update_look(look_id, data)
    if not result:
        return jsonify({'error': 'Look not found'}), 404

    # Async sync to Supabase (non-blocking)
    if SUPABASE_AVAILABLE:
        supabase = get_supabase_service()
        if supabase and supabase.is_enabled():
            cloud_submit(supabase.sync_look, result.to_dict())

    return jsonify({'success': True, 'look': result.to_dict()})

@app.route('/api/looks/<look_id>', methods=['DELETE'])
def delete_look(look_id):
    """Delete a Look"""
    success = looks_sequences_manager.delete_look(look_id)
    if not success:
        return jsonify({'error': 'Look not found'}), 404

    # Note: Supabase delete is not implemented yet (would need to mark as deleted)
    return jsonify({'success': True, 'look_id': look_id})

@app.route('/api/looks/<look_id>/versions', methods=['GET'])
def get_look_versions(look_id):
    """Get version history for a Look"""
    versions = looks_sequences_manager.get_versions(look_id, 'look')
    return jsonify({'success': True, 'versions': versions})

@app.route('/api/looks/<look_id>/versions/<version_id>/revert', methods=['POST'])
def revert_look_version(look_id, version_id):
    """Revert a Look to a specific version"""
    result = looks_sequences_manager.revert_to_version(version_id)
    if not result:
        return jsonify({'error': 'Version not found or revert failed'}), 404
    return jsonify({'success': True, 'look': result})

# ============================================================================
# ✅  TASK-0018 RESOLVED (F06 consolidation)
# ============================================================================
# Previously used RenderEngine directly — now routes through
# UnifiedPlaybackEngine via unified_play_look() for all look types.
# ============================================================================
@app.route('/api/looks/<look_id>/play', methods=['POST'])
def play_look(look_id):
    """
    Play a Look with real-time modifier rendering.

    Routes through UnifiedPlaybackEngine (canonical authority per Hard Rule 1.1).

    POST body:
    {
        "universes": [1, 2],       // Target universes (default: all online)
        "fade_ms": 500,            // Initial fade time (optional)
    }
    """
    data = request.get_json() or {}

    # Get the look
    look = looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'error': 'Look not found'}), 404

    # Acquire arbitration
    if not arbitration.acquire('look', look_id):
        return jsonify({
            'success': False,
            'error': 'Cannot play look - arbitration denied',
            'current_owner': arbitration.current_owner
        }), 409

    # Stop any legacy render engine jobs (migration safety net)
    render_engine.stop_rendering()

    # Determine target universes
    universes = data.get('universes')
    if not universes:
        universes = list(set(
            n.get('universe', 1) for n in node_manager.get_nodes()
            if n.get('is_paired') and n.get('status') == 'online'
        ))
        if not universes:
            universes = [1]

    fade_ms = data.get('fade_ms', look.fade_ms or 0)
    has_modifiers = len(look.modifiers) > 0 and any(m.enabled for m in look.modifiers)

    # Route through UnifiedPlaybackEngine (canonical authority)
    look_data = look.to_dict()
    session_id = unified_play_look(
        look_id,
        look_data,
        universes=universes,
        fade_ms=fade_ms,
    )

    return jsonify({
        'success': True,
        'look_id': look_id,
        'name': look.name,
        'universes': universes,
        'rendering': has_modifiers,
        'modifier_count': len([m for m in look.modifiers if m.enabled]) if has_modifiers else 0,
        'session_id': session_id,
        'engine': 'unified',
    })

@app.route('/api/looks/<look_id>/stop', methods=['POST'])
def stop_look(look_id):
    """Stop playing a Look"""
    # Stop unified engine look sessions
    unified_engine.stop_type(PlaybackType.LOOK)
    # Stop legacy render engine (migration safety net)
    render_engine.stop_rendering()

    # Release arbitration if we own it
    if arbitration.current_owner == 'look' and arbitration.current_id == look_id:
        arbitration.release('look')

    return jsonify({
        'success': True,
        'look_id': look_id,
        'stopped': True,
    })

@app.route('/api/looks/preview', methods=['POST'])
def preview_look():
    """
    Preview a Look without saving - render a single frame.

    POST body:
    {
        "channels": {"1": 255, "2": 128},
        "modifiers": [...],
        "elapsed_time": 0.5,    // Simulated time for preview
        "seed": 12345           // Optional seed
    }

    Returns the computed channel values for preview.
    """
    data = request.get_json() or {}

    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    elapsed_time = data.get('elapsed_time', 0.0)
    seed = data.get('seed', 0)

    if not channels:
        return jsonify({'error': 'Channels required'}), 400

    # Validate modifiers
    for mod in modifiers:
        valid, error = validate_modifier(mod)
        if not valid:
            return jsonify({'error': f'Invalid modifier: {error}'}), 400

    # Render single frame
    result = render_look_frame(
        channels=channels,
        modifiers=[normalize_modifier(m) for m in modifiers],
        elapsed_time=elapsed_time,
        seed=seed,
    )

    return jsonify({
        'success': True,
        'input_channels': channels,
        'output_channels': result,
        'elapsed_time': elapsed_time,
        'modifier_count': len(modifiers),
    })

@app.route('/api/render/status', methods=['GET'])
def get_render_status():
    """Get current render engine status"""
    return jsonify(render_engine.get_status())

@app.route('/api/render/stop', methods=['POST'])
def stop_render():
    """Stop all rendering"""
    render_engine.stop_rendering()
    return jsonify({'success': True, 'stopped': True})

# ─────────────────────────────────────────────────────────
# Sequences Routes (New unified architecture - replaces Chases)
# ─────────────────────────────────────────────────────────
@app.route('/api/sequences', methods=['GET'])
def get_sequences():
    """Get all Sequences"""
    sequences = looks_sequences_manager.get_all_sequences()
    return jsonify([s.to_dict() for s in sequences])

@app.route('/api/sequences', methods=['POST'])
def create_sequence():
    """Create a new Sequence"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_sequence_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create Sequence object
    steps = [SequenceStep.from_dict(s) for s in data.get('steps', [])]
    sequence = Sequence(
        sequence_id=data.get('sequence_id', f"sequence_{int(time.time() * 1000)}"),
        name=data['name'],
        steps=steps,
        bpm=data.get('bpm', 120),
        loop=data.get('loop', True),
        color=data.get('color', 'green'),
        description=data.get('description', ''),
    )

    result = looks_sequences_manager.create_sequence(sequence)

    # Async sync to Supabase (non-blocking)
    if SUPABASE_AVAILABLE:
        supabase = get_supabase_service()
        if supabase and supabase.is_enabled():
            cloud_submit(supabase.sync_sequence, result.to_dict())

    return jsonify({'success': True, 'sequence': result.to_dict()})

@app.route('/api/sequences/<sequence_id>', methods=['GET'])
def get_sequence(sequence_id):
    """Get a Sequence by ID"""
    sequence = looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'error': 'Sequence not found'}), 404
    return jsonify(sequence.to_dict())

@app.route('/api/sequences/<sequence_id>', methods=['PUT'])
def update_sequence(sequence_id):
    """Update an existing Sequence"""
    data = request.get_json() or {}

    # Validate if steps are being updated
    if 'steps' in data:
        valid, error = validate_sequence_data(data)
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = looks_sequences_manager.update_sequence(sequence_id, data)
    if not result:
        return jsonify({'error': 'Sequence not found'}), 404

    # Async sync to Supabase (non-blocking)
    if SUPABASE_AVAILABLE:
        supabase = get_supabase_service()
        if supabase and supabase.is_enabled():
            cloud_submit(supabase.sync_sequence, result.to_dict())

    return jsonify({'success': True, 'sequence': result.to_dict()})

@app.route('/api/sequences/<sequence_id>', methods=['DELETE'])
def delete_sequence(sequence_id):
    """Delete a Sequence"""
    success = looks_sequences_manager.delete_sequence(sequence_id)
    if not success:
        return jsonify({'error': 'Sequence not found'}), 404

    # Note: Supabase delete is not implemented yet
    return jsonify({'success': True, 'sequence_id': sequence_id})

@app.route('/api/sequences/<sequence_id>/versions', methods=['GET'])
def get_sequence_versions(sequence_id):
    """Get version history for a Sequence"""
    versions = looks_sequences_manager.get_versions(sequence_id, 'sequence')
    return jsonify({'success': True, 'versions': versions})

@app.route('/api/sequences/<sequence_id>/versions/<version_id>/revert', methods=['POST'])
def revert_sequence_version(sequence_id, version_id):
    """Revert a Sequence to a specific version"""
    result = looks_sequences_manager.revert_to_version(version_id)
    if not result:
        return jsonify({'error': 'Version not found or revert failed'}), 404
    return jsonify({'success': True, 'sequence': result})

# ─────────────────────────────────────────────────────────
# Unified Playback API (Phase 4)
# ─────────────────────────────────────────────────────────

@app.route('/api/sequences/<sequence_id>/play', methods=['POST'])
def play_sequence(sequence_id):
    """
    Play a Sequence with step timing and modifiers.

    Uses UnifiedPlaybackEngine (unified_playback.py) as the canonical authority.
    playback_controller.py was deleted in Phase 3.

    POST body:
    {
        "universes": [1, 2],        // Target universes (default: all online)
        "loop_mode": "loop",        // one_shot, loop, bounce (default: from sequence)
        "bpm": 120,                 // BPM override (optional)
        "start_step": 0,            // Starting step index (default: 0)
        "seed": 12345               // Random seed for determinism (optional)
    }
    """
    data = request.get_json() or {}

    # Get the sequence
    sequence = looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'error': 'Sequence not found'}), 404

    if not sequence.steps:
        return jsonify({'error': 'Sequence has no steps'}), 400

    # Acquire arbitration
    if not arbitration.acquire('sequence', sequence_id):
        return jsonify({
            'success': False,
            'error': 'Cannot play sequence - arbitration denied',
            'current_owner': arbitration.current_owner
        }), 409

    # Stop any existing playback (all engines)
    render_engine.stop_rendering()
    chase_engine.stop_all()
    unified_stop()  # Canonical authority for playback

    # Determine target universes
    universes = data.get('universes')
    if not universes:
        universes = list(set(
            n.get('universe', 1) for n in node_manager.get_nodes()
            if n.get('is_paired') and n.get('status') == 'online'
        ))
        if not universes:
            universes = [1]

    # Parse loop mode
    loop_mode_str = data.get('loop_mode', 'loop' if sequence.loop else 'one_shot')

    # Get BPM (override or sequence default)
    bpm = data.get('bpm', sequence.bpm)

    # Convert sequence steps to playback format
    steps = []
    for step in sequence.steps:
        step_data = {
            'step_id': step.step_id,
            'name': step.name,
            'look_id': step.look_id,
            'channels': step.channels or {},
            'modifiers': [m.to_dict() for m in step.modifiers],
            'fade_ms': step.fade_ms,
            'hold_ms': step.hold_ms,
        }
        steps.append(step_data)

    # =========================================================================
    # MIGRATION (TASK-0021): Use UnifiedPlaybackEngine as canonical authority
    # =========================================================================
    # Build sequence_data dict for unified_play_sequence
    sequence_data = {
        'name': sequence.name,
        'steps': steps,
        'loop_mode': loop_mode_str,
        'bpm': bpm,
    }

    # Start playback via canonical UnifiedPlaybackEngine
    start_step = data.get('start_step', 0)
    seed = data.get('seed')

    print(f"[MIGRATION] play_sequence: Using UnifiedPlaybackEngine (canonical)", flush=True)
    print(f"[MIGRATION]   sequence_id={sequence_id}, universes={universes}, "
          f"start_step={start_step}, bpm={bpm}, loop_mode={loop_mode_str}", flush=True)

    session_id = unified_play_sequence(
        sequence_id=sequence_id,
        sequence_data=sequence_data,
        universes=universes,
        start_step=start_step,
        seed=seed,
    )

    # Build result compatible with old format
    result = {
        'success': True,
        'job_id': session_id,  # session_id maps to job_id for compatibility
        'session_id': session_id,
        'sequence_id': sequence_id,
        'universes': universes,
        'step_count': len(steps),
        'bpm': bpm,
        'loop_mode': loop_mode_str,
    }

    print(f"[MIGRATION] Sequence '{sequence.name}' playing via UnifiedPlaybackEngine "
          f"(session: {session_id})", flush=True)

    return jsonify({
        **result,
        'name': sequence.name,
        'engine': 'unified',  # Indicates which engine is active
    })


@app.route('/api/sequences/<sequence_id>/stop', methods=['POST'])
def stop_sequence(sequence_id):
    """
    Stop sequence playback.

    MIGRATION NOTE (TASK-0021): Now uses UnifiedPlaybackEngine.
    """
    # Check unified engine (canonical authority)
    unified_status = unified_get_status()
    for session_info in unified_status.get('sessions', []):
        sid = session_info.get('session_id', '')
        # Check if this session matches the sequence
        if f"seq_{sequence_id}_" in sid:
            result = unified_stop(sid)
            arbitration.release('sequence')
            return jsonify({**result, 'sequence_id': sequence_id})

    return jsonify({'success': True, 'message': 'Sequence was not playing'})


# ─────────────────────────────────────────────────────────
# Unified Playback API (New - replaces fragmented playback)
# ─────────────────────────────────────────────────────────

@app.route('/api/unified/status', methods=['GET'])
def get_unified_status():
    """Get unified playback engine status"""
    return jsonify(unified_engine.get_status())


@app.route('/api/unified/play/look/<look_id>', methods=['POST'])
def unified_api_play_look(look_id):
    """Play a Look via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 0)

    look = looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'success': False, 'error': 'Look not found'}), 404

    session = session_factory.from_look(look_id, look.to_dict(), universes, fade_ms)
    session_id = unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'look',
        'name': look.name
    })


@app.route('/api/unified/play/sequence/<sequence_id>', methods=['POST'])
def unified_api_play_sequence(sequence_id):
    """Play a Sequence via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])

    sequence = looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'success': False, 'error': 'Sequence not found'}), 404

    session = session_factory.from_sequence(sequence_id, sequence.to_dict(), universes)
    session_id = unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'sequence',
        'name': sequence.name
    })


@app.route('/api/unified/play/chase/<chase_id>', methods=['POST'])
def unified_api_play_chase(chase_id):
    """Play a Chase via unified engine (legacy support)"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])

    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM chases WHERE chase_id = ?', (chase_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'success': False, 'error': 'Chase not found'}), 404

    chase_data = dict(row)
    chase_data['steps'] = json.loads(chase_data.get('steps', '[]'))

    session = session_factory.from_chase(chase_id, chase_data, universes)
    session_id = unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'chase',
        'name': chase_data.get('name', chase_id)
    })


@app.route('/api/unified/play/scene/<scene_id>', methods=['POST'])
def unified_api_play_scene(scene_id):
    """Play a Scene via unified engine (legacy support)"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 0)

    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'success': False, 'error': 'Scene not found'}), 404

    scene_data = dict(row)
    scene_data['channels'] = json.loads(scene_data.get('channels', '{}'))

    session = session_factory.from_scene(scene_id, scene_data, universes, fade_ms)
    session_id = unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'scene',
        'name': scene_data.get('name', scene_id)
    })


@app.route('/api/unified/play/effect', methods=['POST'])
def unified_api_play_effect():
    """Play a built-in effect via unified engine"""
    data = request.get_json() or {}
    effect_type = data.get('effect_type', 'pulse')
    params = data.get('params', {})
    universes = data.get('universes', [2, 3, 4, 5])

    session = session_factory.from_effect(effect_type, params, universes)
    session_id = unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'effect',
        'effect_type': effect_type
    })


@app.route('/api/unified/blackout', methods=['POST'])
def unified_api_blackout():
    """Trigger blackout via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 1000)

    # Get current state for fade (convert list to channel dict)
    fade_from = {}
    if fade_ms > 0:
        for u in universes:
            universe_data = dmx_state.get_universe(u)
            # Convert list [val0, val1, ...] to dict {1: val0, 2: val1, ...}
            for ch_idx, val in enumerate(universe_data):
                fade_from[ch_idx + 1] = val

    session = session_factory.blackout(universes, fade_ms)
    session_id = unified_engine.play(session, fade_from)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'blackout'
    })


@app.route('/api/unified/stop', methods=['POST'])
def unified_api_stop():
    """Stop unified playback - uses SSOT stop_all_playback for complete stop"""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    fade_ms = data.get('fade_ms', 0)

    if session_id:
        # Stop specific session only
        unified_engine.stop_session(session_id, fade_ms)
        return jsonify({'success': True, 'stopped': session_id})
    else:
        # Stop ALL playback sources (shows, chases, effects, unified engine)
        result = stop_all_playback(blackout=False, fade_ms=fade_ms)
        # Also stop unified engine sessions
        unified_engine.stop_all(fade_ms)
        return jsonify({'success': True, 'stopped': 'all', 'results': result})


@app.route('/api/unified/stop/<playback_type>', methods=['POST'])
def unified_api_stop_type(playback_type):
    """Stop all sessions of a specific type"""
    data = request.get_json() or {}
    fade_ms = data.get('fade_ms', 0)

    try:
        ptype = PlaybackType(playback_type)
        unified_engine.stop_type(ptype, fade_ms)
        return jsonify({'success': True, 'stopped_type': playback_type})
    except ValueError:
        return jsonify({'success': False, 'error': f'Unknown type: {playback_type}'}), 400


@app.route('/api/unified/pause/<session_id>', methods=['POST'])
def unified_api_pause(session_id):
    """Pause a session"""
    result = unified_engine.pause(session_id)
    return jsonify({'success': result, 'session_id': session_id})


@app.route('/api/unified/resume/<session_id>', methods=['POST'])
def unified_api_resume(session_id):
    """Resume a paused session"""
    result = unified_engine.resume(session_id)
    return jsonify({'success': result, 'session_id': session_id})


@app.route('/api/unified/sessions', methods=['GET'])
def unified_api_get_sessions():
    """Get all active sessions"""
    sessions = unified_engine.get_active_sessions()
    return jsonify([{
        'session_id': s.session_id,
        'type': s.playback_type.value,
        'name': s.name,
        'state': s.state.value,
        'priority': s.priority.value,
        'universes': s.universes,
        'elapsed_time': s.elapsed_time,
        'frame_count': s.frame_count,
    } for s in sessions])


# ─────────────────────────────────────────────────────────
# Legacy Playback API (maintains backward compatibility)
# ─────────────────────────────────────────────────────────

@app.route('/api/playback/status', methods=['GET'])
def get_playback_status():
    """Get playback status from UnifiedPlaybackEngine (canonical authority)."""
    status = unified_get_status()
    # Add playing flag for backward compatibility
    status['playing'] = bool(status.get('sessions'))
    return jsonify(status)


@app.route('/api/playback/stop', methods=['POST'])
def api_stop_all_playback():
    """Stop all playback (Look, Sequence, Chase, Effect, Show).

    SSOT: This endpoint uses the unified stop_all_playback function to ensure
    consistent behavior across all stop controls (UI buttons, hotkeys, etc.)

    Optional body params:
        blackout: bool - If true, also send blackout command
        fade_ms: int - Fade time for blackout (default 1000)
        universe: int - Specific universe to stop (default all)
    """
    data = request.get_json(silent=True) or {}
    blackout = data.get('blackout', False)
    fade_ms = data.get('fade_ms', 1000)
    universe = data.get('universe')

    # Use unified SSOT stop function
    result = stop_all_playback(blackout=blackout, fade_ms=fade_ms, universe=universe)

    # Also stop render engine (not in unified function as it's a separate system)
    render_engine.stop_rendering()

    return jsonify(result)


@app.route('/api/playback/pause', methods=['POST'])
def pause_playback():
    """Pause current playback via UnifiedPlaybackEngine (canonical authority)."""
    return jsonify(unified_pause())


@app.route('/api/playback/resume', methods=['POST'])
def resume_playback():
    """Resume paused playback via UnifiedPlaybackEngine (canonical authority)."""
    return jsonify(unified_resume())

    return jsonify({'success': False, 'error': 'Nothing paused to resume'})


# ─────────────────────────────────────────────────────────
# Cue Stacks API (Manual Theatrical Cueing)
# ─────────────────────────────────────────────────────────

@app.route('/api/cue-stacks', methods=['GET'])
def get_cue_stacks():
    """Get all Cue Stacks"""
    stacks = cue_stacks_manager.get_all_cue_stacks()
    return jsonify([s.to_dict() for s in stacks])

@app.route('/api/cue-stacks', methods=['POST'])
def create_cue_stack():
    """Create a new Cue Stack"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_cue_stack_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create CueStack object
    cues = [Cue.from_dict(c) for c in data.get('cues', [])]
    stack = CueStack(
        stack_id=data.get('stack_id', f"stack_{int(time.time() * 1000)}"),
        name=data['name'],
        cues=cues,
        color=data.get('color', 'purple'),
        description=data.get('description', ''),
    )

    result = cue_stacks_manager.create_cue_stack(stack)
    return jsonify({'success': True, 'cue_stack': result.to_dict()})

@app.route('/api/cue-stacks/<stack_id>', methods=['GET'])
def get_cue_stack(stack_id):
    """Get a Cue Stack by ID"""
    stack = cue_stacks_manager.get_cue_stack(stack_id)
    if not stack:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify(stack.to_dict())

@app.route('/api/cue-stacks/<stack_id>', methods=['PUT'])
def update_cue_stack(stack_id):
    """Update an existing Cue Stack"""
    data = request.get_json() or {}

    # Validate if cues are being updated
    if 'cues' in data:
        valid, error = validate_cue_stack_data({**data, 'name': data.get('name', 'temp')})
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = cue_stacks_manager.update_cue_stack(stack_id, data)
    if not result:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify({'success': True, 'cue_stack': result.to_dict()})

@app.route('/api/cue-stacks/<stack_id>', methods=['DELETE'])
def delete_cue_stack(stack_id):
    """Delete a Cue Stack"""
    success = cue_stacks_manager.delete_cue_stack(stack_id)
    if not success:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify({'success': True, 'stack_id': stack_id})

@app.route('/api/cue-stacks/<stack_id>/go', methods=['POST'])
def cue_stack_go(stack_id):
    """
    Execute the next cue (Go button).
    Manually triggers the next cue in the stack.
    """
    # Helper to resolve look_id to channels
    def look_resolver(look_id):
        look = looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = cue_stacks_manager.go(stack_id, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        # Acquire arbitration for cue stack
        if not arbitration.acquire('cue_stack', stack_id):
            return jsonify({
                'success': False,
                'error': 'Cannot execute cue - arbitration denied',
                'current_owner': arbitration.current_owner
            }), 409

        # Determine target universes from channel keys
        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                # Universe:channel format
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                # Just channel number, assume universe 1
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        # If no explicit universes, use all online nodes
        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in node_manager.get_nodes()
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        # Apply the cue with fade via merge layer
        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not merge_layer.get_source(source_id):
            merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            # Convert to int keys for merge layer
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            # Update merge layer and output
            merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = merge_layer.compute_merge(univ)
            if merged:
                merge_layer_output(univ, merged)

    return jsonify(result)

@app.route('/api/cue-stacks/<stack_id>/back', methods=['POST'])
def cue_stack_back(stack_id):
    """
    Go back to previous cue (Back button).
    """
    def look_resolver(look_id):
        look = looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = cue_stacks_manager.back(stack_id, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output (same logic as go)
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in node_manager.get_nodes()
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not merge_layer.get_source(source_id):
            merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = merge_layer.compute_merge(univ)
            if merged:
                merge_layer_output(univ, merged)

    return jsonify(result)

@app.route('/api/cue-stacks/<stack_id>/goto/<cue_number>', methods=['POST'])
def cue_stack_goto(stack_id, cue_number):
    """
    Jump to a specific cue by number.
    """
    def look_resolver(look_id):
        look = looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = cue_stacks_manager.goto(stack_id, cue_number, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        if not arbitration.acquire('cue_stack', stack_id):
            return jsonify({
                'success': False,
                'error': 'Cannot execute cue - arbitration denied',
                'current_owner': arbitration.current_owner
            }), 409

        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in node_manager.get_nodes()
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not merge_layer.get_source(source_id):
            merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = merge_layer.compute_merge(univ)
            if merged:
                merge_layer_output(univ, merged)

    return jsonify(result)

@app.route('/api/cue-stacks/<stack_id>/stop', methods=['POST'])
def cue_stack_stop(stack_id):
    """Stop cue stack playback and release output"""
    result = cue_stacks_manager.stop(stack_id)

    # Release merge layer source
    merge_layer.unregister_source(f"cue_stack_{stack_id}")

    # Release arbitration
    if arbitration.current_owner == ('cue_stack', stack_id):
        arbitration.release()

    return jsonify(result)

@app.route('/api/cue-stacks/<stack_id>/status', methods=['GET'])
def cue_stack_status(stack_id):
    """Get current playback status for a cue stack"""
    result = cue_stacks_manager.get_status(stack_id)
    if not result.get('success'):
        return jsonify(result), 404
    return jsonify(result)


# [F02] Merge Layer routes moved to blueprints/merge_bp.py (5 routes)

# [F02] Preview routes moved to blueprints/preview_bp.py (11 routes)


# [F02] Modifier & Distribution routes moved to blueprints/modifiers_bp.py (11 routes)

# [F02] AI suggestions + render pipeline routes moved to blueprints/ai_bp.py (9 routes)

# ─────────────────────────────────────────────────────────
# Migration Routes (legacy scenes/chases to looks/sequences)
# ─────────────────────────────────────────────────────────
# [F02] Migration routes moved to blueprints/migrate_bp.py (3 routes)

# [F02] Effects routes moved to blueprints/effects_bp.py (14 routes)

# ─────────────────────────────────────────────────────────
# [F02] Fixtures routes moved to blueprints/fixtures_bp.py (8 routes)
# [F02] Fixture Library routes moved to blueprints/fixture_library_bp.py (11 routes)

# [F02] Groups routes moved to blueprints/groups_bp.py (5 routes)

# ─────────────────────────────────────────────────────────
# [F02] Shows routes moved to blueprints/shows_bp.py (10 routes)

# [F02] Schedules + Timers routes moved to blueprints/schedules_bp.py (14 routes)

# Legacy Playback Manager Routes (for backward compatibility)
# ─────────────────────────────────────────────────────────
@app.route('/api/playback-manager/status', methods=['GET'])
def playback_manager_status():
    """Legacy: Get playback manager status (use /api/playback/status for unified controller)"""
    return jsonify(playback_manager.get_status())

@app.route('/api/playback-manager/stop', methods=['POST'])
def stop_playback_manager():
    """Stop all playback via unified SSOT function.

    SSOT: Redirects to unified stop_all_playback for consistent behavior.
    Use /api/playback/stop for full control with blackout option.
    """
    data = request.get_json() or {}
    universe = data.get('universe')
    # Use unified SSOT stop function
    return jsonify(stop_all_playback(blackout=False, universe=universe))

# [F02] Cloud sync routes moved to blueprints/cloud_bp.py (8 routes)


# ─────────────────────────────────────────────────────────
# Settings Routes
# ─────────────────────────────────────────────────────────
# [F02] Settings & Screen Context routes moved to blueprints/settings_bp.py (7 routes)

# [F02] AI SSOT stubs + optimize-playback routes moved to blueprints/ai_bp.py (7 routes)

# ============================================================
# WebSocket Events
# ============================================================
@socketio.on('connect')
def handle_connect():
    print(f"🔌 WebSocket client connected")
    emit('nodes_update', {'nodes': node_manager.get_all_nodes()})
    emit('playback_update', {'playback': playback_manager.get_status()})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 WebSocket client disconnected")

@socketio.on('subscribe_dmx')
def handle_subscribe_dmx(data):
    universe = data.get('universe', 1)
    emit('dmx_state', {'universe': universe, 'channels': dmx_state.get_universe(universe)})

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print(f"  AETHER Core v{AETHER_VERSION} - Local Playback Engine")
    print("  Features: Scene/Chase sync, Universe splitting")
    print("="*60)
    print(f"  SSOT VERIFICATION:")
    print(f"    File:   {AETHER_FILE_PATH}")
    print(f"    Commit: {AETHER_COMMIT}")
    print(f"    CWD:    {os.getcwd()}")
    print("="*60 + "\n")

    # SSOT guardrail check - warn on startup if any bypasses detected
    ssot_startup_verify()

    init_database()
    app.teardown_appcontext(close_db)  # [F04] Clean up thread-local DB connections after each request
    pass  # ai_ssot removed

    # Supabase cloud sync (async, non-blocking)
    if SUPABASE_AVAILABLE:
        supabase = get_supabase_service()
        if supabase and supabase.is_enabled():
            def startup_cloud_sync():
                """Background sync to Supabase on startup"""
                try:
                    # Gather local data for sync
                    conn = get_db()
                    c = conn.cursor()

                    # Get nodes
                    c.execute('SELECT * FROM nodes')
                    nodes = [dict(row) for row in c.fetchall()]

                    # Get scenes
                    c.execute('SELECT * FROM scenes')
                    scenes = [dict(row) for row in c.fetchall()]

                    # Get chases
                    c.execute('SELECT * FROM chases')
                    chases = [dict(row) for row in c.fetchall()]

                    # Get fixtures
                    c.execute('SELECT * FROM fixtures')
                    fixtures = [dict(row) for row in c.fetchall()]

                    # Get shows
                    c.execute('SELECT * FROM shows')
                    shows = [dict(row) for row in c.fetchall()]

                    # Get schedules
                    c.execute('SELECT * FROM schedules')
                    schedules = [dict(row) for row in c.fetchall()]

                    # Get groups
                    c.execute('SELECT * FROM groups')
                    groups = [dict(row) for row in c.fetchall()]

                    conn.close()

                    # Get looks and sequences from LooksSequencesManager
                    looks_list = []
                    sequences_list = []
                    try:
                        looks_list = [l.to_dict() for l in looks_sequences_manager.list_looks()]
                        sequences_list = [s.to_dict() for s in looks_sequences_manager.list_sequences()]
                    except Exception as e:
                        print(f"⚠️ Failed to get looks/sequences for sync: {e}")

                    # Perform initial sync (one-way push to Supabase)
                    result = supabase.initial_sync(
                        nodes=nodes,
                        looks=looks_list,
                        sequences=sequences_list,
                        scenes=scenes,
                        chases=chases,
                        fixtures=fixtures,
                        shows=shows,
                        schedules=schedules,
                        groups=groups
                    )
                    print(f"☁️ Startup sync result: {result}")
                except Exception as e:
                    print(f"⚠️ Startup cloud sync failed (non-fatal): {e}")

            # Run sync in cloud pool (doesn't block startup)
            cloud_submit(startup_cloud_sync)
            print(f"☁️ Supabase cloud sync enabled - syncing in background...")
        else:
            print("☁️ Supabase not configured - running in local-only mode")

    threading.Thread(target=discovery_listener, daemon=True).start()
    threading.Thread(target=stale_checker, daemon=True).start()
    schedule_runner.start()
    node_manager.start_dmx_refresh()  # 40Hz refresh loop sends DMX to nodes via UDPJSON

    # Initialize Unified Playback Engine
    # Cache for fixture expansion (refreshed every 5 seconds)
    _fx_cache = {'fixtures': {}, 'time': 0.0}

    def unified_output_callback(universe: int, channels: dict, fade_ms: int = 0):
        """Route unified playback output directly to dmx_state (SSOT).

        The 40Hz refresh loop handles actual UDP output to nodes.
        This avoids double-sending and blocking I/O in the render path.

        Handles fixture-aware expansion: if channels represent a single-fixture
        pattern (e.g., channels 1-4), expands to all fixtures in the universe.
        Handles both simple channel keys (1, 2, 3) and universe:channel format (4:1, 4:2).
        """
        if not channels:
            return

        # Parse channel keys — handle both "1" and "4:1" formats
        parsed_channels = {}
        for key, value in channels.items():
            key_str = str(key)
            if ':' in key_str:
                parts = key_str.split(':')
                if int(parts[0]) == universe:
                    parsed_channels[int(parts[1])] = value
            else:
                try:
                    parsed_channels[int(key_str)] = value
                except ValueError:
                    pass

        if not parsed_channels:
            return

        # Refresh fixture cache every 5 seconds (not per frame)
        now = time.monotonic()
        if now - _fx_cache['time'] > 5.0:
            try:
                new_cache = {}
                for f in content_manager.get_fixtures():
                    u = f.get('universe', 1)
                    if u not in new_cache:
                        new_cache[u] = []
                    new_cache[u].append(f)
                _fx_cache['fixtures'] = new_cache
                _fx_cache['time'] = now
            except Exception:
                pass

        # Expand single-fixture patterns to all fixtures (safety net)
        # Chase steps are pre-replicated in play_chase(), but this catches
        # any other sources that output single-fixture data.
        fixtures = _fx_cache['fixtures'].get(universe, [])
        if fixtures and len(fixtures) > 1:
            sorted_fixtures = sorted(fixtures, key=lambda f: f.get('start_channel', 1))
            first_start = sorted_fixtures[0].get('start_channel', 1)
            fixture_ch_count = sorted_fixtures[0].get('channel_count', 4)

            # Check how many distinct fixtures have non-zero channels
            nonzero_chs = [ch for ch, v in parsed_channels.items() if v > 0]
            if nonzero_chs:
                fixtures_hit = set()
                for ch in nonzero_chs:
                    fixture_idx = (ch - first_start) // fixture_ch_count
                    fixtures_hit.add(fixture_idx)

                # Only expand if all non-zero data is in ONE fixture (not already expanded)
                if len(fixtures_hit) == 1:
                    # Use modulo to extract base pattern from whichever fixture has data
                    base_pattern = {}
                    for ch, val in parsed_channels.items():
                        offset = (ch - first_start) % fixture_ch_count
                        if val > 0 or offset not in base_pattern:
                            base_pattern[offset] = val

                    expanded = {}
                    for fix in sorted_fixtures:
                        start = fix.get('start_channel', 1)
                        count = fix.get('channel_count', fixture_ch_count)
                        for offset, value in base_pattern.items():
                            if offset < count:
                                expanded[start + offset] = value

                    if expanded:
                        parsed_channels = expanded

        # Write directly to dmx_state — the refresh loop handles UDP output
        dmx_state.set_channels(universe, parsed_channels, fade_ms=fade_ms)

    def unified_look_resolver(look_id: str):
        """Resolve Look ID to Look data"""
        try:
            look = looks_sequences_manager.get_look(look_id)
            return look.to_dict() if look else None
        except Exception as e:
            print(f"⚠️ Look resolver error for {look_id}: {e}")
            return None

    def unified_fixture_resolver(fixture_ids=None):
        """
        Resolve fixture IDs to fixture data for fixture-aware effects.

        Args:
            fixture_ids: List of fixture IDs to resolve, or None for all fixtures

        Returns:
            List of fixture dicts with: fixture_id, universe, start_channel,
            channel_count, channel_map
        """
        try:
            all_fixtures = content_manager.get_fixtures()
            if fixture_ids is None or len(fixture_ids) == 0:
                return all_fixtures
            return [f for f in all_fixtures if f.get('fixture_id') in fixture_ids]
        except Exception as e:
            print(f"⚠️ Fixture resolver error: {e}")
            return []

    unified_engine.set_output_callback(unified_output_callback)
    unified_engine.set_look_resolver(unified_look_resolver)
    unified_engine.set_fixture_resolver(unified_fixture_resolver)
    unified_engine.set_modifier_renderer(ModifierRenderer())

    # DMX state reader - allows modifiers to read current live DMX values
    # when no base session exists (e.g., look played via render_engine or direct set)
    def unified_dmx_state_reader(universe: int) -> dict:
        """Read current DMX state for a universe from dmx_state (the SSOT)"""
        try:
            with dmx_state.lock:
                if universe in dmx_state.universes:
                    raw = dmx_state.universes[universe]
                    # Return only non-zero channels as {channel: value} (1-indexed)
                    return {ch + 1: val for ch, val in enumerate(raw) if val > 0}
            return {}
        except Exception:
            return {}

    unified_engine.set_dmx_state_reader(unified_dmx_state_reader)
    unified_engine.start()
    print("✓ Unified Playback Engine started (30 fps)")

    # Initialize Operator Trust Enforcement (Phase 4 Lane 3)
    # TRUST RULES:
    # 1. Network loss -> Nodes HOLD last DMX value (firmware behavior)
    # 2. Backend crash -> Nodes CONTINUE output (firmware behavior)
    # 3. UI desync -> REALITY wins over UI (backend is authoritative)
    # 4. Partial node failure -> SYSTEM HALTS playback + ALERTS
    def halt_playback_for_trust():
        """Called by trust enforcer when partial node failure detected."""
        print("🚨 TRUST: Halting playback due to node failure", flush=True)
        unified_engine.stop_all()

    def get_playback_status_for_trust():
        """Get playback status for trust enforcement checks."""
        return unified_engine.get_status()

    def get_dmx_state_for_trust():
        """Get current DMX state (reality) for UI sync checks."""
        return content_manager.get_all_universes()

    trust_enforcer.set_halt_playback_callback(halt_playback_for_trust)
    trust_enforcer.set_get_playback_status_callback(get_playback_status_for_trust)
    trust_enforcer.set_get_dmx_state_callback(get_dmx_state_for_trust)
    report_backend_start()
    start_trust_monitoring()
    print("✓ Operator Trust Enforcer started")

    # Initialize consolidated RDM Manager
    rdm_manager.set_node_manager(node_manager)
    rdm_manager.set_socketio(socketio)
    rdm_manager.set_playback_engine(unified_engine)
    print("✓ RDM Manager initialized (consolidated)")

    print(f"✓ API server on port {API_PORT}")
    print(f"✓ Discovery on UDP {DISCOVERY_PORT}")
    print(f"✓ UDPJSON DMX output enabled (40 fps refresh, port {AETHER_UDPJSON_PORT})")
    print(f"⚠️ Universe 1 is OFFLINE - use universes 2-5")

    # [F01] Systemd watchdog integration — auto-restart if process hangs
    _watchdog_usec = os.environ.get('WATCHDOG_USEC')
    if _watchdog_usec:
        _wd_interval = int(_watchdog_usec) / 1_000_000 / 2  # Ping at half the timeout
        def _watchdog_loop():
            """[F01] Background thread pings systemd watchdog to prove liveness."""
            import socket as _wdsock
            notify_addr = os.environ.get('NOTIFY_SOCKET')
            if not notify_addr:
                return
            sock = _wdsock.socket(_wdsock.AF_UNIX, _wdsock.SOCK_DGRAM)
            if notify_addr.startswith('@'):
                notify_addr = '\0' + notify_addr[1:]
            try:
                while True:
                    try:
                        sock.sendto(b'WATCHDOG=1', notify_addr)
                    except Exception:
                        pass
                    time.sleep(_wd_interval)
            finally:
                sock.close()

        # Notify systemd we're ready
        _notify_addr = os.environ.get('NOTIFY_SOCKET')
        if _notify_addr:
            _ns = __import__('socket').socket(__import__('socket').AF_UNIX, __import__('socket').SOCK_DGRAM)
            if _notify_addr.startswith('@'):
                _notify_addr = '\0' + _notify_addr[1:]
            try:
                _ns.sendto(b'READY=1', _notify_addr)
            except Exception:
                pass
            _ns.close()

        _wd_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _wd_thread.start()
        print(f"✓ [F01] Systemd watchdog active (ping every {_wd_interval:.0f}s)")
    else:
        print("ℹ️ Systemd watchdog not configured (standalone mode)")

    # [F01] Graceful shutdown on SIGTERM
    import signal
    def _graceful_shutdown(signum, frame):
        print("\n⏹️ SIGTERM received — graceful shutdown...", flush=True)
        try:
            dmx_state.save_state_now()  # [F09] Persist state before shutdown
            print("  ✓ State saved", flush=True)
        except Exception:
            pass
        try:
            node_manager.stop_dmx_refresh()
            print("  ✓ DMX refresh stopped", flush=True)
        except Exception:
            pass
        try:
            _cloud_pool.shutdown(wait=False)
            _node_pool.shutdown(wait=False)
            print("  ✓ Thread pools shut down", flush=True)
        except Exception:
            pass
        try:
            close_db()
            print("  ✓ Database closed", flush=True)
        except Exception:
            pass
        print("✓ Shutdown complete", flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _graceful_shutdown)

    print("="*60 + "\n")

    # [F01] Production mode: allow_unsafe_werkzeug=True is needed for
    # Flask-SocketIO threading mode. Werkzeug 3.x is production-capable.
    # Real protection comes from systemd watchdog + auto-restart.
    socketio.run(app, host='0.0.0.0', port=API_PORT, debug=False, allow_unsafe_werkzeug=True)
