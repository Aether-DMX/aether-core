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
from datetime import datetime, timedelta
from typing import Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import ai_ssot
import ai_ops_registry
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
from playback_controller import (
    playback_controller, UnifiedPlaybackController,
    LoopMode, PlaybackState,
)
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

    FADE HANDLING:
    - All fades are handled internally via target/current state interpolation
    - The refresh loop reads get_output_values() which returns interpolated values
    - Output goes via UDPJSON to ESP32 nodes
    """
    def __init__(self):
        self.universes = {}  # {universe_num: [512 current values]}
        self.targets = {}    # {universe_num: [512 target values]}
        self.fade_info = {}  # {universe_num: {'start_time': float, 'duration': float, 'start_values': [512]}}
        self.master_level = 100  # 0-100 percent
        self.master_base = {}  # Captured state at 100%
        self.lock = threading.Lock()
        self._save_timer = None
        self._load_state()

    def _load_state(self):
        """On startup: channels start at 0, but we remember what was playing"""
        # Don't restore channel values - start fresh
        # But save active playback info for resume prompt
        try:
            if os.path.exists(DMX_STATE_FILE):
                with open(DMX_STATE_FILE, 'r') as f:
                    saved = json.load(f)
                    # Store what was playing for resume prompt (but don't apply)
                    self.last_session = saved.get('active_playback', None)
                    if self.last_session:
                        print(f"💾 Previous session had active playback: {self.last_session}")
                    else:
                        print("✓ DMX starting fresh (no previous playback)")
        except Exception as e:
            print(f"⚠️ Could not check previous session: {e}")
            self.last_session = None

    def _save_state(self):
        """Save DMX state and active playback info to disk"""
        try:
            with self.lock:
                # Get active playback from playback_manager
                active_playback = None
                try:
                    status = playback_manager.get_status()
                    if status:
                        # Find any active scene or chase
                        for univ, info in status.items():
                            if info and info.get('type'):
                                active_playback = {
                                    'universe': univ,
                                    'type': info.get('type'),
                                    'id': info.get('id'),
                                    'name': info.get('name')
                                }
                                break
                except Exception:
                    pass  # Playback status not critical for state save
                
                data = {
                    'universes': {str(k): v for k, v in self.universes.items()},
                    'active_playback': active_playback,
                    'saved_at': datetime.now().isoformat()
                }
            with open(DMX_STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save DMX state: {e}")
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

        Since the refresh loop is disabled and ESPs handle fades locally via UDPJSON,
        we update SSOT to the TARGET values immediately. The ESP fade engine handles
        the actual interpolation on the hardware side.

        If fade_ms > 0:
          - SSOT updated to target values immediately (ESP fades locally)
          - fade_info stored for any components that need it

        If fade_ms == 0:
          - Immediate snap to new values
        """
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            if universe not in self.targets:
                self.targets[universe] = [0] * 512

            # Update both current and target values for all cases
            # ESP handles fades locally, so SSOT should reflect final state
            for ch_str, value in channels_dict.items():
                ch = int(ch_str)
                if 1 <= ch <= 512:
                    self.universes[universe][ch - 1] = int(value)
                    self.targets[universe][ch - 1] = int(value)

            if fade_ms > 0:
                # Store fade info for any components that need it
                self.fade_info[universe] = {
                    'start_time': time.time(),
                    'duration': fade_ms / 1000.0,
                    'start_values': list(self.universes[universe])  # Already updated to target
                }
            else:
                # Clear any fade in progress
                self.fade_info.pop(universe, None)

        socketio.emit('dmx_state', {
            'universe': universe,
            'channels': self.get_output_values(universe)
        })
        self._schedule_save()

    def get_output_values(self, universe):
        """Get current output values, including fade interpolation

        This is what the refresh loop should use to get the actual values to send.
        It handles fade interpolation automatically.
        """
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
                return [0] * 512

            fade = self.fade_info.get(universe)
            if fade:
                elapsed = time.time() - fade['start_time']
                progress = min(1.0, elapsed / fade['duration'])

                if progress >= 1.0:
                    # Fade complete - snap to target and clear fade
                    if universe in self.targets:
                        self.universes[universe] = list(self.targets[universe])
                    self.fade_info.pop(universe, None)
                    return list(self.universes[universe])
                else:
                    # Interpolate between start and target
                    start = fade['start_values']
                    target = self.targets.get(universe, self.universes[universe])
                    interpolated = [
                        int(start[i] + (target[i] - start[i]) * progress)
                        for i in range(512)
                    ]
                    # Update current state so websocket sees progress
                    self.universes[universe] = interpolated
                    return interpolated
            else:
                return list(self.universes[universe])

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
# Chase Playback Engine (streams steps via UDPJSON)
# ============================================================
class ChaseEngine:
    """Runs chases by streaming each step via UDPJSON to all universes"""
    def __init__(self):
        self.lock = threading.Lock()
        self.running_chases = {}  # {chase_id: thread}
        self.stop_flags = {}  # {chase_id: Event}
        # Health tracking for debugging
        self.chase_health = {}  # {chase_id: {"step": int, "last_time": float, "status": str}}

    def start_chase(self, chase, universes, fade_ms_override=None):
        """Start a chase on the given universes with optional fade override"""
        chase_id = chase['chase_id']

        # ARBITRATION: Acquire chase ownership
        if not arbitration.acquire('chase', chase_id):
            print(f"⚠️ Cannot start chase - arbitration denied (owner: {arbitration.current_owner})", flush=True)
            return False

        # Stop any other running chases first
        self.stop_chase(chase_id)

        # Create stop flag
        stop_flag = threading.Event()
        self.stop_flags[chase_id] = stop_flag

        # Start chase thread with fade override
        thread = threading.Thread(
            target=self._run_chase,
            args=(chase, universes, stop_flag, fade_ms_override),
            daemon=True
        )
        self.running_chases[chase_id] = thread
        thread.start()
        print(f"🏃 Chase engine started: {chase['name']} (fade_override={fade_ms_override})", flush=True)
        return True

    def stop_chase(self, chase_id=None, wait=True):
        """Stop a chase or all chases, optionally waiting for thread to finish"""
        threads_to_join = []
        with self.lock:
            if chase_id:
                if chase_id in self.stop_flags:
                    self.stop_flags[chase_id].set()
                    self.stop_flags.pop(chase_id, None)
                    thread = self.running_chases.pop(chase_id, None)
                    if thread and wait:
                        threads_to_join.append(thread)
            else:
                # Stop all
                for flag in self.stop_flags.values():
                    flag.set()
                if wait:
                    threads_to_join = list(self.running_chases.values())
                self.stop_flags.clear()
                self.running_chases.clear()

            # ARBITRATION: Release chase ownership if no more chases running
            if not self.running_chases:
                arbitration.release('chase')

        # Wait for threads outside of lock to avoid deadlock
        if wait:
            for thread in threads_to_join:
                thread.join(timeout=0.5)  # Max 500ms wait per thread

    def stop_all(self):
        """Stop all running chases"""
        self.stop_chase(None)

    def _run_chase(self, chase, universes, stop_flag, fade_ms_override=None):
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
                def send_to_universe(univ):
                    try:
                        self._send_step(univ, channels, fade_ms, distribution_mode)
                    except Exception as e:
                        print(f"❌ Chase step send error (U{univ}): {e}", flush=True)

                threads = [threading.Thread(target=send_to_universe, args=(univ,)) for univ in universes]
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

    def _send_step(self, universe, channels, fade_ms=0, distribution_mode='unified'):
        """Send chase step with intelligent distribution.
        
        distribution_mode: 'unified' = replicate to all, 'pixel' = unique per fixture"""
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
    Priority: BLACKOUT(100) > MANUAL(80) > EFFECT(60) > CHASE(40) > SCENE(20) > IDLE(0)

    SSOT ENFORCEMENT: All DMX write attempts must check arbitration first.
    Rejected writes are tracked for diagnostics.
    """
    PRIORITY = {'blackout': 100, 'manual': 80, 'effect': 60, 'look': 50, 'sequence': 45, 'chase': 40, 'scene': 20, 'idle': 0}

    def __init__(self):
        self.current_owner = 'idle'
        self.current_id = None
        self.blackout_active = False
        self.last_change = None
        self.lock = threading.Lock()
        self.history = []
        # SSOT diagnostics tracking
        self.rejected_writes = []  # Track rejected acquire attempts
        self.last_writer = None  # Last service that successfully wrote
        self.last_scene_id = None  # Last scene played
        self.last_scene_time = None  # When last scene was played
        self.writes_per_service = {}  # Count writes per service type

    def acquire(self, owner_type, owner_id=None, force=False):
        with self.lock:
            now = datetime.now().isoformat()
            if self.blackout_active and owner_type != 'blackout':
                # Track rejected write
                self._track_rejection(owner_type, owner_id, 'blackout_active', now)
                return False
            new_pri = self.PRIORITY.get(owner_type, 0)
            cur_pri = self.PRIORITY.get(self.current_owner, 0)
            if force or new_pri >= cur_pri:
                old = self.current_owner
                self.current_owner = owner_type
                self.current_id = owner_id
                self.last_change = now
                self.last_writer = owner_type
                # Track scene plays specifically
                if owner_type == 'scene':
                    self.last_scene_id = owner_id
                    self.last_scene_time = now
                # Track writes per service
                self.writes_per_service[owner_type] = self.writes_per_service.get(owner_type, 0) + 1
                self.history.append({'time': now, 'from': old, 'to': owner_type, 'id': owner_id, 'action': 'acquire'})
                if len(self.history) > 50: self.history = self.history[-50:]
                print(f"🎯 Arbitration: {old} → {owner_type}", flush=True)
                return True
            # Track rejected write - lower priority
            self._track_rejection(owner_type, owner_id, f'priority_too_low (current: {self.current_owner})', now)
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
                self.last_change = datetime.now().isoformat()
                self.history.append({'time': self.last_change, 'from': old, 'to': 'idle', 'action': 'release'})
                if len(self.history) > 50: self.history = self.history[-50:]

    def set_blackout(self, active):
        with self.lock:
            self.blackout_active = active
            self.current_owner = 'blackout' if active else 'idle'
            self.last_change = datetime.now().isoformat()
            print(f"{'⬛ BLACKOUT ACTIVE' if active else '🔓 Blackout released'}", flush=True)

    def get_status(self):
        with self.lock:
            return {
                'current_owner': self.current_owner,
                'current_id': self.current_id,
                'blackout_active': self.blackout_active,
                'last_change': self.last_change,
                'last_writer': self.last_writer,
                'last_scene_id': self.last_scene_id,
                'last_scene_time': self.last_scene_time,
                'writes_per_service': dict(self.writes_per_service),
                'rejected_writes': self.rejected_writes[-5:],  # Last 5 rejections
                'history': self.history[-10:]
            }

    def can_write(self, owner_type):
        with self.lock:
            if self.blackout_active and owner_type != 'blackout': return False
            return self.current_owner == owner_type or self.current_owner == 'idle'

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
        """Parse cron string and check if it matches current time"""
        try:
            parts = cron_str.split()
            if len(parts) != 5:
                return False
            minute, hour, day, month, dow = parts
            now = datetime.now()
            
            def match(val, current, max_val):
                if val == '*':
                    return True
                if '/' in val:
                    _, step = val.split('/')
                    return current % int(step) == 0
                if '-' in val:
                    start, end = val.split('-')
                    return int(start) <= current <= int(end)
                if ',' in val:
                    return current in [int(x) for x in val.split(',')]
                return current == int(val)
            
            return (match(minute, now.minute, 59) and
                    match(hour, now.hour, 23) and
                    match(day, now.day, 31) and
                    match(month, now.month, 12) and
                    match(dow, now.weekday(), 6))
        except Exception as e:
            print(f"⚠️ Cron parse error: {e}")
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
        """Execute a schedule's action"""
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM schedules WHERE schedule_id = ?', (schedule_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {'success': False, 'error': 'Schedule not found'}
        
        action_type = row[3]
        action_id = row[4]
        
        # Update last_run
        c.execute('UPDATE schedules SET last_run = ? WHERE schedule_id = ?',
                  (datetime.now().isoformat(), schedule_id))
        conn.commit()
        conn.close()
        
        # Execute action
        try:
            if action_type == 'scene':
                result = content_manager.play_scene(action_id)
            elif action_type == 'chase':
                result = content_manager.play_chase(action_id)
            elif action_type == 'blackout':
                result = content_manager.blackout()
            else:
                result = {'success': False, 'error': f'Unknown action type: {action_type}'}
            print(f"📅 Schedule '{row[1]}' executed: {result.get('success', False)}")
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
class ShowEngine:
    """Plays back timeline-based shows with timed events"""
    
    def __init__(self):
        self.current_show = None
        self.running = False
        self.thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.paused = False
        self.tempo = 1.0
    
    def play_show(self, show_id, universe=1):
        """Play a show timeline"""
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

    def stop_silent(self):
        print(f"🛑 stop_silent called, running={self.running}", flush=True)
        """Stop without blackout (for SSOT transitions)"""
        if self.running:
            self.stop_flag.set()
            self.pause_flag.clear()
            self.running = False
            self.paused = False
            self.current_show = None

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
        """Execute timeline events in sequence, with optional looping"""
        sorted_events = sorted(timeline, key=lambda x: x.get('time_ms', 0))
        
        while self.running and not self.stop_flag.is_set():
            start_time = time.time() * 1000  # Convert to ms
            
            for event in sorted_events:
                if self.stop_flag.is_set():
                    break
                # Wait until event time
                event_time = event.get('time_ms', 0)
                elapsed = (time.time() * 1000) - start_time
                wait_time = (event_time - elapsed) / 1000  # Convert to seconds
                if wait_time > 0:
                    # Wait in small increments, checking stop/pause flags and applying tempo
                    while wait_time > 0 and not self.stop_flag.is_set():
                        # Check pause
                        while self.pause_flag.is_set() and not self.stop_flag.is_set():
                            time.sleep(0.1)
                        if self.stop_flag.is_set():
                            break
                        # Apply tempo to sleep time
                        sleep_chunk = min(wait_time, 0.1) / self.tempo
                        time.sleep(sleep_chunk)
                        wait_time -= 0.1
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
        """Execute a single timeline event"""
        event_type = event.get('type', 'scene')
        
        try:
            if event_type == 'scene':
                scene_id = event.get('scene_id')
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
                chase_id = event.get('chase_id')
                content_manager.play_chase(chase_id, universe=universe)
                print(f"  ▶️ Chase '{chase_id}' at {event.get('time_ms')}ms")
            
            elif event_type == 'blackout':
                fade_ms = event.get('fade_ms', 1000)
                content_manager.blackout(universe=universe, fade_ms=fade_ms)
                print(f"  ⬛ Blackout at {event.get('time_ms')}ms")
            
            elif event_type == 'channels':
                channels = event.get('channels', {})
                fade_ms = event.get('fade_ms', 0)
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
# ============================================================
def get_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_db()
    c = conn.cursor()

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

    # NOTE: Universe 1 built-in node removed - all nodes are WiFi ESP32 via UDPJSON

    print("✓ Database initialized")
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
        """Send a 'panic' command - immediate blackout with no fade."""
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "panic",
            "u": universe,
            "seq": self._next_seq()
        }
        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

    def send_udpjson_ping(self, node_ip):
        """Send a 'ping' command to a node and expect a 'pong' response."""
        payload = {
            "v": self.PROTOCOL_VERSION,
            "type": "ping",
            "seq": self._next_seq()
        }
        return self.send_udpjson(node_ip, AETHER_UDPJSON_PORT, payload)

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

    def _dmx_refresh_loop(self):
        """Background thread that sends DMX data to nodes via UDPJSON.

        All output uses the new UDPJSON protocol on port 6455.
        """
        frame_interval = 1.0 / self._refresh_rate
        frame_count = 0
        print(f"🔄 DMX Refresh loop starting (interval={frame_interval:.3f}s) - UDPJSON on port {AETHER_UDPJSON_PORT}")

        # Cache node IPs by universe
        universe_to_nodes = {}  # {universe: [(node_ip, slice_start, slice_end), ...]}

        while self._refresh_running:
            try:
                loop_start = time.time()

                # Get active universes from dmx_state
                active_universes = set(dmx_state.universes.keys())

                # Build universe->nodes mapping from database (refresh periodically)
                if frame_count % (self._refresh_rate * 2) == 0:  # Every 2 seconds
                    try:
                        conn = get_db()
                        c = conn.cursor()
                        # Only get WiFi nodes with IPs (not built-in universe 1)
                        # Include via_seance and seance_ip for Seance bridge routing
                        c.execute("""
                            SELECT universe, ip, channel_start, channel_end, via_seance, seance_ip
                            FROM nodes
                            WHERE is_paired = 1 AND ip IS NOT NULL AND type = 'wifi'
                        """)
                        universe_to_nodes = {}
                        for row in c.fetchall():
                            u, ip, ch_start, ch_end, via_seance, seance_ip = row
                            # Skip universe 1 (offline)
                            if u == 1:
                                continue
                            if u not in universe_to_nodes:
                                universe_to_nodes[u] = []
                            # Store target_ip: use seance_ip if via_seance is set, otherwise direct ip
                            target_ip = seance_ip if via_seance and seance_ip else ip
                            universe_to_nodes[u].append((target_ip, ch_start or 1, ch_end or 512, ip, via_seance))
                            active_universes.add(u)
                        conn.close()
                    except Exception as e:
                        print(f"⚠️ Node query error: {e}")

                # Log active universes periodically (every 5 seconds)
                frame_count += 1
                if frame_count == 1 or frame_count % (self._refresh_rate * 5) == 0:
                    print(f"🔄 DMX Refresh: universes={sorted(active_universes)}, udpjson sends={self._udpjson_send_count}")

                # Send DMX data for each active universe (skip universe 1)
                for universe in active_universes:
                    if not self._refresh_running:
                        break
                    if universe == 1:
                        continue  # Universe 1 is offline - skip

                    # Get output values (handles fade interpolation internally)
                    dmx_values = dmx_state.get_output_values(universe)

                    # Build channels dict for UDPJSON (only non-zero for efficiency)
                    channels_dict = {}
                    for i, val in enumerate(dmx_values):
                        if val > 0:
                            channels_dict[str(i + 1)] = val

                    # Send to each node in this universe via UDPJSON
                    nodes = universe_to_nodes.get(universe, [])
                    for target_ip, slice_start, slice_end, original_ip, via_seance in nodes:
                        # Filter channels for this node's slice
                        node_channels = {}
                        for ch_str, val in channels_dict.items():
                            ch = int(ch_str)
                            if slice_start <= ch <= slice_end:
                                node_channels[ch_str] = val

                        # Send UDPJSON set command
                        if node_channels or frame_count % self._refresh_rate == 0:
                            # Send either when there's data, or once per second for keepalive
                            # Use target_ip which routes through Seance if via_seance is set
                            self.send_udpjson_set(target_ip, universe, node_channels, source="refresh")

                # Maintain consistent frame rate for fade interpolation
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                # Log errors but don't crash the loop
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
                threading.Thread(
                    target=lambda: supabase.sync_node(node),
                    daemon=True
                ).start()

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
                threading.Thread(
                    target=lambda: supabase.sync_node(node),
                    daemon=True
                ).start()

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
        c.execute('UPDATE nodes SET status = "offline" WHERE last_seen < ? AND status = "online" AND is_builtin = 0', (cutoff,))
        if c.rowcount > 0:
            conn.commit()
            self.broadcast_status()
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
        - Fade interpolation via dmx_state.get_output_values()
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
class RDMManager:
    """RDM (Remote Device Management) for fixture discovery and configuration.

    Sends RDM commands to ESP32 nodes via UDPJSON and processes responses.
    Commands go to nodes, which forward to fixtures via RS485/DMX.
    """

    def __init__(self):
        self.discovery_tasks = {}  # node_id -> discovery status
        self.response_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.response_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_socket.settimeout(10.0)  # 10 second timeout for RDM responses
        self.pending_requests = {}  # request_id -> callback
        print("✓ RDMManager initialized")

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

        # Save discovered devices to database
        if result.get('success') and result.get('devices'):
            universe = node.get('universe', 1)
            self._save_devices(node_id, universe, result['devices'])

            # Fetch detailed info for each device
            for device in result['devices']:
                uid = device if isinstance(device, str) else device.get('uid')
                if uid:
                    try:
                        info = self._send_rdm_command(node['ip'], 'get_info', {"uid": uid})
                        if info.get('success'):
                            self._update_device_info(uid, info)
                            print(f"  📋 Got info for {uid}: Ch{info.get('dmx_address', '?')}, {info.get('footprint', '?')}ch")
                    except Exception as e:
                        print(f"  ⚠️ Failed to get info for {uid}: {e}")

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
            (info.get('dmx_address', 0), info.get('footprint', 0),
             info.get('personality_current', 0), info.get('personality_count', 0),
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
        """Remove a device from the database."""
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM rdm_devices WHERE uid = ?', (uid,))
        c.execute('DELETE FROM rdm_personalities WHERE device_uid = ?', (uid,))
        conn.commit()
        return {"success": True}

rdm_manager = RDMManager()

# ============================================================
# Content Manager
# ============================================================
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
        print(f"🎛️ set_channels: U{universe}, {len(channels)} ch update, fade={fade_ms}ms", flush=True)

        # Update SSOT with the channel changes
        dmx_state.set_channels(universe, channels, fade_ms=fade_ms)

        # Build full 512-channel frame from SSOT
        full_frame = dmx_state.get_output_values(universe)

        # Count non-zero for logging
        non_zero_count = sum(1 for v in full_frame if v > 0)
        print(f"📤 SSOT U{universe}: 512 ch frame ({non_zero_count} non-zero)", flush=True)

        nodes = node_manager.get_nodes_in_universe(universe)
        print(f"📍 Found {len(nodes)} nodes in universe {universe}", flush=True)

        if not nodes:
            print(f"⚠️ No online nodes in universe {universe}", flush=True)
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
            print(f"  📡 {node['name']} ({node_ip}): ch {slice_start}-{slice_end} ({node_non_zero} non-zero, {len(node_channels)} sent)", flush=True)

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
            fade_ms, curve, color, icon, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
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
                threading.Thread(
                    target=lambda: supabase.sync_scene(scene),
                    daemon=True
                ).start()

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
        """Play a scene - broadcasts to specified or all online nodes

        Args:
            scene_id: ID of the scene to play
            fade_ms: Fade time override
            use_local: Use local playback
            target_channels: Optional list of specific channels
            universe: Single universe (legacy, use universes instead)
            universes: List of universes to target (preferred)
            skip_ssot: Skip SSOT lock (internal use)
            replicate: Replicate scene across fixtures

        SSOT COMPLIANCE: All DMX writes go through set_channels which updates dmx_state.
        """
        print(f"▶️ play_scene called: scene_id={scene_id}", flush=True)
        scene = self.get_scene(scene_id)
        if not scene:
            return {'success': False, 'error': 'Scene not found'}

        # ARBITRATION: Acquire scene ownership
        if not arbitration.acquire('scene', scene_id):
            print(f"⚠️ Cannot play scene - arbitration denied (owner: {arbitration.current_owner})", flush=True)
            return {'success': False, 'error': f'Arbitration denied: {arbitration.current_owner} has control'}

        # Capture state before SSOT
        playback_before = dict(self.current_playback)
        all_nodes = node_manager.get_all_nodes(include_offline=False)
        all_universes = set(node.get('universe', 1) for node in all_nodes)

        # SSOT: Acquire lock and stop everything cleanly
        ssot_acquired = False
        if not skip_ssot:
            with self.ssot_lock:
                ssot_acquired = True
                print(f"🔒 SSOT Lock - stopping all before scene", flush=True)
                try:
                    show_engine.stop_silent() if hasattr(show_engine, "stop_silent") else show_engine.stop()
                except Exception as e:
                    print(f"⚠️ Show stop error: {e}", flush=True)
                chase_engine.stop_all()  # Now waits for chase threads to finish
                effects_engine.stop_effect()  # Also stop effects
                self.current_playback = {'type': 'scene', 'id': scene_id, 'universe': universe}
                print(f"✓ SSOT: Now playing scene '{scene_id}'", flush=True)

        fade = fade_ms if fade_ms is not None else scene.get('fade_ms', 500)
        channels_to_apply = scene['channels']

        # Replicate scene pattern across all fixtures (unless targeting specific channels)
        if replicate and not target_channels:
            channels_to_apply = self.replicate_scene_to_fixtures(channels_to_apply)

        if target_channels:
            target_set = set(target_channels)
            channels_to_apply = {k: v for k, v in channels_to_apply.items() if int(k) in target_set}

        # Get target universes - priority: universes array > single universe > all online paired nodes
        if universes is not None and len(universes) > 0:
            universes_with_nodes = set(universes)
        elif universe is not None:
            universes_with_nodes = {universe}
            if universe not in all_universes:
                print(f"⚠️ Universe {universe} requested but no online nodes detected - will still update SSOT", flush=True)
        else:
            # Default: all online PAIRED universes only
            all_nodes = node_manager.get_all_nodes(include_offline=False)
            universes_with_nodes = set(node.get('universe', 1) for node in all_nodes if node.get('is_paired'))
            if not universes_with_nodes:
                universes_with_nodes = {1}

        # Early warning if no universes available
        if not universes_with_nodes:
            print(f"⚠️ No universes available for scene play - defaulting to universe 1", flush=True)
            universes_with_nodes = {1}

        beta_log("play_scene", {
            "requested_universe": universe,
            "selected_universes_at_action_time": sorted(all_universes),
            "expanded_target_universes": sorted(universes_with_nodes),
            "dispatch_targets_final": sorted(universes_with_nodes),
            "playback_state_before": playback_before,
            "playback_state_after": dict(self.current_playback),
            "ssot_lock_acquired": ssot_acquired,
            "scene_id": scene_id
        })

        print(f"🎬 Playing scene '{scene['name']}' on universes: {sorted(universes_with_nodes)}", flush=True)

        # Debug: Show what channels we're about to apply
        non_zero = {k: v for k, v in channels_to_apply.items() if v > 0}
        print(f"  📋 Channels to apply: {len(channels_to_apply)} total, {len(non_zero)} non-zero", flush=True)
        if len(non_zero) <= 10:
            print(f"  📋 Non-zero values: {non_zero}", flush=True)
        else:
            sample = dict(list(non_zero.items())[:5])
            print(f"  📋 Sample non-zero values: {sample}...", flush=True)

        all_results = []

        # Clear ALL previous playback states before setting new one
        # Only one scene can be "playing" at a time globally
        if target_channels is None:
            playback_manager.stop()  # Clear all universes

        for univ in universes_with_nodes:
            if target_channels is None:
                current = playback_manager.get_status(univ)
                if current and current.get('type') == 'chase':
                    print(f"⏹️ Stopping chase on U{univ} before scene play", flush=True)
                    node_manager.stop_playback_on_nodes(univ)
                playback_manager.set_playing(univ, 'scene', scene_id)
            # set_channels already updates dmx_state - no need for duplicate call
            result = self.set_channels(univ, channels_to_apply, fade)
            all_results.extend(result.get('results', []))

        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE scenes SET play_count = play_count + 1 WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()

        return {'success': True, 'results': all_results, 'universes': list(universes_with_nodes)}

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
                    threading.Thread(
                        target=lambda: supabase.sync_chase(chase),
                        daemon=True
                    ).start()
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
        """Start chase playback - streams steps via UDPJSON to specified or all online nodes

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
        # (Allow multiple chases to run simultaneously on different universes)
        with self.ssot_lock:
            print(f"🔒 SSOT Lock - stopping playback on target universes: {universes_with_nodes}", flush=True)
            try:
                show_engine.stop()
            except Exception as e:
                print(f"⚠️ Show stop: {e}", flush=True)
            # Stop ALL running chases before starting new one
            chase_engine.stop_all()  # Stop all running chases
            effects_engine.stop_effect()  # Also stop effects
            for univ in universes_with_nodes:
                playback_manager.stop(univ)
            self.current_playback = {'type': 'chase', 'id': chase_id, 'universe': universe}
            print(f"✓ SSOT: Now playing chase '{chase_id}'", flush=True)


        # Set playback state for all universes
        for univ in universes_with_nodes:
            playback_manager.set_playing(univ, 'chase', chase_id)

        # Start chase engine with apply-time fade override
        chase_engine.start_chase(chase, universes_with_nodes, fade_ms_override=effective_fade_ms)

        return {'success': True, 'universes': universes_with_nodes, 'fade_ms': effective_fade_ms}

    def stop_playback(self, universe=None):
        """Stop all playback"""
        chase_engine.stop_all()  # Stop chase engine
        playback_manager.stop(universe)
        node_results = node_manager.stop_playback_on_nodes(universe)
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
def ssot_send_frame(universe, channels_array, owner_type='effect'):
    """
    Unified output function for all DMX writes.
    Routes through ContentManager to reach ALL nodes via UDPJSON.

    This is the ONLY function that should send DMX output.
    All subsystems (Scenes, Chases, Effects) must go through here.
    """
    try:
        # Check arbitration - if we can't write, skip silently
        if not arbitration.can_write(owner_type):
            return

        # Convert array to dict format for ContentManager
        channels_dict = {}
        for i, value in enumerate(channels_array):
            if value > 0:  # Only include non-zero values
                channels_dict[str(i + 1)] = value

        if not channels_dict:
            return

        # Route through ContentManager - sends to nodes via UDPJSON
        content_manager.set_channels(universe, channels_dict, fade_ms=0)

    except Exception as e:
        print(f"❌ SSOT output error U{universe}: {e}")

# Hook up effects engine to SSOT with arbitration
effects_engine.set_ssot_hooks(dmx_state, ssot_send_frame, arbitration)

# Hook up render engine to SSOT for Look playback
def render_engine_output(universe: int, channels: dict):
    """Callback for render engine to send frames through SSOT"""
    # Check arbitration
    if not arbitration.can_write('look'):
        return
    # Convert dict to format expected by set_channels
    content_manager.set_channels(universe, {str(k): v for k, v in channels.items()}, fade_ms=0)

render_engine.set_output_callback(render_engine_output)

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

# Hook up unified playback controller to merge layer
def playback_output(universe: int, channels: dict):
    """Callback for playback controller - routes through merge layer"""
    # Get the current job from playback controller
    status = playback_controller.get_status()
    job_id = status.get('job_id')

    if not job_id:
        return

    source_id = get_active_source_id(job_id)
    if not source_id:
        # Auto-register if not registered (backward compatibility)
        source_type = status.get('job_type', 'look')
        universes = status.get('universes', [universe])
        source_id = register_playback_source(job_id, source_type, universes)

    # Update merge layer with new channel values
    merge_layer.set_source_channels(source_id, universe, channels)

    # Output merged result
    merged = merge_layer.compute_merge(universe)
    if merged:
        merge_layer_output(universe, merged)

playback_controller.set_output_callback(playback_output)
playback_controller.set_modifier_renderer(ModifierRenderer())

# Set look resolver for Sequence playback (to resolve Look references in steps)
def resolve_look(look_id: str):
    """Resolve a Look ID to Look data"""
    look = looks_sequences_manager.get_look(look_id)
    if look:
        return look.to_dict()
    return None

playback_controller.set_look_resolver(resolve_look)

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
                if msg_type == 'register':
                    print(f"📥 Node registered: {msg.get('hostname', 'Unknown')} @ {addr[0]}")
                    # Auto-sync content to newly registered node if paired
                    node = node_manager.get_node(msg.get('node_id'))
                    if node and node.get('is_paired'):
                        threading.Thread(target=node_manager.sync_content_to_node, args=(node,), daemon=True).start()
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
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 'version': AETHER_VERSION, 'timestamp': datetime.now().isoformat(),
        'services': {'database': True, 'discovery': True, 'udpjson': True}
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

@app.route('/api/session/resume', methods=['GET'])
def get_resume_session():
    """Check if there's a previous session to resume"""
    last_session = getattr(dmx_state, 'last_session', None)
    if last_session:
        return jsonify({'has_session': True, 'playback': last_session})
    return jsonify({'has_session': False, 'playback': None})

@app.route('/api/session/resume', methods=['POST'])
def resume_session():
    """Resume the previous session's playback"""
    last_session = getattr(dmx_state, 'last_session', None)
    if not last_session:
        return jsonify({'success': False, 'error': 'No session to resume'})
    
    playback_type = last_session.get('type')
    playback_id = last_session.get('id')
    universe = last_session.get('universe', 1)
    
    try:
        if playback_type == 'scene':
            result = content_manager.play_scene(playback_id, universe=universe)
        elif playback_type == 'chase':
            result = content_manager.play_chase(playback_id, universe=universe)
        else:
            return jsonify({'success': False, 'error': f'Unknown type: {playback_type}'})
        
        # Clear the last session after resuming
        dmx_state.last_session = None
        return jsonify({'success': True, 'resumed': last_session})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/session/dismiss', methods=['POST'])
def dismiss_session():
    """Dismiss the resume prompt without resuming"""
    dmx_state.last_session = None
    return jsonify({'success': True})


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """System information for frontend mode detection"""
    device_id = get_or_create_device_id()
    is_pi = os.path.exists('/sys/firmware/devicetree/base/model')
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

    # Determine mode from settings
    mode_setting = app_settings.get('system', {}).get('uiMode', 'auto')
    if mode_setting == 'auto':
        mode = 'kiosk' if (is_pi and has_display) else 'desktop'
    else:
        mode = mode_setting

    return jsonify({
        'deviceId': device_id,
        'hostname': platform.node(),
        'platform': 'pi5' if is_pi else platform.system().lower(),
        'version': AETHER_VERSION,
        'commit': AETHER_COMMIT,
        'mode': mode,
        'capabilities': {
            'maxUniverses': 16,
            'rdmSupported': True,
            'sacnSupported': True,
        }
    })


@app.route('/api/system/mode', methods=['POST'])
def set_system_mode():
    """Set UI mode preference"""
    global app_settings
    data = request.get_json() or {}
    mode = data.get('mode', 'auto')
    if mode not in ['auto', 'kiosk', 'desktop', 'mobile']:
        return jsonify({'error': 'Invalid mode'}), 400
    if 'system' not in app_settings:
        app_settings['system'] = {}
    app_settings['system']['uiMode'] = mode
    save_settings()
    return jsonify({'success': True, 'mode': mode})


@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    """Get system statistics (CPU, memory, temperature)"""
    stats = {
        'cpu_percent': None,
        'memory_used': None,
        'memory_total': None,
        'cpu_temp': None,
        'disk_used': None,
        'disk_total': None,
        'uptime': None
    }

    try:
        # CPU usage - read from /proc/stat
        with open('/proc/loadavg', 'r') as f:
            load = f.read().split()
            # Convert 1-min load average to approximate percentage (for 4 cores)
            stats['cpu_percent'] = float(load[0]) * 25  # Rough approximation
    except (OSError, ValueError, IndexError):
        pass  # Not available on this platform

    try:
        # Memory - read from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024  # Convert KB to bytes
            stats['memory_total'] = meminfo.get('MemTotal', 0)
            stats['memory_used'] = stats['memory_total'] - meminfo.get('MemAvailable', 0)
    except (OSError, ValueError, KeyError):
        pass  # Not available on this platform

    try:
        # CPU temperature - Raspberry Pi specific
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats['cpu_temp'] = int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass  # Not available on this platform

    try:
        # Disk usage
        statvfs = os.statvfs('/')
        stats['disk_total'] = statvfs.f_blocks * statvfs.f_frsize
        stats['disk_used'] = (statvfs.f_blocks - statvfs.f_bfree) * statvfs.f_frsize
    except (OSError, AttributeError):
        pass  # Not available on this platform

    try:
        # Uptime
        with open('/proc/uptime', 'r') as f:
            stats['uptime'] = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass  # Not available on this platform

    return jsonify(stats)

@app.route('/api/system/update', methods=['POST'])
def system_update():
    """Pull latest code from git and deploy to runtime location"""
    results = {'steps': [], 'success': False}

    # Determine paths based on environment
    git_dir = os.path.dirname(AETHER_FILE_PATH)
    # If running from /home/ramzt, the git repo is in ~/aether-core-git
    if '/aether-core-git/' not in AETHER_FILE_PATH and '/aether-core/' not in AETHER_FILE_PATH:
        git_dir = os.path.expanduser('~/aether-core-git/aether-core')

    runtime_file = '/home/ramzt/aether-core.py'

    try:
        # Step 1: Git fetch
        fetch_result = subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=git_dir
        )
        results['steps'].append({
            'step': 'git_fetch',
            'success': fetch_result.returncode == 0,
            'output': fetch_result.stdout + fetch_result.stderr
        })

        # Step 2: Check if update available
        status_result = subprocess.run(
            ['git', 'status', '-uno'],
            capture_output=True, text=True, timeout=10,
            cwd=git_dir
        )
        behind = 'behind' in status_result.stdout
        results['update_available'] = behind

        if not behind:
            results['message'] = 'Already up to date'
            results['success'] = True
            return jsonify(results)

        # Step 3: Git pull
        pull_result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            capture_output=True, text=True, timeout=60,
            cwd=git_dir
        )
        results['steps'].append({
            'step': 'git_pull',
            'success': pull_result.returncode == 0,
            'output': pull_result.stdout + pull_result.stderr
        })

        if pull_result.returncode != 0:
            results['message'] = 'Git pull failed'
            return jsonify(results), 500

        # Step 4: Get new commit
        commit_result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=git_dir
        )
        new_commit = commit_result.stdout.strip()
        results['new_commit'] = new_commit
        results['old_commit'] = AETHER_COMMIT

        # Step 5: Deploy to runtime location (critical for Pi setup)
        source_file = os.path.join(git_dir, 'aether-core.py')
        if os.path.exists(source_file) and runtime_file != source_file:
            # Copy the file
            import shutil
            shutil.copy2(source_file, runtime_file)

            # Embed commit hash (since runtime is outside git)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            sed_cmd = f'sed -i "s/AETHER_COMMIT = get_git_commit()/AETHER_COMMIT = \\"{new_commit}\\"  # Baked at deploy: {timestamp}/" {runtime_file}'
            subprocess.run(['bash', '-c', sed_cmd], capture_output=True)

            results['steps'].append({
                'step': 'deploy_to_runtime',
                'success': True,
                'output': f'Copied to {runtime_file} with embedded commit {new_commit}'
            })

        # Step 6: Schedule restart (non-blocking)
        results['message'] = 'Update deployed. Restarting service...'
        results['success'] = True

        # Restart service in background after response is sent
        def restart_service():
            time.sleep(1)  # Give time for response to be sent
            os.system('sudo systemctl restart aether-core')

        restart_thread = threading.Thread(target=restart_service, daemon=True)
        restart_thread.start()

        return jsonify(results)

    except Exception as e:
        results['error'] = str(e)
        return jsonify(results), 500

@app.route('/api/system/update/check', methods=['GET'])
def system_update_check():
    """Check if updates are available without applying them"""
    try:
        # Fetch latest
        subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(AETHER_FILE_PATH)
        )

        # Check status
        result = subprocess.run(
            ['git', 'rev-list', 'HEAD..origin/main', '--count'],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(AETHER_FILE_PATH)
        )

        commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

        # Get latest remote commit message
        log_result = subprocess.run(
            ['git', 'log', 'origin/main', '-1', '--format=%h %s'],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(AETHER_FILE_PATH)
        )

        return jsonify({
            'current_commit': AETHER_COMMIT,
            'update_available': commits_behind > 0,
            'commits_behind': commits_behind,
            'latest_commit': log_result.stdout.strip() if log_result.returncode == 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/autosync', methods=['GET'])
def get_autosync_status():
    """Get auto-sync status"""
    return jsonify({
        'enabled': getattr(app, '_autosync_enabled', False),
        'interval_minutes': getattr(app, '_autosync_interval', 30),
        'last_check': getattr(app, '_autosync_last_check', None),
        'last_update': getattr(app, '_autosync_last_update', None)
    })

@app.route('/api/system/autosync', methods=['POST'])
def set_autosync():
    """Enable/disable auto-sync"""
    data = request.get_json() or {}
    enabled = data.get('enabled', False)
    interval = data.get('interval_minutes', 30)

    app._autosync_enabled = enabled
    app._autosync_interval = max(5, min(1440, interval))  # 5 min to 24 hours

    if enabled:
        start_autosync_thread()
        print(f"✓ Auto-sync enabled: checking every {app._autosync_interval} minutes")
    else:
        print("✓ Auto-sync disabled")

    return jsonify({
        'success': True,
        'enabled': app._autosync_enabled,
        'interval_minutes': app._autosync_interval
    })

def start_autosync_thread():
    """Start background thread for auto-sync"""
    def autosync_worker():
        # Determine paths based on environment
        git_dir = os.path.dirname(AETHER_FILE_PATH)
        if '/aether-core-git/' not in AETHER_FILE_PATH and '/aether-core/' not in AETHER_FILE_PATH:
            git_dir = os.path.expanduser('~/aether-core-git/aether-core')
        runtime_file = '/home/ramzt/aether-core.py'

        while getattr(app, '_autosync_enabled', False):
            try:
                interval = getattr(app, '_autosync_interval', 30) * 60  # Convert to seconds
                time.sleep(interval)

                if not getattr(app, '_autosync_enabled', False):
                    break

                app._autosync_last_check = datetime.now().isoformat()

                # Check for updates
                subprocess.run(['git', 'fetch', 'origin'],
                    capture_output=True, timeout=30,
                    cwd=git_dir)

                result = subprocess.run(
                    ['git', 'rev-list', 'HEAD..origin/main', '--count'],
                    capture_output=True, text=True, timeout=10,
                    cwd=git_dir)

                commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

                if commits_behind > 0:
                    print(f"🔄 Auto-sync: {commits_behind} updates available, pulling...")

                    pull_result = subprocess.run(
                        ['git', 'pull', 'origin', 'main'],
                        capture_output=True, text=True, timeout=60,
                        cwd=git_dir)

                    if pull_result.returncode == 0:
                        # Deploy to runtime location
                        source_file = os.path.join(git_dir, 'aether-core.py')
                        if os.path.exists(source_file) and runtime_file != source_file:
                            import shutil
                            shutil.copy2(source_file, runtime_file)

                            # Embed commit hash
                            commit_result = subprocess.run(
                                ['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5,
                                cwd=git_dir)
                            new_commit = commit_result.stdout.strip()
                            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                            sed_cmd = f'sed -i "s/AETHER_COMMIT = get_git_commit()/AETHER_COMMIT = \\"{new_commit}\\"  # Baked at deploy: {timestamp}/" {runtime_file}'
                            subprocess.run(['bash', '-c', sed_cmd], capture_output=True)

                            print(f"✓ Auto-sync: deployed {new_commit} to {runtime_file}")

                        app._autosync_last_update = datetime.now().isoformat()
                        print("✓ Auto-sync: restarting service...")
                        time.sleep(1)
                        os.system('sudo systemctl restart aether-core')
                    else:
                        print(f"❌ Auto-sync pull failed: {pull_result.stderr}")

            except Exception as e:
                print(f"❌ Auto-sync error: {e}")

    # Only start if not already running
    if not getattr(app, '_autosync_thread', None) or not app._autosync_thread.is_alive():
        app._autosync_thread = threading.Thread(target=autosync_worker, daemon=True)
        app._autosync_thread.start()

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

@app.route('/api/nodes/<node_id>/sync', methods=['POST'])
def sync_node(node_id):
    """Force sync content to a specific node"""
    node = node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    threading.Thread(target=node_manager.sync_content_to_node, args=(node,), daemon=True).start()
    return jsonify({'success': True, 'message': 'Sync started'})

@app.route('/api/nodes/sync', methods=['POST'])
def sync_all_nodes():
    """Force sync content to all nodes"""
    threading.Thread(target=node_manager.sync_all_content, daemon=True).start()
    return jsonify({'success': True, 'message': 'Full sync started'})

# ─────────────────────────────────────────────────────────
# RDM Routes - Remote Device Management
# ─────────────────────────────────────────────────────────
@app.route('/api/nodes/<node_id>/rdm/discover', methods=['POST'])
def rdm_discover(node_id):
    """Start RDM discovery on a node - finds all RDM fixtures on DMX bus"""
    result = rdm_manager.discover_devices(node_id)
    return jsonify(result)

@app.route('/api/nodes/<node_id>/rdm/discover/status', methods=['GET'])
def rdm_discover_status(node_id):
    """Get RDM discovery status for a node"""
    status = rdm_manager.discovery_tasks.get(node_id, {"status": "idle"})
    return jsonify(status)

@app.route('/api/nodes/<node_id>/rdm/devices', methods=['GET'])
def rdm_devices_for_node(node_id):
    """List all RDM devices discovered on a node"""
    devices = rdm_manager.get_devices_for_node(node_id)
    return jsonify(devices)

@app.route('/api/rdm/devices', methods=['GET'])
def rdm_all_devices():
    """List all RDM devices across all nodes"""
    devices = rdm_manager.get_all_devices()
    return jsonify(devices)

@app.route('/api/rdm/devices/<uid>', methods=['GET'])
def rdm_device_detail(uid):
    """Get detailed info for a specific RDM device"""
    # Find which node has this device
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT node_id FROM rdm_devices WHERE uid = ?', (uid,))
    row = c.fetchone()
    if not row:
        return jsonify({'error': 'Device not found'}), 404

    result = rdm_manager.get_device_info(row[0], uid)
    return jsonify(result)

@app.route('/api/rdm/devices/<uid>/identify', methods=['POST'])
def rdm_identify(uid):
    """Identify a device (flash its LED)"""
    data = request.get_json() or {}
    state = data.get('state', True)

    # Find which node has this device
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT node_id FROM rdm_devices WHERE uid = ?', (uid,))
    row = c.fetchone()
    if not row:
        return jsonify({'error': 'Device not found'}), 404

    result = rdm_manager.identify_device(row[0], uid, state)
    return jsonify(result)

@app.route('/api/rdm/devices/<uid>/address', methods=['POST'])
def rdm_set_address(uid):
    """Set DMX start address for a device"""
    data = request.get_json() or {}
    address = data.get('address')
    if not address or address < 1 or address > 512:
        return jsonify({'error': 'Address must be between 1 and 512'}), 400

    # Find which node has this device
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT node_id FROM rdm_devices WHERE uid = ?', (uid,))
    row = c.fetchone()
    if not row:
        return jsonify({'error': 'Device not found'}), 404

    result = rdm_manager.set_dmx_address(row[0], uid, address)
    return jsonify(result)

@app.route('/api/rdm/devices/<uid>/label', methods=['POST'])
def rdm_set_label(uid):
    """Set device label (stored in database, not on fixture)"""
    data = request.get_json() or {}
    label = data.get('label', '')

    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE rdm_devices SET device_label = ? WHERE uid = ?', (label, uid))
    conn.commit()

    return jsonify({'success': True, 'uid': uid, 'label': label})

@app.route('/api/rdm/devices/<uid>', methods=['DELETE'])
def rdm_delete_device(uid):
    """Remove a stale RDM device from database"""
    result = rdm_manager.delete_device(uid)
    return jsonify(result)

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

# ─────────────────────────────────────────────────────────
# Pixel Array Routes - Multi-fixture pixel-style control
# ─────────────────────────────────────────────────────────

def _get_pixel_array_send_callback():
    """Create a callback that routes pixel array output through SSOT"""
    def callback(universe: int, channels: Dict[int, int]):
        # Route through ContentManager for proper SSOT handling
        content_manager.set_channels(universe, channels, fade_ms=0)
    return callback


@app.route('/api/pixel-arrays', methods=['GET'])
def get_pixel_arrays():
    """List all active pixel array controllers"""
    result = {}
    for array_id, controller in _pixel_arrays.items():
        result[array_id] = controller.get_status()
    return jsonify({
        'pixel_arrays': result,
        'count': len(_pixel_arrays),
    })


@app.route('/api/pixel-arrays', methods=['POST'])
def create_pixel_array():
    """
    Create a new pixel array controller.

    Request body:
    {
        "id": "my_array",           // Optional, auto-generated if not provided
        "universe": 4,              // Target DMX universe (default 4)
        "fixture_count": 8,         // Number of RGBW fixtures (default 8)
        "start_channel": 1,         // First fixture start channel (default 1)
        "channel_spacing": 4        // Channels between fixtures (default 4)
    }
    """
    data = request.get_json() or {}

    array_id = data.get('id', f"array_{int(time.time())}")

    # Check if already exists
    if array_id in _pixel_arrays:
        return jsonify({
            'success': False,
            'error': f'Pixel array {array_id} already exists'
        }), 400

    try:
        controller = create_pixel_controller(
            fixture_count=data.get('fixture_count', 8),
            universe=data.get('universe', 4),
            start_channel=data.get('start_channel', 1),
            channel_spacing=data.get('channel_spacing', 4),
            send_callback=_get_pixel_array_send_callback(),
        )
        _pixel_arrays[array_id] = controller

        return jsonify({
            'success': True,
            'id': array_id,
            'status': controller.get_status(),
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/pixel-arrays/<array_id>', methods=['GET'])
def get_pixel_array(array_id):
    """Get status of a specific pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404
    return jsonify(_pixel_arrays[array_id].get_status())


@app.route('/api/pixel-arrays/<array_id>', methods=['DELETE'])
def delete_pixel_array(array_id):
    """Delete a pixel array controller"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.stop()  # Stop render loop if running
    del _pixel_arrays[array_id]

    return jsonify({'success': True, 'id': array_id})


@app.route('/api/pixel-arrays/<array_id>/mode', methods=['POST'])
def set_pixel_array_mode(array_id):
    """
    Set operation mode for a pixel array.

    Request body:
    {
        "mode": "grouped" | "pixel_array"
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    mode_str = data.get('mode', 'grouped')

    controller = _pixel_arrays[array_id]

    if mode_str == 'grouped':
        controller.set_grouped_mode()
    elif mode_str == 'pixel_array':
        controller.set_pixel_array_mode()
    else:
        return jsonify({'error': f'Invalid mode: {mode_str}'}), 400

    return jsonify({
        'success': True,
        'mode': controller.mode.value,
    })


@app.route('/api/pixel-arrays/<array_id>/pixels', methods=['POST'])
def set_pixel_array_pixels(array_id):
    """
    Set pixel values in the array.

    Request body (grouped mode - sets all pixels):
    {
        "r": 255, "g": 0, "b": 0, "w": 0
    }

    Request body (pixel array mode - set individual pixels):
    {
        "pixels": [
            {"index": 0, "r": 255, "g": 0, "b": 0, "w": 0},
            {"index": 1, "r": 0, "g": 255, "b": 0, "w": 0},
            ...
        ]
    }

    Or set all pixels to same color in pixel array mode:
    {
        "all": {"r": 255, "g": 0, "b": 0, "w": 0}
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    controller = _pixel_arrays[array_id]

    # Set all pixels to same color
    if 'all' in data:
        color = data['all']
        controller.set_all_rgbw(
            color.get('r', 0),
            color.get('g', 0),
            color.get('b', 0),
            color.get('w', 0),
        )
    # Set all pixels (grouped mode shorthand)
    elif 'r' in data or 'g' in data or 'b' in data or 'w' in data:
        controller.set_all_rgbw(
            data.get('r', 0),
            data.get('g', 0),
            data.get('b', 0),
            data.get('w', 0),
        )
    # Set individual pixels
    elif 'pixels' in data:
        for pixel_data in data['pixels']:
            index = pixel_data.get('index', 0)
            controller.set_pixel_rgbw(
                index,
                pixel_data.get('r', 0),
                pixel_data.get('g', 0),
                pixel_data.get('b', 0),
                pixel_data.get('w', 0),
            )

    # Send frame immediately
    controller.populate_dmx_buffer()
    controller._send_frame()

    return jsonify({
        'success': True,
        'status': controller.get_status(),
    })


@app.route('/api/pixel-arrays/<array_id>/effect', methods=['POST'])
def set_pixel_array_effect(array_id):
    """
    Start an effect on the pixel array.

    Request body:
    {
        "type": "wave" | "chase" | "bounce" | "rainbow_wave" | "none",
        "color": {"r": 255, "g": 0, "b": 0, "w": 0},  // Base color for effect
        "speed": 1.0,                                  // Hz
        "params": {                                    // Effect-specific params
            "tail_length": 2                           // For chase/bounce
        }
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    controller = _pixel_arrays[array_id]

    effect_type_str = data.get('type', 'none')
    effect_types = {
        'none': EffectType.NONE,
        'wave': EffectType.WAVE,
        'chase': EffectType.CHASE,
        'bounce': EffectType.BOUNCE,
        'rainbow_wave': EffectType.RAINBOW_WAVE,
    }

    if effect_type_str not in effect_types:
        return jsonify({'error': f'Invalid effect type: {effect_type_str}'}), 400

    effect_type = effect_types[effect_type_str]

    # Parse color
    color_data = data.get('color', {})
    color = Pixel(
        color_data.get('r', 255),
        color_data.get('g', 0),
        color_data.get('b', 0),
        color_data.get('w', 0),
    )

    speed = data.get('speed', 1.0)
    params = data.get('params', {})

    controller.set_effect(effect_type, color=color, speed=speed, **params)

    return jsonify({
        'success': True,
        'effect_type': effect_type.value,
        'status': controller.get_status(),
    })


@app.route('/api/pixel-arrays/<array_id>/start', methods=['POST'])
def start_pixel_array(array_id):
    """Start the render loop for a pixel array (30 FPS)"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.start()

    return jsonify({
        'success': True,
        'running': controller.is_running,
        'status': controller.get_status(),
    })


@app.route('/api/pixel-arrays/<array_id>/stop', methods=['POST'])
def stop_pixel_array(array_id):
    """Stop the render loop for a pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.stop()

    return jsonify({
        'success': True,
        'running': controller.is_running,
        'status': controller.get_status(),
    })


@app.route('/api/pixel-arrays/<array_id>/blackout', methods=['POST'])
def blackout_pixel_array(array_id):
    """Set all pixels to black and send frame"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.clear_all()
    controller.populate_dmx_buffer()
    controller._send_frame()

    return jsonify({
        'success': True,
        'status': controller.get_status(),
    })


@app.route('/api/pixel-arrays/<array_id>/fixture-map', methods=['GET'])
def get_pixel_array_fixture_map(array_id):
    """Get the fixture map for a pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    fixture_map = controller.get_fixture_map()

    entries = []
    for entry in fixture_map.get_all_entries():
        entries.append({
            'fixture_index': entry.fixture_index,
            'start_channel': entry.start_channel,
            'r_channel': entry.r_channel,
            'g_channel': entry.g_channel,
            'b_channel': entry.b_channel,
            'w_channel': entry.w_channel,
        })

    return jsonify({
        'fixture_count': fixture_map.fixture_count,
        'is_valid': fixture_map.is_valid,
        'overflow_error': fixture_map.overflow_error,
        'entries': entries,
    })


# ─────────────────────────────────────────────────────────
# Scene Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/scenes', methods=['GET'])
def get_scenes():
    return jsonify(content_manager.get_scenes())

@app.route('/api/scenes', methods=['POST'])
def create_scene():
    return jsonify(content_manager.create_scene(request.get_json()))

@app.route('/api/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    scene = content_manager.get_scene(scene_id)
    return jsonify(scene) if scene else (jsonify({'error': 'Scene not found'}), 404)

@app.route('/api/scenes/<scene_id>', methods=['PUT'])
def update_scene(scene_id):
    """Update an existing scene"""
    data = request.get_json() or {}
    data['scene_id'] = scene_id  # Ensure scene_id is set for the update
    return jsonify(content_manager.create_scene(data))

@app.route('/api/scenes/<scene_id>', methods=['DELETE'])
def delete_scene(scene_id):
    return jsonify(content_manager.delete_scene(scene_id))

@app.route('/api/scenes/<scene_id>/play', methods=['POST'])
def play_scene(scene_id):
    data = request.get_json() or {}
    print(f"📥 Scene play request: scene_id={scene_id}, payload={data}", flush=True)
    return jsonify(content_manager.play_scene(
        scene_id,
        fade_ms=data.get('fade_ms'),
        use_local=data.get('use_local', True),
        target_channels=data.get('target_channels'),
        universe=data.get('universe'),
        universes=data.get('universes')  # NEW: Accept universes array
    ))

# ─────────────────────────────────────────────────────────
# Chase Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/chases', methods=['GET'])
def get_chases():
    return jsonify(content_manager.get_chases())

@app.route('/api/chases', methods=['POST'])
def create_chase():
    return jsonify(content_manager.create_chase(request.get_json()))

@app.route('/api/chases/<chase_id>', methods=['GET'])
def get_chase(chase_id):
    chase = content_manager.get_chase(chase_id)
    return jsonify(chase) if chase else (jsonify({'error': 'Chase not found'}), 404)

@app.route('/api/chases/<chase_id>', methods=['PUT'])
def update_chase(chase_id):
    """Update an existing chase"""
    data = request.get_json() or {}
    data['chase_id'] = chase_id  # Ensure chase_id is set for the update
    return jsonify(content_manager.create_chase(data))

@app.route('/api/chases/<chase_id>', methods=['DELETE'])
def delete_chase(chase_id):
    return jsonify(content_manager.delete_chase(chase_id))

@app.route('/api/chases/<chase_id>/play', methods=['POST'])
def play_chase(chase_id):
    data = request.get_json() or {}
    print(f"📥 Chase play request: chase_id={chase_id}, payload={data}", flush=True)
    return jsonify(content_manager.play_chase(
        chase_id,
        target_channels=data.get('target_channels'),
        universe=data.get('universe'),
        universes=data.get('universes'),  # NEW: Accept universes array
        fade_ms=data.get('fade_ms')
    ))

@app.route('/api/chases/<chase_id>/stop', methods=['POST'])
def stop_chase(chase_id):
    """Stop a specific chase"""
    chase_engine.stop_all()  # Stop all running chases
    return jsonify({'success': True, 'chase_id': chase_id})

@app.route('/api/chases/health', methods=['GET'])
def get_chase_health():
    """Get health status of all running chases (for debugging)"""
    return jsonify({
        'running': list(chase_engine.running_chases.keys()),
        'health': chase_engine.chase_health,
        'timestamp': datetime.now().isoformat()
    })

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
            threading.Thread(
                target=lambda: supabase.sync_look(result.to_dict()),
                daemon=True
            ).start()

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
            threading.Thread(
                target=lambda: supabase.sync_look(result.to_dict()),
                daemon=True
            ).start()

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

@app.route('/api/looks/<look_id>/play', methods=['POST'])
def play_look(look_id):
    """
    Play a Look with real-time modifier rendering.

    POST body:
    {
        "universes": [1, 2],       // Target universes (default: all online)
        "fade_ms": 500,            // Initial fade time (optional)
        "seed": 12345              // Random seed for determinism (optional)
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

    # Stop any existing renders
    render_engine.stop_rendering()

    # Determine target universes
    universes = data.get('universes')
    if not universes:
        # Default to all online paired nodes
        universes = list(set(
            n.get('universe', 1) for n in node_manager.get_nodes()
            if n.get('is_paired') and n.get('status') == 'online'
        ))
        if not universes:
            universes = [1]

    # Check if look has modifiers
    has_modifiers = len(look.modifiers) > 0 and any(m.enabled for m in look.modifiers)

    if has_modifiers:
        # Start render engine for continuous modifier rendering
        if not render_engine._running:
            render_engine.start()

        # Start rendering this look
        render_engine.render_look(
            look_id=look_id,
            channels=look.channels,
            modifiers=[m.to_dict() for m in look.modifiers],
            universes=universes,
            seed=data.get('seed'),
        )

        return jsonify({
            'success': True,
            'look_id': look_id,
            'name': look.name,
            'universes': universes,
            'rendering': True,
            'modifier_count': len([m for m in look.modifiers if m.enabled]),
            'fps': render_engine.target_fps,
        })
    else:
        # No modifiers - just set static channels (like a scene)
        fade_ms = data.get('fade_ms', look.fade_ms or 0)

        for universe in universes:
            content_manager.set_channels(universe, look.channels, fade_ms=fade_ms)

        return jsonify({
            'success': True,
            'look_id': look_id,
            'name': look.name,
            'universes': universes,
            'rendering': False,
            'fade_ms': fade_ms,
        })

@app.route('/api/looks/<look_id>/stop', methods=['POST'])
def stop_look(look_id):
    """Stop playing a Look"""
    # Stop render engine
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
            threading.Thread(
                target=lambda: supabase.sync_sequence(result.to_dict()),
                daemon=True
            ).start()

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
            threading.Thread(
                target=lambda: supabase.sync_sequence(result.to_dict()),
                daemon=True
            ).start()

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

    # Stop any existing playback
    render_engine.stop_rendering()
    playback_controller.stop()
    chase_engine.stop_all()

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
    try:
        loop_mode = LoopMode(loop_mode_str)
    except ValueError:
        loop_mode = LoopMode.LOOP

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

    # Start playback
    result = playback_controller.play_sequence(
        sequence_id=sequence_id,
        steps=steps,
        universes=universes,
        bpm=bpm,
        loop_mode=loop_mode,
        seed=data.get('seed'),
        start_step=data.get('start_step', 0),
    )

    if result.get('success'):
        print(f"🎬 Playing sequence '{sequence.name}' on universes {universes}", flush=True)

    return jsonify({
        **result,
        'name': sequence.name,
    })


@app.route('/api/sequences/<sequence_id>/stop', methods=['POST'])
def stop_sequence(sequence_id):
    """Stop sequence playback"""
    status = playback_controller.get_status()
    if status.get('sequence_id') == sequence_id:
        result = playback_controller.stop()
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
    """Stop unified playback"""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    fade_ms = data.get('fade_ms', 0)

    if session_id:
        unified_engine.stop_session(session_id, fade_ms)
    else:
        unified_engine.stop_all(fade_ms)

    return jsonify({'success': True, 'stopped': session_id or 'all'})


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
    """Get unified playback controller status"""
    return jsonify(playback_controller.get_status())


@app.route('/api/playback/stop', methods=['POST'])
def stop_all_playback():
    """Stop all playback (Look, Sequence, Chase, Effect)"""
    # Stop unified playback controller
    playback_result = playback_controller.stop()

    # Stop render engine
    render_engine.stop_rendering()

    # Stop legacy chase engine
    chase_engine.stop_all()

    # Stop effects engine
    effects_engine.stop_effect()

    # Release all arbitration
    arbitration.release('look')
    arbitration.release('sequence')
    arbitration.release('chase')
    arbitration.release('effect')

    return jsonify({
        'success': True,
        'stopped': {
            'playback': playback_result.get('stopped'),
            'message': 'All playback stopped'
        }
    })


@app.route('/api/playback/pause', methods=['POST'])
def pause_playback():
    """Pause current playback"""
    return jsonify(playback_controller.pause())


@app.route('/api/playback/resume', methods=['POST'])
def resume_playback():
    """Resume paused playback"""
    return jsonify(playback_controller.resume())


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


# ─────────────────────────────────────────────────────────
# Merge Layer API (Phase 5 - Multi-source playback)
# ─────────────────────────────────────────────────────────

@app.route('/api/merge/status', methods=['GET'])
def get_merge_status():
    """
    Get merge layer status including all active sources.

    Returns:
    {
        "source_count": 2,
        "active_count": 2,
        "blackout_active": false,
        "sources": [
            {"source_id": "look_xxx", "source_type": "look", "priority": 50, ...},
            {"source_id": "effect_yyy", "source_type": "effect", "priority": 60, ...}
        ]
    }
    """
    return jsonify(merge_layer.get_status())


@app.route('/api/merge/channel/<int:universe>/<int:channel>', methods=['GET'])
def get_channel_breakdown(universe, channel):
    """
    Debug endpoint: Show which sources contribute to a specific channel.

    Returns the merge breakdown with:
    - All contributing sources
    - Channel type (dimmer/color/etc)
    - Merge mode (HTP/LTP)
    - Final merged value
    """
    return jsonify(merge_layer.get_source_breakdown(universe, channel))


@app.route('/api/merge/blackout', methods=['POST'])
def merge_blackout():
    """
    Activate merge layer blackout (highest priority override).

    POST body:
    {
        "active": true,           // Enable/disable blackout
        "universes": [1, 2]       // Optional: specific universes (null = all)
    }
    """
    data = request.get_json() or {}
    active = data.get('active', True)
    universes = data.get('universes')

    merge_layer.set_blackout(active, universes)

    # Also trigger SSOT blackout for physical output
    if active:
        if universes:
            for univ in universes:
                content_manager.blackout(universe=univ, fade_ms=0)
        else:
            content_manager.blackout(fade_ms=0)

    return jsonify({
        'success': True,
        'blackout_active': merge_layer.is_blackout(),
        'universes': universes
    })


@app.route('/api/merge/sources', methods=['GET'])
def get_merge_sources():
    """List all registered merge sources with their priorities"""
    status = merge_layer.get_status()
    return jsonify({
        'sources': status.get('sources', []),
        'priority_levels': {
            'blackout': 100,
            'manual': 80,
            'effect': 60,
            'look': 50,
            'sequence': 45,
            'chase': 40,
            'scene': 20,
            'background': 10
        }
    })


@app.route('/api/merge/classifier/reload', methods=['POST'])
def reload_fixture_classifier():
    """Reload fixture definitions into the channel classifier"""
    load_fixtures_into_classifier()
    return jsonify({
        'success': True,
        'message': 'Fixture classifier reloaded'
    })


# ─────────────────────────────────────────────────────────
# Preview Service API (Phase 6 - Live editing preview)
# ─────────────────────────────────────────────────────────

@app.route('/api/preview/sessions', methods=['GET'])
def list_preview_sessions():
    """List all preview sessions"""
    return jsonify({
        'sessions': preview_service.list_sessions(),
        'status': preview_service.get_status()
    })


@app.route('/api/preview/session', methods=['POST'])
def create_preview_session():
    """
    Create a new preview session for editing.

    POST body:
    {
        "session_id": "edit_look_123",     // Unique session ID
        "preview_type": "look",             // look or sequence
        "channels": {"1": 255, "2": 128},   // Base channels
        "modifiers": [...],                 // Modifier configs
        "universes": [1, 2],                // Target universes
        "fixture_filter": ["fixture_1"]     // Optional: specific fixtures
    }
    """
    data = request.get_json() or {}

    session_id = data.get('session_id', f"preview_{int(time.time() * 1000)}")
    preview_type = data.get('preview_type', 'look')
    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    universes = data.get('universes', [1])
    fixture_filter = data.get('fixture_filter')

    session = preview_service.create_session(
        session_id=session_id,
        preview_type=preview_type,
        channels=channels,
        modifiers=modifiers,
        universes=universes,
        fixture_filter=fixture_filter,
    )

    return jsonify({
        'success': True,
        'session_id': session.session_id,
        'mode': session.mode.value,
        'universes': session.universes,
    })


@app.route('/api/preview/session/<session_id>', methods=['GET'])
def get_preview_session(session_id):
    """Get a preview session's current state"""
    session = preview_service.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session.session_id,
        'preview_type': session.preview_type,
        'mode': session.mode.value,
        'running': session.running,
        'channels': session.channels,
        'modifiers': session.modifiers,
        'universes': session.universes,
        'frame_count': session.frame_count,
        'last_frame': {
            'channels': session.last_frame.channels,
            'elapsed_ms': session.last_frame.elapsed_ms,
        } if session.last_frame else None,
    })


@app.route('/api/preview/session/<session_id>', methods=['PUT'])
def update_preview_session(session_id):
    """
    Update preview session content (immediate re-render).

    PUT body:
    {
        "channels": {"1": 255},    // Optional: update base channels
        "modifiers": [...],         // Optional: update modifiers
        "universes": [1, 2]         // Optional: update targets
    }
    """
    data = request.get_json() or {}

    success = preview_service.update_session(
        session_id=session_id,
        channels=data.get('channels'),
        modifiers=data.get('modifiers'),
        universes=data.get('universes'),
    )

    if not success:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/preview/session/<session_id>', methods=['DELETE'])
def delete_preview_session(session_id):
    """Delete a preview session"""
    success = preview_service.delete_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/preview/session/<session_id>/start', methods=['POST'])
def start_preview_session(session_id):
    """Start preview playback (begins rendering and streaming)"""
    success = preview_service.start_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id, 'running': True})


@app.route('/api/preview/session/<session_id>/stop', methods=['POST'])
def stop_preview_session(session_id):
    """Stop preview playback"""
    success = preview_service.stop_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id, 'running': False})


@app.route('/api/preview/session/<session_id>/arm', methods=['POST'])
def arm_preview_session(session_id):
    """
    Arm a preview session for live output.
    WARNING: Armed sessions output to real universes!
    """
    success = preview_service.arm_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'success': True,
        'session_id': session_id,
        'mode': 'armed',
        'warning': 'Session is now outputting to live universes!'
    })


@app.route('/api/preview/session/<session_id>/disarm', methods=['POST'])
def disarm_preview_session(session_id):
    """Disarm a preview session (return to sandbox mode)"""
    success = preview_service.disarm_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'success': True,
        'session_id': session_id,
        'mode': 'sandbox'
    })


@app.route('/api/preview/frame', methods=['POST'])
def render_single_preview_frame():
    """
    Render a single preview frame without creating a session.
    Useful for instant feedback when editing params.

    POST body:
    {
        "channels": {"1": 255, "2": 128},
        "modifiers": [...],
        "elapsed_time": 0.0,     // Optional: time offset for animation
        "seed": 12345            // Optional: random seed
    }

    Returns:
    {
        "success": true,
        "channels": {1: 255, 2: 128, ...},  // Rendered channel values
    }
    """
    data = request.get_json() or {}

    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    elapsed_time = data.get('elapsed_time', 0.0)
    seed = data.get('seed', 0)

    rendered = preview_service.render_preview_frame(
        channels=channels,
        modifiers=modifiers,
        elapsed_time=elapsed_time,
        seed=seed,
    )

    return jsonify({
        'success': True,
        'channels': rendered,
        'modifier_count': len([m for m in modifiers if m.get('enabled', True)]),
    })


@app.route('/api/preview/status', methods=['GET'])
def get_preview_status():
    """Get preview service status"""
    return jsonify(preview_service.get_status())


# ─────────────────────────────────────────────────────────
# Modifier Registry API (schemas, presets, validation)
# ─────────────────────────────────────────────────────────
@app.route('/api/modifiers/schemas', methods=['GET'])
def get_modifier_schemas_route():
    """
    Get all modifier schemas for UI generation.
    Returns complete schema definitions including params, presets, and categories.
    """
    return jsonify(modifier_registry.to_api_response())

@app.route('/api/modifiers/types', methods=['GET'])
def get_modifier_types_route():
    """Get list of available modifier types"""
    return jsonify({
        'types': modifier_registry.get_types(),
        'categories': modifier_registry.get_categories()
    })

@app.route('/api/modifiers/<modifier_type>/presets', methods=['GET'])
def get_modifier_presets_route(modifier_type):
    """Get all presets for a specific modifier type"""
    presets = modifier_registry.get_presets(modifier_type)
    if not presets and modifier_type not in modifier_registry.get_types():
        return jsonify({'error': f'Unknown modifier type: {modifier_type}'}), 404
    return jsonify({
        'modifier_type': modifier_type,
        'presets': presets
    })

@app.route('/api/modifiers/<modifier_type>/presets/<preset_id>', methods=['GET'])
def get_modifier_preset_route(modifier_type, preset_id):
    """Get a specific preset and create a modifier from it"""
    modifier_data = modifier_registry.create_from_preset(modifier_type, preset_id)
    if not modifier_data:
        return jsonify({'error': f'Preset not found: {modifier_type}/{preset_id}'}), 404
    return jsonify({
        'success': True,
        'modifier': modifier_data
    })

@app.route('/api/modifiers/validate', methods=['POST'])
def validate_modifier_route():
    """
    Validate a modifier against its schema.
    Returns validation result with detailed error if invalid.
    """
    data = request.get_json() or {}
    is_valid, error = validate_modifier(data)
    if not is_valid:
        return jsonify({
            'valid': False,
            'error': error
        }), 400
    return jsonify({
        'valid': True,
        'normalized': normalize_modifier(data)
    })

@app.route('/api/modifiers/normalize', methods=['POST'])
def normalize_modifier_route():
    """
    Normalize a modifier by applying defaults and generating ID.
    Use this before saving a modifier to ensure all fields are populated.
    """
    data = request.get_json() or {}
    # Validate first
    is_valid, error = validate_modifier(data)
    if not is_valid:
        return jsonify({'success': False, 'error': error}), 400
    return jsonify({
        'success': True,
        'modifier': normalize_modifier(data)
    })


# ─────────────────────────────────────────────────────────
# Distribution Modes API (Phase 1 - Fixture-Centric Architecture)
# ─────────────────────────────────────────────────────────
@app.route('/api/modifiers/distribution-modes', methods=['GET'])
def get_distribution_modes_route():
    """
    Get all available distribution modes and their descriptions.

    Distribution modes control how modifiers are applied across multiple fixtures:
    - SYNCED: All fixtures identical (default)
    - INDEXED: Scaled by fixture index
    - PHASED: Time offset per fixture
    - PIXELATED: Unique per fixture
    - RANDOM: Deterministic random per fixture
    """
    modes = [
        {
            'mode': mode.value,
            'name': mode.name,
            'description': {
                'synced': 'All fixtures receive identical effect values',
                'indexed': 'Effect values scale linearly with fixture position',
                'phased': 'Time offset between fixtures for traveling effects',
                'pixelated': 'Each fixture has unique, independent effect values',
                'random': 'Deterministic random variation per fixture',
                'grouped': 'Same value for fixtures in same group',
            }.get(mode.value, '')
        }
        for mode in DistributionMode
    ]
    return jsonify({
        'modes': modes,
        'presets': list_distribution_presets()
    })


@app.route('/api/modifiers/distribution-modes/<modifier_type>', methods=['GET'])
def get_modifier_distribution_modes_route(modifier_type):
    """Get supported distribution modes for a specific modifier type"""
    supported = get_supported_distributions(modifier_type)
    return jsonify({
        'modifier_type': modifier_type,
        'supported_modes': [mode.value for mode in supported]
    })


@app.route('/api/modifiers/distribution-modes/suggest', methods=['POST'])
def suggest_distribution_mode_route():
    """
    Get AI-suggested distribution mode for a modifier and fixture selection.

    Request body:
    {
        "modifier_type": "wave",
        "fixture_count": 8,
        "effect_intent": "chase" (optional)
    }
    """
    data = request.get_json() or {}
    modifier_type = data.get('modifier_type', 'pulse')
    fixture_count = data.get('fixture_count', 1)
    effect_intent = data.get('effect_intent')

    suggestion = suggest_distribution_for_effect(
        modifier_type=modifier_type,
        fixture_count=fixture_count,
        effect_intent=effect_intent
    )

    return jsonify({
        'suggestion': suggestion.to_dict(),
        'modifier_type': modifier_type,
        'fixture_count': fixture_count
    })


@app.route('/api/modifiers/distribution-presets', methods=['GET'])
def get_distribution_presets_route():
    """Get all distribution presets"""
    return jsonify({
        'presets': list_distribution_presets()
    })


@app.route('/api/modifiers/distribution-presets/<preset_name>', methods=['GET'])
def get_distribution_preset_route(preset_name):
    """Get a specific distribution preset"""
    preset = get_distribution_preset(preset_name)
    if not preset:
        return jsonify({'error': f'Preset not found: {preset_name}'}), 404
    return jsonify({
        'name': preset_name,
        'config': preset.to_dict()
    })


# ─────────────────────────────────────────────────────────
# AI Fixture Advisor API (Phase 2 - Fixture-Centric Architecture)
# ─────────────────────────────────────────────────────────
@app.route('/api/ai/suggestions/distribution', methods=['POST'])
def get_ai_distribution_suggestions_route():
    """
    Get AI suggestions for distribution modes.

    IMPORTANT: AI suggestions are NEVER auto-applied.
    Returns suggestions that the user must explicitly Apply or Dismiss.

    Request body:
    {
        "modifier_type": "rainbow",
        "modifier_params": {"speed": 0.5},
        "fixture_count": 6
    }
    """
    data = request.get_json() or {}
    modifier_type = data.get('modifier_type', 'pulse')
    modifier_params = data.get('modifier_params', {})
    fixture_count = data.get('fixture_count', 1)

    suggestions = get_distribution_suggestions(
        modifier_type=modifier_type,
        fixture_count=fixture_count,
        modifier_params=modifier_params
    )

    return jsonify({
        'suggestions': suggestions,
        'note': 'AI suggestions require explicit user approval. Apply or Dismiss each suggestion.'
    })


@app.route('/api/ai/suggestions/apply/<suggestion_id>', methods=['POST'])
def apply_ai_suggestion_route(suggestion_id):
    """
    Mark an AI suggestion as applied.

    This does NOT apply the suggestion automatically - it only marks it as applied
    after the user has explicitly chosen to apply it in the UI.
    """
    success = apply_ai_suggestion(suggestion_id)
    if not success:
        return jsonify({'error': 'Suggestion not found'}), 404
    return jsonify({
        'success': True,
        'suggestion_id': suggestion_id,
        'status': 'applied'
    })


@app.route('/api/ai/suggestions/dismiss/<suggestion_id>', methods=['POST'])
def dismiss_ai_suggestion_route(suggestion_id):
    """Dismiss an AI suggestion"""
    success = dismiss_ai_suggestion(suggestion_id)
    if not success:
        return jsonify({'error': 'Suggestion not found'}), 404
    return jsonify({
        'success': True,
        'suggestion_id': suggestion_id,
        'status': 'dismissed'
    })


@app.route('/api/ai/suggestions/pending', methods=['GET'])
def get_pending_ai_suggestions_route():
    """Get all pending (not applied/dismissed) AI suggestions"""
    advisor = get_ai_advisor()
    suggestions = advisor.get_pending_suggestions()
    return jsonify({
        'suggestions': [s.to_dict() for s in suggestions]
    })


@app.route('/api/ai/suggestions/transition', methods=['POST'])
def get_ai_transition_suggestions_route():
    """
    Get AI suggestions for transition/crossfade times.

    Request body:
    {
        "effect_type": "wave",      // Required: wave, chase, pulse, rainbow, etc.
        "fixture_count": 5,         // Required: number of fixtures
        "step_duration_ms": 350     // Optional: step duration for timing
    }

    Returns suggestions for smooth, ultra_smooth, default, and snappy transitions.
    AI suggests but NEVER auto-applies - user must explicitly apply.
    """
    from ai_fixture_advisor import get_transition_suggestions

    data = request.get_json() or {}
    effect_type = data.get('effect_type', 'wave')
    fixture_count = data.get('fixture_count', 1)
    step_duration_ms = data.get('step_duration_ms')

    suggestions = get_transition_suggestions(
        effect_type=effect_type,
        fixture_count=fixture_count,
        step_duration_ms=step_duration_ms
    )

    return jsonify({
        'effect_type': effect_type,
        'fixture_count': fixture_count,
        'suggestions': suggestions
    })


@app.route('/api/ai/transition/recommend', methods=['GET'])
def get_recommended_transition_route():
    """
    Quick endpoint to get recommended transition for an effect.

    Query params:
    - effect_type: wave, chase, pulse, rainbow, etc.
    - fixture_count: number of fixtures
    - smoothness: snappy, default, smooth, ultra_smooth (default: smooth)

    Returns:
    {
        "transition_ms": 400,
        "transition_easing": "ease-in-out"
    }
    """
    from ai_fixture_advisor import get_recommended_transition_for_effect

    effect_type = request.args.get('effect_type', 'wave')
    fixture_count = int(request.args.get('fixture_count', 1))
    smoothness = request.args.get('smoothness', 'smooth')

    result = get_recommended_transition_for_effect(
        effect_type=effect_type,
        fixture_count=fixture_count,
        smoothness=smoothness
    )

    return jsonify({
        'effect_type': effect_type,
        'fixture_count': fixture_count,
        'smoothness': smoothness,
        **result
    })


# ─────────────────────────────────────────────────────────
# Render Pipeline API (Phase 3 - Fixture-Centric Architecture)
# ─────────────────────────────────────────────────────────
@app.route('/api/render/status', methods=['GET'])
def get_render_pipeline_status_route():
    """Get status of the final render pipeline"""
    pipeline = get_render_pipeline()
    return jsonify(pipeline.get_status())


@app.route('/api/render/features', methods=['GET'])
def get_render_features_route():
    """Get feature flags for the render pipeline"""
    pipeline = get_render_pipeline()
    return jsonify({
        'fixture_centric_enabled': pipeline.features.FIXTURE_CENTRIC_ENABLED,
        'legacy_channel_fallback': pipeline.features.LEGACY_CHANNEL_FALLBACK,
        'ai_suggestions_enabled': pipeline.features.AI_SUGGESTIONS_ENABLED,
        'distribution_modes_enabled': pipeline.features.DISTRIBUTION_MODES_ENABLED,
    })


@app.route('/api/render/features', methods=['POST'])
def set_render_features_route():
    """
    Update feature flags for the render pipeline.

    Request body:
    {
        "fixture_centric_enabled": true,
        "distribution_modes_enabled": true
    }
    """
    data = request.get_json() or {}
    pipeline = get_render_pipeline()

    if 'fixture_centric_enabled' in data:
        pipeline.features.FIXTURE_CENTRIC_ENABLED = bool(data['fixture_centric_enabled'])
    if 'legacy_channel_fallback' in data:
        pipeline.features.LEGACY_CHANNEL_FALLBACK = bool(data['legacy_channel_fallback'])
    if 'ai_suggestions_enabled' in data:
        pipeline.features.AI_SUGGESTIONS_ENABLED = bool(data['ai_suggestions_enabled'])
    if 'distribution_modes_enabled' in data:
        pipeline.features.DISTRIBUTION_MODES_ENABLED = bool(data['distribution_modes_enabled'])

    return jsonify({
        'success': True,
        'features': {
            'fixture_centric_enabled': pipeline.features.FIXTURE_CENTRIC_ENABLED,
            'legacy_channel_fallback': pipeline.features.LEGACY_CHANNEL_FALLBACK,
            'ai_suggestions_enabled': pipeline.features.AI_SUGGESTIONS_ENABLED,
            'distribution_modes_enabled': pipeline.features.DISTRIBUTION_MODES_ENABLED,
        }
    })


# ─────────────────────────────────────────────────────────
# Migration Routes (legacy scenes/chases to looks/sequences)
# ─────────────────────────────────────────────────────────
@app.route('/api/migrate/scenes-to-looks', methods=['POST'])
def migrate_scenes_to_looks_route():
    """Migrate legacy scenes to new looks format"""
    from looks_sequences import migrate_scenes_to_looks
    report = migrate_scenes_to_looks(DATABASE, looks_sequences_manager)
    return jsonify({'success': True, 'report': report})

@app.route('/api/migrate/chases-to-sequences', methods=['POST'])
def migrate_chases_to_sequences_route():
    """Migrate legacy chases to new sequences format"""
    from looks_sequences import migrate_chases_to_sequences
    report = migrate_chases_to_sequences(DATABASE, looks_sequences_manager)
    return jsonify({'success': True, 'report': report})

@app.route('/api/migrate/all', methods=['POST'])
def migrate_all_route():
    """Run full migration from legacy to new format"""
    report = run_full_migration(DATABASE)
    return jsonify({'success': True, 'report': report})

# ─────────────────────────────────────────────────────────
# Dynamic Effects Routes (smooth fades, staggered patterns)
# ─────────────────────────────────────────────────────────
@app.route('/api/effects/christmas', methods=['POST'])
def start_christmas_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.christmas_stagger(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('fade_ms', 1500),
        data.get('hold_ms', 1000),
        data.get('stagger_ms', 300)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/twinkle', methods=['POST'])
def start_twinkle_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.random_twinkle(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('min_fade_ms', 500),
        data.get('max_fade_ms', 2000)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/smooth', methods=['POST'])
def start_smooth_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.smooth_chase(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('fade_ms', 1500),
        data.get('hold_ms', 500)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/wave', methods=['POST'])
def start_wave_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.wave(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 0, 0, 0]),
        data.get('wave_speed_ms', 2000),
        data.get('tail_length', 2)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/strobe', methods=['POST'])
def start_strobe_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.strobe(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 255, 255, 0]),
        data.get('on_ms', 50),
        data.get('off_ms', 50)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/pulse', methods=['POST'])
def start_pulse_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.pulse(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 255, 255, 0]),
        data.get('pulse_ms', 2000),
        data.get('min_brightness', 0),
        data.get('max_brightness', 255)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/fade', methods=['POST'])
def start_fade_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.fade(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('cycle_ms', 10000)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/fire', methods=['POST'])
def start_fire_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = effects_engine.fire(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('intensity', 0.8)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@app.route('/api/effects/stop', methods=['POST'])
def stop_effects():
    data = request.get_json() or {}
    effects_engine.stop_effect(data.get('effect_id'))
    return jsonify({'success': True})

@app.route('/api/effects', methods=['GET'])
def get_running_effects():
    return jsonify({'running': list(effects_engine.running.keys()), 'count': len(effects_engine.running)})

# ─────────────────────────────────────────────────────────
# Fixture Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/fixtures', methods=['GET'])
def get_fixtures():
    return jsonify(content_manager.get_fixtures())

@app.route('/api/fixtures', methods=['POST'])
def create_fixture():
    return jsonify(content_manager.create_fixture(request.get_json()))

@app.route('/api/fixtures/<fixture_id>', methods=['GET'])
def get_fixture(fixture_id):
    fixture = content_manager.get_fixture(fixture_id)
    return jsonify(fixture) if fixture else (jsonify({'error': 'Fixture not found'}), 404)

@app.route('/api/fixtures/<fixture_id>', methods=['PUT'])
def update_fixture(fixture_id):
    result = content_manager.update_fixture(fixture_id, request.get_json())
    if result.get('error'):
        return jsonify(result), 404
    return jsonify(result)

@app.route('/api/fixtures/<fixture_id>', methods=['DELETE'])
def delete_fixture(fixture_id):
    return jsonify(content_manager.delete_fixture(fixture_id))

@app.route('/api/fixtures/universe/<int:universe>', methods=['GET'])
def get_fixtures_by_universe(universe):
    fixtures = content_manager.get_fixtures()
    filtered = [f for f in fixtures if f.get('universe') == universe]
    return jsonify(filtered)

@app.route('/api/fixtures/channels', methods=['POST'])
def get_fixtures_for_channels():
    """Get fixtures that cover specific channel ranges"""
    data = request.get_json() or {}
    universe = data.get('universe', 1)
    channels = data.get('channels', [])
    return jsonify(content_manager.get_fixtures_for_channels(universe, channels))

# ─────────────────────────────────────────────────────────
# Fixture Library Routes (profiles, OFL integration)
# ─────────────────────────────────────────────────────────
@app.route('/api/fixture-library/profiles', methods=['GET'])
def get_fixture_profiles():
    """Get all available fixture profiles"""
    category = request.args.get('category')
    profiles = fixture_library.get_all_profiles(category)
    return jsonify([{
        'profile_id': p.profile_id,
        'manufacturer': p.manufacturer,
        'model': p.model,
        'category': p.category,
        'modes': [{'mode_id': m.mode_id, 'name': m.name, 'channel_count': m.channel_count} for m in p.modes],
        'source': p.source
    } for p in profiles])

@app.route('/api/fixture-library/profiles/<profile_id>', methods=['GET'])
def get_fixture_profile(profile_id):
    """Get a specific fixture profile with full details"""
    profile = fixture_library.get_profile(profile_id)
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({
        'profile_id': profile.profile_id,
        'manufacturer': profile.manufacturer,
        'model': profile.model,
        'category': profile.category,
        'modes': [{
            'mode_id': m.mode_id,
            'name': m.name,
            'channel_count': m.channel_count,
            'channels': [{'name': ch.name, 'type': ch.type, 'default': ch.default} for ch in m.channels]
        } for m in profile.modes],
        'source': profile.source,
        'ofl_key': profile.ofl_key,
        'rdm_manufacturer_id': profile.rdm_manufacturer_id,
        'rdm_device_model_id': profile.rdm_device_model_id
    })

@app.route('/api/fixture-library/profiles', methods=['POST'])
def create_fixture_profile():
    """Create a new fixture profile"""
    data = request.get_json() or {}
    try:
        modes = []
        for m in data.get('modes', []):
            channels = [ChannelCapability(
                name=ch.get('name', f'Channel {i+1}'),
                type=ch.get('type', 'control'),
                default=ch.get('default', 0)
            ) for i, ch in enumerate(m.get('channels', []))]
            modes.append(FixtureMode(
                mode_id=m.get('mode_id', 'default'),
                name=m.get('name', 'Default'),
                channel_count=len(channels),
                channels=channels
            ))

        profile = FixtureProfile(
            profile_id=data.get('profile_id', f"custom-{int(time.time())}"),
            manufacturer=data.get('manufacturer', 'Custom'),
            model=data.get('model', 'Fixture'),
            category=data.get('category', 'generic'),
            modes=modes,
            source='manual'
        )
        fixture_library.save_profile(profile)
        return jsonify({'success': True, 'profile_id': profile.profile_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/fixture-library/ofl/search', methods=['GET'])
def search_ofl():
    """Search Open Fixture Library for fixtures"""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify({'error': 'Query must be at least 2 characters'}), 400
    results = fixture_library.search_ofl(query)
    return jsonify(results)

@app.route('/api/fixture-library/ofl/manufacturers', methods=['GET'])
def get_ofl_manufacturers():
    """Get list of manufacturers from Open Fixture Library"""
    manufacturers = fixture_library.get_ofl_manufacturers()
    return jsonify(manufacturers)

@app.route('/api/fixture-library/ofl/import', methods=['POST'])
def import_ofl_fixture():
    """Import a fixture from Open Fixture Library"""
    data = request.get_json() or {}
    manufacturer = data.get('manufacturer')
    fixture = data.get('fixture')
    if not manufacturer or not fixture:
        return jsonify({'error': 'manufacturer and fixture required'}), 400

    profile = fixture_library.import_from_ofl(manufacturer, fixture)
    if profile:
        return jsonify({
            'success': True,
            'profile_id': profile.profile_id,
            'manufacturer': profile.manufacturer,
            'model': profile.model,
            'modes': [{'mode_id': m.mode_id, 'name': m.name, 'channel_count': m.channel_count} for m in profile.modes]
        })
    return jsonify({'error': 'Failed to import fixture'}), 500

@app.route('/api/fixture-library/rdm/auto-configure', methods=['POST'])
def auto_configure_from_rdm():
    """Auto-configure fixtures from RDM devices"""
    data = request.get_json() or {}
    rdm_uid = data.get('rdm_uid')

    # Get RDM device info
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM rdm_devices WHERE uid = ?', (rdm_uid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'RDM device not found'}), 404

    rdm_device = dict(row)

    # Try to find matching profile
    profile = fixture_library.find_profile_by_rdm(
        rdm_device.get('manufacturer_id', 0),
        rdm_device.get('device_model_id', 0)
    )

    # Create fixture instance
    instance = fixture_library.create_fixture_from_rdm(rdm_device, profile)

    # Save to fixtures table (existing system)
    mode = None
    if profile:
        mode = profile.get_mode(instance.mode_id)

    fixture_data = {
        'fixture_id': instance.fixture_id,
        'name': instance.name,
        'type': profile.category if profile else 'generic',
        'manufacturer': profile.manufacturer if profile else fixture_library.get_rdm_manufacturer_name(rdm_device.get('manufacturer_id', 0)),
        'model': profile.model if profile else 'Unknown',
        'universe': instance.universe,
        'start_channel': instance.start_channel,
        'channel_count': mode.channel_count if mode else rdm_device.get('dmx_footprint', 4),
        'channel_map': [ch.name for ch in mode.channels] if mode else [],
        'rdm_uid': rdm_uid
    }
    content_manager.create_fixture(fixture_data)

    return jsonify({
        'success': True,
        'fixture': fixture_data,
        'profile_matched': profile is not None,
        'profile_id': profile.profile_id if profile else None
    })

@app.route('/api/fixture-library/apply-color', methods=['POST'])
def apply_color_to_fixtures():
    """Apply a color to specified fixtures intelligently"""
    data = request.get_json() or {}
    fixture_ids = data.get('fixture_ids', [])
    color = data.get('color', {})  # {r, g, b, w, dimmer}
    fade_ms = data.get('fade_ms', 0)
    universe = data.get('universe', 1)

    if not fixture_ids:
        return jsonify({'error': 'No fixtures specified'}), 400

    # Get fixtures
    fixtures = []
    for fid in fixture_ids:
        fixture_data = content_manager.get_fixture(fid)
        if fixture_data:
            # Convert to FixtureInstance
            profile = fixture_library.get_profile(fixture_data.get('profile_id', 'generic-rgbw'))
            if not profile:
                # Use legacy data to create ad-hoc profile
                profile = fixture_library._create_generic_profile_for_footprint(
                    fixture_data.get('channel_count', 4)
                )
            fixtures.append(FixtureInstance(
                fixture_id=fixture_data['fixture_id'],
                name=fixture_data['name'],
                profile_id=profile.profile_id,
                mode_id=profile.modes[0].mode_id if profile.modes else 'default',
                universe=fixture_data.get('universe', 1),
                start_channel=fixture_data.get('start_channel', 1)
            ))

    if not fixtures:
        return jsonify({'error': 'No valid fixtures found'}), 404

    # Apply color using channel mapper
    channels = channel_mapper.apply_color_to_fixtures(
        fixtures,
        r=color.get('r', 0),
        g=color.get('g', 0),
        b=color.get('b', 0),
        w=color.get('w', 0),
        dimmer=color.get('dimmer', 255)
    )

    # Send DMX
    content_manager.set_channels(universe, channels, fade_ms=fade_ms)

    return jsonify({
        'success': True,
        'channels': channels,
        'fixture_count': len(fixtures)
    })

@app.route('/api/fixture-library/apply-scene-to-all', methods=['POST'])
def apply_scene_to_all_fixtures():
    """
    Apply a scene's color pattern to ALL configured fixtures intelligently.

    This extracts the color intent from the scene (RGB values from first few channels)
    and applies it to all fixtures using their proper channel mappings.
    """
    data = request.get_json() or {}
    scene_id = data.get('scene_id')
    fade_ms = data.get('fade_ms', 1000)
    universes = data.get('universes', [])

    if not scene_id:
        return jsonify({'error': 'scene_id required'}), 400

    # Get scene
    scene = content_manager.get_scene(scene_id)
    if not scene:
        return jsonify({'error': 'Scene not found'}), 404

    # Extract color intent from scene channels
    scene_channels = scene.get('channels', {})
    if not scene_channels:
        return jsonify({'error': 'Scene has no channels'}), 400

    # Try to detect RGB(W) values from scene
    # Assume typical fixture layout: R, G, B, W, D or similar
    ch_values = sorted([(int(k), v) for k, v in scene_channels.items()], key=lambda x: x[0])

    # Extract first fixture's values as the "intent"
    color = {'r': 0, 'g': 0, 'b': 0, 'w': 0, 'dimmer': 255}
    if len(ch_values) >= 1:
        color['r'] = ch_values[0][1]
    if len(ch_values) >= 2:
        color['g'] = ch_values[1][1]
    if len(ch_values) >= 3:
        color['b'] = ch_values[2][1]
    if len(ch_values) >= 4:
        color['w'] = ch_values[3][1]
    if len(ch_values) >= 5:
        color['dimmer'] = ch_values[4][1]

    # Get ALL configured fixtures
    all_fixtures = content_manager.get_fixtures()

    # Filter by universe if specified
    if universes:
        all_fixtures = [f for f in all_fixtures if f.get('universe') in universes]

    if not all_fixtures:
        return jsonify({'error': 'No fixtures configured'}), 404

    # Group fixtures by universe
    fixtures_by_universe = {}
    for fixture_data in all_fixtures:
        u = fixture_data.get('universe', 1)
        if u not in fixtures_by_universe:
            fixtures_by_universe[u] = []

        # Convert to FixtureInstance
        profile = fixture_library.get_profile(fixture_data.get('profile_id'))
        if not profile:
            profile = fixture_library._create_generic_profile_for_footprint(
                fixture_data.get('channel_count', 4)
            )

        fixtures_by_universe[u].append(FixtureInstance(
            fixture_id=fixture_data['fixture_id'],
            name=fixture_data['name'],
            profile_id=profile.profile_id,
            mode_id=profile.modes[0].mode_id if profile.modes else 'default',
            universe=u,
            start_channel=fixture_data.get('start_channel', 1)
        ))

    # Apply to each universe
    total_fixtures = 0
    all_channels = {}

    for universe, fixtures in fixtures_by_universe.items():
        channels = channel_mapper.apply_color_to_fixtures(
            fixtures,
            r=color['r'],
            g=color['g'],
            b=color['b'],
            w=color['w'],
            dimmer=color['dimmer']
        )

        content_manager.set_channels(universe, channels, fade_ms=fade_ms)
        total_fixtures += len(fixtures)
        all_channels[universe] = channels

    return jsonify({
        'success': True,
        'scene_name': scene.get('name'),
        'color_extracted': color,
        'fixture_count': total_fixtures,
        'universes': list(fixtures_by_universe.keys()),
        'channels_by_universe': all_channels
    })

@app.route('/api/fixture-library/apply-color-to-all', methods=['POST'])
def apply_color_to_all_fixtures():
    """
    Apply a color to ALL configured fixtures.

    This is the simplest way to set all lights to a color.
    """
    data = request.get_json() or {}
    color = data.get('color', {})  # {r, g, b, w, dimmer}
    fade_ms = data.get('fade_ms', 0)
    universes = data.get('universes', [])

    if not color:
        return jsonify({'error': 'color required'}), 400

    # Get ALL configured fixtures
    all_fixtures = content_manager.get_fixtures()

    # Filter by universe if specified
    if universes:
        all_fixtures = [f for f in all_fixtures if f.get('universe') in universes]

    if not all_fixtures:
        return jsonify({'error': 'No fixtures configured'}), 404

    # Group fixtures by universe
    fixtures_by_universe = {}
    for fixture_data in all_fixtures:
        u = fixture_data.get('universe', 1)
        if u not in fixtures_by_universe:
            fixtures_by_universe[u] = []

        # Convert to FixtureInstance
        profile = fixture_library.get_profile(fixture_data.get('profile_id'))
        if not profile:
            profile = fixture_library._create_generic_profile_for_footprint(
                fixture_data.get('channel_count', 4)
            )

        fixtures_by_universe[u].append(FixtureInstance(
            fixture_id=fixture_data['fixture_id'],
            name=fixture_data['name'],
            profile_id=profile.profile_id,
            mode_id=profile.modes[0].mode_id if profile.modes else 'default',
            universe=u,
            start_channel=fixture_data.get('start_channel', 1)
        ))

    # Apply to each universe
    total_fixtures = 0

    for universe, fixtures in fixtures_by_universe.items():
        channels = channel_mapper.apply_color_to_fixtures(
            fixtures,
            r=color.get('r', 0),
            g=color.get('g', 0),
            b=color.get('b', 0),
            w=color.get('w', 0),
            dimmer=color.get('dimmer', 255)
        )

        content_manager.set_channels(universe, channels, fade_ms=fade_ms)
        total_fixtures += len(fixtures)

    return jsonify({
        'success': True,
        'color': color,
        'fixture_count': total_fixtures,
        'universes': list(fixtures_by_universe.keys())
    })

# ─────────────────────────────────────────────────────────
# Groups Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all fixture groups"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM groups ORDER BY name')
    rows = c.fetchall()
    conn.close()
    groups = []
    for row in rows:
        groups.append({
            'group_id': row[0], 'name': row[1], 'universe': row[2],
            'channels': json.loads(row[3]) if row[3] else [],
            'color': row[4], 'created_at': row[5]
        })
    return jsonify(groups)

@app.route('/api/groups', methods=['POST'])
def create_group():
    """Create a fixture group"""
    data = request.get_json()
    group_id = data.get('group_id', f"group_{int(time.time())}")
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO groups (group_id, name, universe, channels, color)
        VALUES (?, ?, ?, ?, ?)''',
        (group_id, data.get('name', 'New Group'), data.get('universe', 1),
         json.dumps(data.get('channels', [])), data.get('color', '#8b5cf6')))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'group_id': group_id})

@app.route('/api/groups/<group_id>', methods=['GET'])
def get_group(group_id):
    """Get a single group"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM groups WHERE group_id = ?', (group_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Group not found'}), 404
    return jsonify({
        'group_id': row[0], 'name': row[1], 'universe': row[2],
        'channels': json.loads(row[3]) if row[3] else [],
        'color': row[4], 'created_at': row[5]
    })

@app.route('/api/groups/<group_id>', methods=['PUT'])
def update_group(group_id):
    """Update a group"""
    data = request.get_json()
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE groups SET name=?, universe=?, channels=?, color=? WHERE group_id=?''',
        (data.get('name'), data.get('universe', 1),
         json.dumps(data.get('channels', [])), data.get('color', '#8b5cf6'), group_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/groups/<group_id>', methods=['DELETE'])
def delete_group(group_id):
    """Delete a group"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM groups WHERE group_id = ?', (group_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# Shows Routes (Timeline Playback)
# ─────────────────────────────────────────────────────────
@app.route('/api/shows', methods=['GET'])
def get_shows():
    """Get all shows"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM shows ORDER BY updated_at DESC')
    rows = c.fetchall()
    conn.close()
    shows = []
    for row in rows:
        shows.append({
            'show_id': row[0], 'name': row[1], 'description': row[2],
            'timeline': json.loads(row[3]) if row[3] else [],
            'duration_ms': row[4], 'created_at': row[5], 'updated_at': row[6], 'distributed': row[7] if len(row) > 7 else 0
        })
    return jsonify(shows)

@app.route('/api/shows', methods=['POST'])
def create_show():
    """Create a show"""
    data = request.get_json()
    show_id = data.get('show_id', f"show_{int(time.time())}")
    timeline = data.get('timeline', [])
    # Calculate duration from timeline
    duration_ms = max([e.get('time_ms', 0) for e in timeline]) if timeline else 0
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO shows 
        (show_id, name, description, timeline, duration_ms, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (show_id, data.get('name', 'New Show'), data.get('description', ''),
         json.dumps(timeline), duration_ms, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'show_id': show_id})

@app.route('/api/shows/<show_id>', methods=['GET'])
def get_show(show_id):
    """Get a single show"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM shows WHERE show_id = ?', (show_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Show not found'}), 404
    return jsonify({
        'show_id': row[0], 'name': row[1], 'description': row[2],
        'timeline': json.loads(row[3]) if row[3] else [],
        'duration_ms': row[4], 'created_at': row[5], 'updated_at': row[6], 'distributed': row[7] if len(row) > 7 else 0
    })

@app.route('/api/shows/<show_id>', methods=['PUT'])
def update_show(show_id):
    """Update a show"""
    data = request.get_json()
    timeline = data.get('timeline', [])
    duration_ms = max([e.get('time_ms', 0) for e in timeline]) if timeline else 0
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE shows SET name=?, description=?, timeline=?, duration_ms=?, updated_at=? 
        WHERE show_id=?''',
        (data.get('name'), data.get('description', ''), json.dumps(timeline),
         duration_ms, datetime.now().isoformat(), show_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/shows/<show_id>', methods=['DELETE'])
def delete_show(show_id):
    """Delete a show"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM shows WHERE show_id = ?', (show_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/shows/<show_id>/play', methods=['POST'])
def play_show(show_id):
    """Play a show timeline"""
    data = request.get_json() or {}
    universe = data.get('universe', 1)
    return jsonify(show_engine.play_show(show_id, universe))

@app.route('/api/shows/stop', methods=['POST'])
def stop_show():
    """Stop current show"""
    show_engine.stop()
    return jsonify({'success': True})


@app.route('/api/shows/pause', methods=['POST'])
def pause_show():
    """Pause current show"""
    show_engine.pause()
    return jsonify({'success': True, 'paused': True})

@app.route('/api/shows/resume', methods=['POST'])
def resume_show():
    """Resume current show"""
    show_engine.resume()
    return jsonify({'success': True, 'paused': False})

@app.route('/api/shows/tempo', methods=['POST'])
def set_show_tempo():
    """Set show tempo (0.25 to 4.0)"""
    data = request.get_json() or {}
    tempo = data.get('tempo', 1.0)
    show_engine.set_tempo(tempo)
    return jsonify({'success': True, 'tempo': show_engine.tempo})

# Schedules Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/schedules', methods=['GET'])
def get_schedules():
    """Get all schedules"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM schedules ORDER BY name')
    rows = c.fetchall()
    conn.close()
    schedules = []
    for row in rows:
        schedules.append({
            'schedule_id': row[0], 'name': row[1], 'cron': row[2],
            'action_type': row[3], 'action_id': row[4], 'enabled': bool(row[5]),
            'last_run': row[6], 'next_run': row[7], 'created_at': row[8]
        })
    return jsonify(schedules)

@app.route('/api/schedules', methods=['POST'])
def create_schedule():
    """Create a schedule"""
    data = request.get_json()
    schedule_id = data.get('schedule_id', f"sched_{int(time.time())}")
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO schedules 
        (schedule_id, name, cron, action_type, action_id, enabled)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (schedule_id, data.get('name', 'New Schedule'), data.get('cron', '0 8 * * *'),
         data.get('action_type', 'scene'), data.get('action_id'), data.get('enabled', True)))
    conn.commit()
    conn.close()
    schedule_runner.update_schedules()
    return jsonify({'success': True, 'schedule_id': schedule_id})

@app.route('/api/schedules/<schedule_id>', methods=['GET'])
def get_schedule(schedule_id):
    """Get a single schedule"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM schedules WHERE schedule_id = ?', (schedule_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Schedule not found'}), 404
    return jsonify({
        'schedule_id': row[0], 'name': row[1], 'cron': row[2],
        'action_type': row[3], 'action_id': row[4], 'enabled': bool(row[5]),
        'last_run': row[6], 'next_run': row[7], 'created_at': row[8]
    })

@app.route('/api/schedules/<schedule_id>', methods=['PUT'])
def update_schedule(schedule_id):
    """Update a schedule"""
    data = request.get_json()
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE schedules SET name=?, cron=?, action_type=?, action_id=?, enabled=? 
        WHERE schedule_id=?''',
        (data.get('name'), data.get('cron'), data.get('action_type'),
         data.get('action_id'), data.get('enabled', True), schedule_id))
    conn.commit()
    conn.close()
    schedule_runner.update_schedules()
    return jsonify({'success': True})

@app.route('/api/schedules/<schedule_id>', methods=['DELETE'])
def delete_schedule(schedule_id):
    """Delete a schedule"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM schedules WHERE schedule_id = ?', (schedule_id,))
    conn.commit()
    conn.close()
    schedule_runner.update_schedules()
    return jsonify({'success': True})

@app.route('/api/schedules/<schedule_id>/trigger', methods=['POST'])
def trigger_schedule(schedule_id):
    """Manually trigger a schedule"""
    return jsonify(schedule_runner.run_schedule(schedule_id))

# Timers Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/timers', methods=['GET'])
def get_timers():
    """Get all timers with current remaining time"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM timers ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()

    timers = []
    now = time.time() * 1000

    for row in rows:
        timer_id, name, duration_ms, remaining_ms, action_type, action_id, status, started_at, created_at = row

        # Calculate actual remaining time for running timers
        actual_remaining = remaining_ms
        if status == 'running' and started_at:
            started_timestamp = datetime.fromisoformat(started_at).timestamp() * 1000
            elapsed = now - started_timestamp
            actual_remaining = max(0, remaining_ms - elapsed)
            if actual_remaining <= 0:
                # Timer completed, update status
                status = 'completed'

        timers.append({
            'timer_id': timer_id,
            'name': name,
            'duration_ms': duration_ms,
            'remaining_ms': int(actual_remaining),
            'action_type': action_type,
            'action_id': action_id,
            'status': status,
            'started_at': started_at,
            'created_at': created_at,
            'running': status == 'running'
        })

    return jsonify(timers)

@app.route('/api/timers', methods=['POST'])
def create_timer():
    """Create a new timer"""
    data = request.get_json() or {}
    timer_id = data.get('timer_id', f"timer_{int(time.time() * 1000)}")
    name = data.get('name', 'New Timer')
    duration_ms = data.get('duration_ms', 60000)  # Default 1 minute

    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO timers
        (timer_id, name, duration_ms, remaining_ms, action_type, action_id, status)
        VALUES (?, ?, ?, ?, ?, ?, 'stopped')''',
        (timer_id, name, duration_ms, duration_ms,
         data.get('action_type'), data.get('action_id')))
    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'timer_id': timer_id,
        'name': name,
        'duration_ms': duration_ms,
        'remaining_ms': duration_ms,
        'status': 'stopped'
    })

@app.route('/api/timers/<timer_id>', methods=['GET'])
def get_timer(timer_id):
    """Get a single timer"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Timer not found'}), 404

    timer_id, name, duration_ms, remaining_ms, action_type, action_id, status, started_at, created_at = row
    return jsonify({
        'timer_id': timer_id,
        'name': name,
        'duration_ms': duration_ms,
        'remaining_ms': remaining_ms,
        'action_type': action_type,
        'action_id': action_id,
        'status': status,
        'started_at': started_at,
        'created_at': created_at
    })

@app.route('/api/timers/<timer_id>', methods=['PUT'])
def update_timer(timer_id):
    """Update a timer"""
    data = request.get_json() or {}
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE timers SET name=?, duration_ms=?, action_type=?, action_id=?
        WHERE timer_id=?''',
        (data.get('name'), data.get('duration_ms'),
         data.get('action_type'), data.get('action_id'), timer_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/timers/<timer_id>', methods=['DELETE'])
def delete_timer(timer_id):
    """Delete a timer"""
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM timers WHERE timer_id = ?', (timer_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/timers/<timer_id>/start', methods=['POST'])
def start_timer(timer_id):
    """Start/resume a timer"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT remaining_ms, status FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    remaining_ms, current_status = row

    if current_status == 'running':
        conn.close()
        return jsonify({'success': True, 'message': 'Timer already running'})

    # If completed, reset to full duration
    if remaining_ms <= 0 or current_status == 'completed':
        c.execute('SELECT duration_ms FROM timers WHERE timer_id = ?', (timer_id,))
        remaining_ms = c.fetchone()[0]

    now = datetime.now().isoformat()
    c.execute('''UPDATE timers SET status='running', started_at=?, remaining_ms=?
        WHERE timer_id=?''', (now, remaining_ms, timer_id))
    conn.commit()
    conn.close()

    # Start background timer check
    timer_runner.start_timer(timer_id, remaining_ms)

    return jsonify({'success': True, 'status': 'running', 'remaining_ms': remaining_ms})

@app.route('/api/timers/<timer_id>/pause', methods=['POST'])
def pause_timer(timer_id):
    """Pause a running timer"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT remaining_ms, started_at, status FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    remaining_ms, started_at, status = row

    if status != 'running':
        conn.close()
        return jsonify({'success': True, 'message': 'Timer not running'})

    # Calculate actual remaining
    if started_at:
        started_timestamp = datetime.fromisoformat(started_at).timestamp() * 1000
        elapsed = (time.time() * 1000) - started_timestamp
        remaining_ms = max(0, remaining_ms - elapsed)

    c.execute('''UPDATE timers SET status='paused', remaining_ms=?, started_at=NULL
        WHERE timer_id=?''', (int(remaining_ms), timer_id))
    conn.commit()
    conn.close()

    timer_runner.stop_timer(timer_id)

    return jsonify({'success': True, 'status': 'paused', 'remaining_ms': int(remaining_ms)})

@app.route('/api/timers/<timer_id>/reset', methods=['POST'])
def reset_timer(timer_id):
    """Reset a timer to its full duration"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT duration_ms FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    duration_ms = row[0]

    c.execute('''UPDATE timers SET status='stopped', remaining_ms=?, started_at=NULL
        WHERE timer_id=?''', (duration_ms, timer_id))
    conn.commit()
    conn.close()

    timer_runner.stop_timer(timer_id)

    return jsonify({'success': True, 'status': 'stopped', 'remaining_ms': duration_ms})

# Legacy Playback Manager Routes (for backward compatibility)
# ─────────────────────────────────────────────────────────
@app.route('/api/playback-manager/status', methods=['GET'])
def playback_manager_status():
    """Legacy: Get playback manager status (use /api/playback/status for unified controller)"""
    return jsonify(playback_manager.get_status())

@app.route('/api/playback-manager/stop', methods=['POST'])
def stop_playback_manager():
    """Legacy: Stop via playback manager (use /api/playback/stop for unified controller)"""
    data = request.get_json() or {}
    return jsonify(content_manager.stop_playback(data.get('universe')))

# ─────────────────────────────────────────────────────────
# Supabase Cloud Sync Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/cloud/status', methods=['GET'])
def get_cloud_status():
    """Get Supabase cloud sync status"""
    if not SUPABASE_AVAILABLE:
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': 'Supabase service not available'
        })

    supabase = get_supabase_service()
    if not supabase:
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': 'Supabase service not initialized'
        })

    return jsonify(supabase.get_status())

@app.route('/api/cloud/sync', methods=['POST'])
def trigger_cloud_sync():
    """Manually trigger a cloud sync"""
    if not SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    # Gather local data
    conn = get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM nodes')
    nodes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM scenes')
    scenes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM chases')
    chases = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM fixtures')
    fixtures = [dict(row) for row in c.fetchall()]

    conn.close()

    # Get looks and sequences
    looks_list = []
    sequences_list = []
    try:
        looks_list = [l.to_dict() for l in looks_sequences_manager.list_looks()]
        sequences_list = [s.to_dict() for s in looks_sequences_manager.list_sequences()]
    except Exception as e:
        print(f"⚠️ Failed to get looks/sequences for sync: {e}")

    # Perform sync
    result = supabase.initial_sync(
        nodes=nodes,
        looks=looks_list,
        sequences=sequences_list,
        scenes=scenes,
        chases=chases,
        fixtures=fixtures
    )

    return jsonify({'success': True, 'result': result})

@app.route('/api/cloud/retry-pending', methods=['POST'])
def retry_pending_sync():
    """Retry pending sync operations"""
    if not SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    result = supabase.retry_pending()
    return jsonify({'success': True, 'result': result})

# ─────────────────────────────────────────────────────────
# Settings Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/settings/all', methods=['GET'])
def get_all_settings():
    return jsonify(app_settings)

@app.route('/api/settings/<category>', methods=['GET'])
def get_settings_category(category):
    return jsonify(app_settings.get(category, {}))

@app.route('/api/settings/<category>', methods=['POST', 'PUT'])
def update_settings_category(category):
    global app_settings
    data = request.get_json()
    if category in app_settings:
        app_settings[category].update(data)
        save_settings(app_settings)
        socketio.emit('settings_update', {'category': category, 'data': app_settings[category]})
        return jsonify({'success': True, category: app_settings[category]})
    return jsonify({'error': 'Category not found'}), 404

@app.route('/api/screen-context', methods=['POST'])
def screen_context():
    data = request.get_json()
    socketio.emit('screen:context', {'page': data.get('page', 'Unknown'),
                                      'action': data.get('action'),
                                      'timestamp': datetime.now().isoformat()})
    return jsonify({'success': True})

# ─────────────────────────────────────────────────────────
# Setup Complete Routes (for browser onboarding persistence)
# ─────────────────────────────────────────────────────────
@app.route('/api/settings/setup-complete', methods=['GET'])
def get_setup_complete():
    """Get setup completion status - shared across all browsers"""
    setup = app_settings.get('setup', {'complete': False})
    return jsonify(setup)

@app.route('/api/settings/setup-complete', methods=['POST'])
def set_setup_complete():
    """Mark setup as complete - persists on server for all browsers"""
    global app_settings
    data = request.get_json() or {}
    if 'setup' not in app_settings:
        app_settings['setup'] = {'complete': False, 'mode': None, 'userProfile': {}}
    app_settings['setup']['complete'] = data.get('complete', True)
    if 'mode' in data:
        app_settings['setup']['mode'] = data['mode']
    if 'userProfile' in data:
        app_settings['setup']['userProfile'].update(data['userProfile'])
    save_settings(app_settings)
    socketio.emit('settings_update', {'category': 'setup', 'data': app_settings['setup']})
    return jsonify({'success': True, 'setup': app_settings['setup']})

@app.route('/api/settings/setup-reset', methods=['POST'])
def reset_setup():
    """Reset setup (for debugging/testing)"""
    global app_settings
    app_settings['setup'] = {'complete': False, 'mode': None, 'userProfile': {}}
    save_settings(app_settings)
    return jsonify({'success': True, 'setup': app_settings['setup']})


# ============================================================
# AI SSOT Routes
# ============================================================
@app.route('/api/ai/preferences', methods=['GET'])
def ai_get_prefs():
    return jsonify(ai_ssot.get_all_preferences())

@app.route('/api/ai/preferences/<key>', methods=['GET', 'POST'])
def ai_pref(key):
    if request.method == 'POST':
        data = request.get_json()
        ai_ssot.set_preference(key, data.get('value'))
        return jsonify({'success': True})
    return jsonify({'value': ai_ssot.get_preference(key)})

@app.route('/api/ai/budget', methods=['GET'])
def ai_budget():
    return jsonify(ai_ssot.check_budget())

@app.route('/api/ai/outcomes', methods=['GET', 'POST'])
def ai_outcomes():
    if request.method == 'POST':
        d = request.get_json()
        ai_ssot.record_outcome(d.get('cid'), d.get('intent'),
            d.get('suggested'), d.get('user_scope'), d.get('action'), d.get('success'))
        return jsonify({'success': True})
    return jsonify({'outcomes': ai_ssot.get_outcomes(limit=50)})

@app.route('/api/ai/audit', methods=['GET'])
def ai_audit():
    return jsonify({'log': ai_ssot.get_audit_log(limit=100)})

@app.route('/api/ai/ops', methods=['GET'])
def ai_ops():
    return jsonify({'ops': ai_ops_registry.list_ops()})
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
    ai_ssot.init_ai_db()

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
                        fixtures=fixtures
                    )
                    print(f"☁️ Startup sync result: {result}")
                except Exception as e:
                    print(f"⚠️ Startup cloud sync failed (non-fatal): {e}")

            # Run sync in background thread (doesn't block startup)
            threading.Thread(target=startup_cloud_sync, daemon=True).start()
            print(f"☁️ Supabase cloud sync enabled - syncing in background...")
        else:
            print("☁️ Supabase not configured - running in local-only mode")

    threading.Thread(target=discovery_listener, daemon=True).start()
    threading.Thread(target=stale_checker, daemon=True).start()
    schedule_runner.start()
    # node_manager.start_dmx_refresh()  # Disabled - UDPJSON is on-demand

    # Initialize Unified Playback Engine
    def unified_output_callback(universe: int, channels: dict, fade_ms: int = 0):
        """Route unified playback output through SSOT"""
        content_manager.set_channels(universe, channels, fade_ms=fade_ms)

    def unified_look_resolver(look_id: str):
        """Resolve Look ID to Look data"""
        try:
            look = looks_sequences_manager.get_look(look_id)
            return look.to_dict() if look else None
        except Exception as e:
            print(f"⚠️ Look resolver error for {look_id}: {e}")
            return None

    unified_engine.set_output_callback(unified_output_callback)
    unified_engine.set_look_resolver(unified_look_resolver)
    unified_engine.set_modifier_renderer(ModifierRenderer())
    unified_engine.start()
    print("✓ Unified Playback Engine started (30 fps)")

    print(f"✓ API server on port {API_PORT}")
    print(f"✓ Discovery on UDP {DISCOVERY_PORT}")
    print(f"✓ UDPJSON DMX output enabled (40 fps refresh, port {AETHER_UDPJSON_PORT})")
    print(f"⚠️ Universe 1 is OFFLINE - use universes 2-5")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=API_PORT, debug=False, allow_unsafe_werkzeug=True)
