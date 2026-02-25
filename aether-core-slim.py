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
import re
import hmac
import hashlib
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

# ── Extracted Modules (Phase 2b refactoring) ──
# These classes were extracted from this file for modularity.
# They use core_registry for cross-module references.
import core_registry as reg
from dmx_state_manager import DMXStateManager
from playback_state import PlaybackManager
from chase_engine_module import ChaseEngine
from arbitration_manager import ArbitrationManager
from schedulers import ScheduleRunner, TimerRunner
from show_engine_module import ShowEngine
from node_manager_module import NodeManager
from rdm_manager_module import RDMDevice, RDMManager
from content_manager_module import ContentManager

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

# [F10] Discovery security configuration
DISCOVERY_SUBNET = os.environ.get('AETHER_NODE_SUBNET', '192.168.50.')  # Allowed source subnet
DISCOVERY_SECRET = os.environ.get('AETHER_DISCOVERY_SECRET', '').encode()  # HMAC shared secret (optional)
DISCOVERY_STRICT_HMAC = os.environ.get('AETHER_DISCOVERY_STRICT', '').lower() == 'true'  # Require HMAC
DISCOVERY_NODE_ID_PATTERN = re.compile(r'^(pulse|gateway|seance|universe)-[0-9A-Fa-f]{4,8}$|^universe-\d+-builtin$')
DISCOVERY_RATE_LIMIT = 10  # Max packets per second per IP
_discovery_stats = {'rejections': 0, 'warnings': 0, 'accepted': 0}  # [F10] Module-level stats

# [F16] Persistent audit log with file rotation
from logging.handlers import RotatingFileHandler
AUDIT_LOG_DIR = os.path.join(os.path.expanduser("~"), "aether-logs")
os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
_audit_logger = logging.getLogger('aether.audit')
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False  # Don't spam console
_audit_handler = RotatingFileHandler(
    os.path.join(AUDIT_LOG_DIR, 'audit.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,              # Keep 5 rotated files (25 MB total)
    encoding='utf-8'
)
_audit_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S'))
_audit_logger.addHandler(_audit_handler)

def audit_log(event_type, **kwargs):
    """[F16] Write a structured audit log entry. Persists to ~/aether-logs/audit.log with rotation."""
    entry = json.dumps({'event': event_type, **kwargs}, separators=(',', ':'))
    _audit_logger.info(entry)

# ── Wire audit/beta logging into registry ──
reg.audit_log = audit_log


# Dynamic paths - works for any user
HOME_DIR = os.path.expanduser("~")
DATABASE = os.path.join(HOME_DIR, "aether-core.db")
SETTINGS_FILE = os.path.join(HOME_DIR, "aether-settings.json")
DMX_STATE_FILE = os.path.join(HOME_DIR, "aether-dmx-state.json")

# ── Populate core_registry with constants (for extracted modules) ──
reg.DMX_STATE_FILE = DMX_STATE_FILE
reg.DATA_DIR = HOME_DIR
reg.DB_PATH = DATABASE
reg.AETHER_UDPJSON_PORT = AETHER_UDPJSON_PORT
reg.AETHER_CONFIG_PORT = WIFI_COMMAND_PORT
reg.AETHER_DISCOVERY_PORT = DISCOVERY_PORT


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

# ── Wire socketio into registry for extracted modules ──
reg.socketio = socketio


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

reg.beta_log = beta_log



# ============================================================
# DMX State - THE SINGLE SOURCE OF TRUTH FOR CHANNEL VALUES
# ============================================================
# [Phase 2b] DMXStateManager class extracted to separate module
dmx_state = DMXStateManager()

# ============================================================
# Playback State Manager
# ============================================================
# [Phase 2b] PlaybackManager class extracted to separate module
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
# [Phase 2b] ChaseEngine class extracted to separate module
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
# [Phase 2b] ArbitrationManager class extracted to separate module
arbitration = ArbitrationManager()

# ============================================================
# Schedule Runner
# ============================================================
# [Phase 2b] ScheduleRunner class extracted to separate module
schedule_runner = ScheduleRunner()

# ============================================================
# Timer Runner (Countdown Timers)
# ============================================================

# [Phase 2b] TimerRunner class extracted to separate module
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
# [Phase 2b] ShowEngine class extracted to separate module
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
    conn = sqlite3.connect(DATABASE, check_same_thread=False, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')       # [F04] WAL for concurrent access
    conn.execute('PRAGMA busy_timeout=15000')       # [F04] Wait up to 15s instead of failing
    conn.execute('PRAGMA synchronous=NORMAL')      # [F04] Safe with WAL, faster than FULL
    conn.execute('PRAGMA cache_size=-8000')         # [F04] 8MB cache per connection
    _db_local.connection = conn
    return _ThreadLocalConnection(conn)

# ── Wire database access into registry for extracted modules ──
reg.get_db = get_db


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

    # [N02 fix] Shows table — timeline-based show playback
    c.execute('''CREATE TABLE IF NOT EXISTS shows (
        show_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
        timeline TEXT, duration_ms INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        distributed BOOLEAN DEFAULT 0
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
# [Phase 2b] NodeManager class extracted to separate module
node_manager = NodeManager()

# ============================================================
# RDM Manager - Remote Device Management
# ============================================================
# [Phase 2b] RDMDevice + RDMManager classes extracted to separate module
rdm_manager = RDMManager()

# ============================================================
# Content Manager
# ============================================================
CHUNK_SIZE = 50
CHUNK_DELAY = 0.05

# [Phase 2b] ContentManager class extracted to separate module
content_manager = ContentManager()

# ── Wire all instances into core_registry ──
# Extracted modules access these via: import core_registry as reg; reg.xxx
reg.dmx_state = dmx_state
reg.playback_manager = playback_manager
reg.chase_engine = chase_engine
reg.arbitration = arbitration
reg.schedule_runner = schedule_runner
reg.timer_runner = timer_runner
reg.show_engine = show_engine
reg.node_manager = node_manager
reg.content_manager = content_manager
reg.effects_engine = effects_engine
reg.merge_layer = merge_layer
reg.channel_classifier = channel_classifier
reg.render_engine = render_engine


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

    # Broadcast stop event to all connected clients [N01 fix]
    try:
        socketio.emit('playback_stopped', {
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
from blueprints.nodes_bp import nodes_bp, init_app as nodes_init
from blueprints.looks_bp import looks_bp, init_app as looks_init
from blueprints.sequences_bp import sequences_bp, init_app as sequences_init
from blueprints.unified_bp import unified_bp, init_app as unified_init
from blueprints.cue_stacks_bp import cue_stacks_bp, init_app as cue_stacks_init
from blueprints.dmx_bp import dmx_bp, init_app as dmx_init
from blueprints.playback_bp import playback_bp, init_app as playback_init

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
ai_init(get_ai_advisor, get_render_pipeline, get_db)
fixtures_init(content_manager, get_db, rdm_manager)
fixture_library_init(fixture_library, content_manager, channel_mapper, looks_sequences_manager, get_db)
node_groups_init(get_db)
rdm_init(rdm_manager, get_db)
preview_init(preview_service)
effects_init(effects_engine, content_manager, unified_engine)
nodes_init(node_manager, get_db, node_submit)
looks_init(looks_sequences_manager, arbitration, render_engine, node_manager,
           unified_play_look, unified_engine, content_manager, cloud_submit,
           SUPABASE_AVAILABLE, get_supabase_service)
sequences_init(looks_sequences_manager, arbitration, render_engine, chase_engine,
               node_manager, unified_play_sequence, unified_stop, unified_get_status,
               unified_engine, cloud_submit, SUPABASE_AVAILABLE, get_supabase_service)
unified_init(unified_engine, session_factory, looks_sequences_manager, dmx_state,
             stop_all_playback, get_db)
cue_stacks_init(cue_stacks_manager, looks_sequences_manager, arbitration, merge_layer,
                merge_layer_output, node_manager)
dmx_init(content_manager, node_manager, dmx_state, stop_all_playback,
         get_render_pipeline, get_db, arbitration, ArbitrationManager,
         effects_engine, chase_engine, show_engine, schedule_runner,
         playback_manager, NodeManager, AETHER_UDPJSON_PORT, WIFI_COMMAND_PORT,
         AETHER_VERSION, AETHER_COMMIT, AETHER_START_TIME, AETHER_FILE_PATH)
playback_init(unified_get_status, unified_pause, unified_resume,
              stop_all_playback, render_engine, playback_manager)

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
app.register_blueprint(nodes_bp)
app.register_blueprint(looks_bp)
app.register_blueprint(sequences_bp)
app.register_blueprint(unified_bp)
app.register_blueprint(cue_stacks_bp)
app.register_blueprint(dmx_bp)
app.register_blueprint(playback_bp)

print(f"[F02] 28 blueprints registered — god file decomposition nearly complete")

# Restore persisted auto-sync state from settings
_autosync_cfg = app_settings.get('autosync', {})
if _autosync_cfg.get('enabled', False):
    app._autosync_enabled = True
    app._autosync_interval = _autosync_cfg.get('interval_minutes', 30)
    from blueprints.system_bp import _start_autosync_thread
    _start_autosync_thread()
    print(f"✓ Auto-sync restored from settings: every {app._autosync_interval} min")

# ============================================================
# Background Services
# ============================================================
def discovery_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', DISCOVERY_PORT))
    sock.settimeout(1.0)

    # [F10] Security state
    _rate_tracker = {}  # {ip: [timestamp, count]}

    security_mode = 'strict-hmac' if DISCOVERY_STRICT_HMAC else ('hmac-optional' if DISCOVERY_SECRET else 'subnet-only')
    print(f"✓ Discovery listening on UDP {DISCOVERY_PORT} [F10 security={security_mode}, subnet={DISCOVERY_SUBNET}]")

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            source_ip = addr[0]

            # [F10] Gate 1: Subnet validation
            if DISCOVERY_SUBNET and not source_ip.startswith(DISCOVERY_SUBNET) and source_ip != '127.0.0.1':
                _discovery_stats['rejections'] += 1
                if _discovery_stats['rejections'] <= 10:  # Don't flood logs
                    print(f"[F10 REJECT] Discovery from outside subnet: {source_ip}")
                    audit_log('discovery_reject', reason='subnet', ip=source_ip)  # [F16]
                continue

            # [F10] Gate 2: Rate limiting per IP
            now_mono = time.monotonic()
            ip_rate = _rate_tracker.get(source_ip)
            if ip_rate and now_mono - ip_rate[0] < 1.0:
                ip_rate[1] += 1
                if ip_rate[1] > DISCOVERY_RATE_LIMIT:
                    if ip_rate[1] == DISCOVERY_RATE_LIMIT + 1:
                        print(f"[F10 RATE] Throttling discovery from {source_ip} (>{DISCOVERY_RATE_LIMIT}/sec)")
                    _discovery_stats['rejections'] += 1
                    continue
            else:
                _rate_tracker[source_ip] = [now_mono, 1]

            msg = json.loads(data.decode())

            # [F10] Gate 3: HMAC validation (if secret configured)
            if DISCOVERY_SECRET:
                token = msg.pop('hmac', None)
                # Compute HMAC over the raw JSON minus the hmac field
                check_data = json.dumps({k: v for k, v in msg.items()}, separators=(',', ':')).encode()
                if token:
                    expected = hmac.new(DISCOVERY_SECRET, check_data, hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(expected, token):
                        _discovery_stats['rejections'] += 1
                        print(f"[F10 REJECT] Bad HMAC from {source_ip} node_id={msg.get('node_id')}")
                        continue
                elif DISCOVERY_STRICT_HMAC:
                    _discovery_stats['rejections'] += 1
                    print(f"[F10 REJECT] Missing HMAC (strict mode) from {source_ip} node_id={msg.get('node_id')}")
                    continue
                else:
                    _discovery_stats['warnings'] += 1
                    if _discovery_stats['warnings'] <= 20:
                        print(f"[F10 WARN] No HMAC from {source_ip} node_id={msg.get('node_id')} — accepting (non-strict)")

            # [F10] Gate 4: node_id format validation
            node_id = msg.get('node_id', '')
            if not DISCOVERY_NODE_ID_PATTERN.match(str(node_id)):
                _discovery_stats['rejections'] += 1
                print(f"[F10 REJECT] Malformed node_id from {source_ip}: '{node_id}'")
                audit_log('discovery_reject', reason='malformed_id', ip=source_ip, node_id=str(node_id))  # [F16]
                continue

            msg['ip'] = source_ip
            msg_type = msg.get('type', 'unknown')
            if msg_type in ('register', 'heartbeat'):
                _discovery_stats['accepted'] += 1
                # Retry register_node up to 3 times on DB lock
                for _attempt in range(3):
                    try:
                        node_manager.register_node(msg)
                        break
                    except Exception as db_err:
                        if 'database is locked' in str(db_err) and _attempt < 2:
                            time.sleep(0.1 * (_attempt + 1))
                            continue
                        raise
                # Report heartbeat to Trust Enforcer (Phase 4 Lane 3)
                report_node_heartbeat(node_id, {
                    'ip': source_ip,
                    'rssi': msg.get('rssi'),
                    'uptime': msg.get('uptime'),
                    'stale': msg.get('stale', False),
                    'type': msg_type
                })
                if msg_type == 'register':
                    print(f"📥 Node registered: {msg.get('hostname', 'Unknown')} @ {source_ip}")
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

def supabase_retry_loop():
    """[F19] Periodically retry pending Supabase operations."""
    while True:
        time.sleep(60)  # Retry every 60 seconds
        if SUPABASE_AVAILABLE:
            try:
                supabase = get_supabase_service()
                if supabase and supabase.is_enabled():
                    result = supabase.retry_pending()
                    if result and result.get('completed', 0) > 0:
                        print(f"☁️ [F19] Supabase retry: {result['completed']} synced, {result.get('remaining', 0)} pending")
            except Exception as e:
                pass  # Silent — don't crash background thread

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
        'udp_delivery': node_manager.get_delivery_stats(),
        'discovery_security': {
            'mode': 'strict-hmac' if DISCOVERY_STRICT_HMAC else ('hmac-optional' if DISCOVERY_SECRET else 'subnet-only'),
            'subnet': DISCOVERY_SUBNET,
            'accepted': _discovery_stats['accepted'],
            'rejections': _discovery_stats['rejections'],
            'warnings': _discovery_stats['warnings'],
        },
        'audit_log': os.path.join(AUDIT_LOG_DIR, 'audit.log'),
    })

@app.route('/api/audit', methods=['GET'])
def get_audit_log():
    """[F16] Read recent audit log entries. ?lines=N (default 100)"""
    lines_requested = request.args.get('lines', 100, type=int)
    lines_requested = min(lines_requested, 1000)  # Cap at 1000
    log_path = os.path.join(AUDIT_LOG_DIR, 'audit.log')
    if not os.path.exists(log_path):
        return jsonify({'entries': [], 'total_lines': 0})
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        recent = all_lines[-lines_requested:] if len(all_lines) > lines_requested else all_lines
        entries = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            # Format: "2026-02-18T02:30:00 {json}"
            parts = line.split(' ', 1)
            if len(parts) == 2:
                try:
                    entry = json.loads(parts[1])
                    entry['_timestamp'] = parts[0]
                    entries.append(entry)
                except json.JSONDecodeError:
                    entries.append({'_timestamp': parts[0], 'raw': parts[1]})
        return jsonify({'entries': entries, 'total_lines': len(all_lines)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# [F02] Node routes moved to blueprints/nodes_bp.py (10 routes)


# [F02] Node groups routes moved to blueprints/node_groups_bp.py (10 routes)

# [F02] RDM node-scoped routes moved to blueprints/rdm_bp.py (3 routes)

# [F02] DMX routes moved to blueprints/dmx_bp.py (9 routes)

# [F02] Pixel Array route handlers removed — see blueprints/pixel_bp.py

# [F02] Scene routes moved to blueprints/scenes_bp.py (6 routes)

# [F02] Chase routes moved to blueprints/chases_bp.py (8 routes)

# [F02] Looks routes moved to blueprints/looks_bp.py (10 routes)
# [F02] Render routes moved to blueprints/playback_bp.py (2 routes)
# [F02] Sequences routes moved to blueprints/sequences_bp.py (9 routes)
# [F02] Unified routes moved to blueprints/unified_bp.py (12 routes)
# [F02] Playback routes moved to blueprints/playback_bp.py (4 routes)
# [F02] Cue Stacks routes moved to blueprints/cue_stacks_bp.py (10 routes)
# [F02] Playback Manager routes moved to blueprints/playback_bp.py (2 routes)

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
    threading.Thread(target=supabase_retry_loop, daemon=True).start()  # [F19]
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
    print("✓ Unified Playback Engine started (40 fps)")  # [F17] Aligned with DMX refresh

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
        # [N07 fix] Stop all playback engines before saving state
        try:
            effects_engine.stop_effect()
            print("  ✓ Effects stopped", flush=True)
        except Exception:
            pass
        try:
            chase_engine.stop_all()
            print("  ✓ Chases stopped", flush=True)
        except Exception:
            pass
        try:
            show_engine.stop()
            print("  ✓ Shows stopped", flush=True)
        except Exception:
            pass
        try:
            unified_engine.stop_all()
            print("  ✓ Unified playback stopped", flush=True)
        except Exception:
            pass
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
