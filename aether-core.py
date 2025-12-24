#!/usr/bin/env python3
"""
AETHER Core v3.0 - Unified Control System with Local Playback
Single source of truth for ALL system functionality
Features:
- Scene/Chase sync to ESP32 nodes for local playback
- Universe splitting (multiple nodes per universe)
- Coordinated play/stop across all nodes
"""

import socket
import json
import sqlite3
import serial
import threading
import time
import os
import subprocess
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import ai_ssot
import ai_ops_registry

# ============================================================
# Configuration - Dynamic paths based on user home directory
# ============================================================
API_PORT = 8891
DISCOVERY_PORT = 9999
WIFI_COMMAND_PORT = 8888

# Dynamic paths - works for any user
HOME_DIR = os.path.expanduser("~")
DATABASE = os.path.join(HOME_DIR, "aether-core.db")
SETTINGS_FILE = os.path.join(HOME_DIR, "aether-settings.json")
DMX_STATE_FILE = os.path.join(HOME_DIR, "aether-dmx-state.json")

# ============================================================
# Version/Runtime Info - For SSOT verification
# ============================================================
AETHER_VERSION = "3.1.0"
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
    except:
        pass
    return "unknown"

AETHER_COMMIT = get_git_commit()

# Serial port for hardwired node
HARDWIRED_UART = "/dev/serial0"
HARDWIRED_BAUD = 115200

# Timing configuration
STALE_TIMEOUT = 60
CHUNK_SIZE = 5  # Max channels per UDP packet
CHUNK_DELAY = 0.05  # Delay between chunks (50ms)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
    """Manages DMX state for all universes - this is the SSOT for channel values"""
    def __init__(self):
        self.universes = {}  # {universe_num: [512 values]}
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
                except:
                    pass
                
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

    def set_channels(self, universe, channels_dict):
        """Update specific channels, preserving others"""
        with self.lock:
            if universe not in self.universes:
                self.universes[universe] = [0] * 512
            for ch_str, value in channels_dict.items():
                ch = int(ch_str)
                if 1 <= ch <= 512:
                    self.universes[universe][ch - 1] = int(value)
        socketio.emit('dmx_state', {
            'universe': universe,
            'channels': self.universes[universe]
        })
        self._schedule_save()

    def blackout(self, universe):
        """Set all channels to 0"""
        with self.lock:
            self.universes[universe] = [0] * 512
        socketio.emit('dmx_state', {
            'universe': universe,
            'channels': self.universes[universe]
        })
        self._schedule_save()

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
            if universe:
                return self.current.get(universe)
            return self.current.copy()

playback_manager = PlaybackManager()

# ============================================================
# Chase Playback Engine (streams steps via OLA/sACN)
# ============================================================
class ChaseEngine:
    """Runs chases by streaming each step via OLA to all universes"""
    def __init__(self):
        self.lock = threading.Lock()
        self.running_chases = {}  # {chase_id: thread}
        self.stop_flags = {}  # {chase_id: Event}
        # Health tracking for debugging
        self.chase_health = {}  # {chase_id: {"step": int, "last_time": float, "status": str}}

    def start_chase(self, chase, universes):
        """Start a chase on the given universes"""
        chase_id = chase['chase_id']

        # Stop if already running
        self.stop_chase(chase_id)

        # Create stop flag
        stop_flag = threading.Event()
        self.stop_flags[chase_id] = stop_flag

        # Start chase thread
        thread = threading.Thread(
            target=self._run_chase,
            args=(chase, universes, stop_flag),
            daemon=True
        )
        self.running_chases[chase_id] = thread
        thread.start()
        print(f"🏃 Chase engine started: {chase['name']}", flush=True)

    def stop_chase(self, chase_id=None):
        """Stop a chase or all chases"""
        with self.lock:
            if chase_id:
                if chase_id in self.stop_flags:
                    self.stop_flags[chase_id].set()
                    self.stop_flags.pop(chase_id, None)
                    self.running_chases.pop(chase_id, None)
            else:
                # Stop all
                for flag in self.stop_flags.values():
                    flag.set()
                self.stop_flags.clear()
                self.running_chases.clear()

    def stop_all(self):
        """Stop all running chases"""
        self.stop_chase(None)

    def _run_chase(self, chase, universes, stop_flag):
        """Chase playback loop - runs in background thread"""
        chase_id = chase['chase_id']
        steps = chase.get('steps', [])
        bpm = chase.get('bpm', 120)
        loop = chase.get('loop', True)
        chase_fade_ms = chase.get('fade_ms', 0)  # Global fade for chase

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
                # Use step duration if available, otherwise fall back to BPM timing
                step_duration_ms = step.get('duration', int(default_interval * 1000))
                # Use step fade or chase fade
                fade_ms = step.get('fade_ms', chase_fade_ms)

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

                # Send step to all universes with fade
                for univ in universes:
                    try:
                        self._send_step(univ, channels, fade_ms)
                    except Exception as e:
                        print(f"❌ Chase step send error (U{univ}): {e}", flush=True)
                        import traceback
                        traceback.print_exc()

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

    def _send_step(self, universe, channels, fade_ms=0):
        """Send a single chase step - routes through SSOT (same path as /api/dmx/set)"""
        if not channels:
            print(f"  ⚠️ _send_step: no channels to send for U{universe}", flush=True)
            return
        parsed_channels = {}
        for key, value in channels.items():
            key_str = str(key)
            if ':' in key_str:
                # Format: "universe:channel" -> only apply to matching universe
                parts = key_str.split(':')
                ch_univ = int(parts[0])
                ch_num = int(parts[1])
                if ch_univ == universe:
                    parsed_channels[ch_num] = value
            else:
                # Simple channel number - apply to all universes
                parsed_channels[int(key_str)] = value
        if not parsed_channels:
            print(f"  ⚠️ _send_step: no channels matched U{universe} after parsing", flush=True)
            return

        # SSOT trace: log what we're about to send
        sample_ch = list(parsed_channels.items())[:3]  # First 3 for brevity
        print(f"  📤 _send_step -> SSOT: U{universe}, {len(parsed_channels)} ch, fade={fade_ms}ms, sample={sample_ch}", flush=True)

        # Route through SSOT - same function as /api/dmx/set
        result = content_manager.set_channels(universe, parsed_channels, fade_ms=fade_ms)

        # Log result
        if result.get('success'):
            print(f"  ✓ _send_step: SSOT accepted, {len(result.get('results', []))} nodes updated", flush=True)
        else:
            print(f"  ❌ _send_step: SSOT failed: {result.get('error', 'unknown')}", flush=True)

chase_engine = ChaseEngine()
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
    "security": {"pinEnabled": False, "sessionTimeout": 3600}
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
        mode TEXT DEFAULT 'output', type TEXT DEFAULT 'wifi', connection TEXT, firmware TEXT,
        status TEXT DEFAULT 'offline', is_builtin BOOLEAN DEFAULT 0, is_paired BOOLEAN DEFAULT 0,
        can_delete BOOLEAN DEFAULT 1, uptime INTEGER DEFAULT 0, rssi INTEGER DEFAULT 0, fps REAL DEFAULT 0,
        last_seen TIMESTAMP, first_seen TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS scenes (
        scene_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, universe INTEGER DEFAULT 1,
        channels TEXT, fade_ms INTEGER DEFAULT 500, curve TEXT DEFAULT 'linear', color TEXT DEFAULT '#3b82f6',
        icon TEXT DEFAULT 'lightbulb', is_favorite BOOLEAN DEFAULT 0, play_count INTEGER DEFAULT 0,
        synced_to_nodes BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS chases (
        chase_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, universe INTEGER DEFAULT 1,
        bpm INTEGER DEFAULT 120, loop BOOLEAN DEFAULT 1, steps TEXT, color TEXT DEFAULT '#10b981',
        fade_ms INTEGER DEFAULT 0,
        synced_to_nodes BOOLEAN DEFAULT 0,
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()

    # Add synced_to_nodes column if missing (migration)
    try:
        c.execute('ALTER TABLE scenes ADD COLUMN synced_to_nodes BOOLEAN DEFAULT 0')
        conn.commit()
    except:
        pass
    try:
        c.execute('ALTER TABLE chases ADD COLUMN synced_to_nodes BOOLEAN DEFAULT 0')
        conn.commit()
    except:
        pass

    # Add fade_ms column to chases table for smooth transitions
    try:
        c.execute('ALTER TABLE chases ADD COLUMN fade_ms INTEGER DEFAULT 0')
        conn.commit()
        print("✓ Added fade_ms column to chases table")
    except:
        pass  # Column already exists


    # Add node_id to fixtures (which Pulse node outputs this fixture)
    try:
        c.execute('ALTER TABLE fixtures ADD COLUMN node_id TEXT')
        conn.commit()
    except:
        pass
    # Add built-in hardwired node
    # Add fixture_ids to groups
    try:
        c.execute('ALTER TABLE groups ADD COLUMN fixture_ids TEXT')
        conn.commit()
    except:
        pass
    c.execute('SELECT * FROM nodes WHERE node_id = ?', ('universe-1-builtin',))
    if not c.fetchone():
        c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start, type, channel_end,
            mode, type, connection, firmware, status, is_builtin, is_paired, can_delete, first_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ('universe-1-builtin', 'Universe 1 (Built-in)', 'aether-pi', 'UART', 'localhost', 1, 1, 512,
             'output', 'hardwired', 'Serial UART', 'AETHER v5.1', 'online', True, True, False, datetime.now().isoformat()))
        conn.commit()

    print("✓ Database initialized")
    conn.close()

# ============================================================
# Node Manager
# ============================================================
class NodeManager:
    # Packet protocol version - increment if packet format changes
    PACKET_VERSION = 3  # v3: full 512-ch frames, ESP32 firmware v1.1 has 2500-byte buffer

    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.lock = threading.Lock()
        self._serial = None
        # Diagnostics tracking
        self._last_udp_send = None
        self._last_uart_send = None
        self._udp_send_count = 0
        self._uart_send_count = 0

    def _get_serial(self):
        if self._serial is None or not self._serial.is_open:
            try:
                self._serial = serial.Serial(HARDWIRED_UART, HARDWIRED_BAUD, timeout=0.1)
                print(f"✓ Serial connected: {HARDWIRED_UART} @ {HARDWIRED_BAUD}")
            except Exception as e:
                print(f"❌ Serial connection failed: {e}")
                self._serial = None
        return self._serial

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
        """Get all paired/builtin online nodes in a universe"""
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT * FROM nodes WHERE universe = ? AND (is_paired = 1 OR is_builtin = 1)
                     AND status = "online" ORDER BY channel_start''', (universe,))
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
        if existing:
            # Check if node was offline before updating
            c.execute('SELECT status FROM nodes WHERE node_id = ?', (node_id,))
            row = c.fetchone()
            was_offline = row and row[0] == 'offline'
            
            c.execute('''UPDATE nodes SET hostname = COALESCE(?, hostname), mac = COALESCE(?, mac),
                ip = COALESCE(?, ip), uptime = COALESCE(?, uptime), rssi = COALESCE(?, rssi),
                fps = COALESCE(?, fps), firmware = COALESCE(?, firmware), status = 'online', last_seen = ?
                WHERE node_id = ?''',
                (data.get('hostname'), data.get('mac'), data.get('ip'), data.get('uptime'),
                 data.get('rssi'), data.get('fps'), data.get('version'), now, node_id))
        else:
            c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start, type,
                channel_end, status, is_paired, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, 'wifi', ?, ?, ?, ?, ?)''',
                (node_id, data.get('hostname', f'Node-{node_id[-4:]}'), data.get('hostname'),
                 data.get('mac'), data.get('ip'), data.get('universe', 1), data.get('startChannel', 1),
                 data.get('channelCount', 512), 'online', False, now, now))
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
                'channel_end': node.get('channel_end', 512)
            })
        self.broadcast_status()
        return node

    def pair_node(self, node_id, config):
        conn = get_db()
        c = conn.cursor()
        c.execute('''UPDATE nodes SET name = COALESCE(?, name), universe = ?, channel_start = ?,
            channel_end = ?, mode = COALESCE(?, 'output'), is_paired = 1 WHERE node_id = ?''',
            (config.get('name'), config.get('universe', 1), config.get('channel_start', 1),
             config.get('channel_end', 512), config.get('mode'), str(node_id)))
        conn.commit()
        conn.close()
        
        # Send config to node
        node = self.get_node(node_id)
        if node and node.get('type') == 'wifi':
            self.send_config_to_node(node, config)
            # Sync all content to newly paired node
            self.sync_content_to_node(node)
        
        self.broadcast_status()
        return node

    def unpair_node(self, node_id):
        # Get node info before updating DB
        node = self.get_node(node_id)

        # Send unpair command to WiFi node to clear its config
        if node and node.get('type') == 'wifi' and node.get('ip'):
            self.send_command_to_wifi(node['ip'], {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({node['ip']})")

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
            self.send_command_to_wifi(node['ip'], {'cmd': 'unpair'})
            print(f"📤 Unpair sent to {node.get('name', node_id)} ({node['ip']})")

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
    # Send Commands to Nodes
    # ─────────────────────────────────────────────────────────
    def send_to_node(self, node, channels_dict, fade_ms=0):
        """Send DMX values to a node"""
        universe = node.get("universe", 1)
        if node.get('type') == 'hardwired' or node.get('is_builtin'):
            return self.send_to_hardwired(universe, channels_dict, fade_ms)
        else:
            # WiFi nodes - send UDP JSON command with fade support
            ip = node.get('ip')
            if not ip:
                print(f"⚠️ No IP for WiFi node {node.get('name')}")
                return False

            # Get full 512-channel universe from SSOT
            # NOTE: ESP32 firmware v1.1+ has 2500-byte buffer, can handle full frames
            data = dmx_state.get_universe(universe)

            # Apply any new channel updates
            if channels_dict:
                for ch_str, value in channels_dict.items():
                    ch = int(ch_str)
                    if 1 <= ch <= 512:
                        data[ch - 1] = int(value)

            esp_cmd = {"cmd": "scene", "ch": 1, "data": data}
            if fade_ms > 0:
                esp_cmd["fade"] = fade_ms

            # Log packet size for debugging
            packet_json = json.dumps(esp_cmd)
            packet_size = len(packet_json)

            print(f"📤 UDP -> {node.get('name')} ({ip}): 512 ch, {packet_size} bytes, fade={fade_ms}ms")

            # Track for diagnostics
            self._last_udp_send = {
                'time': datetime.now().isoformat(),
                'node': node.get('name'),
                'ip': ip,
                'channels': 512,
                'packet_size': packet_size,
                'fade_ms': fade_ms
            }
            self._udp_send_count += 1

            return self.send_command_to_wifi(ip, esp_cmd)

    def send_to_hardwired(self, universe, channels_dict, fade_ms=0):
        """Send command to hardwired ESP32 via UART - always sends full 512-channel frame"""
        try:
            ser = self._get_serial()
            if ser is None:
                return False

            # Get full 512-channel universe from SSOT
            data = dmx_state.get_universe(universe)

            # Apply any new channel updates
            if channels_dict:
                for ch_str, value in channels_dict.items():
                    ch = int(ch_str)
                    if 1 <= ch <= 512:
                        data[ch - 1] = int(value)

            esp_cmd = {"cmd": "scene", "ch": 1, "data": data}
            if fade_ms > 0:
                esp_cmd["fade"] = fade_ms

            json_cmd = json.dumps(esp_cmd) + '\n'
            ser.write(json_cmd.encode())
            ser.flush()
            print(f"📤 UART -> 512 channels (full frame), fade={fade_ms}ms")

            # Track for diagnostics
            self._last_uart_send = {
                'time': datetime.now().isoformat(),
                'universe': universe,
                'channels': 512,
                'packet_size': len(json_cmd),
                'fade_ms': fade_ms
            }
            self._uart_send_count += 1
            return True

        except Exception as e:
            print(f"❌ UART error: {e}")
            self._serial = None
            return False

    def send_to_wifi(self, ip, channels_dict, fade_ms=0):
        """Send DMX via OLA sACN - wireless nodes listen to their universe"""
        # This method is now deprecated - we use send_via_ola instead
        return True

    def send_via_ola(self, universe, channels_dict):
        """Send DMX to a universe via OLA (sACN output)"""
        try:
            # Build full 512 channel array from current state
            current = dmx_state.get_universe(universe)
            
            # Apply changes
            for ch_str, value in channels_dict.items():
                ch = int(ch_str)
                if 1 <= ch <= 512:
                    current[ch - 1] = int(value)
            
            # Send via OLA CLI
            data_str = ','.join(str(v) for v in current)
            result = subprocess.run(
                ['ola_set_dmx', '-u', str(universe), '-d', data_str],
                capture_output=True, text=True, timeout=2
            )
            
            if result.returncode == 0:
                print(f"📤 OLA U{universe} -> {len(channels_dict)} channels")
                return True
            else:
                print(f"❌ OLA error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ OLA error: {e}")
            return False

    def send_command_to_wifi(self, ip, command):
        """Send any command to WiFi node"""
        try:
            json_data = json.dumps(command)
            self.udp_socket.sendto(json_data.encode(), (ip, WIFI_COMMAND_PORT))
            return True
        except Exception as e:
            print(f"❌ UDP command error to {ip}: {e}")
            return False

    def send_blackout(self, node, fade_ms=1000):
        """Send blackout command to a node"""
        if node.get('type') == 'hardwired' or node.get('is_builtin'):
            try:
                ser = self._get_serial()
                if ser:
                    esp_cmd = {"cmd": "blackout"}
                    if fade_ms > 0:
                        esp_cmd["fade"] = fade_ms
                    ser.write((json.dumps(esp_cmd) + '\n').encode())
                    ser.flush()
                    return True
            except Exception as e:
                print(f"❌ Blackout error: {e}")
                return False
        else:
            # WiFi nodes use OLA/sACN - send all zeros
            universe = node.get('universe', 1)
            all_zeros = {str(ch): 0 for ch in range(1, 513)}
            return self.send_via_ola(universe, all_zeros)

    def send_config_to_node(self, node, config):
        """Send configuration update to a WiFi node"""
        if node.get('type') != 'wifi':
            return False
        
        universe = config.get('universe', node.get('universe', 1))
        
        # Send config to ESP32
        command = {
            'cmd': 'config',
            'name': config.get('name', node.get('name')),
            'universe': universe,
            'channel_start': config.get('channel_start', node.get('channel_start', 1)),
            'channel_end': config.get('channel_end', node.get('channel_end', 512))
        }
        result = self.send_command_to_wifi(node['ip'], command)
        
        # Auto-configure OLA universe
        self.configure_ola_universe(universe)
        
        return result

    def configure_ola_universe(self, universe):
        """Ensure OLA has this universe configured for E1.31 output"""
        try:
            # Get E1.31 device info
            result = subprocess.run(
                ['ola_dev_info'],
                capture_output=True, text=True, timeout=5
            )
            
            # Find E1.31 device ID
            e131_device = None
            for line in result.stdout.split('\n'):
                if 'E1.31' in line and 'Device' in line:
                    parts = line.split(':')
                    if parts:
                        dev_part = parts[0].replace('Device', '').strip()
                        try:
                            e131_device = int(dev_part)
                        except:
                            pass
                    break
            
            if e131_device is None:
                print(f"⚠️ E1.31 device not found in OLA")
                return False
            
            # Patch universe to E1.31 output
            patch_result = subprocess.run(
                ['ola_patch', '-d', str(e131_device), '-p', str(universe - 1), '-u', str(universe)],
                capture_output=True, text=True, timeout=5
            )
            
            if patch_result.returncode == 0:
                print(f"✓ OLA universe {universe} patched to E1.31")
                return True
            else:
                print(f"✓ OLA universe {universe} likely already configured")
                return True
                
        except Exception as e:
            print(f"❌ OLA config error: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    # Sync Content to Nodes (Scenes/Chases)
    # ─────────────────────────────────────────────────────────
    def sync_scene_to_node(self, node, scene):
        """Send a scene to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False
        
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
            self.send_command_to_wifi(node['ip'], meta_cmd)
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
                self.send_command_to_wifi(node['ip'], chunk_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']} (chunked)")
        else:
            self.send_command_to_wifi(node['ip'], command)
            print(f"  📤 Scene '{scene['name']}' -> {node['name']}")
        
        return True

    def sync_chase_to_node(self, node, chase):
        """Send a chase to a WiFi node for local storage"""
        if node.get('type') != 'wifi':
            return False
        
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
            self.send_command_to_wifi(node['ip'], meta_cmd)
            time.sleep(CHUNK_DELAY)
            
            # Send steps in batches
            for i in range(0, len(filtered_steps), 5):
                batch_steps = filtered_steps[i:i+5]
                batch_cmd = {
                    'cmd': 'append_chase_steps',
                    'id': chase['chase_id'],
                    'steps': batch_steps
                }
                self.send_command_to_wifi(node['ip'], batch_cmd)
                time.sleep(CHUNK_DELAY)
            print(f"  📤 Chase '{chase['name']}' -> {node['name']} (chunked, {len(filtered_steps)} steps)")
        else:
            self.send_command_to_wifi(node['ip'], command)
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
                command = {'cmd': 'play_scene', 'id': scene_id}
                if fade_ms is not None:
                    command['fade_ms'] = fade_ms
                success = self.send_command_to_wifi(node['ip'], command)
                results.append({'node': node['name'], 'success': success})
        
        return results

    def play_chase_on_nodes(self, universe, chase_id):
        """Tell all nodes in universe to play a stored chase"""
        nodes = self.get_nodes_in_universe(universe)
        results = []
        
        for node in nodes:
            if node.get('type') == 'wifi':
                command = {'cmd': 'play_chase', 'id': chase_id}
                success = self.send_command_to_wifi(node['ip'], command)
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
                success = self.send_command_to_wifi(node['ip'], {'cmd': 'stop'})
                results.append({'node': node['name'], 'success': success})
        
        return results

node_manager = NodeManager()

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
        """Set DMX channels - updates state and sends to nodes"""
        import sys
        print(f"🎛️ set_channels: universe={universe}, channels={len(channels)}, fade={fade_ms}", flush=True)
        dmx_state.set_channels(universe, channels)
        nodes = node_manager.get_nodes_in_universe(universe)
        print(f"📍 Found {len(nodes)} nodes in universe {universe}", flush=True)

        if not nodes:
            print(f"⚠️ No online nodes in universe {universe}", flush=True)
            return {'success': False, 'error': 'No nodes online'}

        results = []
        for node in nodes:
            print(f"  → Processing node: {node['name']} ({node.get('type', 'unknown')})", flush=True)
            local_channels = node_manager.translate_channels_for_node(node, channels)
            if local_channels:
                print(f"    Translated {len(local_channels)} channels for {node['name']}", flush=True)
                success = node_manager.send_to_node(node, local_channels, fade_ms)
                print(f"    Send result: {success}", flush=True)
                results.append({'node': node['name'], 'success': success})

        return {'success': True, 'results': results}

    def blackout(self, universe=None, fade_ms=1000):
        """Blackout all channels - if universe is None, blackout ALL universes"""
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
            dmx_state.blackout(univ)
            playback_manager.stop(univ)
            nodes = node_manager.get_nodes_in_universe(univ)
            for node in nodes:
                success = node_manager.send_blackout(node, fade_ms)
                results.append({'node': node['name'], 'success': success})

        if hasattr(self, 'current_playback'):
            self.current_playback = {"type": None, "id": None, "universe": None}

        beta_log("blackout_complete", {
            "dispatch_targets_final": sorted(universes_to_blackout),
            "playback_state_after": {"type": None, "id": None, "universe": None}
        })

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

    def play_scene(self, scene_id, fade_ms=None, use_local=True, target_channels=None, universe=None, skip_ssot=False):
        """Play a scene - broadcasts to ALL online nodes across all universes"""
        print(f"▶️ play_scene called: scene_id={scene_id}", flush=True)
        scene = self.get_scene(scene_id)
        if not scene:
            return {'success': False, 'error': 'Scene not found'}

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
                chase_engine.stop_all()
                time.sleep(0.15)
                self.current_playback = {'type': 'scene', 'id': scene_id, 'universe': universe}
                print(f"✓ SSOT: Now playing scene '{scene_id}'", flush=True)

        fade = fade_ms if fade_ms is not None else scene.get('fade_ms', 500)
        channels_to_apply = scene['channels']
        if target_channels:
            target_set = set(target_channels)
            channels_to_apply = {k: v for k, v in scene['channels'].items() if int(k) in target_set}

        if universe is not None:
            universes_with_nodes = {universe} if universe in all_universes else set()
        else:
            universes_with_nodes = all_universes

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

        all_results = []
        for univ in universes_with_nodes:
            if target_channels is None:
                current = playback_manager.get_status(univ)
                if current and current.get('type') == 'chase':
                    print(f"⏹️ Stopping chase on U{univ} before scene play", flush=True)
                    node_manager.stop_playback_on_nodes(univ)
                playback_manager.set_playing(univ, 'scene', scene_id)
            result = self.set_channels(univ, channels_to_apply, fade)
            all_results.extend(result.get('results', []))
            dmx_state.set_channels(univ, channels_to_apply)

        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE scenes SET play_count = play_count + 1 WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()

        return {'success': True, 'results': all_results}

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
            steps, color, fade_ms, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (chase_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, data.get('bpm', 120), data.get('loop', True),
             json.dumps(steps), data.get('color', '#10b981'), data.get('fade_ms', 0), False, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        # Sync to nodes
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
        
        socketio.emit('chases_update', {'chases': self.get_chases()})
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

    def play_chase(self, chase_id, target_channels=None, universe=None):
        """Start chase playback - streams steps via OLA to ALL online nodes"""
        chase = self.get_chase(chase_id)
        if not chase:
            return {'success': False, 'error': 'Chase not found'}

        # Get ALL online nodes and their universes
        all_nodes = node_manager.get_all_nodes(include_offline=False)
        universes_with_nodes = list(set(node.get('universe', 1) for node in all_nodes))
        print(f"🎬 Playing chase '{chase['name']}' on universes: {sorted(universes_with_nodes)}", flush=True)

        # SSOT: Acquire lock and stop everything cleanly
        with self.ssot_lock:
            print(f"🔒 SSOT Lock - stopping all before chase", flush=True)
            try:
                show_engine.stop()
            except Exception as e:
                print(f"⚠️ Show stop: {e}", flush=True)
            chase_engine.stop_all()
            for univ in universes_with_nodes:
                playback_manager.stop(univ)
            time.sleep(0.15)
            self.current_playback = {'type': 'chase', 'id': chase_id, 'universe': universe}
            print(f"✓ SSOT: Now playing chase '{chase_id}'", flush=True)


        # Set playback state for all universes
        for univ in universes_with_nodes:
            playback_manager.set_playing(univ, 'chase', chase_id)

        # Start chase engine (streams steps via OLA)
        chase_engine.start_chase(chase, universes_with_nodes)

        return {'success': True, 'universes': universes_with_nodes}

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
            universe, start_channel, channel_count, channel_map, color, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (fixture_id, data.get('name', 'Untitled Fixture'), data.get('type', 'generic'),
             data.get('manufacturer', ''), data.get('model', ''),
             data.get('universe', 1), data.get('start_channel', 1), data.get('channel_count', 1),
             json.dumps(channel_map), data.get('color', '#8b5cf6'),
             data.get('notes', ''), datetime.now().isoformat()))
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
        'services': {'database': True, 'discovery': True,
                     'serial': node_manager._serial is not None and node_manager._serial.is_open}
    })

@app.route('/api/version', methods=['GET'])
def version():
    """Get version info for SSOT verification - confirms which backend file is running"""
    uptime_seconds = (datetime.now() - AETHER_START_TIME).total_seconds()
    return jsonify({
        'version': AETHER_VERSION,
        'commit': AETHER_COMMIT,
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
    except:
        pass

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
    except:
        pass

    try:
        # CPU temperature - Raspberry Pi specific
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats['cpu_temp'] = int(f.read().strip()) / 1000.0
    except:
        pass

    try:
        # Disk usage
        statvfs = os.statvfs('/')
        stats['disk_total'] = statvfs.f_blocks * statvfs.f_frsize
        stats['disk_used'] = (statvfs.f_blocks - statvfs.f_bfree) * statvfs.f_frsize
    except:
        pass

    try:
        # Uptime
        with open('/proc/uptime', 'r') as f:
            stats['uptime'] = float(f.read().split()[0])
    except:
        pass

    return jsonify(stats)

@app.route('/api/system/update', methods=['POST'])
def system_update():
    """Pull latest code from git and restart the service"""
    results = {'steps': [], 'success': False}

    try:
        # Step 1: Git fetch
        fetch_result = subprocess.run(
            ['git', 'fetch', 'origin'],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(AETHER_FILE_PATH)
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
            cwd=os.path.dirname(AETHER_FILE_PATH)
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
            cwd=os.path.dirname(AETHER_FILE_PATH)
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
            cwd=os.path.dirname(AETHER_FILE_PATH)
        )
        results['new_commit'] = commit_result.stdout.strip()
        results['old_commit'] = AETHER_COMMIT

        # Step 5: Schedule restart (non-blocking)
        results['message'] = 'Update pulled. Restarting service...'
        results['success'] = True

        # Restart service in background after response is sent
        def restart_service():
            time.sleep(1)  # Give time for response to be sent
            os.system('sudo systemctl restart aether')

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
                    cwd=os.path.dirname(AETHER_FILE_PATH))

                result = subprocess.run(
                    ['git', 'rev-list', 'HEAD..origin/main', '--count'],
                    capture_output=True, text=True, timeout=10,
                    cwd=os.path.dirname(AETHER_FILE_PATH))

                commits_behind = int(result.stdout.strip()) if result.returncode == 0 else 0

                if commits_behind > 0:
                    print(f"🔄 Auto-sync: {commits_behind} updates available, pulling...")

                    pull_result = subprocess.run(
                        ['git', 'pull', 'origin', 'main'],
                        capture_output=True, text=True, timeout=60,
                        cwd=os.path.dirname(AETHER_FILE_PATH))

                    if pull_result.returncode == 0:
                        app._autosync_last_update = datetime.now().isoformat()
                        print("✓ Auto-sync: update pulled, restarting service...")
                        time.sleep(1)
                        os.system('sudo systemctl restart aether')
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
    
    # Update database
    conn = get_db()
    c = conn.cursor()
    c.execute('''UPDATE nodes SET 
        name = COALESCE(?, name),
        universe = COALESCE(?, universe), 
        channel_start = COALESCE(?, channel_start),
        channel_end = COALESCE(?, channel_end)
        WHERE node_id = ?''',
        (config.get('name'), config.get('universe'), 
         config.get('channelStart'), config.get('channelEnd'), str(node_id)))
    conn.commit()
    conn.close()
    
    # Send config to node if it's WiFi
    node = node_manager.get_node(node_id)
    if node and node.get('type') == 'wifi':
        node_manager.configure_ola_universe(node.get('universe', 1))
        node_manager.send_config_to_node(node, {
            'name': node.get('name'),
            'universe': node.get('universe'),
            'channel_start': node.get('channel_start'),
            'channel_end': node.get('channel_end')
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
# DMX Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/dmx/set', methods=['POST'])
def dmx_set():
    data = request.get_json()
    return jsonify(content_manager.set_channels(
        data.get('universe', 1), data.get('channels', {}), data.get('fade_ms', 0)))

@app.route('/api/dmx/blackout', methods=['POST'])
def dmx_blackout():
    data = request.get_json() or {}
    # If no universe specified, blackout ALL universes (pass None)
    universe = data.get('universe')  # None = all universes
    return jsonify(content_manager.blackout(universe, data.get('fade_ms', 1000)))


@app.route('/api/dmx/master', methods=['POST'])
def dmx_master():
    """Master dimmer - scales all output proportionally"""
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
    
    for univ, base in dmx_state.master_base.items():
        scaled = {}
        for ch_idx, base_val in enumerate(base):
            if base_val > 0:
                scaled[ch_idx + 1] = int(base_val * scale)
        
        if scaled:
            print(f"   🔧 Scaling universe {univ}: {len(scaled)} channels at {level}%", flush=True)
            dmx_state.set_channels(univ, scaled)
            nodes = node_manager.get_nodes_in_universe(univ)
            for node in nodes:
                local_ch = node_manager.translate_channels_for_node(node, scaled)
                if local_ch:
                    node_manager.send_to_node(node, local_ch, fade_ms=0)
    
    return jsonify({'success': True, 'level': level})

@app.route('/api/dmx/master/reset', methods=['POST'])
def dmx_master_reset():
    dmx_state.master_base = {}
    dmx_state.master_level = 100
    return jsonify({'success': True})


@app.route('/api/dmx/universe/<int:universe>', methods=['GET'])
def dmx_get_universe(universe):
    return jsonify({'universe': universe, 'channels': dmx_state.get_universe(universe)})

@app.route('/api/dmx/diagnostics', methods=['GET'])
def dmx_diagnostics():
    """Diagnostics endpoint for debugging DMX output issues"""
    return jsonify({
        'packet_version': NodeManager.PACKET_VERSION,
        'packet_version_info': 'v3: full 512-ch frames, ESP32 firmware v1.1 has 2500-byte buffer',
        'udp': {
            'last_send': node_manager._last_udp_send,
            'total_sends': node_manager._udp_send_count,
            'port': WIFI_COMMAND_PORT,
            'max_packet_size': 2500,
            'channels_per_frame': 512
        },
        'uart': {
            'last_send': node_manager._last_uart_send,
            'total_sends': node_manager._uart_send_count,
            'port': HARDWIRED_UART,
            'baud': HARDWIRED_BAUD
        },
        'chase_engine': {
            'running_chases': list(chase_engine.running_chases.keys()),
            'health': chase_engine.chase_health
        },
        'playback': playback_manager.get_status(),
        'ssot': {
            'universes_active': list(dmx_state.universes.keys()),
            'master_level': dmx_state.master_level
        },
        'system': {
            'version': AETHER_VERSION,
            'commit': AETHER_COMMIT,
            'uptime_seconds': (datetime.now() - AETHER_START_TIME).total_seconds(),
            'file_path': AETHER_FILE_PATH
        }
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

@app.route('/api/scenes/<scene_id>', methods=['DELETE'])
def delete_scene(scene_id):
    return jsonify(content_manager.delete_scene(scene_id))

@app.route('/api/scenes/<scene_id>/play', methods=['POST'])
def play_scene(scene_id):
    data = request.get_json() or {}
    return jsonify(content_manager.play_scene(
        scene_id,
        fade_ms=data.get('fade_ms'),
        use_local=data.get('use_local', True),
        target_channels=data.get('target_channels'),
        universe=data.get('universe')
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

@app.route('/api/chases/<chase_id>', methods=['DELETE'])
def delete_chase(chase_id):
    return jsonify(content_manager.delete_chase(chase_id))

@app.route('/api/chases/<chase_id>/play', methods=['POST'])
def play_chase(chase_id):
    data = request.get_json() or {}
    return jsonify(content_manager.play_chase(
        chase_id,
        target_channels=data.get('target_channels'),
        universe=data.get('universe')
    ))

@app.route('/api/chases/<chase_id>/stop', methods=['POST'])
def stop_chase(chase_id):
    """Stop a specific chase"""
    chase_engine.stop_chase(chase_id)
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

# Playback Routes
# ─────────────────────────────────────────────────────────
@app.route('/api/playback/status', methods=['GET'])
def playback_status():
    return jsonify(playback_manager.get_status())

@app.route('/api/playback/stop', methods=['POST'])
def stop_playback():
    data = request.get_json() or {}
    return jsonify(content_manager.stop_playback(data.get('universe')))

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

    init_database()
    ai_ssot.init_ai_db()
    threading.Thread(target=discovery_listener, daemon=True).start()
    threading.Thread(target=stale_checker, daemon=True).start()
    schedule_runner.start()


    print(f"✓ API server on port {API_PORT}")
    print(f"✓ Discovery on UDP {DISCOVERY_PORT}")
    print(f"✓ Serial: {HARDWIRED_UART}")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=API_PORT, debug=False, allow_unsafe_werkzeug=True)
