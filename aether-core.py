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
# DMX State - THE SINGLE SOURCE OF TRUTH FOR CHANNEL VALUES
# ============================================================
class DMXStateManager:
    """Manages DMX state for all universes - this is the SSOT for channel values"""
    def __init__(self):
        self.universes = {}  # {universe_num: [512 values]}
        self.lock = threading.Lock()
        self._save_timer = None
        self._load_state()

    def _load_state(self):
        """Load DMX state from disk on startup"""
        try:
            if os.path.exists(DMX_STATE_FILE):
                with open(DMX_STATE_FILE, 'r') as f:
                    saved = json.load(f)
                    # Convert string keys back to integers
                    for universe_str, channels in saved.get('universes', {}).items():
                        universe = int(universe_str)
                        self.universes[universe] = channels[:512]  # Ensure 512 max
                        # Pad if needed
                        while len(self.universes[universe]) < 512:
                            self.universes[universe].append(0)
                print(f"✓ Loaded DMX state: {len(self.universes)} universes")
        except Exception as e:
            print(f"⚠️ Could not load DMX state: {e}")

    def _save_state(self):
        """Save DMX state to disk (debounced)"""
        try:
            with self.lock:
                data = {'universes': {str(k): v for k, v in self.universes.items()},
                        'saved_at': datetime.now().isoformat()}
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

    # Add built-in hardwired node
    c.execute('SELECT * FROM nodes WHERE node_id = ?', ('universe-1-builtin',))
    if not c.fetchone():
        c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start, channel_end,
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
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.lock = threading.Lock()
        self._serial = None

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

        if existing:
            c.execute('''UPDATE nodes SET hostname = COALESCE(?, hostname), mac = COALESCE(?, mac),
                ip = COALESCE(?, ip), uptime = COALESCE(?, uptime), rssi = COALESCE(?, rssi),
                fps = COALESCE(?, fps), firmware = COALESCE(?, firmware), status = 'online', last_seen = ?
                WHERE node_id = ?''',
                (data.get('hostname'), data.get('mac'), data.get('ip'), data.get('uptime'),
                 data.get('rssi'), data.get('fps'), data.get('version'), now, node_id))
        else:
            c.execute('''INSERT INTO nodes (node_id, name, hostname, mac, ip, universe, channel_start,
                channel_end, status, is_paired, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (node_id, data.get('hostname', f'Node-{node_id[-4:]}'), data.get('hostname'),
                 data.get('mac'), data.get('ip'), data.get('universe', 1), data.get('startChannel', 1),
                 data.get('channelCount', 512), 'online', False, now, now))

        conn.commit()
        conn.close()
        self.broadcast_status()
        return self.get_node(node_id)

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
        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE nodes SET is_paired = 0 WHERE node_id = ? AND can_delete = 1', (str(node_id),))
        conn.commit()
        conn.close()
        self.broadcast_status()

    def delete_node(self, node_id):
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
        if node.get('type') == 'hardwired' or node.get('is_builtin'):
            return self.send_to_hardwired(channels_dict, fade_ms)
        else:
            # WiFi nodes use OLA/sACN - send to their universe
            universe = node.get('universe', 1)
            return self.send_via_ola(universe, channels_dict)

    def send_to_hardwired(self, channels_dict, fade_ms=0):
        """Send command to hardwired ESP32 via UART"""
        try:
            ser = self._get_serial()
            if ser is None:
                return False

            if not channels_dict:
                return True

            max_ch = max(int(k) for k in channels_dict.keys())
            data = dmx_state.get_channels_for_esp(1, max_ch)

            for ch_str, value in channels_dict.items():
                ch = int(ch_str)
                if 1 <= ch <= len(data):
                    data[ch - 1] = int(value)

            esp_cmd = {"cmd": "scene", "ch": 1, "data": data}
            if fade_ms > 0:
                esp_cmd["fade"] = fade_ms

            json_cmd = json.dumps(esp_cmd) + '\n'
            ser.write(json_cmd.encode())
            ser.flush()
            print(f"📤 UART -> {len(data)} channels, fade={fade_ms}ms")
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
        print(f"🔄 Syncing content to {node['name']} (U{universe})")
        
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
        
        print(f"✓ Synced {len(scenes)} scenes, {len(chases)} chases to {node['name']}")

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
    def set_channels(self, universe, channels, fade_ms=0):
        """Set DMX channels - updates state and sends to nodes"""
        dmx_state.set_channels(universe, channels)
        nodes = node_manager.get_nodes_in_universe(universe)
        
        if not nodes:
            print(f"⚠️ No online nodes in universe {universe}")
            return {'success': False, 'error': 'No nodes online'}

        results = []
        for node in nodes:
            local_channels = node_manager.translate_channels_for_node(node, channels)
            if local_channels:
                success = node_manager.send_to_node(node, local_channels, fade_ms)
                results.append({'node': node['name'], 'success': success})

        return {'success': True, 'results': results}

    def blackout(self, universe, fade_ms=1000):
        """Blackout all channels in universe"""
        dmx_state.blackout(universe)
        playback_manager.stop(universe)
        
        nodes = node_manager.get_nodes_in_universe(universe)
        results = []
        for node in nodes:
            success = node_manager.send_blackout(node, fade_ms)
            results.append({'node': node['name'], 'success': success})
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

    def play_scene(self, scene_id, fade_ms=None, use_local=True, target_channels=None, universe=None):
        """Play a scene - either via local playback or direct channel control

        If target_channels is provided, only those channels are affected and
        other playback continues. If None, this is a full scene play which
        stops any running chase first (SSOT enforcement).

        universe parameter allows playing the scene on any universe (overrides stored universe)
        """
        scene = self.get_scene(scene_id)
        if not scene:
            return {'success': False, 'error': 'Scene not found'}

        # Use provided universe or fall back to scene's stored universe
        universe = universe if universe is not None else scene['universe']
        fade = fade_ms if fade_ms is not None else scene.get('fade_ms', 500)
        print(f"🎬 Playing scene '{scene['name']}' on universe {universe}")

        # SSOT: If playing full scene (no target channels), stop any running chase first
        if target_channels is None:
            current = playback_manager.get_status(universe)
            if current and current.get('type') == 'chase':
                print(f"⏹️ Stopping chase before scene play (SSOT)")
                node_manager.stop_playback_on_nodes(universe)

        # Update playback state (only if full scene, not targeted)
        if target_channels is None:
            playback_manager.set_playing(universe, 'scene', scene_id)
        
        # Filter channels if targeting specific ones
        channels_to_apply = scene['channels']
        if target_channels:
            target_set = set(target_channels)
            channels_to_apply = {k: v for k, v in scene['channels'].items() if int(k) in target_set}
            print(f"🎯 Targeted play: {len(channels_to_apply)} of {len(scene['channels'])} channels")

        # Send to all nodes via set_channels (handles both hardwired UART and WiFi sACN/OLA)
        result = self.set_channels(universe, channels_to_apply, fade)
        node_results = result.get('results', [])

        # Update state and play count
        dmx_state.set_channels(universe, channels_to_apply)
        conn = get_db()
        c = conn.cursor()
        c.execute('UPDATE scenes SET play_count = play_count + 1 WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()
        
        return {'success': True, 'results': node_results}

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
            steps, color, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (chase_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, data.get('bpm', 120), data.get('loop', True),
             json.dumps(steps), data.get('color', '#10b981'), False, datetime.now().isoformat()))
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
        """Start chase playback on all nodes

        SSOT: Only one scene OR chase can run at a time (per universe).
        If target_channels is provided, this is a targeted play which
        allows coexistence with other playback on different channels.

        universe parameter allows playing the chase on any universe (overrides stored universe)
        """
        chase = self.get_chase(chase_id)
        if not chase:
            return {'success': False, 'error': 'Chase not found'}

        # Use provided universe or fall back to chase's stored universe
        universe = universe if universe is not None else chase['universe']
        print(f"🎬 Playing chase '{chase['name']}' on universe {universe}")

        # SSOT: Stop any current playback if this is a full chase (not targeted)
        if target_channels is None:
            current = playback_manager.get_status(universe)
            if current:
                print(f"⏹️ Stopping {current.get('type')} before chase play (SSOT)")
            playback_manager.stop(universe)
            node_manager.stop_playback_on_nodes(universe)
            time.sleep(0.1)

        # Start chase on nodes
        if target_channels is None:
            playback_manager.set_playing(universe, 'chase', chase_id)
        node_results = node_manager.play_chase_on_nodes(universe, chase_id)

        return {'success': True, 'results': node_results}

    def stop_playback(self, universe=None):
        """Stop all playback"""
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
        'status': 'healthy', 'version': '3.0.0', 'timestamp': datetime.now().isoformat(),
        'services': {'database': True, 'discovery': True,
                     'serial': node_manager._serial is not None and node_manager._serial.is_open}
    })

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
    return jsonify(node_manager.pair_node(node_id, request.get_json() or {}))

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
    return jsonify(content_manager.blackout(data.get('universe', 1), data.get('fade_ms', 1000)))

@app.route('/api/dmx/universe/<int:universe>', methods=['GET'])
def dmx_get_universe(universe):
    return jsonify({'universe': universe, 'channels': dmx_state.get_universe(universe)})

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
    print("  AETHER Core v3.0 - Local Playback Engine")
    print("  Features: Scene/Chase sync, Universe splitting")
    print("="*60)

    init_database()
    threading.Thread(target=discovery_listener, daemon=True).start()
    threading.Thread(target=stale_checker, daemon=True).start()

    # DMX state is loaded from disk automatically - ESPs hold their own state
    # so we don't re-send on startup (that would cause a flash)

    print(f"✓ API server on port {API_PORT}")
    print(f"✓ Discovery on UDP {DISCOVERY_PORT}")
    print(f"✓ Serial: {HARDWIRED_UART}")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=API_PORT, debug=False, allow_unsafe_werkzeug=True)
