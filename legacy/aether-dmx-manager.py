#!/usr/bin/env python3
"""
AETHER Unified DMX Manager
Single source of truth for all DMX control across hardwired + WiFi nodes
"""

import socket
import json
import sqlite3
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import threading
import time

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
API_PORT = 8891
NODES_DB = "/home/ramzt/aether-nodes.json"
SCENES_DB = "/home/ramzt/aether-scenes.db"
HARDWIRED_UART = "/dev/serial0"
WIFI_COMMAND_PORT = 8888

class UnifiedDMXManager:
    def __init__(self):
        self.nodes = {}
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.hardwired_process = None
        self.init_database()
        self.load_nodes()
        
        # Start background tasks
        self.reload_thread = threading.Thread(target=self.auto_reload_nodes, daemon=True)
        self.reload_thread.start()
    
    def init_database(self):
        """Initialize SQLite database for scenes and chases"""
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        
        # Scenes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                universe INTEGER,
                channels TEXT,
                fade_ms INTEGER DEFAULT 0,
                curve TEXT DEFAULT 'linear',
                color TEXT DEFAULT 'blue',
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        # Chases table
        c.execute('''
            CREATE TABLE IF NOT EXISTS chases (
                chase_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                universe INTEGER,
                bpm INTEGER DEFAULT 120,
                loop BOOLEAN DEFAULT 1,
                steps TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    
    def load_nodes(self):
        """Load node registry"""
        try:
            with open(NODES_DB, 'r') as f:
                self.nodes = json.load(f)
                print(f"✓ Loaded {len(self.nodes)} nodes")
        except:
            self.nodes = {}
            print("⚠️  No nodes found")
    
    def auto_reload_nodes(self):
        """Auto-reload nodes every 10 seconds"""
        while True:
            time.sleep(10)
            self.load_nodes()
    
    def get_nodes_in_universe(self, universe):
        """Get all nodes controlling a universe"""
        result = {'hardwired': [], 'wifi': []}
        for node_id, node in self.nodes.items():
            if node.get('universe') == universe:
                if node.get('type') == 'hardwired' or node.get('isBuiltIn'):
                    result['hardwired'].append(node)
                else:
                    result['wifi'].append(node)
        return result
    
    def send_to_hardwired(self, command):
        """Send command to hardwired ESP32 via UART"""
        try:
            # Use the existing persistentDMX approach
            # Send JSON command via UART at 250000 baud
            json_cmd = json.dumps(command) + '\n'
            
            with open(HARDWIRED_UART, 'w') as uart:
                uart.write(json_cmd)
                uart.flush()
            
            print(f"📤 Hardwired <- {json_cmd.strip()}")
            return True
        except Exception as e:
            print(f"❌ Hardwired error: {e}")
            return False
    
    def send_to_wifi(self, node_ip, command):
        """Send JSON command to WiFi node via UDP"""
        try:
            json_data = json.dumps(command)
            self.udp_socket.sendto(json_data.encode(), (node_ip, WIFI_COMMAND_PORT))
            print(f"📤 WiFi {node_ip}:{WIFI_COMMAND_PORT} <- {json_data}")
            return True
        except Exception as e:
            print(f"❌ WiFi {node_ip} error: {e}")
            return False
    
    def translate_channels_for_node(self, node, channels):
        """Translate global universe channels to node-local channels"""
        node_start = node.get('channelStart', 1)
        node_end = node.get('channelEnd', 512)
        
        local_channels = {}
        for ch_str, value in channels.items():
            ch = int(ch_str)
            if node_start <= ch <= node_end:
                local_ch = ch - node_start + 1
                local_channels[str(local_ch)] = value
        
        return local_channels
    
    # ==================== SCENE MANAGEMENT ====================
    
    def create_scene(self, scene_data):
        """Create and store a new scene"""
        scene_id = scene_data.get('scene_id', f"scene_{int(time.time())}")
        
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO scenes 
            (scene_id, name, description, universe, channels, fade_ms, curve, color, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scene_id,
            scene_data.get('name', 'Untitled Scene'),
            scene_data.get('description', ''),
            scene_data.get('universe', 1),
            json.dumps(scene_data.get('channels', {})),
            scene_data.get('fade_ms', 0),
            scene_data.get('curve', 'linear'),
            scene_data.get('color', 'blue'),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Push scene to nodes
        self.push_scene_to_nodes(scene_id, scene_data)
        
        return {'success': True, 'scene_id': scene_id}
    
    def push_scene_to_nodes(self, scene_id, scene_data):
        """Push scene definition to all nodes in the universe"""
        universe = scene_data.get('universe', 1)
        channels = scene_data.get('channels', {})
        nodes = self.get_nodes_in_universe(universe)
        
        results = []
        
        # Push to hardwired nodes
        for node in nodes['hardwired']:
            local_channels = self.translate_channels_for_node(node, channels)
            if local_channels:
                command = {
                    'cmd': 'store_scene',
                    'scene_id': scene_id,
                    'channels': local_channels
                }
                success = self.send_to_hardwired(command)
                results.append({'node': node.get('name'), 'type': 'hardwired', 'success': success})
        
        # Push to WiFi nodes
        for node in nodes['wifi']:
            local_channels = self.translate_channels_for_node(node, channels)
            if local_channels:
                command = {
                    'cmd': 'store_scene',
                    'scene_id': scene_id,
                    'channels': local_channels
                }
                node_ip = node.get('ip')
                if node_ip:
                    success = self.send_to_wifi(node_ip, command)
                    results.append({'node': node.get('name'), 'type': 'wifi', 'success': success})
        
        return results
    
    def play_scene(self, scene_id, fade_ms=None, curve=None):
        """Play a scene across all nodes"""
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        
        c.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return {'success': False, 'error': 'Scene not found'}
        
        universe = row[3]
        fade_ms = fade_ms or row[5]
        curve = curve or row[6]
        
        nodes = self.get_nodes_in_universe(universe)
        results = []
        
        # Command to load scene
        command = {
            'cmd': 'load_scene',
            'scene_id': scene_id,
            'fade_ms': fade_ms,
            'curve': curve
        }
        
        # Send to hardwired
        for node in nodes['hardwired']:
            success = self.send_to_hardwired(command)
            results.append({'node': node.get('name'), 'success': success})
        
        # Send to WiFi
        for node in nodes['wifi']:
            node_ip = node.get('ip')
            if node_ip:
                success = self.send_to_wifi(node_ip, command)
                results.append({'node': node.get('name'), 'success': success})
        
        return {'success': True, 'results': results}
    
    def list_scenes(self):
        """List all scenes"""
        conn = sqlite3.connect(SCENES_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM scenes ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        
        scenes = []
        for row in rows:
            scenes.append({
                'scene_id': row['scene_id'],
                'name': row['name'],
                'description': row['description'],
                'universe': row['universe'],
                'channels': json.loads(row['channels']),
                'fade_ms': row['fade_ms'],
                'curve': row['curve'],
                'color': row['color'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            })
        
        return scenes
    
    def delete_scene(self, scene_id):
        """Delete a scene"""
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        c.execute('DELETE FROM scenes WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()
        
        return {'success': True, 'scene_id': scene_id}
    
    # ==================== CHASE MANAGEMENT ====================
    
    def create_chase(self, chase_data):
        """Create and store a new chase"""
        chase_id = chase_data.get('chase_id', f"chase_{int(time.time())}")
        
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO chases
            (chase_id, name, description, universe, bpm, loop, steps, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chase_id,
            chase_data.get('name', 'Untitled Chase'),
            chase_data.get('description', ''),
            chase_data.get('universe', 1),
            chase_data.get('bpm', 120),
            chase_data.get('loop', True),
            json.dumps(chase_data.get('steps', [])),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'chase_id': chase_id}
    
    def play_chase(self, chase_id):
        """Play a chase across all nodes"""
        conn = sqlite3.connect(SCENES_DB)
        c = conn.cursor()
        
        c.execute('SELECT * FROM chases WHERE chase_id = ?', (chase_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return {'success': False, 'error': 'Chase not found'}
        
        universe = row[3]
        bpm = row[4]
        loop = bool(row[5])
        steps = json.loads(row[6])
        
        nodes = self.get_nodes_in_universe(universe)
        results = []
        
        command = {
            'cmd': 'start_chase',
            'chase_id': chase_id,
            'bpm': bpm,
            'loop': loop,
            'steps': steps
        }
        
        # Send to hardwired
        for node in nodes['hardwired']:
            success = self.send_to_hardwired(command)
            results.append({'node': node.get('name'), 'success': success})
        
        # Send to WiFi
        for node in nodes['wifi']:
            node_ip = node.get('ip')
            if node_ip:
                success = self.send_to_wifi(node_ip, command)
                results.append({'node': node.get('name'), 'success': success})
        
        return {'success': True, 'results': results}
    
    def list_chases(self):
        """List all chases"""
        conn = sqlite3.connect(SCENES_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM chases ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        
        chases = []
        for row in rows:
            chases.append({
                'chase_id': row['chase_id'],
                'name': row['name'],
                'description': row['description'],
                'universe': row['universe'],
                'bpm': row['bpm'],
                'loop': bool(row['loop']),
                'steps': json.loads(row['steps']),
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            })
        
        return chases
    
    # ==================== DIRECT DMX CONTROL ====================
    
    def set_channels(self, universe, channels, fade_ms=0, curve='linear'):
        """Set channels directly across all nodes in universe"""
        nodes = self.get_nodes_in_universe(universe)
        results = []
        
        # Hardwired nodes
        for node in nodes['hardwired']:
            local_channels = self.translate_channels_for_node(node, channels)
            if local_channels:
                command = {
                    'cmd': 'set_channels',
                    'channels': local_channels,
                    'fade_ms': fade_ms,
                    'curve': curve
                }
                success = self.send_to_hardwired(command)
                results.append({'node': node.get('name'), 'success': success})
        
        # WiFi nodes
        for node in nodes['wifi']:
            local_channels = self.translate_channels_for_node(node, channels)
            if local_channels:
                command = {
                    'cmd': 'set_channels',
                    'channels': local_channels,
                    'fade_ms': fade_ms,
                    'curve': curve
                }
                node_ip = node.get('ip')
                if node_ip:
                    success = self.send_to_wifi(node_ip, command)
                    results.append({'node': node.get('name'), 'success': success})
        
        return {'success': True, 'results': results}
    
    def blackout(self, universe, fade_ms=1000):
        """Blackout all nodes in universe"""
        nodes = self.get_nodes_in_universe(universe)
        results = []
        
        command = {'cmd': 'stop', 'fade_ms': fade_ms}
        
        # Hardwired
        for node in nodes['hardwired']:
            success = self.send_to_hardwired(command)
            results.append({'node': node.get('name'), 'success': success})
        
        # WiFi
        for node in nodes['wifi']:
            node_ip = node.get('ip')
            if node_ip:
                success = self.send_to_wifi(node_ip, command)
                results.append({'node': node.get('name'), 'success': success})
        
        return {'success': True, 'results': results}

# Global manager instance
manager = UnifiedDMXManager()

# ==================== REST API ====================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'nodes': len(manager.nodes)})

# Scene endpoints
@app.route('/api/scenes', methods=['GET'])
def get_scenes():
    return jsonify(manager.list_scenes())

@app.route('/api/scenes', methods=['POST'])
def create_scene():
    result = manager.create_scene(request.json)
    return jsonify(result)

@app.route('/api/scenes/<scene_id>', methods=['DELETE'])
def delete_scene(scene_id):
    result = manager.delete_scene(scene_id)
    return jsonify(result)

@app.route('/api/scenes/<scene_id>/play', methods=['POST'])
def play_scene(scene_id):
    data = request.json or {}
    result = manager.play_scene(scene_id, data.get('fade_ms'), data.get('curve'))
    return jsonify(result)

# Chase endpoints
@app.route('/api/chases', methods=['GET'])
def get_chases():
    return jsonify(manager.list_chases())

@app.route('/api/chases', methods=['POST'])
def create_chase():
    result = manager.create_chase(request.json)
    return jsonify(result)

@app.route('/api/chases/<chase_id>/play', methods=['POST'])
def play_chase(chase_id):
    result = manager.play_chase(chase_id)
    return jsonify(result)

# Direct control endpoints
@app.route('/api/dmx/set', methods=['POST'])
def set_channels():
    data = request.json
    result = manager.set_channels(
        data.get('universe', 1),
        data.get('channels', {}),
        data.get('fade_ms', 0),
        data.get('curve', 'linear')
    )
    return jsonify(result)

@app.route('/api/dmx/blackout', methods=['POST'])
def blackout():
    data = request.json
    result = manager.blackout(data.get('universe', 1), data.get('fade_ms', 1000))
    return jsonify(result)

if __name__ == '__main__':
    print("=" * 70)
    print("AETHER Unified DMX Manager")
    print("=" * 70)
    print(f"API: http://0.0.0.0:{API_PORT}")
    print(f"Database: {SCENES_DB}")
    print(f"Nodes: {len(manager.nodes)}")
    print("=" * 70)
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
