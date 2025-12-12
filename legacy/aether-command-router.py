#!/usr/bin/env python3
"""
AETHER Command Router - DMX Universe/Channel to Node JSON Translator
"""

import socket
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

NODES_DB = "/home/ramzt/aether-nodes.json"
COMMAND_PORT = 8888
API_PORT = 8889

class NodeRouter:
    def __init__(self):
        self.nodes = {}
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.load_nodes()
        self.reload_thread = threading.Thread(target=self.auto_reload_nodes, daemon=True)
        self.reload_thread.start()
    
    def load_nodes(self):
        try:
            with open(NODES_DB, 'r') as f:
                data = json.load(f)
                self.nodes = data
                print(f"✓ Loaded {len(self.nodes)} nodes")
        except Exception as e:
            print(f"⚠️  Could not load nodes: {e}")
            self.nodes = {}
    
    def auto_reload_nodes(self):
        while True:
            time.sleep(10)
            self.load_nodes()
    
    def find_node_for_channel(self, universe, channel):
        for node_id, node in self.nodes.items():
            if node.get('universe') == universe:
                start = node.get('channelStart', 1)
                end = node.get('channelEnd', 512)
                if start <= channel <= end:
                    local_channel = channel - start + 1
                    return {'node_id': node_id, 'node': node, 'local_channel': local_channel}
        return None
    
    def get_all_nodes_in_universe(self, universe):
        result = []
        for node_id, node in self.nodes.items():
            if node.get('universe') == universe:
                result.append({'node_id': node_id, 'node': node})
        return result
    
    def send_command_to_node(self, node_ip, command):
        try:
            json_data = json.dumps(command)
            self.udp_socket.sendto(json_data.encode(), (node_ip, COMMAND_PORT))
            print(f"📤 {node_ip}:{COMMAND_PORT} <- {json_data}")
            return True
        except Exception as e:
            print(f"❌ Failed to send to {node_ip}: {e}")
            return False
    
    def set_channel(self, universe, channel, value, fade_ms=0, curve="linear"):
        target = self.find_node_for_channel(universe, channel)
        if not target:
            return {'success': False, 'error': f'No node for U{universe} Ch{channel}'}
        
        command = {
            'cmd': 'set_channels',
            'channels': {str(target['local_channel']): value},
            'fade_ms': fade_ms,
            'curve': curve
        }
        
        node_ip = target['node'].get('ip')
        if not node_ip:
            return {'success': False, 'error': 'Node has no IP'}
        
        success = self.send_command_to_node(node_ip, command)
        return {'success': success, 'node': target['node'].get('name'), 
                'universe': universe, 'channel': channel, 'value': value}
    
    def set_channels_bulk(self, universe, channels, fade_ms=0, curve="linear"):
        node_commands = {}
        
        for channel, value in channels.items():
            channel = int(channel)
            target = self.find_node_for_channel(universe, channel)
            if target:
                node_id = target['node_id']
                if node_id not in node_commands:
                    node_commands[node_id] = {'node': target['node'], 'channels': {}}
                node_commands[node_id]['channels'][str(target['local_channel'])] = value
        
        results = []
        for node_id, data in node_commands.items():
            command = {
                'cmd': 'set_channels',
                'channels': data['channels'],
                'fade_ms': fade_ms,
                'curve': curve
            }
            node_ip = data['node'].get('ip')
            if node_ip:
                success = self.send_command_to_node(node_ip, command)
                results.append({'node': data['node'].get('name'), 'success': success})
        
        return {'success': True, 'nodes_updated': results}
    
    def load_scene_on_universe(self, universe, scene_id, fade_ms=2000, curve="s_curve"):
        nodes = self.get_all_nodes_in_universe(universe)
        if not nodes:
            return {'success': False, 'error': f'No nodes in U{universe}'}
        
        results = []
        for item in nodes:
            command = {'cmd': 'load_scene', 'scene_id': scene_id, 'fade_ms': fade_ms, 'curve': curve}
            node_ip = item['node'].get('ip')
            if node_ip:
                success = self.send_command_to_node(node_ip, command)
                results.append({'node': item['node'].get('name'), 'success': success})
        
        return {'success': True, 'nodes_updated': results}
    
    def start_chase_on_universe(self, universe, chase_data):
        nodes = self.get_all_nodes_in_universe(universe)
        if not nodes:
            return {'success': False, 'error': f'No nodes in U{universe}'}
        
        results = []
        for item in nodes:
            command = {
                'cmd': 'start_chase',
                'chase_id': chase_data.get('chase_id', 'chase'),
                'bpm': chase_data.get('bpm', 120),
                'loop': chase_data.get('loop', True),
                'steps': chase_data.get('steps', [])
            }
            node_ip = item['node'].get('ip')
            if node_ip:
                success = self.send_command_to_node(node_ip, command)
                results.append({'node': item['node'].get('name'), 'success': success})
        
        return {'success': True, 'nodes_updated': results}
    
    def blackout_universe(self, universe, fade_ms=1000):
        nodes = self.get_all_nodes_in_universe(universe)
        if not nodes:
            return {'success': False, 'error': f'No nodes in U{universe}'}
        
        results = []
        for item in nodes:
            command = {'cmd': 'stop', 'fade_ms': fade_ms}
            node_ip = item['node'].get('ip')
            if node_ip:
                success = self.send_command_to_node(node_ip, command)
                results.append({'node': item['node'].get('name'), 'success': success})
        
        return {'success': True, 'nodes_updated': results}

router = NodeRouter()

@app.route('/api/dmx/set', methods=['POST'])
def api_set_channel():
    data = request.json
    universe = data.get('universe')
    channel = data.get('channel')
    value = data.get('value')
    fade_ms = data.get('fade_ms', 0)
    curve = data.get('curve', 'linear')
    
    if universe is None or channel is None or value is None:
        return jsonify({'error': 'Missing: universe, channel, value'}), 400
    
    result = router.set_channel(universe, channel, value, fade_ms, curve)
    return jsonify(result)

@app.route('/api/dmx/set-bulk', methods=['POST'])
def api_set_channels_bulk():
    data = request.json
    universe = data.get('universe')
    channels = data.get('channels')
    fade_ms = data.get('fade_ms', 0)
    curve = data.get('curve', 'linear')
    
    if universe is None or channels is None:
        return jsonify({'error': 'Missing: universe, channels'}), 400
    
    result = router.set_channels_bulk(universe, channels, fade_ms, curve)
    return jsonify(result)

@app.route('/api/dmx/scene', methods=['POST'])
def api_load_scene():
    data = request.json
    universe = data.get('universe')
    scene_id = data.get('scene_id')
    fade_ms = data.get('fade_ms', 2000)
    curve = data.get('curve', 's_curve')
    
    if universe is None or scene_id is None:
        return jsonify({'error': 'Missing: universe, scene_id'}), 400
    
    result = router.load_scene_on_universe(universe, scene_id, fade_ms, curve)
    return jsonify(result)

@app.route('/api/dmx/start-chase', methods=['POST'])
def api_start_chase():
    data = request.json
    universe = data.get('universe')
    
    if universe is None:
        return jsonify({'error': 'Missing: universe'}), 400
    
    result = router.start_chase_on_universe(universe, data)
    return jsonify(result)

@app.route('/api/dmx/blackout', methods=['POST'])
def api_blackout():
    data = request.json
    universe = data.get('universe')
    fade_ms = data.get('fade_ms', 1000)
    
    if universe is None:
        return jsonify({'error': 'Missing: universe'}), 400
    
    result = router.blackout_universe(universe, fade_ms)
    return jsonify(result)

@app.route('/api/nodes', methods=['GET'])
def api_get_nodes():
    return jsonify(router.nodes)

@app.route('/api/nodes/reload', methods=['POST'])
def api_reload_nodes():
    router.load_nodes()
    return jsonify({'success': True, 'node_count': len(router.nodes)})

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'running', 'nodes': len(router.nodes), 'port': API_PORT})

if __name__ == '__main__':
    print("=" * 60)
    print("AETHER Command Router v2")
    print("=" * 60)
    print(f"REST API: http://0.0.0.0:{API_PORT}")
    print(f"UDP Target: ESP32 nodes on port {COMMAND_PORT}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
