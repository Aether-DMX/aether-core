#!/usr/bin/env python3
"""
AETHER DMX - Pairing Server
With universe channel splitting and auto-rebalancing
"""

import requests
import json
import socket
import subprocess
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

NODES_DB = "/home/ramzt/aether-nodes.json"
PENDING_NODES_FILE = "/home/ramzt/aether-discovered.json"
CONFIG_PORT = 5556

def load_nodes():
    try:
        with open(NODES_DB, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_nodes(nodes):
    with open(NODES_DB, 'w') as f:
        json.dump(nodes, f, indent=2)

def load_pending():
    try:
        with open(PENDING_NODES_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_pending(pending):
    with open(PENDING_NODES_FILE, 'w') as f:
        json.dump(pending, f, indent=2)

def get_universe_conflicts(universe):
    """Check if universe is already used and return existing nodes"""
    nodes = load_nodes()
    conflicts = []
    for node_id, node in nodes.items():
        if node.get('universe') == universe:
            conflicts.append({
                'node_id': node_id,
                'name': node.get('name'),
                'channelStart': node.get('channelStart', 1),
                'channelEnd': node.get('channelEnd', 512)
            })
    return conflicts

def calculate_available_channels(conflicts):
    """Find gaps in channel allocation"""
    if not conflicts:
        return [[1, 512]]

    used = set()
    for conflict in conflicts:
        start = conflict.get('channelStart', 1)
        end = conflict.get('channelEnd', 512)
        for ch in range(start, end + 1):
            used.add(ch)

    # Find contiguous free ranges
    free_ranges = []
    in_range = False
    range_start = None

    for ch in range(1, 513):
        if ch not in used:
            if not in_range:
                range_start = ch
                in_range = True
        else:
            if in_range:
                free_ranges.append([range_start, ch - 1])
                in_range = False

    if in_range:
        free_ranges.append([range_start, 512])

    return free_ranges

def validate_channel_range(universe, start, end, exclude_node_id=None):
    """Validate that channel range doesn't overlap with existing nodes"""
    nodes = load_nodes()
    for node_id, node in nodes.items():
        if node_id == exclude_node_id:
            continue
        if node.get('universe') == universe:
            node_start = node.get('channelStart', 1)
            node_end = node.get('channelEnd', 512)

            # Check for overlap
            if not (end < node_start or start > node_end):
                return False, f"Overlaps with {node.get('name')}"

    return True, None

# ============================================================================
# REBALANCING FUNCTIONS
# ============================================================================

def get_universe_nodes(universe):
    """Get all nodes in a specific universe (for rebalancing)"""
    nodes = load_nodes()
    universe_nodes = []
    for node_id, node in nodes.items():
        if node.get('universe') == universe:
            universe_nodes.append({
                'node_id': node_id,
                'name': node.get('name'),
                'channelStart': node.get('channelStart', 1),
                'channelEnd': node.get('channelEnd', 512),
                'ip': node.get('ip', '0.0.0.0'),
                'online': node.get('online', False)
            })
    return sorted(universe_nodes, key=lambda x: x['channelStart'])

def calculate_even_split(num_nodes):
    """Calculate even channel split for N nodes"""
    if num_nodes <= 0:
        return []

    channels_per_node = 512 // num_nodes
    remainder = 512 % num_nodes

    splits = []
    current_start = 1

    for i in range(num_nodes):
        # Give extra channels to first nodes if there's remainder
        node_channels = channels_per_node + (1 if i < remainder else 0)
        splits.append({
            'start': current_start,
            'end': current_start + node_channels - 1
        })
        current_start += node_channels

    return splits

def rebalance_universe(universe, include_new_node=True):
    """Calculate rebalance plan for all nodes in a universe"""
    universe_nodes = get_universe_nodes(universe)
    num_nodes = len(universe_nodes) + (1 if include_new_node else 0)

    if num_nodes == 0:
        return []

    splits = calculate_even_split(num_nodes)

    rebalance_plan = []
    for i, node in enumerate(universe_nodes):
        rebalance_plan.append({
            'node_id': node['node_id'],
            'name': node['name'],
            'old_range': [node['channelStart'], node['channelEnd']],
            'new_range': [splits[i]['start'], splits[i]['end']]
        })

    if include_new_node and len(splits) > len(universe_nodes):
        rebalance_plan.append({
            'node_id': 'NEW',
            'name': 'New Node',
            'old_range': None,
            'new_range': [splits[-1]['start'], splits[-1]['end']]
        })

    return rebalance_plan

def apply_rebalance(rebalance_plan, new_node_id=None):
    """Apply rebalanced channel allocations to all nodes"""
    nodes = load_nodes()

    for item in rebalance_plan:
        node_id = item['node_id']
        if node_id == 'NEW' and new_node_id:
            node_id = new_node_id

        if node_id in nodes:
            old_range = item.get('old_range')
            new_range = item['new_range']

            nodes[node_id]['channelStart'] = new_range[0]
            nodes[node_id]['channelEnd'] = new_range[1]

            # Send config to node (handles offline gracefully)
            send_config_to_node(nodes[node_id])

            if old_range:
                print(f"✅ Rebalanced {nodes[node_id].get('name')}: Ch{old_range[0]}-{old_range[1]} → Ch{new_range[0]}-{new_range[1]}")
            else:
                print(f"✅ Configured {nodes[node_id].get('name')}: Ch{new_range[0]}-{new_range[1]}")

    save_nodes(nodes)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def send_config_to_node(node):
    """Send configuration to node (handles offline nodes gracefully)"""
    try:
        ip = node.get('ip', '0.0.0.0')

        # Don't try to send to offline nodes
        if ip == '0.0.0.0' or not node.get('online', False):
            print(f"⚠️  Node {node.get('name')} is offline, config will be sent when it reconnects")
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        config_data = json.dumps({'action': 'configure', 'config': node}).encode()
        sock.sendto(config_data, (ip, CONFIG_PORT))
        print(f"📤 Config sent to {node['name']} at {ip}")
    except Exception as e:
        print(f"⚠️  Error sending config to {node.get('name')}: {e}")

def send_unpair_to_node(node):
    """Send unpair command to clear node config"""
    try:
        ip = node.get('ip', '0.0.0.0')
        if ip == '0.0.0.0':
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unpair_data = json.dumps({'action': 'unpair'}).encode()
        sock.sendto(unpair_data, (ip, CONFIG_PORT))
        print(f"📤 Unpair sent to {node.get('name', node.get('ip'))}")
    except Exception as e:
        print(f"Error sending unpair: {e}")

def configure_ola_for_node(node):
    """Automatically configure OLA for a paired node"""
    universe = node.get('universe', 1)
    node_name = node.get('name', 'Unknown')

    try:
        # Get E1.31 device ID
        result = subprocess.run(['ola_dev_info'], capture_output=True, text=True)
        e131_device = None
        for line in result.stdout.split('\n'):
            if 'E1.31' in line:
                # Extract "Device 7:" → "7"
                parts = line.split()
                if len(parts) > 1 and parts[0] == 'Device':
                    e131_device = parts[1].rstrip(':')
                    break

        if not e131_device:
            print(f"⚠️  E1.31 device not found in OLA")
            return

        # Create universe via web API if it doesn't exist
        import urllib.request
        import urllib.parse
        try:
            data = urllib.parse.urlencode({'id': universe, 'name': f'AETHER-U{universe}', 'add_ports': 'on'}).encode()
            req = urllib.request.Request('http://localhost:9090/new_universe', data=data)
            urllib.request.urlopen(req, timeout=2)
        except:
            pass  # Universe might already exist

        # Patch universe to E1.31 device (port = universe - 1)
        port = universe - 1
        subprocess.run([
            'ola_patch',
            '-d', str(e131_device),
            '-p', str(port),
            '-u', str(universe)
        ], check=True)

        print(f"✅ OLA Universe {universe} configured for {node_name}")

    except Exception as e:
        print(f"⚠️  Error configuring OLA: {e}")

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('pairing.html')

@app.route('/api/nodes')
def get_nodes():
    return jsonify(load_nodes())

@app.route('/api/pending')
def get_pending():
    return jsonify(load_pending())

@app.route('/api/check_universe', methods=['POST'])
def check_universe():
    """Check if universe has conflicts"""
    data = request.json
    universe = data.get('universe', 1)
    conflicts = get_universe_conflicts(universe)

    return jsonify({
        'has_conflict': len(conflicts) > 0,
        'conflicts': conflicts,
        'available_channels': calculate_available_channels(conflicts)
    })

@app.route('/api/universe/<int:universe>/nodes')
def get_universe_nodes_api(universe):
    """Get all nodes in a specific universe"""
    return jsonify(get_universe_nodes(universe))

@app.route('/api/universe/<int:universe>/rebalance', methods=['POST'])
def preview_rebalance(universe):
    """Preview what rebalancing would look like"""
    data = request.json or {}
    include_new = data.get('include_new_node', True)

    plan = rebalance_universe(universe, include_new)

    return jsonify({
        'universe': universe,
        'total_nodes': len(plan),
        'rebalance_plan': plan
    })

@app.route('/api/approve', methods=['POST'])
def approve_node():
    data = request.json
    node_id = data.get('node_id')
    config = data.get('config', {})

    nodes = load_nodes()
    pending = load_pending()

    # Find node in pending list
    node = None
    for nid, n in pending.items():
        if nid == node_id:
            node = n
            break

    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    # Use assigned_name as default if user didn't provide a name
    if not config.get('name') or config.get('name') == node.get('hostname'):
        config['name'] = node.get('assigned_name', 'AETHER-PULSE-1')

    universe = config.get('universe', 1)
    auto_rebalance = config.get('auto_rebalance', False)

    # Check if auto-rebalance is requested
    if auto_rebalance:
        universe_nodes = get_universe_nodes(universe)

        if len(universe_nodes) > 0:
            # Calculate even split for all nodes including new one
            plan = rebalance_universe(universe, include_new_node=True)

            if plan:
                # Get new node's allocation from plan
                new_node_allocation = plan[-1]['new_range']

                config['channelStart'] = new_node_allocation[0]
                config['channelEnd'] = new_node_allocation[1]

                # Update new node
                node.update(config)
                nodes[node_id] = node
                save_nodes(nodes)

                # Apply rebalance to ALL nodes (including new one)
                apply_rebalance(plan, new_node_id=node_id)

                print(f"✅ Auto-rebalanced Universe {universe} with {len(plan)} nodes")
        else:
            # First node in universe - gets full range
            config['channelStart'] = 1
            config['channelEnd'] = 512
            node.update(config)
            nodes[node_id] = node
            save_nodes(nodes)
    else:
        # Manual allocation - validate
        channel_start = config.get('channelStart', 1)
        channel_end = config.get('channelEnd', 512)

        # Check for conflicts
        conflicts = get_universe_conflicts(universe)
        if conflicts:
            # Universe is split - MUST specify channels
            if channel_start == 1 and channel_end == 512:
                return jsonify({
                    'success': False,
                    'error': 'Universe {} is split. Must specify channel range.'.format(universe),
                    'conflicts': conflicts,
                    'available': calculate_available_channels(conflicts)
                }), 400

            # Validate no overlap
            valid, error = validate_channel_range(universe, channel_start, channel_end)
            if not valid:
                return jsonify({
                    'success': False,
                    'error': f'Channel overlap: {error}'
                }), 400

        node.update(config)
        nodes[node_id] = node
        save_nodes(nodes)

    # Remove from pending
    pending_dict = load_pending()
    if node_id in pending_dict:
        del pending_dict[node_id]
        save_pending(pending_dict)

    send_config_to_node(node)
    configure_ola_for_node(node)

    return jsonify({'success': True, 'name': config['name'], 'node': node})

@app.route('/api/update_channel_ranges', methods=['POST'])
def update_channel_ranges():
    """Update channel ranges for universe splitting"""
    data = request.json
    updates = data.get('updates', [])

    nodes = load_nodes()

    for update in updates:
        node_id = update.get('node_id')
        if node_id in nodes:
            nodes[node_id]['channelStart'] = update.get('channelStart')
            nodes[node_id]['channelEnd'] = update.get('channelEnd')
            send_config_to_node(nodes[node_id])

    save_nodes(nodes)
    return jsonify({'success': True})

@app.route('/api/update_node', methods=['POST'])
def update_node():
    """Update an existing node configuration"""
    data = request.json
    node_id = data.get('node_id')
    config = data.get('config', {})
    
    nodes = load_nodes()
    
    if node_id not in nodes:
        return jsonify({'success': False, 'error': 'Node not found'}), 404
    
    # Update node configuration
    if 'universe' in config:
        nodes[node_id]['universe'] = config['universe']
    if 'channelStart' in config:
        nodes[node_id]['channelStart'] = config['channelStart']
    if 'channelEnd' in config:
        nodes[node_id]['channelEnd'] = config['channelEnd']
    if 'deviceRole' in config:
        nodes[node_id]['deviceRole'] = config['deviceRole']
    if 'name' in config:
        nodes[node_id]['name'] = config['name']
    
    # Send updated config to node (skip for built-in nodes)
    if not nodes[node_id].get('isBuiltIn', False) and nodes[node_id].get('type') != 'hardwired':
        send_config_to_node(nodes[node_id])
    else:
        print(f"⚠️  Skipping config send for built-in node {nodes[node_id].get('name')}")
    
    # Save to database
    save_nodes(nodes)
    
    print(f"✅ Updated node {nodes[node_id].get('name')}: Universe {nodes[node_id].get('universe')}, Ch{nodes[node_id].get('channelStart')}-{nodes[node_id].get('channelEnd')}")
    
    return jsonify({'success': True, 'node': nodes[node_id]})


@app.route('/api/remove', methods=['POST'])
def remove_node():
    data = request.json
    node_id = data.get('node_id')

    nodes = load_nodes()
    if node_id in nodes:
        node = nodes[node_id]

        # Send unpair command to node
        send_unpair_to_node(node)

        del nodes[node_id]
        save_nodes(nodes)
        return jsonify({'success': True})

    return jsonify({'success': False}), 404

@app.route('/api/replace', methods=['POST'])
def replace_node():
    """Replace a failed/corrupted node with a new one"""
    data = request.json
    old_node_id = data.get('old_node_id')
    new_node_id = data.get('new_node_id')

    nodes = load_nodes()

    if old_node_id not in nodes:
        return jsonify({'success': False, 'error': 'Old node not found'}), 404

    # Get old node config
    old_config = nodes[old_node_id]

    # Transfer config to new node
    if new_node_id in nodes:
        nodes[new_node_id].update({
            'universe': old_config['universe'],
            'channelStart': old_config['channelStart'],
            'channelEnd': old_config['channelEnd'],
            'assigned_name': old_config['assigned_name'] + '-REPLACED'
        })

        # Remove old node
        del nodes[old_node_id]

        save_nodes(nodes)
        send_config_to_node(nodes[new_node_id])

        return jsonify({
            'success': True,
            'replaced_node': nodes[new_node_id]
        })

    return jsonify({'success': False, 'error': 'New node not found'}), 404

if __name__ == '__main__':
    print("🚀 AETHER Pairing Server with Auto-Rebalancing")
    print("   Listening on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
