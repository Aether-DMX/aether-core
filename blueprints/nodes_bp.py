"""
AETHER Core — Nodes Blueprint
Routes: /api/nodes/*
Dependencies: node_manager (also aliased as ssot), get_db, node_submit
"""

from flask import Blueprint, jsonify, request

nodes_bp = Blueprint('nodes', __name__)

_node_manager = None
_ssot = None  # Same as node_manager, used for UDP commands
_get_db = None
_node_submit = None


def init_app(node_manager, get_db_fn, node_submit_fn):
    """Initialize blueprint with required dependencies."""
    global _node_manager, _ssot, _get_db, _node_submit
    _node_manager = node_manager
    _ssot = node_manager  # ssot is an alias for node_manager
    _get_db = get_db_fn
    _node_submit = node_submit_fn


@nodes_bp.route('/api/nodes', methods=['GET'])
def get_nodes():
    return jsonify(_node_manager.get_all_nodes())

@nodes_bp.route('/api/nodes/online', methods=['GET'])
def get_online_nodes():
    return jsonify(_node_manager.get_all_nodes(include_offline=False))

@nodes_bp.route('/api/nodes/<node_id>', methods=['GET'])
def get_node(node_id):
    node = _node_manager.get_node(node_id)
    return jsonify(node) if node else (jsonify({'error': 'Node not found'}), 404)

@nodes_bp.route('/api/nodes/<node_id>/pair', methods=['POST'])
def pair_node(node_id):
    try:
        node = _node_manager.pair_node(node_id, request.get_json() or {})
        if node:
            return jsonify(node)
        else:
            return jsonify({'error': 'Node not found - it may not have registered yet'}), 404
    except Exception as e:
        print(f"Error pairing node {node_id}: {e}")
        return jsonify({'error': str(e)}), 500

@nodes_bp.route('/api/nodes/<node_id>/configure', methods=['POST'])
def configure_node(node_id):
    """Update configuration for an already-paired node"""
    config = request.get_json() or {}
    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404

    channel_start = config.get('channelStart') or config.get('channel_start')
    channel_end = config.get('channelEnd') or config.get('channel_end')
    slice_mode = config.get('slice_mode')

    conn = _get_db()
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

    node = _node_manager.get_node(node_id)

    if node and node.get('type') == 'wifi':
        _node_manager.send_config_to_node(node, {
            'name': node.get('name'),
            'universe': node.get('universe'),
            'channel_start': node.get('channel_start'),
            'channel_end': node.get('channel_end'),
            'slice_mode': node.get('slice_mode', 'zero_outside')
        })

    _node_manager.broadcast_status()
    return jsonify({'success': True, 'node': node})

@nodes_bp.route('/api/nodes/<node_id>/unpair', methods=['POST'])
def unpair_node(node_id):
    _node_manager.unpair_node(node_id)
    return jsonify({'success': True})

@nodes_bp.route('/api/nodes/<node_id>', methods=['DELETE'])
def delete_node(node_id):
    _node_manager.delete_node(node_id)
    return jsonify({'success': True})

@nodes_bp.route('/api/nodes/<node_id>/toggle-visibility', methods=['POST'])
def toggle_node_visibility(node_id):
    """Toggle whether a node is hidden from the dashboard"""
    data = request.get_json() or {}
    hidden = data.get('hidden', False)

    conn = _get_db()
    c = conn.cursor()
    c.execute('UPDATE nodes SET hidden_from_dashboard = ? WHERE node_id = ?', (1 if hidden else 0, node_id))
    conn.commit()
    conn.close()

    _node_manager.broadcast_status()
    return jsonify({'success': True, 'hidden_from_dashboard': hidden})

@nodes_bp.route('/api/nodes/<node_id>/sync', methods=['POST'])
def sync_node(node_id):
    """Force sync content to a specific node"""
    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    _node_submit(_node_manager.sync_content_to_node, node)
    return jsonify({'success': True, 'message': 'Sync started'})

@nodes_bp.route('/api/nodes/sync', methods=['POST'])
def sync_all_nodes():
    """Force sync content to all nodes"""
    _node_submit(_node_manager.sync_all_content)
    return jsonify({'success': True, 'message': 'Full sync started'})

@nodes_bp.route('/api/nodes/<node_id>/ping', methods=['POST'])
def ping_node(node_id):
    """SAFETY ACTION: Health check for a specific node."""
    print(f"PING: /api/nodes/{node_id}/ping called", flush=True)

    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = _ssot.send_udpjson_ping(node_ip)
        # [F03] result is now a dict: {success, attempts, rtt_ms, response}
        success = result.get('success', False) if isinstance(result, dict) else result is not None
        response = {
            'success': success,
            'node_id': node_id,
            'ip': node_ip,
            'action': 'ping'
        }
        if isinstance(result, dict):
            response['attempts'] = result.get('attempts')
            response['rtt_ms'] = result.get('rtt_ms')
            if result.get('response'):
                response['pong'] = result['response']
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@nodes_bp.route('/api/nodes/<node_id>/reset', methods=['POST'])
def reset_node(node_id):
    """SAFETY ACTION: Reset a specific node."""
    print(f"RESET: /api/nodes/{node_id}/reset called", flush=True)

    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = _ssot.send_udpjson_reset(node_ip)
        # [F03] result is now a dict: {success, attempts, rtt_ms, response}
        success = result.get('success', False) if isinstance(result, dict) else result is not None
        response = {
            'success': success,
            'node_id': node_id,
            'ip': node_ip,
            'action': 'reset'
        }
        if isinstance(result, dict):
            response['attempts'] = result.get('attempts')
            response['rtt_ms'] = result.get('rtt_ms')
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@nodes_bp.route('/api/nodes/<node_id>/stats', methods=['GET'])
def node_stats(node_id):
    """Get real-time stats for a node from stored heartbeat data."""
    node = _node_manager.get_node(node_id)
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

@nodes_bp.route('/api/nodes/<node_id>/identify', methods=['POST'])
def identify_node(node_id):
    """Send identify command to flash the node's LED."""
    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = _node_manager.send_command_to_wifi(node_ip, {"cmd": "identify"})
        return jsonify({'success': bool(result), 'node_id': node_id, 'action': 'identify'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@nodes_bp.route('/api/nodes/<node_id>/reboot', methods=['POST'])
def reboot_node(node_id):
    """Send reboot command to restart the node."""
    print(f"REBOOT: /api/nodes/{node_id}/reboot called", flush=True)
    node = _node_manager.get_node(node_id)
    if not node:
        return jsonify({'success': False, 'error': 'Node not found'}), 404

    node_ip = node.get('ip')
    if not node_ip:
        return jsonify({'success': False, 'error': 'Node has no IP address'}), 400

    try:
        result = _node_manager.send_command_to_wifi(node_ip, {"cmd": "reboot"})
        print(f"   Reboot sent to {node_id} ({node_ip}): {'OK' if result else 'FAIL'}", flush=True)
        return jsonify({'success': bool(result), 'node_id': node_id, 'action': 'reboot'})
    except Exception as e:
        print(f"   Reboot {node_id} failed: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@nodes_bp.route('/api/nodes/<node_id>/update', methods=['POST'])
def update_node_firmware(node_id):
    """Firmware OTA update - not yet supported over network."""
    return jsonify({
        'success': False,
        'error': 'OTA firmware update not yet supported. Flash via USB.',
        'node_id': node_id,
        'action': 'update'
    }), 501

@nodes_bp.route('/api/nodes/ping', methods=['POST'])
def ping_all_nodes():
    """SAFETY ACTION: Health check for all online nodes."""
    print("PING ALL: /api/nodes/ping (all) called", flush=True)

    all_nodes = _node_manager.get_all_nodes(include_offline=False)
    results = {'success': True, 'nodes': [], 'total': len(all_nodes), 'responded': 0}

    for node in all_nodes:
        node_id = node.get('node_id')
        node_ip = node.get('ip')

        if not node_ip:
            results['nodes'].append({'node_id': node_id, 'success': False, 'error': 'No IP'})
            continue

        try:
            result = _ssot.send_udpjson_ping(node_ip)
            # [F03] result is now a dict: {success, attempts, rtt_ms, response}
            success = result.get('success', False) if isinstance(result, dict) else result is not None
            node_result = {'node_id': node_id, 'ip': node_ip, 'success': success}
            if isinstance(result, dict) and result.get('rtt_ms'):
                node_result['rtt_ms'] = result['rtt_ms']
            results['nodes'].append(node_result)
            if success:
                results['responded'] += 1
        except Exception as e:
            results['nodes'].append({'node_id': node_id, 'ip': node_ip, 'success': False, 'error': str(e)})

    results['success'] = results['responded'] == results['total']
    print(f"PING ALL complete: {results['responded']}/{results['total']} nodes responded", flush=True)
    return jsonify(results)
