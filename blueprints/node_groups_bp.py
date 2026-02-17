"""
AETHER Core — Node Groups Blueprint
Routes: /api/node-groups/*
Dependencies: get_db
"""

import json
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request

node_groups_bp = Blueprint('node_groups', __name__)

_get_db = None


def init_app(get_db_fn):
    """Initialize blueprint with required dependencies."""
    global _get_db
    _get_db = get_db_fn


@node_groups_bp.route('/api/node-groups', methods=['GET'])
def get_node_groups():
    """List all node groups with their nodes and computed stats"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM node_groups ORDER BY name')
    groups = [dict(row) for row in c.fetchall()]

    for group in groups:
        c.execute('SELECT * FROM nodes WHERE group_id = ? ORDER BY channel_offset', (group['group_id'],))
        group['nodes'] = [dict(row) for row in c.fetchall()]
        group['node_count'] = len(group['nodes'])
        group['total_capacity'] = len(group['nodes']) * 512

        c.execute('''SELECT MAX(start_channel + channel_count - 1) as max_ch
                     FROM fixtures WHERE universe IN (
                         SELECT universe FROM nodes WHERE group_id = ?
                     )''', (group['group_id'],))
        result = c.fetchone()
        group['channel_ceiling'] = result['max_ch'] if result and result['max_ch'] else 0

        c.execute('''SELECT COUNT(*) as cnt FROM fixtures WHERE universe IN (
                         SELECT universe FROM nodes WHERE group_id = ?
                     )''', (group['group_id'],))
        result = c.fetchone()
        group['fixture_count'] = result['cnt'] if result else 0

    conn.close()
    return jsonify(groups)


@node_groups_bp.route('/api/node-groups', methods=['POST'])
def create_node_group():
    """Create a new node group"""
    data = request.get_json() or {}
    group_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    conn = _get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO node_groups (group_id, name, description, channel_mode, manual_channel_count, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (group_id, data.get('name', 'New Group'), data.get('description', ''),
               data.get('channel_mode', 'auto'), data.get('manual_channel_count', 512), now, now))
    conn.commit()
    conn.close()

    return jsonify({'success': True, 'group_id': group_id})


@node_groups_bp.route('/api/node-groups/<group_id>', methods=['GET'])
def get_node_group(group_id):
    """Get a single node group with full details"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM node_groups WHERE group_id = ?', (group_id,))
    group = c.fetchone()
    if not group:
        conn.close()
        return jsonify({'error': 'Node group not found'}), 404

    group = dict(group)

    c.execute('SELECT * FROM nodes WHERE group_id = ? ORDER BY channel_offset', (group_id,))
    group['nodes'] = [dict(row) for row in c.fetchall()]

    c.execute('''SELECT f.*, n.name as node_name FROM fixtures f
                 JOIN nodes n ON f.universe = n.universe
                 WHERE n.group_id = ?
                 ORDER BY f.start_channel''', (group_id,))
    group['fixtures'] = [dict(row) for row in c.fetchall()]

    conn.close()
    return jsonify(group)


@node_groups_bp.route('/api/node-groups/<group_id>', methods=['PUT'])
def update_node_group(group_id):
    """Update a node group"""
    data = request.get_json() or {}
    now = datetime.now().isoformat()

    conn = _get_db()
    c = conn.cursor()
    c.execute('''UPDATE node_groups SET
                 name = COALESCE(?, name),
                 description = COALESCE(?, description),
                 channel_mode = COALESCE(?, channel_mode),
                 manual_channel_count = COALESCE(?, manual_channel_count),
                 node_order = COALESCE(?, node_order),
                 updated_at = ?
                 WHERE group_id = ?''',
              (data.get('name'), data.get('description'), data.get('channel_mode'),
               data.get('manual_channel_count'), data.get('node_order'), now, group_id))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@node_groups_bp.route('/api/node-groups/<group_id>', methods=['DELETE'])
def delete_node_group(group_id):
    """Delete a node group (removes nodes from group first)"""
    conn = _get_db()
    c = conn.cursor()

    c.execute('UPDATE nodes SET group_id = NULL, channel_offset = 0 WHERE group_id = ?', (group_id,))
    c.execute('DELETE FROM node_groups WHERE group_id = ?', (group_id,))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@node_groups_bp.route('/api/node-groups/<group_id>/nodes', methods=['POST'])
def add_node_to_group(group_id):
    """Add a node to a group"""
    data = request.get_json() or {}
    node_id = data.get('node_id')
    if not node_id:
        return jsonify({'error': 'node_id is required'}), 400

    conn = _get_db()
    c = conn.cursor()

    c.execute('SELECT MAX(channel_offset) as max_offset FROM nodes WHERE group_id = ?', (group_id,))
    result = c.fetchone()
    next_offset = (result['max_offset'] or 0) + 512 if result['max_offset'] is not None else 0

    c.execute('UPDATE nodes SET group_id = ?, channel_offset = ? WHERE node_id = ?',
              (group_id, next_offset, node_id))
    conn.commit()
    conn.close()

    return jsonify({'success': True, 'channel_offset': next_offset})


@node_groups_bp.route('/api/node-groups/<group_id>/nodes/<node_id>', methods=['DELETE'])
def remove_node_from_group(group_id, node_id):
    """Remove a node from a group"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('UPDATE nodes SET group_id = NULL, channel_offset = 0 WHERE node_id = ? AND group_id = ?',
              (node_id, group_id))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@node_groups_bp.route('/api/node-groups/<group_id>/reorder', methods=['POST'])
def reorder_nodes_in_group(group_id):
    """Reorder nodes in a group (updates channel offsets)"""
    data = request.get_json() or {}
    node_ids = data.get('node_ids', [])

    conn = _get_db()
    c = conn.cursor()

    for i, node_id in enumerate(node_ids):
        offset = i * 512
        c.execute('UPDATE nodes SET channel_offset = ? WHERE node_id = ? AND group_id = ?',
                  (offset, node_id, group_id))

    c.execute('UPDATE node_groups SET node_order = ?, updated_at = ? WHERE group_id = ?',
              (json.dumps(node_ids), datetime.now().isoformat(), group_id))

    conn.commit()
    conn.close()

    return jsonify({'success': True})


@node_groups_bp.route('/api/node-groups/<group_id>/channel-map', methods=['GET'])
def get_group_channel_map(group_id):
    """Get channel usage visualization for a node group"""
    conn = _get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM nodes WHERE group_id = ? ORDER BY channel_offset', (group_id,))
    nodes = [dict(row) for row in c.fetchall()]

    channel_map = []
    for node in nodes:
        c.execute('''SELECT fixture_id, name, start_channel, channel_count
                     FROM fixtures WHERE universe = ? ORDER BY start_channel''', (node['universe'],))
        fixtures = [dict(row) for row in c.fetchall()]

        ceiling = 0
        for f in fixtures:
            end_ch = f['start_channel'] + f['channel_count'] - 1
            if end_ch > ceiling:
                ceiling = end_ch

        channel_map.append({
            'node_id': node['node_id'],
            'node_name': node['name'] or node['node_id'],
            'ip': node['ip'],
            'universe': node['universe'],
            'channel_offset': node['channel_offset'],
            'channel_ceiling': ceiling,
            'fixtures': fixtures,
            'used_channels': ceiling,
            'total_channels': 512
        })

    conn.close()
    return jsonify(channel_map)
