"""
AETHER Core — Groups Blueprint
Routes: /api/groups/*
Dependencies: get_db, SUPABASE_AVAILABLE, get_supabase_service, cloud_submit
"""

import json
import time
from flask import Blueprint, jsonify, request

groups_bp = Blueprint('groups', __name__)

_get_db = None
_SUPABASE_AVAILABLE = False
_get_supabase_service = None
_cloud_submit = None


def init_app(get_db_fn, supabase_available, get_supabase_service_fn, cloud_submit_fn):
    """Initialize blueprint with required dependencies."""
    global _get_db, _SUPABASE_AVAILABLE, _get_supabase_service, _cloud_submit
    _get_db = get_db_fn
    _SUPABASE_AVAILABLE = supabase_available
    _get_supabase_service = get_supabase_service_fn
    _cloud_submit = cloud_submit_fn


@groups_bp.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all fixture groups"""
    conn = _get_db()
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

@groups_bp.route('/api/groups', methods=['POST'])
def create_group():
    """Create a fixture group"""
    data = request.get_json()
    group_id = data.get('group_id', f"group_{int(time.time())}")
    conn = _get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO groups (group_id, name, universe, channels, color)
        VALUES (?, ?, ?, ?, ?)''',
        (group_id, data.get('name', 'New Group'), data.get('universe', 1),
         json.dumps(data.get('channels', [])), data.get('color', '#8b5cf6')))
    conn.commit()
    conn.close()

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                group_data = {'group_id': group_id, 'name': data.get('name', 'New Group'),
                              'universe': data.get('universe', 1), 'channels': data.get('channels', []),
                              'color': data.get('color', '#8b5cf6')}
                _cloud_submit(supabase.sync_group, group_data)
        except Exception as e:
            print(f"Supabase group sync error: {e}", flush=True)

    return jsonify({'success': True, 'group_id': group_id})

@groups_bp.route('/api/groups/<group_id>', methods=['GET'])
def get_group(group_id):
    """Get a single group"""
    conn = _get_db()
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

@groups_bp.route('/api/groups/<group_id>', methods=['PUT'])
def update_group(group_id):
    """Update a group"""
    data = request.get_json()
    conn = _get_db()
    c = conn.cursor()
    c.execute('''UPDATE groups SET name=?, universe=?, channels=?, color=? WHERE group_id=?''',
        (data.get('name'), data.get('universe', 1),
         json.dumps(data.get('channels', [])), data.get('color', '#8b5cf6'), group_id))
    conn.commit()
    conn.close()

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                group_data = {'group_id': group_id, 'name': data.get('name'),
                              'universe': data.get('universe', 1), 'channels': data.get('channels', []),
                              'color': data.get('color', '#8b5cf6')}
                _cloud_submit(supabase.sync_group, group_data)
        except Exception as e:
            print(f"Supabase group sync error: {e}", flush=True)

    return jsonify({'success': True})

@groups_bp.route('/api/groups/<group_id>', methods=['DELETE'])
def delete_group(group_id):
    """Delete a group"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('DELETE FROM groups WHERE group_id = ?', (group_id,))
    conn.commit()
    conn.close()

    # Async delete from Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                _cloud_submit(supabase.delete_group, group_id)
        except Exception as e:
            print(f"Supabase group delete error: {e}", flush=True)

    return jsonify({'success': True})
