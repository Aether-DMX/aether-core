"""
AETHER Core — Shows & Timeline Playback Blueprint
Routes: /api/shows/*
Dependencies: get_db, show_engine, cloud_submit, supabase_available, get_supabase_service
"""

from flask import Blueprint, jsonify, request
import json
import time
from datetime import datetime

shows_bp = Blueprint('shows', __name__)

# Dependencies injected at registration time
_get_db = None
_show_engine = None
_cloud_submit = None
_supabase_available = False
_get_supabase_service = None


def init_app(get_db, show_engine, cloud_submit, supabase_available, get_supabase_service_fn):
    """Initialize blueprint with required dependencies."""
    global _get_db, _show_engine, _cloud_submit, _supabase_available, _get_supabase_service
    _get_db = get_db
    _show_engine = show_engine
    _cloud_submit = cloud_submit
    _supabase_available = supabase_available
    _get_supabase_service = get_supabase_service_fn


# ─────────────────────────────────────────────────────────
# Shows CRUD
# ─────────────────────────────────────────────────────────

@shows_bp.route('/api/shows', methods=['GET'])
def get_shows():
    """Get all shows"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM shows ORDER BY updated_at DESC')
    rows = c.fetchall()
    conn.close()
    shows = []
    for row in rows:
        shows.append({
            'show_id': row[0], 'name': row[1], 'description': row[2],
            'timeline': json.loads(row[3]) if row[3] else [],
            'duration_ms': row[4], 'created_at': row[5], 'updated_at': row[6],
            'distributed': row[7] if len(row) > 7 else 0
        })
    return jsonify(shows)


@shows_bp.route('/api/shows', methods=['POST'])
def create_show():
    """Create a show"""
    data = request.get_json()
    show_id = data.get('show_id', f"show_{int(time.time())}")
    timeline = data.get('timeline', [])
    # Calculate duration from timeline
    duration_ms = max([e.get('time_ms', 0) for e in timeline]) if timeline else 0
    conn = _get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO shows
        (show_id, name, description, timeline, duration_ms, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (show_id, data.get('name', 'New Show'), data.get('description', ''),
         json.dumps(timeline), duration_ms, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    # Async sync to Supabase (non-blocking)
    if _supabase_available:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                show_data = {'show_id': show_id, 'name': data.get('name', 'New Show'),
                             'description': data.get('description', ''), 'timeline': timeline,
                             'duration_ms': duration_ms}
                _cloud_submit(supabase.sync_show, show_data)
        except Exception as e:
            print(f"Supabase show sync error: {e}", flush=True)

    return jsonify({'success': True, 'show_id': show_id})


@shows_bp.route('/api/shows/<show_id>', methods=['GET'])
def get_show(show_id):
    """Get a single show"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM shows WHERE show_id = ?', (show_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Show not found'}), 404
    return jsonify({
        'show_id': row[0], 'name': row[1], 'description': row[2],
        'timeline': json.loads(row[3]) if row[3] else [],
        'duration_ms': row[4], 'created_at': row[5], 'updated_at': row[6],
        'distributed': row[7] if len(row) > 7 else 0
    })


@shows_bp.route('/api/shows/<show_id>', methods=['PUT'])
def update_show(show_id):
    """Update a show"""
    data = request.get_json()
    timeline = data.get('timeline', [])
    duration_ms = max([e.get('time_ms', 0) for e in timeline]) if timeline else 0
    conn = _get_db()
    c = conn.cursor()
    c.execute('''UPDATE shows SET name=?, description=?, timeline=?, duration_ms=?, updated_at=?
        WHERE show_id=?''',
        (data.get('name'), data.get('description', ''), json.dumps(timeline),
         duration_ms, datetime.now().isoformat(), show_id))
    conn.commit()
    conn.close()

    # Async sync to Supabase (non-blocking)
    if _supabase_available:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                show_data = {'show_id': show_id, 'name': data.get('name'),
                             'description': data.get('description', ''), 'timeline': timeline,
                             'duration_ms': duration_ms}
                _cloud_submit(supabase.sync_show, show_data)
        except Exception as e:
            print(f"Supabase show sync error: {e}", flush=True)

    return jsonify({'success': True})


@shows_bp.route('/api/shows/<show_id>', methods=['DELETE'])
def delete_show(show_id):
    """Delete a show"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('DELETE FROM shows WHERE show_id = ?', (show_id,))
    conn.commit()
    conn.close()

    # Async delete from Supabase (non-blocking)
    if _supabase_available:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                _cloud_submit(supabase.delete_show, show_id)
        except Exception as e:
            print(f"Supabase show delete error: {e}", flush=True)

    return jsonify({'success': True})


# ─────────────────────────────────────────────────────────
# Show Playback Control
# ─────────────────────────────────────────────────────────

@shows_bp.route('/api/shows/<show_id>/play', methods=['POST'])
def play_show(show_id):
    """Play a show timeline"""
    data = request.get_json() or {}
    universe = data.get('universe', 1)
    return jsonify(_show_engine.play_show(show_id, universe))


@shows_bp.route('/api/shows/stop', methods=['POST'])
def stop_show():
    """Stop current show"""
    _show_engine.stop()
    return jsonify({'success': True})


@shows_bp.route('/api/shows/pause', methods=['POST'])
def pause_show():
    """Pause current show"""
    _show_engine.pause()
    return jsonify({'success': True, 'paused': True})


@shows_bp.route('/api/shows/resume', methods=['POST'])
def resume_show():
    """Resume current show"""
    _show_engine.resume()
    return jsonify({'success': True, 'paused': False})


@shows_bp.route('/api/shows/tempo', methods=['POST'])
def set_show_tempo():
    """Set show tempo (0.25 to 4.0)"""
    data = request.get_json() or {}
    tempo = data.get('tempo', 1.0)
    _show_engine.set_tempo(tempo)
    return jsonify({'success': True, 'tempo': _show_engine.tempo})
