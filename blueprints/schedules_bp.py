"""
AETHER Core — Schedules & Timers Blueprint
Routes: /api/schedules/*, /api/timers/*
Dependencies: get_db, schedule_runner, timer_runner, cloud_submit,
              SUPABASE_AVAILABLE, get_supabase_service, socketio
"""

from flask import Blueprint, jsonify, request
import json
import time
from datetime import datetime

schedules_bp = Blueprint('schedules', __name__)

# Dependencies injected at registration time
_get_db = None
_schedule_runner = None
_timer_runner = None
_cloud_submit = None
_SUPABASE_AVAILABLE = False
_get_supabase_service = None
_socketio = None


def init_app(get_db, schedule_runner, timer_runner, cloud_submit,
             SUPABASE_AVAILABLE, get_supabase_service, socketio):
    """Initialize blueprint with required dependencies."""
    global _get_db, _schedule_runner, _timer_runner, _cloud_submit
    global _SUPABASE_AVAILABLE, _get_supabase_service, _socketio
    _get_db = get_db
    _schedule_runner = schedule_runner
    _timer_runner = timer_runner
    _cloud_submit = cloud_submit
    _SUPABASE_AVAILABLE = SUPABASE_AVAILABLE
    _get_supabase_service = get_supabase_service
    _socketio = socketio


# ─────────────────────────────────────────────────────────
# Schedules Routes
# ─────────────────────────────────────────────────────────

@schedules_bp.route('/api/schedules', methods=['GET'])
def get_schedules():
    """Get all schedules"""
    conn = _get_db()
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


@schedules_bp.route('/api/schedules', methods=['POST'])
def create_schedule():
    """Create a schedule"""
    data = request.get_json()
    schedule_id = data.get('schedule_id', f"sched_{int(time.time())}")
    conn = _get_db()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO schedules
        (schedule_id, name, cron, action_type, action_id, enabled)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (schedule_id, data.get('name', 'New Schedule'), data.get('cron', '0 8 * * *'),
         data.get('action_type', 'scene'), data.get('action_id'), data.get('enabled', True)))
    conn.commit()
    conn.close()
    _schedule_runner.update_schedules()

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                sched_data = {'schedule_id': schedule_id, 'name': data.get('name', 'New Schedule'),
                              'cron': data.get('cron', '0 8 * * *'), 'action_type': data.get('action_type', 'scene'),
                              'action_id': data.get('action_id'), 'enabled': data.get('enabled', True)}
                _cloud_submit(supabase.sync_schedule, sched_data)
        except Exception as e:
            print(f"Supabase schedule sync error: {e}", flush=True)

    return jsonify({'success': True, 'schedule_id': schedule_id})


@schedules_bp.route('/api/schedules/<schedule_id>', methods=['GET'])
def get_schedule(schedule_id):
    """Get a single schedule"""
    conn = _get_db()
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


@schedules_bp.route('/api/schedules/<schedule_id>', methods=['PUT'])
def update_schedule(schedule_id):
    """Update a schedule"""
    data = request.get_json()
    conn = _get_db()
    c = conn.cursor()
    c.execute('''UPDATE schedules SET name=?, cron=?, action_type=?, action_id=?, enabled=?
        WHERE schedule_id=?''',
        (data.get('name'), data.get('cron'), data.get('action_type'),
         data.get('action_id'), data.get('enabled', True), schedule_id))
    conn.commit()
    conn.close()
    _schedule_runner.update_schedules()

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                sched_data = {'schedule_id': schedule_id, 'name': data.get('name'),
                              'cron': data.get('cron'), 'action_type': data.get('action_type'),
                              'action_id': data.get('action_id'), 'enabled': data.get('enabled', True)}
                _cloud_submit(supabase.sync_schedule, sched_data)
        except Exception as e:
            print(f"Supabase schedule sync error: {e}", flush=True)

    return jsonify({'success': True})


@schedules_bp.route('/api/schedules/<schedule_id>', methods=['DELETE'])
def delete_schedule(schedule_id):
    """Delete a schedule"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('DELETE FROM schedules WHERE schedule_id = ?', (schedule_id,))
    conn.commit()
    conn.close()
    _schedule_runner.update_schedules()

    # Async delete from Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        try:
            supabase = _get_supabase_service()
            if supabase and supabase.is_enabled():
                _cloud_submit(supabase.delete_schedule, schedule_id)
        except Exception as e:
            print(f"Supabase schedule delete error: {e}", flush=True)

    return jsonify({'success': True})


@schedules_bp.route('/api/schedules/<schedule_id>/trigger', methods=['POST'])
def trigger_schedule(schedule_id):
    """Manually trigger a schedule"""
    return jsonify(_schedule_runner.run_schedule(schedule_id))


# ─────────────────────────────────────────────────────────
# Timers Routes
# ─────────────────────────────────────────────────────────

@schedules_bp.route('/api/timers', methods=['GET'])
def get_timers():
    """Get all timers with current remaining time"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM timers ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()

    timers = []
    now = time.time() * 1000

    for row in rows:
        timer_id, name, duration_ms, remaining_ms, action_type, action_id, status, started_at, created_at = row

        # Calculate actual remaining time for running timers
        actual_remaining = remaining_ms
        if status == 'running' and started_at:
            started_timestamp = datetime.fromisoformat(started_at).timestamp() * 1000
            elapsed = now - started_timestamp
            actual_remaining = max(0, remaining_ms - elapsed)
            if actual_remaining <= 0:
                # Timer completed, update status
                status = 'completed'

        timers.append({
            'timer_id': timer_id,
            'name': name,
            'duration_ms': duration_ms,
            'remaining_ms': int(actual_remaining),
            'action_type': action_type,
            'action_id': action_id,
            'status': status,
            'started_at': started_at,
            'created_at': created_at,
            'running': status == 'running'
        })

    return jsonify(timers)


@schedules_bp.route('/api/timers', methods=['POST'])
def create_timer():
    """Create a new timer"""
    data = request.get_json() or {}
    timer_id = data.get('timer_id', f"timer_{int(time.time() * 1000)}")
    name = data.get('name', 'New Timer')
    duration_ms = data.get('duration_ms', 60000)  # Default 1 minute

    conn = _get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO timers
        (timer_id, name, duration_ms, remaining_ms, action_type, action_id, status)
        VALUES (?, ?, ?, ?, ?, ?, 'stopped')''',
        (timer_id, name, duration_ms, duration_ms,
         data.get('action_type'), data.get('action_id')))
    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'timer_id': timer_id,
        'name': name,
        'duration_ms': duration_ms,
        'remaining_ms': duration_ms,
        'status': 'stopped'
    })


@schedules_bp.route('/api/timers/<timer_id>', methods=['GET'])
def get_timer(timer_id):
    """Get a single timer"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Timer not found'}), 404

    timer_id, name, duration_ms, remaining_ms, action_type, action_id, status, started_at, created_at = row
    return jsonify({
        'timer_id': timer_id,
        'name': name,
        'duration_ms': duration_ms,
        'remaining_ms': remaining_ms,
        'action_type': action_type,
        'action_id': action_id,
        'status': status,
        'started_at': started_at,
        'created_at': created_at
    })


@schedules_bp.route('/api/timers/<timer_id>', methods=['PUT'])
def update_timer(timer_id):
    """Update a timer"""
    data = request.get_json() or {}
    conn = _get_db()
    c = conn.cursor()
    c.execute('''UPDATE timers SET name=?, duration_ms=?, action_type=?, action_id=?
        WHERE timer_id=?''',
        (data.get('name'), data.get('duration_ms'),
         data.get('action_type'), data.get('action_id'), timer_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@schedules_bp.route('/api/timers/<timer_id>', methods=['DELETE'])
def delete_timer(timer_id):
    """Delete a timer"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('DELETE FROM timers WHERE timer_id = ?', (timer_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@schedules_bp.route('/api/timers/<timer_id>/start', methods=['POST'])
def start_timer(timer_id):
    """Start/resume a timer"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT remaining_ms, status FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    remaining_ms, current_status = row

    if current_status == 'running':
        conn.close()
        return jsonify({'success': True, 'message': 'Timer already running'})

    # If completed or zero remaining, reset to full duration
    if remaining_ms <= 0 or current_status == 'completed':
        c.execute('SELECT duration_ms FROM timers WHERE timer_id = ?', (timer_id,))
        remaining_ms = c.fetchone()[0]

    now = datetime.now().isoformat()
    c.execute('''UPDATE timers SET status='running', started_at=?, remaining_ms=?
        WHERE timer_id=?''', (now, remaining_ms, timer_id))
    conn.commit()
    conn.close()

    # Start background timer check
    _timer_runner.start_timer(timer_id, remaining_ms)

    return jsonify({'success': True, 'status': 'running', 'remaining_ms': remaining_ms})


@schedules_bp.route('/api/timers/<timer_id>/pause', methods=['POST'])
def pause_timer(timer_id):
    """Pause a running timer"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT remaining_ms, started_at, status FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    remaining_ms, started_at, status = row

    if status != 'running':
        conn.close()
        return jsonify({'success': True, 'message': 'Timer not running'})

    # Calculate actual remaining
    if started_at:
        started_timestamp = datetime.fromisoformat(started_at).timestamp() * 1000
        elapsed = (time.time() * 1000) - started_timestamp
        remaining_ms = max(0, remaining_ms - elapsed)

    c.execute('''UPDATE timers SET status='paused', remaining_ms=?, started_at=NULL
        WHERE timer_id=?''', (int(remaining_ms), timer_id))
    conn.commit()
    conn.close()

    _timer_runner.stop_timer(timer_id)

    return jsonify({'success': True, 'status': 'paused', 'remaining_ms': int(remaining_ms)})


@schedules_bp.route('/api/timers/<timer_id>/reset', methods=['POST'])
def reset_timer(timer_id):
    """Reset a timer to its full duration"""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT duration_ms FROM timers WHERE timer_id = ?', (timer_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Timer not found'}), 404

    duration_ms = row[0]

    c.execute('''UPDATE timers SET status='stopped', remaining_ms=?, started_at=NULL
        WHERE timer_id=?''', (duration_ms, timer_id))
    conn.commit()
    conn.close()

    _timer_runner.stop_timer(timer_id)

    return jsonify({'success': True, 'status': 'stopped', 'remaining_ms': duration_ms})
