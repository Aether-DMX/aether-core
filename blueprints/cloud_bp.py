"""
AETHER Core — Cloud Sync Blueprint
Routes: /api/cloud/*
Dependencies: SUPABASE_AVAILABLE, get_supabase_service, get_db, cloud_submit,
              looks_sequences_manager, content_manager, app_settings, save_settings, socketio
"""

import json
from flask import Blueprint, jsonify, request

cloud_bp = Blueprint('cloud', __name__)

_SUPABASE_AVAILABLE = False
_get_supabase_service = None
_get_db = None
_cloud_submit = None
_looks_sequences_manager = None
_content_manager = None
_app_settings = None
_save_settings = None
_socketio = None


def _is_cloud_backup_enabled():
    """Check if the cloudBackup premium feature is enabled in app settings."""
    if not _app_settings:
        return False
    return bool(_app_settings.get('features', {}).get('cloudBackup', False))


def init_app(supabase_available, get_supabase_service_fn, get_db_fn, cloud_submit_fn,
             looks_sequences_manager, content_manager=None, app_settings=None,
             save_settings=None, socketio=None):
    """Initialize blueprint with required dependencies."""
    global _SUPABASE_AVAILABLE, _get_supabase_service, _get_db, _cloud_submit
    global _looks_sequences_manager, _content_manager, _app_settings, _save_settings, _socketio
    _SUPABASE_AVAILABLE = supabase_available
    _get_supabase_service = get_supabase_service_fn
    _get_db = get_db_fn
    _cloud_submit = cloud_submit_fn
    _looks_sequences_manager = looks_sequences_manager
    _content_manager = content_manager
    _app_settings = app_settings
    _save_settings = save_settings
    _socketio = socketio


@cloud_bp.route('/api/cloud/status', methods=['GET'])
def get_cloud_status():
    """Get Supabase cloud sync status (always accessible, includes feature flag)"""
    feature_enabled = _is_cloud_backup_enabled()

    if not _SUPABASE_AVAILABLE:
        return jsonify({
            'enabled': False,
            'connected': False,
            'featureEnabled': feature_enabled,
            'error': 'Supabase service not available'
        })

    supabase = _get_supabase_service()
    if not supabase:
        return jsonify({
            'enabled': False,
            'connected': False,
            'featureEnabled': feature_enabled,
            'error': 'Supabase service not initialized'
        })

    status = supabase.get_status()
    status['featureEnabled'] = feature_enabled
    return jsonify(status)

@cloud_bp.route('/api/cloud/sync', methods=['POST'])
def trigger_cloud_sync():
    """Manually trigger a cloud sync"""
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    # Gather local data
    conn = _get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM nodes')
    nodes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM scenes')
    scenes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM chases')
    chases = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM fixtures')
    fixtures = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM shows')
    shows = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM schedules')
    schedules = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM groups')
    groups = [dict(row) for row in c.fetchall()]

    conn.close()

    # Get looks and sequences
    looks_list = []
    sequences_list = []
    try:
        looks_list = [l.to_dict() for l in _looks_sequences_manager.get_all_looks()]
        sequences_list = [s.to_dict() for s in _looks_sequences_manager.get_all_sequences()]
    except Exception as e:
        print(f"⚠️ Failed to get looks/sequences for sync: {e}")

    # Perform sync
    result = supabase.initial_sync(
        nodes=nodes,
        looks=looks_list,
        sequences=sequences_list,
        scenes=scenes,
        chases=chases,
        fixtures=fixtures,
        shows=shows,
        schedules=schedules,
        groups=groups
    )

    return jsonify({'success': True, 'result': result})

@cloud_bp.route('/api/cloud/retry-pending', methods=['POST'])
def retry_pending_sync():
    """Retry pending sync operations"""
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    result = supabase.retry_pending()
    return jsonify({'success': True, 'result': result})


@cloud_bp.route('/api/cloud/backup', methods=['POST'])
def cloud_backup():
    """
    Full backup: push ALL local data + settings → Supabase.
    Gathers from SQLite + in-memory looks/sequences + app settings.
    """
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    if not supabase.is_connected():
        # Try reconnect first
        reconnect_result = supabase.reconnect()
        if not reconnect_result.get("success"):
            return jsonify({'success': False, 'error': 'Not connected to Supabase'}), 503

    # Gather local data from SQLite
    conn = _get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM nodes')
    nodes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM scenes')
    scenes = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM chases')
    chases = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM fixtures')
    fixtures = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM shows')
    shows = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM schedules')
    schedules = [dict(row) for row in c.fetchall()]

    c.execute('SELECT * FROM groups')
    groups = [dict(row) for row in c.fetchall()]

    conn.close()

    # Get looks and sequences from in-memory manager
    looks_list = []
    sequences_list = []
    try:
        looks_list = [l.to_dict() for l in _looks_sequences_manager.get_all_looks()]
        sequences_list = [s.to_dict() for s in _looks_sequences_manager.get_all_sequences()]
    except Exception as e:
        print(f"⚠️ Failed to get looks/sequences for backup: {e}")

    # Perform sync
    result = supabase.initial_sync(
        nodes=nodes,
        looks=looks_list,
        sequences=sequences_list,
        scenes=scenes,
        chases=chases,
        fixtures=fixtures,
        shows=shows,
        schedules=schedules,
        groups=groups
    )

    # Also backup settings
    if _app_settings:
        supabase.sync_settings(_app_settings)
        result["settings"] = True

    return jsonify({'success': True, 'result': result})


@cloud_bp.route('/api/cloud/restore', methods=['POST'])
def cloud_restore():
    """
    Full restore: pull ALL data from Supabase → local SQLite + settings.
    Overwrites local data with cloud data. Emits Socket.IO refresh events.
    """
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    if not supabase.is_connected():
        reconnect_result = supabase.reconnect()
        if not reconnect_result.get("success"):
            return jsonify({'success': False, 'error': 'Not connected to Supabase'}), 503

    # Fetch all data from cloud
    cloud_data = supabase.fetch_all()
    if "error" in cloud_data:
        return jsonify({'success': False, 'error': cloud_data["error"]}), 500

    results = {
        "scenes": 0, "chases": 0, "shows": 0,
        "fixtures": 0, "schedules": 0, "groups": 0,
        "nodes": 0, "looks": 0, "sequences": 0,
        "settings": False
    }

    conn = _get_db()
    c = conn.cursor()

    try:
        # Restore scenes
        for scene in cloud_data.get("scenes", []):
            channels = scene.get("channels")
            if isinstance(channels, (dict, list)):
                channels = json.dumps(channels)
            c.execute('''INSERT OR REPLACE INTO scenes
                        (scene_id, name, description, channels, universe, fade_ms, curve, color, icon)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (scene.get("scene_id"), scene.get("name"), scene.get("description"),
                      channels, scene.get("universe", 1),
                      scene.get("fade_ms", 500), scene.get("curve", "linear"),
                      scene.get("color"), scene.get("icon")))
            results["scenes"] += 1

        # Restore chases
        for chase in cloud_data.get("chases", []):
            steps = chase.get("steps")
            if isinstance(steps, (dict, list)):
                steps = json.dumps(steps)
            c.execute('''INSERT OR REPLACE INTO chases
                        (chase_id, name, description, steps, bpm, loop, fade_ms, universe, color)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (chase.get("chase_id"), chase.get("name"), chase.get("description"),
                      steps, chase.get("bpm", 120), chase.get("loop", True),
                      chase.get("fade_ms", 0), chase.get("universe", 1),
                      chase.get("color")))
            results["chases"] += 1

        # Restore shows
        for show in cloud_data.get("shows", []):
            timeline = show.get("timeline")
            if isinstance(timeline, (dict, list)):
                timeline = json.dumps(timeline)
            c.execute('''INSERT OR REPLACE INTO shows
                        (show_id, name, description, timeline, duration_ms)
                        VALUES (?, ?, ?, ?, ?)''',
                     (show.get("show_id"), show.get("name"), show.get("description"),
                      timeline, show.get("duration_ms", 0)))
            results["shows"] += 1

        # Restore fixtures
        for fixture in cloud_data.get("fixtures", []):
            channel_map = fixture.get("channel_map")
            if isinstance(channel_map, (dict, list)):
                channel_map = json.dumps(channel_map)
            c.execute('''INSERT OR REPLACE INTO fixtures
                        (fixture_id, name, type, manufacturer, model,
                         channel_count, channel_map, start_channel, universe)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (fixture.get("fixture_id"), fixture.get("name"),
                      fixture.get("type", "generic"), fixture.get("manufacturer"),
                      fixture.get("model"), fixture.get("channel_count", 1),
                      channel_map, fixture.get("start_channel", 1),
                      fixture.get("universe", 1)))
            results["fixtures"] += 1

        # Restore schedules
        for sched in cloud_data.get("schedules", []):
            action_params = sched.get("action_params")
            if isinstance(action_params, (dict, list)):
                action_params = json.dumps(action_params)
            c.execute('''INSERT OR REPLACE INTO schedules
                        (schedule_id, name, cron, action_type, action_id,
                         action_params, enabled, last_run, next_run)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (sched.get("schedule_id"), sched.get("name"),
                      sched.get("cron"), sched.get("action_type"),
                      sched.get("action_id"), action_params,
                      sched.get("enabled", True),
                      sched.get("last_run"), sched.get("next_run")))
            results["schedules"] += 1

        # Restore groups
        for group in cloud_data.get("groups", []):
            channels = group.get("channels")
            if isinstance(channels, (dict, list)):
                channels = json.dumps(channels)
            c.execute('''INSERT OR REPLACE INTO groups
                        (group_id, name, universe, channels, color)
                        VALUES (?, ?, ?, ?, ?)''',
                     (group.get("group_id"), group.get("name"),
                      group.get("universe", 1), channels,
                      group.get("color", "#8b5cf6")))
            results["groups"] += 1

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"⚠️ Cloud restore failed during SQLite writes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

    # Restore settings
    cloud_settings = supabase.fetch_settings()
    if cloud_settings and _app_settings is not None and _save_settings is not None:
        _app_settings.update(cloud_settings)
        _save_settings(_app_settings)
        results["settings"] = True

    # Emit Socket.IO refresh events so the UI reloads
    if _socketio:
        _socketio.emit('cloud_restore_complete', results)
        _socketio.emit('settings_update', {'category': 'all', 'data': _app_settings})
        _socketio.emit('content_refresh', {'source': 'cloud_restore'})

    total = sum(v for k, v in results.items() if isinstance(v, int))
    print(f"☁️ Cloud restore complete: {total} records restored")

    return jsonify({'success': True, 'result': results})


@cloud_bp.route('/api/cloud/reconnect', methods=['POST'])
def cloud_reconnect():
    """Attempt to reconnect the Supabase client"""
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase service not initialized'}), 503

    result = supabase.reconnect()
    return jsonify(result)


@cloud_bp.route('/api/cloud/clear-pending', methods=['POST'])
def cloud_clear_pending():
    """Clear all pending sync operations"""
    if not _is_cloud_backup_enabled():
        return jsonify({'success': False, 'error': 'Cloud Backup is a premium feature', 'premium_required': True}), 403
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase service not initialized'}), 503

    count_before = supabase.get_pending_count()
    supabase._pending_queue.clear()
    return jsonify({
        'success': True,
        'cleared': count_before,
        'remaining': supabase.get_pending_count()
    })


@cloud_bp.route('/api/cloud/log-conversation', methods=['POST'])
def log_cloud_conversation():
    """Log an AI conversation to Supabase."""
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503
    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503
    data = request.get_json() or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': 'session_id required'}), 400
    messages = data.get('messages', [])
    metadata = data.get('metadata', {})
    def _log():
        try:
            supabase.log_conversation(session_id, messages, metadata)
        except Exception as e:
            print(f"⚠️ Conversation logging failed (will retry via queue): {e}")
    _cloud_submit(_log)
    return jsonify({'success': True, 'queued': True})

@cloud_bp.route('/api/cloud/log-message', methods=['POST'])
def log_cloud_message():
    """Log a single AI message to Supabase"""
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503
    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503
    data = request.get_json() or {}
    conversation_id = data.get('conversation_id')
    role = data.get('role')
    if not conversation_id or not role:
        return jsonify({'success': False, 'error': 'conversation_id and role required'}), 400
    def _log():
        try:
            supabase.log_message(
                conversation_id=conversation_id, role=role,
                content=data.get('content', ''),
                tokens_used=data.get('tokens_used'),
                tool_calls=data.get('tool_calls'),
            )
        except Exception as e:
            print(f"Message logging failed: {e}")
    _cloud_submit(_log)
    return jsonify({'success': True, 'queued': True})

@cloud_bp.route('/api/cloud/log-learning', methods=['POST'])
def log_cloud_learning():
    """Log an AI learning event to Supabase."""
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503
    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503
    data = request.get_json() or {}
    def _log():
        try:
            supabase.log_learning(data)
        except Exception as e:
            print(f"⚠️ Cloud learning log failed (will retry via queue): {e}")
    _cloud_submit(_log)
    return jsonify({'success': True, 'queued': True})

@cloud_bp.route('/api/cloud/feedback', methods=['POST'])
def log_cloud_feedback():
    """Log user feedback on an AI response"""
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503
    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503
    data = request.get_json() or {}
    def _log():
        try:
            supabase.log_feedback(data)
        except Exception as e:
            print(f"Feedback log failed: {e}")
    _cloud_submit(_log)
    return jsonify({'success': True, 'queued': True})

@cloud_bp.route('/api/cloud/learnings', methods=['GET'])
def get_cloud_learnings():
    """Fetch AI learnings from Supabase"""
    if not _SUPABASE_AVAILABLE:
        return jsonify({'learnings': [], 'source': 'unavailable'})
    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'learnings': [], 'source': 'disabled'})
    category = request.args.get('category')
    limit = int(request.args.get('limit', 20))
    learnings = supabase.get_learned_patterns(category=category, limit=limit)
    return jsonify({'learnings': learnings, 'source': 'cloud'})
