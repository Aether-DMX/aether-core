"""
AETHER Core — Cloud Sync Blueprint
Routes: /api/cloud/*
Dependencies: SUPABASE_AVAILABLE, get_supabase_service, get_db, cloud_submit, looks_sequences_manager
"""

from flask import Blueprint, jsonify, request

cloud_bp = Blueprint('cloud', __name__)

_SUPABASE_AVAILABLE = False
_get_supabase_service = None
_get_db = None
_cloud_submit = None
_looks_sequences_manager = None


def init_app(supabase_available, get_supabase_service_fn, get_db_fn, cloud_submit_fn, looks_sequences_manager):
    """Initialize blueprint with required dependencies."""
    global _SUPABASE_AVAILABLE, _get_supabase_service, _get_db, _cloud_submit, _looks_sequences_manager
    _SUPABASE_AVAILABLE = supabase_available
    _get_supabase_service = get_supabase_service_fn
    _get_db = get_db_fn
    _cloud_submit = cloud_submit_fn
    _looks_sequences_manager = looks_sequences_manager


@cloud_bp.route('/api/cloud/status', methods=['GET'])
def get_cloud_status():
    """Get Supabase cloud sync status"""
    if not _SUPABASE_AVAILABLE:
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': 'Supabase service not available'
        })

    supabase = _get_supabase_service()
    if not supabase:
        return jsonify({
            'enabled': False,
            'connected': False,
            'error': 'Supabase service not initialized'
        })

    return jsonify(supabase.get_status())

@cloud_bp.route('/api/cloud/sync', methods=['POST'])
def trigger_cloud_sync():
    """Manually trigger a cloud sync"""
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
        looks_list = [l.to_dict() for l in _looks_sequences_manager.list_looks()]
        sequences_list = [s.to_dict() for s in _looks_sequences_manager.list_sequences()]
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
    if not _SUPABASE_AVAILABLE:
        return jsonify({'success': False, 'error': 'Supabase not available'}), 503

    supabase = _get_supabase_service()
    if not supabase or not supabase.is_enabled():
        return jsonify({'success': False, 'error': 'Supabase not enabled'}), 503

    result = supabase.retry_pending()
    return jsonify({'success': True, 'result': result})


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
