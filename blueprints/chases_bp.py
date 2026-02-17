"""
AETHER Core — Chases Blueprint (Legacy)
Routes: /api/chases/*
Dependencies: content_manager, chase_engine, unified_engine, PlaybackType
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from unified_playback import PlaybackType

chases_bp = Blueprint('chases', __name__)

_content_manager = None
_chase_engine = None
_unified_engine = None


def init_app(content_manager, chase_engine, unified_engine):
    """Initialize blueprint with required dependencies."""
    global _content_manager, _chase_engine, _unified_engine
    _content_manager = content_manager
    _chase_engine = chase_engine
    _unified_engine = unified_engine


@chases_bp.route('/api/chases', methods=['GET'])
def get_chases():
    return jsonify(_content_manager.get_chases())

@chases_bp.route('/api/chases', methods=['POST'])
def create_chase():
    return jsonify(_content_manager.create_chase(request.get_json()))

@chases_bp.route('/api/chases/<chase_id>', methods=['GET'])
def get_chase(chase_id):
    chase = _content_manager.get_chase(chase_id)
    return jsonify(chase) if chase else (jsonify({'error': 'Chase not found'}), 404)

@chases_bp.route('/api/chases/<chase_id>', methods=['PUT'])
def update_chase(chase_id):
    """Update an existing chase"""
    data = request.get_json() or {}
    data['chase_id'] = chase_id  # Ensure chase_id is set for the update
    return jsonify(_content_manager.create_chase(data))

@chases_bp.route('/api/chases/<chase_id>', methods=['DELETE'])
def delete_chase(chase_id):
    return jsonify(_content_manager.delete_chase(chase_id))

@chases_bp.route('/api/chases/<chase_id>/play', methods=['POST'])
def play_chase(chase_id):
    data = request.get_json() or {}
    print(f"Chase play request: chase_id={chase_id}, payload={data}", flush=True)
    return jsonify(_content_manager.play_chase(
        chase_id,
        target_channels=data.get('target_channels'),
        universe=data.get('universe'),
        universes=data.get('universes'),
        fade_ms=data.get('fade_ms')
    ))

@chases_bp.route('/api/chases/<chase_id>/stop', methods=['POST'])
def stop_chase(chase_id):
    """Stop a specific chase - stops both old chase engine and unified engine sessions"""
    _chase_engine.stop_all()  # Stop old chase engine (legacy)

    # Stop unified engine sessions that match this chase
    for session_id in list(_unified_engine._sessions.keys()):
        if chase_id in session_id:
            _unified_engine.stop_session(session_id)

    # Also stop all CHASE type sessions in unified engine
    _unified_engine.stop_type(PlaybackType.CHASE)

    return jsonify({'success': True, 'chase_id': chase_id})

@chases_bp.route('/api/chases/health', methods=['GET'])
def get_chase_health():
    """Get health status of all running chases (for debugging)"""
    return jsonify({
        'running': list(_chase_engine.running_chases.keys()),
        'health': _chase_engine.chase_health,
        'timestamp': datetime.now().isoformat()
    })
