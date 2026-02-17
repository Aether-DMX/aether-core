"""
AETHER Core — Preview Blueprint
Routes: /api/preview/*
Dependencies: preview_service
"""

import time
from flask import Blueprint, jsonify, request

preview_bp = Blueprint('preview', __name__)

_preview_service = None


def init_app(preview_service):
    """Initialize blueprint with required dependencies."""
    global _preview_service
    _preview_service = preview_service


@preview_bp.route('/api/preview/sessions', methods=['GET'])
def list_preview_sessions():
    """List all preview sessions"""
    return jsonify({
        'sessions': _preview_service.list_sessions(),
        'status': _preview_service.get_status()
    })


@preview_bp.route('/api/preview/session', methods=['POST'])
def create_preview_session():
    """Create a new preview session for editing."""
    data = request.get_json() or {}

    session_id = data.get('session_id', f"preview_{int(time.time() * 1000)}")
    preview_type = data.get('preview_type', 'look')
    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    universes = data.get('universes', [1])
    fixture_filter = data.get('fixture_filter')

    session = _preview_service.create_session(
        session_id=session_id,
        preview_type=preview_type,
        channels=channels,
        modifiers=modifiers,
        universes=universes,
        fixture_filter=fixture_filter,
    )

    return jsonify({
        'success': True,
        'session_id': session.session_id,
        'mode': session.mode.value,
        'universes': session.universes,
    })


@preview_bp.route('/api/preview/session/<session_id>', methods=['GET'])
def get_preview_session(session_id):
    """Get a preview session's current state"""
    session = _preview_service.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session.session_id,
        'preview_type': session.preview_type,
        'mode': session.mode.value,
        'running': session.running,
        'channels': session.channels,
        'modifiers': session.modifiers,
        'universes': session.universes,
        'frame_count': session.frame_count,
        'last_frame': {
            'channels': session.last_frame.channels,
            'elapsed_ms': session.last_frame.elapsed_ms,
        } if session.last_frame else None,
    })


@preview_bp.route('/api/preview/session/<session_id>', methods=['PUT'])
def update_preview_session(session_id):
    """Update preview session content (immediate re-render)."""
    data = request.get_json() or {}

    success = _preview_service.update_session(
        session_id=session_id,
        channels=data.get('channels'),
        modifiers=data.get('modifiers'),
        universes=data.get('universes'),
    )

    if not success:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({'success': True, 'session_id': session_id})


@preview_bp.route('/api/preview/session/<session_id>', methods=['DELETE'])
def delete_preview_session(session_id):
    """Delete a preview session"""
    success = _preview_service.delete_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id})


@preview_bp.route('/api/preview/session/<session_id>/start', methods=['POST'])
def start_preview_session(session_id):
    """Start preview playback (begins rendering and streaming)"""
    success = _preview_service.start_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id, 'running': True})


@preview_bp.route('/api/preview/session/<session_id>/stop', methods=['POST'])
def stop_preview_session(session_id):
    """Stop preview playback"""
    success = _preview_service.stop_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session_id': session_id, 'running': False})


@preview_bp.route('/api/preview/session/<session_id>/arm', methods=['POST'])
def arm_preview_session(session_id):
    """Arm a preview session for live output. WARNING: Armed sessions output to real universes!"""
    success = _preview_service.arm_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'success': True,
        'session_id': session_id,
        'mode': 'armed',
        'warning': 'Session is now outputting to live universes!'
    })


@preview_bp.route('/api/preview/session/<session_id>/disarm', methods=['POST'])
def disarm_preview_session(session_id):
    """Disarm a preview session (return to sandbox mode)"""
    success = _preview_service.disarm_session(session_id)
    if not success:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'success': True,
        'session_id': session_id,
        'mode': 'sandbox'
    })


@preview_bp.route('/api/preview/frame', methods=['POST'])
def render_single_preview_frame():
    """Render a single preview frame without creating a session."""
    data = request.get_json() or {}

    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    elapsed_time = data.get('elapsed_time', 0.0)
    seed = data.get('seed', 0)

    rendered = _preview_service.render_preview_frame(
        channels=channels,
        modifiers=modifiers,
        elapsed_time=elapsed_time,
        seed=seed,
    )

    return jsonify({
        'success': True,
        'channels': rendered,
        'modifier_count': len([m for m in modifiers if m.get('enabled', True)]),
    })


@preview_bp.route('/api/preview/status', methods=['GET'])
def get_preview_status():
    """Get preview service status"""
    return jsonify(_preview_service.get_status())
