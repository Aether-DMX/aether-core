"""
AETHER Core — Unified Playback Blueprint
Routes: /api/unified/*
Dependencies: unified_engine, session_factory, looks_sequences_manager, dmx_state, stop_all_playback, get_db
"""

import json
from flask import Blueprint, jsonify, request
from unified_playback import PlaybackType

unified_bp = Blueprint('unified', __name__)

_unified_engine = None
_session_factory = None
_looks_sequences_manager = None
_dmx_state = None
_stop_all_playback = None
_get_db = None


def init_app(unified_engine, session_factory, looks_sequences_manager, dmx_state, stop_all_playback_fn, get_db_fn):
    """Initialize blueprint with required dependencies."""
    global _unified_engine, _session_factory, _looks_sequences_manager, _dmx_state, _stop_all_playback, _get_db
    _unified_engine = unified_engine
    _session_factory = session_factory
    _looks_sequences_manager = looks_sequences_manager
    _dmx_state = dmx_state
    _stop_all_playback = stop_all_playback_fn
    _get_db = get_db_fn


@unified_bp.route('/api/unified/status', methods=['GET'])
def get_unified_status():
    """Get unified playback engine status"""
    return jsonify(_unified_engine.get_status())


@unified_bp.route('/api/unified/play/look/<look_id>', methods=['POST'])
def unified_api_play_look(look_id):
    """Play a Look via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 0)

    look = _looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'success': False, 'error': 'Look not found'}), 404

    session = _session_factory.from_look(look_id, look.to_dict(), universes, fade_ms)
    session_id = _unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'look',
        'name': look.name
    })


@unified_bp.route('/api/unified/play/sequence/<sequence_id>', methods=['POST'])
def unified_api_play_sequence(sequence_id):
    """Play a Sequence via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])

    sequence = _looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'success': False, 'error': 'Sequence not found'}), 404

    session = _session_factory.from_sequence(sequence_id, sequence.to_dict(), universes)
    session_id = _unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'sequence',
        'name': sequence.name
    })


@unified_bp.route('/api/unified/play/chase/<chase_id>', methods=['POST'])
def unified_api_play_chase(chase_id):
    """Play a Chase via unified engine (legacy support)"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])

    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM chases WHERE chase_id = ?', (chase_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'success': False, 'error': 'Chase not found'}), 404

    chase_data = dict(row)
    chase_data['steps'] = json.loads(chase_data.get('steps', '[]'))

    session = _session_factory.from_chase(chase_id, chase_data, universes)
    session_id = _unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'chase',
        'name': chase_data.get('name', chase_id)
    })


@unified_bp.route('/api/unified/play/scene/<scene_id>', methods=['POST'])
def unified_api_play_scene(scene_id):
    """Play a Scene via unified engine (legacy support)"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 0)

    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'success': False, 'error': 'Scene not found'}), 404

    scene_data = dict(row)
    scene_data['channels'] = json.loads(scene_data.get('channels', '{}'))

    session = _session_factory.from_scene(scene_id, scene_data, universes, fade_ms)
    session_id = _unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'scene',
        'name': scene_data.get('name', scene_id)
    })


@unified_bp.route('/api/unified/play/effect', methods=['POST'])
def unified_api_play_effect():
    """Play a built-in effect via unified engine"""
    data = request.get_json() or {}
    effect_type = data.get('effect_type', 'pulse')
    params = data.get('params', {})
    universes = data.get('universes', [2, 3, 4, 5])

    session = _session_factory.from_effect(effect_type, params, universes)
    session_id = _unified_engine.play(session)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'effect',
        'effect_type': effect_type
    })


@unified_bp.route('/api/unified/blackout', methods=['POST'])
def unified_api_blackout():
    """Trigger blackout via unified engine"""
    data = request.get_json() or {}
    universes = data.get('universes', [2, 3, 4, 5])
    fade_ms = data.get('fade_ms', 1000)

    # Get current state for fade (convert list to channel dict)
    fade_from = {}
    if fade_ms > 0:
        for u in universes:
            universe_data = _dmx_state.get_universe(u)
            # Convert list [val0, val1, ...] to dict {1: val0, 2: val1, ...}
            for ch_idx, val in enumerate(universe_data):
                fade_from[ch_idx + 1] = val

    session = _session_factory.blackout(universes, fade_ms)
    session_id = _unified_engine.play(session, fade_from)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'type': 'blackout'
    })


@unified_bp.route('/api/unified/stop', methods=['POST'])
def unified_api_stop():
    """Stop unified playback - uses SSOT stop_all_playback for complete stop"""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    fade_ms = data.get('fade_ms', 0)

    if session_id:
        # Stop specific session only
        _unified_engine.stop_session(session_id, fade_ms)
        return jsonify({'success': True, 'stopped': session_id})
    else:
        # Stop ALL playback sources (shows, chases, effects, unified engine)
        result = _stop_all_playback(blackout=False, fade_ms=fade_ms)
        # Also stop unified engine sessions
        _unified_engine.stop_all(fade_ms)
        return jsonify({'success': True, 'stopped': 'all', 'results': result})


@unified_bp.route('/api/unified/stop/<playback_type>', methods=['POST'])
def unified_api_stop_type(playback_type):
    """Stop all sessions of a specific type"""
    data = request.get_json() or {}
    fade_ms = data.get('fade_ms', 0)

    try:
        ptype = PlaybackType(playback_type)
        _unified_engine.stop_type(ptype, fade_ms)
        return jsonify({'success': True, 'stopped_type': playback_type})
    except ValueError:
        return jsonify({'success': False, 'error': f'Unknown type: {playback_type}'}), 400


@unified_bp.route('/api/unified/pause/<session_id>', methods=['POST'])
def unified_api_pause(session_id):
    """Pause a session"""
    result = _unified_engine.pause(session_id)
    return jsonify({'success': result, 'session_id': session_id})


@unified_bp.route('/api/unified/resume/<session_id>', methods=['POST'])
def unified_api_resume(session_id):
    """Resume a paused session"""
    result = _unified_engine.resume(session_id)
    return jsonify({'success': result, 'session_id': session_id})


@unified_bp.route('/api/unified/sessions', methods=['GET'])
def unified_api_get_sessions():
    """Get all active sessions"""
    sessions = _unified_engine.get_active_sessions()
    return jsonify([{
        'session_id': s.session_id,
        'type': s.playback_type.value,
        'name': s.name,
        'state': s.state.value,
        'priority': s.priority.value,
        'universes': s.universes,
        'elapsed_time': s.elapsed_time,
        'frame_count': s.frame_count,
    } for s in sessions])
