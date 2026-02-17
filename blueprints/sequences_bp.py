"""
AETHER Core — Sequences Blueprint
Routes: /api/sequences/*
Dependencies: looks_sequences_manager, arbitration, render_engine, chase_engine,
              node_manager, unified_play_sequence, unified_stop, unified_get_status,
              unified_engine, cloud_submit, SUPABASE_AVAILABLE, get_supabase_service
"""

import time
from flask import Blueprint, jsonify, request
from looks_sequences import Sequence, SequenceStep, validate_sequence_data
from unified_playback import PlaybackType

sequences_bp = Blueprint('sequences', __name__)

_looks_sequences_manager = None
_arbitration = None
_render_engine = None
_chase_engine = None
_node_manager = None
_unified_play_sequence = None
_unified_stop = None
_unified_get_status = None
_unified_engine = None
_cloud_submit = None
_SUPABASE_AVAILABLE = False
_get_supabase_service = None


def init_app(looks_sequences_manager, arbitration, render_engine, chase_engine,
             node_manager, unified_play_sequence_fn, unified_stop_fn,
             unified_get_status_fn, unified_engine, cloud_submit_fn,
             SUPABASE_AVAILABLE_flag, get_supabase_service_fn):
    """Initialize blueprint with required dependencies."""
    global _looks_sequences_manager, _arbitration, _render_engine, _chase_engine
    global _node_manager, _unified_play_sequence, _unified_stop, _unified_get_status
    global _unified_engine, _cloud_submit, _SUPABASE_AVAILABLE, _get_supabase_service
    _looks_sequences_manager = looks_sequences_manager
    _arbitration = arbitration
    _render_engine = render_engine
    _chase_engine = chase_engine
    _node_manager = node_manager
    _unified_play_sequence = unified_play_sequence_fn
    _unified_stop = unified_stop_fn
    _unified_get_status = unified_get_status_fn
    _unified_engine = unified_engine
    _cloud_submit = cloud_submit_fn
    _SUPABASE_AVAILABLE = SUPABASE_AVAILABLE_flag
    _get_supabase_service = get_supabase_service_fn


@sequences_bp.route('/api/sequences', methods=['GET'])
def get_sequences():
    """Get all Sequences"""
    sequences = _looks_sequences_manager.get_all_sequences()
    return jsonify([s.to_dict() for s in sequences])

@sequences_bp.route('/api/sequences', methods=['POST'])
def create_sequence():
    """Create a new Sequence"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_sequence_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create Sequence object
    steps = [SequenceStep.from_dict(s) for s in data.get('steps', [])]
    sequence = Sequence(
        sequence_id=data.get('sequence_id', f"sequence_{int(time.time() * 1000)}"),
        name=data['name'],
        steps=steps,
        bpm=data.get('bpm', 120),
        loop=data.get('loop', True),
        color=data.get('color', 'green'),
        description=data.get('description', ''),
    )

    result = _looks_sequences_manager.create_sequence(sequence)

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        supabase = _get_supabase_service()
        if supabase and supabase.is_enabled():
            _cloud_submit(supabase.sync_sequence, result.to_dict())

    return jsonify({'success': True, 'sequence': result.to_dict()})

@sequences_bp.route('/api/sequences/<sequence_id>', methods=['GET'])
def get_sequence(sequence_id):
    """Get a Sequence by ID"""
    sequence = _looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'error': 'Sequence not found'}), 404
    return jsonify(sequence.to_dict())

@sequences_bp.route('/api/sequences/<sequence_id>', methods=['PUT'])
def update_sequence(sequence_id):
    """Update an existing Sequence"""
    data = request.get_json() or {}

    # Validate if steps are being updated
    if 'steps' in data:
        valid, error = validate_sequence_data(data)
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = _looks_sequences_manager.update_sequence(sequence_id, data)
    if not result:
        return jsonify({'error': 'Sequence not found'}), 404

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        supabase = _get_supabase_service()
        if supabase and supabase.is_enabled():
            _cloud_submit(supabase.sync_sequence, result.to_dict())

    return jsonify({'success': True, 'sequence': result.to_dict()})

@sequences_bp.route('/api/sequences/<sequence_id>', methods=['DELETE'])
def delete_sequence(sequence_id):
    """Delete a Sequence"""
    success = _looks_sequences_manager.delete_sequence(sequence_id)
    if not success:
        return jsonify({'error': 'Sequence not found'}), 404

    # Note: Supabase delete is not implemented yet
    return jsonify({'success': True, 'sequence_id': sequence_id})

@sequences_bp.route('/api/sequences/<sequence_id>/versions', methods=['GET'])
def get_sequence_versions(sequence_id):
    """Get version history for a Sequence"""
    versions = _looks_sequences_manager.get_versions(sequence_id, 'sequence')
    return jsonify({'success': True, 'versions': versions})

@sequences_bp.route('/api/sequences/<sequence_id>/versions/<version_id>/revert', methods=['POST'])
def revert_sequence_version(sequence_id, version_id):
    """Revert a Sequence to a specific version"""
    result = _looks_sequences_manager.revert_to_version(version_id)
    if not result:
        return jsonify({'error': 'Version not found or revert failed'}), 404
    return jsonify({'success': True, 'sequence': result})

# ─────────────────────────────────────────────────────────
# Unified Playback API (Phase 4)
# ─────────────────────────────────────────────────────────

@sequences_bp.route('/api/sequences/<sequence_id>/play', methods=['POST'])
def play_sequence(sequence_id):
    """
    Play a Sequence with step timing and modifiers.

    Uses UnifiedPlaybackEngine (unified_playback.py) as the canonical authority.
    playback_controller.py was deleted in Phase 3.

    POST body:
    {
        "universes": [1, 2],        // Target universes (default: all online)
        "loop_mode": "loop",        // one_shot, loop, bounce (default: from sequence)
        "bpm": 120,                 // BPM override (optional)
        "start_step": 0,            // Starting step index (default: 0)
        "seed": 12345               // Random seed for determinism (optional)
    }
    """
    data = request.get_json() or {}

    # Get the sequence
    sequence = _looks_sequences_manager.get_sequence(sequence_id)
    if not sequence:
        return jsonify({'error': 'Sequence not found'}), 404

    if not sequence.steps:
        return jsonify({'error': 'Sequence has no steps'}), 400

    # Acquire arbitration
    if not _arbitration.acquire('sequence', sequence_id):
        return jsonify({
            'success': False,
            'error': 'Cannot play sequence - arbitration denied',
            'current_owner': _arbitration.current_owner
        }), 409

    # Stop any existing playback (all engines)
    _render_engine.stop_rendering()
    _chase_engine.stop_all()
    _unified_stop()  # Canonical authority for playback

    # Determine target universes
    universes = data.get('universes')
    if not universes:
        universes = list(set(
            n.get('universe', 1) for n in _node_manager.get_all_nodes(include_offline=False)
            if n.get('is_paired') and n.get('status') == 'online'
        ))
        if not universes:
            universes = [1]

    # Parse loop mode
    loop_mode_str = data.get('loop_mode', 'loop' if sequence.loop else 'one_shot')

    # Get BPM (override or sequence default)
    bpm = data.get('bpm', sequence.bpm)

    # Convert sequence steps to playback format
    steps = []
    for step in sequence.steps:
        step_data = {
            'step_id': step.step_id,
            'name': step.name,
            'look_id': step.look_id,
            'channels': step.channels or {},
            'modifiers': [m.to_dict() for m in step.modifiers],
            'fade_ms': step.fade_ms,
            'hold_ms': step.hold_ms,
        }
        steps.append(step_data)

    # =========================================================================
    # MIGRATION (TASK-0021): Use UnifiedPlaybackEngine as canonical authority
    # =========================================================================
    # Build sequence_data dict for unified_play_sequence
    sequence_data = {
        'name': sequence.name,
        'steps': steps,
        'loop_mode': loop_mode_str,
        'bpm': bpm,
    }

    # Start playback via canonical UnifiedPlaybackEngine
    start_step = data.get('start_step', 0)
    seed = data.get('seed')

    print(f"[MIGRATION] play_sequence: Using UnifiedPlaybackEngine (canonical)", flush=True)
    print(f"[MIGRATION]   sequence_id={sequence_id}, universes={universes}, "
          f"start_step={start_step}, bpm={bpm}, loop_mode={loop_mode_str}", flush=True)

    session_id = _unified_play_sequence(
        sequence_id=sequence_id,
        sequence_data=sequence_data,
        universes=universes,
        start_step=start_step,
        seed=seed,
    )

    # Build result compatible with old format
    result = {
        'success': True,
        'job_id': session_id,  # session_id maps to job_id for compatibility
        'session_id': session_id,
        'sequence_id': sequence_id,
        'universes': universes,
        'step_count': len(steps),
        'bpm': bpm,
        'loop_mode': loop_mode_str,
    }

    print(f"[MIGRATION] Sequence '{sequence.name}' playing via UnifiedPlaybackEngine "
          f"(session: {session_id})", flush=True)

    return jsonify({
        **result,
        'name': sequence.name,
        'engine': 'unified',  # Indicates which engine is active
    })


@sequences_bp.route('/api/sequences/<sequence_id>/stop', methods=['POST'])
def stop_sequence(sequence_id):
    """
    Stop sequence playback.

    MIGRATION NOTE (TASK-0021): Now uses UnifiedPlaybackEngine.
    """
    # Check unified engine (canonical authority)
    unified_status = _unified_get_status()
    for session_info in unified_status.get('sessions', []):
        sid = session_info.get('session_id', '')
        # Check if this session matches the sequence
        if f"seq_{sequence_id}_" in sid:
            result = _unified_stop(sid)
            _arbitration.release('sequence')
            return jsonify({**result, 'sequence_id': sequence_id})

    return jsonify({'success': True, 'message': 'Sequence was not playing'})
