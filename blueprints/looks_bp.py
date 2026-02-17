"""
AETHER Core — Looks Blueprint
Routes: /api/looks/*
Dependencies: looks_sequences_manager, arbitration, render_engine, node_manager,
              unified_play_look, content_manager, cloud_submit,
              SUPABASE_AVAILABLE, get_supabase_service
Imports: Look, Modifier from looks_sequences
         validate_look_data from looks_sequences
         validate_modifier, normalize_modifier from modifier_registry
         render_look_frame from render_engine module
"""

import time
from flask import Blueprint, jsonify, request
from looks_sequences import Look, Modifier, validate_look_data
from modifier_registry import validate_modifier, normalize_modifier
from render_engine import render_look_frame
from unified_playback import PlaybackType

looks_bp = Blueprint('looks', __name__)

# Dependencies injected at registration time
_looks_sequences_manager = None
_arbitration = None
_render_engine = None
_node_manager = None
_unified_play_look = None
_unified_engine = None
_content_manager = None
_cloud_submit = None
_SUPABASE_AVAILABLE = False
_get_supabase_service = None


def init_app(looks_sequences_manager, arbitration, render_engine, node_manager,
             unified_play_look_fn, unified_engine, content_manager, cloud_submit_fn,
             SUPABASE_AVAILABLE_flag, get_supabase_service_fn):
    """Initialize blueprint with required dependencies."""
    global _looks_sequences_manager, _arbitration, _render_engine, _node_manager
    global _unified_play_look, _unified_engine, _content_manager, _cloud_submit
    global _SUPABASE_AVAILABLE, _get_supabase_service
    _looks_sequences_manager = looks_sequences_manager
    _arbitration = arbitration
    _render_engine = render_engine
    _node_manager = node_manager
    _unified_play_look = unified_play_look_fn
    _unified_engine = unified_engine
    _content_manager = content_manager
    _cloud_submit = cloud_submit_fn
    _SUPABASE_AVAILABLE = SUPABASE_AVAILABLE_flag
    _get_supabase_service = get_supabase_service_fn


@looks_bp.route('/api/looks', methods=['GET'])
def get_looks():
    """Get all Looks"""
    looks = _looks_sequences_manager.get_all_looks()
    return jsonify([l.to_dict() for l in looks])

@looks_bp.route('/api/looks', methods=['POST'])
def create_look():
    """Create a new Look"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_look_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create Look object
    look = Look(
        look_id=data.get('look_id', f"look_{int(time.time() * 1000)}"),
        name=data['name'],
        channels=data['channels'],
        modifiers=[Modifier.from_dict(m) for m in data.get('modifiers', [])],
        fade_ms=data.get('fade_ms', 0),
        color=data.get('color', 'blue'),
        icon=data.get('icon', 'lightbulb'),
        description=data.get('description', ''),
    )

    result = _looks_sequences_manager.create_look(look)

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        supabase = _get_supabase_service()
        if supabase and supabase.is_enabled():
            _cloud_submit(supabase.sync_look, result.to_dict())

    return jsonify({'success': True, 'look': result.to_dict()})

@looks_bp.route('/api/looks/<look_id>', methods=['GET'])
def get_look(look_id):
    """Get a Look by ID"""
    look = _looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'error': 'Look not found'}), 404
    return jsonify(look.to_dict())

@looks_bp.route('/api/looks/<look_id>', methods=['PUT'])
def update_look(look_id):
    """Update an existing Look"""
    data = request.get_json() or {}

    # Validate if full replacement
    if 'channels' in data:
        valid, error = validate_look_data(data)
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = _looks_sequences_manager.update_look(look_id, data)
    if not result:
        return jsonify({'error': 'Look not found'}), 404

    # Async sync to Supabase (non-blocking)
    if _SUPABASE_AVAILABLE:
        supabase = _get_supabase_service()
        if supabase and supabase.is_enabled():
            _cloud_submit(supabase.sync_look, result.to_dict())

    return jsonify({'success': True, 'look': result.to_dict()})

@looks_bp.route('/api/looks/<look_id>', methods=['DELETE'])
def delete_look(look_id):
    """Delete a Look"""
    success = _looks_sequences_manager.delete_look(look_id)
    if not success:
        return jsonify({'error': 'Look not found'}), 404

    # Note: Supabase delete is not implemented yet (would need to mark as deleted)
    return jsonify({'success': True, 'look_id': look_id})

@looks_bp.route('/api/looks/<look_id>/versions', methods=['GET'])
def get_look_versions(look_id):
    """Get version history for a Look"""
    versions = _looks_sequences_manager.get_versions(look_id, 'look')
    return jsonify({'success': True, 'versions': versions})

@looks_bp.route('/api/looks/<look_id>/versions/<version_id>/revert', methods=['POST'])
def revert_look_version(look_id, version_id):
    """Revert a Look to a specific version"""
    result = _looks_sequences_manager.revert_to_version(version_id)
    if not result:
        return jsonify({'error': 'Version not found or revert failed'}), 404
    return jsonify({'success': True, 'look': result})

# ============================================================================
# TASK-0018 RESOLVED (F06 consolidation)
# ============================================================================
# Previously used RenderEngine directly — now routes through
# UnifiedPlaybackEngine via unified_play_look() for all look types.
# ============================================================================
@looks_bp.route('/api/looks/<look_id>/play', methods=['POST'])
def play_look(look_id):
    """
    Play a Look with real-time modifier rendering.

    Routes through UnifiedPlaybackEngine (canonical authority per Hard Rule 1.1).

    POST body:
    {
        "universes": [1, 2],       // Target universes (default: all online)
        "fade_ms": 500,            // Initial fade time (optional)
    }
    """
    data = request.get_json() or {}

    # Get the look
    look = _looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'error': 'Look not found'}), 404

    # Acquire arbitration
    if not _arbitration.acquire('look', look_id):
        return jsonify({
            'success': False,
            'error': 'Cannot play look - arbitration denied',
            'current_owner': _arbitration.current_owner
        }), 409

    # Stop any legacy render engine jobs (migration safety net)
    _render_engine.stop_rendering()

    # Determine target universes
    universes = data.get('universes')
    if not universes:
        universes = list(set(
            n.get('universe', 1) for n in _node_manager.get_nodes()
            if n.get('is_paired') and n.get('status') == 'online'
        ))
        if not universes:
            universes = [1]

    fade_ms = data.get('fade_ms', look.fade_ms or 0)
    has_modifiers = len(look.modifiers) > 0 and any(m.enabled for m in look.modifiers)

    # Stop existing look/scene sessions before playing new one (replace mode)
    # This prevents HTP stacking where old look channels persist
    if _unified_engine:
        from unified_playback import PlaybackType
        _unified_engine.stop_type(PlaybackType.LOOK)
        _unified_engine.stop_type(PlaybackType.SCENE)

    # Route through UnifiedPlaybackEngine (canonical authority)
    look_data = look.to_dict()
    session_id = _unified_play_look(
        look_id,
        look_data,
        universes=universes,
        fade_ms=fade_ms,
    )

    return jsonify({
        'success': True,
        'look_id': look_id,
        'name': look.name,
        'universes': universes,
        'rendering': has_modifiers,
        'modifier_count': len([m for m in look.modifiers if m.enabled]) if has_modifiers else 0,
        'session_id': session_id,
        'engine': 'unified',
    })

@looks_bp.route('/api/looks/<look_id>/stop', methods=['POST'])
def stop_look(look_id):
    """Stop playing a Look"""
    # Stop unified engine look sessions
    _unified_engine.stop_type(PlaybackType.LOOK)
    # Stop legacy render engine (migration safety net)
    _render_engine.stop_rendering()

    # Release arbitration if we own it
    if _arbitration.current_owner == 'look' and _arbitration.current_id == look_id:
        _arbitration.release('look')

    return jsonify({
        'success': True,
        'look_id': look_id,
        'stopped': True,
    })

@looks_bp.route('/api/looks/preview', methods=['POST'])
def preview_look():
    """
    Preview a Look without saving - render a single frame.

    POST body:
    {
        "channels": {"1": 255, "2": 128},
        "modifiers": [...],
        "elapsed_time": 0.5,    // Simulated time for preview
        "seed": 12345           // Optional seed
    }

    Returns the computed channel values for preview.
    """
    data = request.get_json() or {}

    channels = data.get('channels', {})
    modifiers = data.get('modifiers', [])
    elapsed_time = data.get('elapsed_time', 0.0)
    seed = data.get('seed', 0)

    if not channels:
        return jsonify({'error': 'Channels required'}), 400

    # Validate modifiers
    for mod in modifiers:
        valid, error = validate_modifier(mod)
        if not valid:
            return jsonify({'error': f'Invalid modifier: {error}'}), 400

    # Render single frame
    result = render_look_frame(
        channels=channels,
        modifiers=[normalize_modifier(m) for m in modifiers],
        elapsed_time=elapsed_time,
        seed=seed,
    )

    return jsonify({
        'success': True,
        'input_channels': channels,
        'output_channels': result,
        'elapsed_time': elapsed_time,
        'modifier_count': len(modifiers),
    })
