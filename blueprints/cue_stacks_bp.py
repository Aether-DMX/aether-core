"""
AETHER Core — Cue Stacks Blueprint
Routes: /api/cue-stacks/*
Dependencies: cue_stacks_manager, looks_sequences_manager, arbitration, merge_layer, merge_layer_output, node_manager
"""

import time
from flask import Blueprint, jsonify, request
from cue_stacks import CueStack, Cue, validate_cue_stack_data

cue_stacks_bp = Blueprint('cue_stacks', __name__)

_cue_stacks_manager = None
_looks_sequences_manager = None
_arbitration = None
_merge_layer = None
_merge_layer_output = None
_node_manager = None


def init_app(cue_stacks_manager, looks_sequences_manager, arbitration, merge_layer, merge_layer_output_fn, node_manager):
    """Initialize blueprint with required dependencies."""
    global _cue_stacks_manager, _looks_sequences_manager, _arbitration, _merge_layer, _merge_layer_output, _node_manager
    _cue_stacks_manager = cue_stacks_manager
    _looks_sequences_manager = looks_sequences_manager
    _arbitration = arbitration
    _merge_layer = merge_layer
    _merge_layer_output = merge_layer_output_fn
    _node_manager = node_manager


@cue_stacks_bp.route('/api/cue-stacks', methods=['GET'])
def get_cue_stacks():
    """Get all Cue Stacks"""
    stacks = _cue_stacks_manager.get_all_cue_stacks()
    return jsonify([s.to_dict() for s in stacks])

@cue_stacks_bp.route('/api/cue-stacks', methods=['POST'])
def create_cue_stack():
    """Create a new Cue Stack"""
    data = request.get_json() or {}

    # Validate
    valid, error = validate_cue_stack_data(data)
    if not valid:
        return jsonify({'success': False, 'error': error}), 400

    # Create CueStack object
    cues = [Cue.from_dict(c) for c in data.get('cues', [])]
    stack = CueStack(
        stack_id=data.get('stack_id', f"stack_{int(time.time() * 1000)}"),
        name=data['name'],
        cues=cues,
        color=data.get('color', 'purple'),
        description=data.get('description', ''),
    )

    result = _cue_stacks_manager.create_cue_stack(stack)
    return jsonify({'success': True, 'cue_stack': result.to_dict()})

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>', methods=['GET'])
def get_cue_stack(stack_id):
    """Get a Cue Stack by ID"""
    stack = _cue_stacks_manager.get_cue_stack(stack_id)
    if not stack:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify(stack.to_dict())

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>', methods=['PUT'])
def update_cue_stack(stack_id):
    """Update an existing Cue Stack"""
    data = request.get_json() or {}

    # Validate if cues are being updated
    if 'cues' in data:
        valid, error = validate_cue_stack_data({**data, 'name': data.get('name', 'temp')})
        if not valid:
            return jsonify({'success': False, 'error': error}), 400

    result = _cue_stacks_manager.update_cue_stack(stack_id, data)
    if not result:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify({'success': True, 'cue_stack': result.to_dict()})

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>', methods=['DELETE'])
def delete_cue_stack(stack_id):
    """Delete a Cue Stack"""
    success = _cue_stacks_manager.delete_cue_stack(stack_id)
    if not success:
        return jsonify({'error': 'Cue Stack not found'}), 404
    return jsonify({'success': True, 'stack_id': stack_id})

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>/go', methods=['POST'])
def cue_stack_go(stack_id):
    """
    Execute the next cue (Go button).
    Manually triggers the next cue in the stack.
    """
    # Helper to resolve look_id to channels
    def look_resolver(look_id):
        look = _looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = _cue_stacks_manager.go(stack_id, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        # Acquire arbitration for cue stack
        if not _arbitration.acquire('cue_stack', stack_id):
            return jsonify({
                'success': False,
                'error': 'Cannot execute cue - arbitration denied',
                'current_owner': _arbitration.current_owner
            }), 409

        # Determine target universes from channel keys
        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                # Universe:channel format
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                # Just channel number, assume universe 1
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        # If no explicit universes, use all online nodes
        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in _node_manager.get_all_nodes(include_offline=False)
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        # Apply the cue with fade via merge layer
        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not _merge_layer.get_source(source_id):
            _merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            # Convert to int keys for merge layer
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            # Update merge layer and output
            _merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = _merge_layer.compute_merge(univ)
            if merged:
                _merge_layer_output(univ, merged)

    return jsonify(result)

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>/back', methods=['POST'])
def cue_stack_back(stack_id):
    """
    Go back to previous cue (Back button).
    """
    def look_resolver(look_id):
        look = _looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = _cue_stacks_manager.back(stack_id, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output (same logic as go)
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in _node_manager.get_all_nodes(include_offline=False)
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not _merge_layer.get_source(source_id):
            _merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            _merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = _merge_layer.compute_merge(univ)
            if merged:
                _merge_layer_output(univ, merged)

    return jsonify(result)

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>/goto/<cue_number>', methods=['POST'])
def cue_stack_goto(stack_id, cue_number):
    """
    Jump to a specific cue by number.
    """
    def look_resolver(look_id):
        look = _looks_sequences_manager.get_look(look_id)
        if look:
            return look.channels
        return {}

    result = _cue_stacks_manager.goto(stack_id, cue_number, look_resolver=look_resolver)

    if not result.get('success'):
        return jsonify(result), 404

    # If we got channels, send them to DMX output
    channels = result.get('channels')
    fade_time_ms = result.get('fade_time_ms', 1000)

    if channels:
        if not _arbitration.acquire('cue_stack', stack_id):
            return jsonify({
                'success': False,
                'error': 'Cannot execute cue - arbitration denied',
                'current_owner': _arbitration.current_owner
            }), 409

        universes = set()
        flat_channels = {}

        for key, value in channels.items():
            if ':' in str(key):
                parts = str(key).split(':')
                univ = int(parts[0])
                ch = int(parts[1])
                universes.add(univ)
                if univ not in flat_channels:
                    flat_channels[univ] = {}
                flat_channels[univ][ch] = value
            else:
                universes.add(1)
                if 1 not in flat_channels:
                    flat_channels[1] = {}
                flat_channels[1][int(key)] = value

        if not universes:
            universes = list(set(
                n.get('universe', 1) for n in _node_manager.get_all_nodes(include_offline=False)
                if n.get('is_paired') and n.get('status') == 'online'
            ))
            if not universes:
                universes = [1]

        source_id = f"cue_stack_{stack_id}"

        # Register source if not already registered
        if not _merge_layer.get_source(source_id):
            _merge_layer.register_source(source_id, 'cue_stack', list(universes))

        for univ in universes:
            univ_channels = flat_channels.get(univ, channels)
            channel_dict = {}
            for ch_str, val in univ_channels.items():
                ch = int(ch_str) if not isinstance(ch_str, int) else ch_str
                if 1 <= ch <= 512:
                    channel_dict[ch] = val

            _merge_layer.set_source_channels(source_id, univ, channel_dict)
            merged = _merge_layer.compute_merge(univ)
            if merged:
                _merge_layer_output(univ, merged)

    return jsonify(result)

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>/stop', methods=['POST'])
def cue_stack_stop(stack_id):
    """Stop cue stack playback and release output"""
    result = _cue_stacks_manager.stop(stack_id)

    # Release merge layer source
    _merge_layer.unregister_source(f"cue_stack_{stack_id}")

    # Release arbitration if we own it
    if _arbitration.current_owner == 'cue_stack' and _arbitration.current_id == stack_id:
        _arbitration.release('cue_stack')

    return jsonify(result)

@cue_stacks_bp.route('/api/cue-stacks/<stack_id>/status', methods=['GET'])
def cue_stack_status(stack_id):
    """Get current playback status for a cue stack"""
    result = _cue_stacks_manager.get_status(stack_id)
    if not result.get('success'):
        return jsonify(result), 404
    return jsonify(result)
