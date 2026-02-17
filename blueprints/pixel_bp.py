"""
AETHER Core — Pixel Array Control Blueprint
Routes: /api/pixel-arrays/*
Dependencies: _pixel_arrays dict, content_manager, create_pixel_controller, EffectType, Pixel
"""

import time
from typing import Dict
from flask import Blueprint, jsonify, request
from pixel_mapper import (
    PixelArrayController, Pixel, OperationMode, EffectType,
    create_pixel_controller
)

pixel_bp = Blueprint('pixel', __name__)

# Module-level state — _pixel_arrays dict is shared
_pixel_arrays = None
_content_manager = None


def init_app(pixel_arrays_dict, content_manager):
    """Initialize blueprint with required dependencies."""
    global _pixel_arrays, _content_manager
    _pixel_arrays = pixel_arrays_dict
    _content_manager = content_manager


def _get_pixel_array_send_callback():
    """Create a callback that routes pixel array output through SSOT"""
    def callback(universe: int, channels: Dict[int, int]):
        _content_manager.set_channels(universe, channels, fade_ms=0)
    return callback


@pixel_bp.route('/api/pixel-arrays', methods=['GET'])
def get_pixel_arrays():
    """List all active pixel array controllers"""
    result = {}
    for array_id, controller in _pixel_arrays.items():
        result[array_id] = controller.get_status()
    return jsonify({
        'pixel_arrays': result,
        'count': len(_pixel_arrays),
    })


@pixel_bp.route('/api/pixel-arrays', methods=['POST'])
def create_pixel_array():
    """
    Create a new pixel array controller.

    Request body:
    {
        "id": "my_array",           // Optional, auto-generated if not provided
        "universe": 4,              // Target DMX universe (default 4)
        "fixture_count": 8,         // Number of RGBW fixtures (default 8)
        "start_channel": 1,         // First fixture start channel (default 1)
        "channel_spacing": 4        // Channels between fixtures (default 4)
    }
    """
    data = request.get_json() or {}

    array_id = data.get('id', f"array_{int(time.time())}")

    if array_id in _pixel_arrays:
        return jsonify({
            'success': False,
            'error': f'Pixel array {array_id} already exists'
        }), 400

    try:
        controller = create_pixel_controller(
            fixture_count=data.get('fixture_count', 8),
            universe=data.get('universe', 4),
            start_channel=data.get('start_channel', 1),
            channel_spacing=data.get('channel_spacing', 4),
            send_callback=_get_pixel_array_send_callback(),
        )
        _pixel_arrays[array_id] = controller

        return jsonify({
            'success': True,
            'id': array_id,
            'status': controller.get_status(),
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@pixel_bp.route('/api/pixel-arrays/<array_id>', methods=['GET'])
def get_pixel_array(array_id):
    """Get status of a specific pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404
    return jsonify(_pixel_arrays[array_id].get_status())


@pixel_bp.route('/api/pixel-arrays/<array_id>', methods=['DELETE'])
def delete_pixel_array(array_id):
    """Delete a pixel array controller"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.stop()
    del _pixel_arrays[array_id]

    return jsonify({'success': True, 'id': array_id})


@pixel_bp.route('/api/pixel-arrays/<array_id>/mode', methods=['POST'])
def set_pixel_array_mode(array_id):
    """
    Set operation mode for a pixel array.

    Request body:
    {
        "mode": "grouped" | "pixel_array"
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    mode_str = data.get('mode', 'grouped')

    controller = _pixel_arrays[array_id]

    if mode_str == 'grouped':
        controller.set_grouped_mode()
    elif mode_str == 'pixel_array':
        controller.set_pixel_array_mode()
    else:
        return jsonify({'error': f'Invalid mode: {mode_str}'}), 400

    return jsonify({
        'success': True,
        'mode': controller.mode.value,
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/pixels', methods=['POST'])
def set_pixel_array_pixels(array_id):
    """
    Set pixel values in the array.

    Request body (grouped mode - sets all pixels):
    {
        "r": 255, "g": 0, "b": 0, "w": 0
    }

    Request body (pixel array mode - set individual pixels):
    {
        "pixels": [
            {"index": 0, "r": 255, "g": 0, "b": 0, "w": 0},
            {"index": 1, "r": 0, "g": 255, "b": 0, "w": 0},
            ...
        ]
    }

    Or set all pixels to same color in pixel array mode:
    {
        "all": {"r": 255, "g": 0, "b": 0, "w": 0}
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    controller = _pixel_arrays[array_id]

    # Set all pixels to same color
    if 'all' in data:
        color = data['all']
        controller.set_all_rgbw(
            color.get('r', 0),
            color.get('g', 0),
            color.get('b', 0),
            color.get('w', 0),
        )
    # Set all pixels (grouped mode shorthand)
    elif 'r' in data or 'g' in data or 'b' in data or 'w' in data:
        controller.set_all_rgbw(
            data.get('r', 0),
            data.get('g', 0),
            data.get('b', 0),
            data.get('w', 0),
        )
    # Set individual pixels
    elif 'pixels' in data:
        for pixel_data in data['pixels']:
            index = pixel_data.get('index', 0)
            controller.set_pixel_rgbw(
                index,
                pixel_data.get('r', 0),
                pixel_data.get('g', 0),
                pixel_data.get('b', 0),
                pixel_data.get('w', 0),
            )

    # Send frame immediately
    controller.populate_dmx_buffer()
    controller._send_frame()

    return jsonify({
        'success': True,
        'status': controller.get_status(),
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/effect', methods=['POST'])
def set_pixel_array_effect(array_id):
    """
    Start an effect on the pixel array.

    Request body:
    {
        "type": "wave" | "chase" | "bounce" | "rainbow_wave" | "none",
        "color": {"r": 255, "g": 0, "b": 0, "w": 0},
        "speed": 1.0,
        "params": {
            "tail_length": 2
        }
    }
    """
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    data = request.get_json() or {}
    controller = _pixel_arrays[array_id]

    effect_type_str = data.get('type', 'none')
    effect_types = {
        'none': EffectType.NONE,
        'wave': EffectType.WAVE,
        'chase': EffectType.CHASE,
        'bounce': EffectType.BOUNCE,
        'rainbow_wave': EffectType.RAINBOW_WAVE,
    }

    if effect_type_str not in effect_types:
        return jsonify({'error': f'Invalid effect type: {effect_type_str}'}), 400

    effect_type = effect_types[effect_type_str]

    # Parse color
    color_data = data.get('color', {})
    color = Pixel(
        color_data.get('r', 255),
        color_data.get('g', 0),
        color_data.get('b', 0),
        color_data.get('w', 0),
    )

    speed = data.get('speed', 1.0)
    params = data.get('params', {})

    controller.set_effect(effect_type, color=color, speed=speed, **params)

    return jsonify({
        'success': True,
        'effect_type': effect_type.value,
        'status': controller.get_status(),
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/start', methods=['POST'])
def start_pixel_array(array_id):
    """Start the render loop for a pixel array (30 FPS)"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.start()

    return jsonify({
        'success': True,
        'running': controller.is_running,
        'status': controller.get_status(),
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/stop', methods=['POST'])
def stop_pixel_array(array_id):
    """Stop the render loop for a pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.stop()

    return jsonify({
        'success': True,
        'running': controller.is_running,
        'status': controller.get_status(),
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/blackout', methods=['POST'])
def blackout_pixel_array(array_id):
    """Set all pixels to black and send frame"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    controller.clear_all()
    controller.populate_dmx_buffer()
    controller._send_frame()

    return jsonify({
        'success': True,
        'status': controller.get_status(),
    })


@pixel_bp.route('/api/pixel-arrays/<array_id>/fixture-map', methods=['GET'])
def get_pixel_array_fixture_map(array_id):
    """Get the fixture map for a pixel array"""
    if array_id not in _pixel_arrays:
        return jsonify({'error': f'Pixel array {array_id} not found'}), 404

    controller = _pixel_arrays[array_id]
    fixture_map = controller.get_fixture_map()

    entries = []
    for entry in fixture_map.get_all_entries():
        entries.append({
            'fixture_index': entry.fixture_index,
            'start_channel': entry.start_channel,
            'r_channel': entry.r_channel,
            'g_channel': entry.g_channel,
            'b_channel': entry.b_channel,
            'w_channel': entry.w_channel,
        })

    return jsonify({
        'fixture_count': fixture_map.fixture_count,
        'is_valid': fixture_map.is_valid,
        'overflow_error': fixture_map.overflow_error,
        'entries': entries,
    })
