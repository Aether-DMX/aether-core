"""
AETHER Core — Merge Layer Blueprint
Routes: /api/merge/*
Dependencies: merge_layer, content_manager, load_fixtures_into_classifier
"""

from flask import Blueprint, jsonify, request

merge_bp = Blueprint('merge', __name__)

# Dependencies injected at registration time
_merge_layer = None
_content_manager = None
_load_fixtures_into_classifier = None


def init_app(merge_layer, content_manager, load_fixtures_into_classifier_fn):
    """Initialize blueprint with required dependencies."""
    global _merge_layer, _content_manager, _load_fixtures_into_classifier
    _merge_layer = merge_layer
    _content_manager = content_manager
    _load_fixtures_into_classifier = load_fixtures_into_classifier_fn


@merge_bp.route('/api/merge/status', methods=['GET'])
def get_merge_status():
    """
    Get merge layer status including all active sources.

    Returns:
    {
        "source_count": 2,
        "active_count": 2,
        "blackout_active": false,
        "sources": [
            {"source_id": "look_xxx", "source_type": "look", "priority": 50, ...},
            {"source_id": "effect_yyy", "source_type": "effect", "priority": 60, ...}
        ]
    }
    """
    return jsonify(_merge_layer.get_status())


@merge_bp.route('/api/merge/channel/<int:universe>/<int:channel>', methods=['GET'])
def get_channel_breakdown(universe, channel):
    """
    Debug endpoint: Show which sources contribute to a specific channel.

    Returns the merge breakdown with:
    - All contributing sources
    - Channel type (dimmer/color/etc)
    - Merge mode (HTP/LTP)
    - Final merged value
    """
    return jsonify(_merge_layer.get_source_breakdown(universe, channel))


@merge_bp.route('/api/merge/blackout', methods=['POST'])
def merge_blackout():
    """
    Activate merge layer blackout (highest priority override).

    POST body:
    {
        "active": true,           // Enable/disable blackout
        "universes": [1, 2]       // Optional: specific universes (null = all)
    }
    """
    data = request.get_json() or {}
    active = data.get('active', True)
    universes = data.get('universes')

    _merge_layer.set_blackout(active, universes)

    # Also trigger SSOT blackout for physical output
    if active:
        if universes:
            for univ in universes:
                _content_manager.blackout(universe=univ, fade_ms=0)
        else:
            _content_manager.blackout(fade_ms=0)

    return jsonify({
        'success': True,
        'blackout_active': _merge_layer.is_blackout(),
        'universes': universes
    })


@merge_bp.route('/api/merge/sources', methods=['GET'])
def get_merge_sources():
    """List all registered merge sources with their priorities"""
    status = _merge_layer.get_status()
    return jsonify({
        'sources': status.get('sources', []),
        'priority_levels': {
            'blackout': 100,
            'manual': 80,
            'effect': 60,
            'look': 50,
            'sequence': 45,
            'chase': 40,
            'scene': 20,
            'background': 10
        }
    })


@merge_bp.route('/api/merge/classifier/reload', methods=['POST'])
def reload_fixture_classifier():
    """Reload fixture definitions into the channel classifier"""
    _load_fixtures_into_classifier()
    return jsonify({
        'success': True,
        'message': 'Fixture classifier reloaded'
    })
