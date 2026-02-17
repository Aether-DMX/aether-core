"""
AETHER Core — Modifiers & Distribution Modes Blueprint
Routes: /api/modifiers/*
Dependencies: modifier_registry, distribution_modes (library-only, no shared state)
"""

from flask import Blueprint, jsonify, request
from modifier_registry import (
    modifier_registry, validate_modifier, normalize_modifier, get_modifier_presets
)
from distribution_modes import (
    DistributionMode, DistributionConfig, DistributionCalculator,
    DISTRIBUTION_PRESETS, get_distribution_preset, list_distribution_presets,
    get_supported_distributions, suggest_distribution_for_effect
)

modifiers_bp = Blueprint('modifiers', __name__)


@modifiers_bp.route('/api/modifiers/schemas', methods=['GET'])
def get_modifier_schemas_route():
    """
    Get all modifier schemas for UI generation.
    Returns complete schema definitions including params, presets, and categories.
    """
    return jsonify(modifier_registry.to_api_response())

@modifiers_bp.route('/api/modifiers/types', methods=['GET'])
def get_modifier_types_route():
    """Get list of available modifier types"""
    return jsonify({
        'types': modifier_registry.get_types(),
        'categories': modifier_registry.get_categories()
    })

@modifiers_bp.route('/api/modifiers/<modifier_type>/presets', methods=['GET'])
def get_modifier_presets_route(modifier_type):
    """Get all presets for a specific modifier type"""
    presets = modifier_registry.get_presets(modifier_type)
    if not presets and modifier_type not in modifier_registry.get_types():
        return jsonify({'error': f'Unknown modifier type: {modifier_type}'}), 404
    return jsonify({
        'modifier_type': modifier_type,
        'presets': presets
    })

@modifiers_bp.route('/api/modifiers/<modifier_type>/presets/<preset_id>', methods=['GET'])
def get_modifier_preset_route(modifier_type, preset_id):
    """Get a specific preset and create a modifier from it"""
    modifier_data = modifier_registry.create_from_preset(modifier_type, preset_id)
    if not modifier_data:
        return jsonify({'error': f'Preset not found: {modifier_type}/{preset_id}'}), 404
    return jsonify({
        'success': True,
        'modifier': modifier_data
    })

@modifiers_bp.route('/api/modifiers/validate', methods=['POST'])
def validate_modifier_route():
    """
    Validate a modifier against its schema.
    Returns validation result with detailed error if invalid.
    """
    data = request.get_json() or {}
    is_valid, error = validate_modifier(data)
    if not is_valid:
        return jsonify({
            'valid': False,
            'error': error
        }), 400
    return jsonify({
        'valid': True,
        'normalized': normalize_modifier(data)
    })

@modifiers_bp.route('/api/modifiers/normalize', methods=['POST'])
def normalize_modifier_route():
    """
    Normalize a modifier by applying defaults and generating ID.
    Use this before saving a modifier to ensure all fields are populated.
    """
    data = request.get_json() or {}
    # Validate first
    is_valid, error = validate_modifier(data)
    if not is_valid:
        return jsonify({'success': False, 'error': error}), 400
    return jsonify({
        'success': True,
        'modifier': normalize_modifier(data)
    })


# Distribution Modes API
@modifiers_bp.route('/api/modifiers/distribution-modes', methods=['GET'])
def get_distribution_modes_route():
    """
    Get all available distribution modes and their descriptions.

    Distribution modes control how modifiers are applied across multiple fixtures:
    - SYNCED: All fixtures identical (default)
    - INDEXED: Scaled by fixture index
    - PHASED: Time offset per fixture
    - PIXELATED: Unique per fixture
    - RANDOM: Deterministic random per fixture
    """
    modes = [
        {
            'mode': mode.value,
            'name': mode.name,
            'description': {
                'synced': 'All fixtures receive identical effect values',
                'indexed': 'Effect values scale linearly with fixture position',
                'phased': 'Time offset between fixtures for traveling effects',
                'pixelated': 'Each fixture has unique, independent effect values',
                'random': 'Deterministic random variation per fixture',
                'grouped': 'Same value for fixtures in same group',
            }.get(mode.value, '')
        }
        for mode in DistributionMode
    ]
    return jsonify({
        'modes': modes,
        'presets': list_distribution_presets()
    })


@modifiers_bp.route('/api/modifiers/distribution-modes/<modifier_type>', methods=['GET'])
def get_modifier_distribution_modes_route(modifier_type):
    """Get supported distribution modes for a specific modifier type"""
    supported = get_supported_distributions(modifier_type)
    return jsonify({
        'modifier_type': modifier_type,
        'supported_modes': [mode.value for mode in supported]
    })


@modifiers_bp.route('/api/modifiers/distribution-modes/suggest', methods=['POST'])
def suggest_distribution_mode_route():
    """
    Get AI-suggested distribution mode for a modifier and fixture selection.

    Request body:
    {
        "modifier_type": "wave",
        "fixture_count": 8,
        "effect_intent": "chase" (optional)
    }
    """
    data = request.get_json() or {}
    modifier_type = data.get('modifier_type', 'pulse')
    fixture_count = data.get('fixture_count', 1)
    effect_intent = data.get('effect_intent')

    suggestion = suggest_distribution_for_effect(
        modifier_type=modifier_type,
        fixture_count=fixture_count,
        effect_intent=effect_intent
    )

    return jsonify({
        'suggestion': suggestion.to_dict(),
        'modifier_type': modifier_type,
        'fixture_count': fixture_count
    })


@modifiers_bp.route('/api/modifiers/distribution-presets', methods=['GET'])
def get_distribution_presets_route():
    """Get all distribution presets"""
    return jsonify({
        'presets': list_distribution_presets()
    })


@modifiers_bp.route('/api/modifiers/distribution-presets/<preset_name>', methods=['GET'])
def get_distribution_preset_route(preset_name):
    """Get a specific distribution preset"""
    preset = get_distribution_preset(preset_name)
    if not preset:
        return jsonify({'error': f'Preset not found: {preset_name}'}), 404
    return jsonify({
        'name': preset_name,
        'config': preset.to_dict()
    })
