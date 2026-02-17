"""
AETHER Core — AI & Render Pipeline Blueprint
Routes: /api/ai/*, /api/render/pipeline/*, /api/render/features
Dependencies: ai_fixture_advisor functions, get_ai_advisor, get_render_pipeline
"""

from flask import Blueprint, jsonify, request
from ai_fixture_advisor import (
    get_distribution_suggestions,
    apply_ai_suggestion,
    dismiss_ai_suggestion,
)

ai_bp = Blueprint('ai', __name__)

_get_ai_advisor = None
_get_render_pipeline = None


def init_app(get_ai_advisor_fn, get_render_pipeline_fn):
    """Initialize blueprint with required dependencies."""
    global _get_ai_advisor, _get_render_pipeline
    _get_ai_advisor = get_ai_advisor_fn
    _get_render_pipeline = get_render_pipeline_fn


# ============================================================
# AI SSOT Stub Routes
# ============================================================
@ai_bp.route('/api/ai/preferences', methods=['GET'])
def ai_get_prefs():
    return jsonify({})

@ai_bp.route('/api/ai/preferences/<key>', methods=['GET', 'POST'])
def ai_pref(key):
    if request.method == 'POST':
        data = request.get_json()
        pass  # stubbed
        return jsonify({'success': True})
    return jsonify({'value': None})

@ai_bp.route('/api/ai/budget', methods=['GET'])
def ai_budget():
    return jsonify({"remaining":999,"used":0})

@ai_bp.route('/api/ai/outcomes', methods=['GET', 'POST'])
def ai_outcomes():
    if request.method == 'POST':
        d = request.get_json()
        pass  # stubbed
        return jsonify({'success': True})
    return jsonify({'outcomes': []})

@ai_bp.route('/api/ai/audit', methods=['GET'])
def ai_audit():
    return jsonify({'log': []})

@ai_bp.route('/api/ai/ops', methods=['GET'])
def ai_ops():
    return jsonify({'ops': {}})


# ============================================================
# AI Optimize Playback
# ============================================================
@ai_bp.route('/api/ai/optimize-playback', methods=['POST'])
def ai_optimize_playback():
    """
    AI-powered playback optimization suggestions.
    Takes current playback state and an intent/vibe, returns effect suggestions.
    """
    data = request.get_json() or {}
    current_state = data.get('current_state', {})
    intent = data.get('intent', '')

    # Map vibes to effect suggestions
    VIBE_EFFECTS = {
        'chill': {
            'explanation': 'Adding gentle color fades and subtle pulse for a relaxed atmosphere.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_fade', 'params': {'speed': 0.1, 'depth': 60}},
                {'type': 'add_effect', 'effect_id': 'fixture_pulse', 'params': {'speed': 0.3, 'depth': 30}},
            ]
        },
        'party': {
            'explanation': 'Cranking up the energy with rainbow chase and strobe accents!',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_rainbow', 'params': {'speed': 0.5, 'depth': 100}},
                {'type': 'add_effect', 'effect_id': 'fixture_chase', 'params': {'speed': 4, 'depth': 100}},
            ]
        },
        'romantic': {
            'explanation': 'Warm, gentle pulses with color temperature shift for intimacy.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_temp', 'params': {'temperature': 80, 'depth': 70}},
                {'type': 'add_effect', 'effect_id': 'fixture_pulse', 'params': {'speed': 0.2, 'depth': 40}},
            ]
        },
        'dramatic': {
            'explanation': 'Bold contrasts with lightning effects and scanner sweeps.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_lightning', 'params': {'frequency': 0.8, 'depth': 100}},
                {'type': 'add_effect', 'effect_id': 'fixture_scanner', 'params': {'speed': 2, 'depth': 80}},
            ]
        },
        'concert': {
            'explanation': 'High-energy strobes and chase patterns like a live show.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'strobe', 'params': {'speed': 0.2, 'depth': 80}},
                {'type': 'add_effect', 'effect_id': 'fixture_chase', 'params': {'speed': 6, 'depth': 100}},
            ]
        },
        'sunset': {
            'explanation': 'Warm color temperature fading through golden hour hues.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_temp', 'params': {'temperature': 100, 'depth': 80}},
                {'type': 'add_effect', 'effect_id': 'fixture_hue_shift', 'params': {'speed': 0.05, 'depth': 50}},
            ]
        },
        'focus': {
            'explanation': 'Clean, steady lighting with minimal distraction.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_temp', 'params': {'temperature': -30, 'depth': 40}},
            ]
        },
        'spooky': {
            'explanation': 'Eerie flickers and cold color shifts for Halloween vibes.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_lightning', 'params': {'frequency': 0.3, 'depth': 100}},
                {'type': 'add_effect', 'effect_id': 'fixture_color_temp', 'params': {'temperature': -80, 'depth': 60}},
            ]
        },
    }

    # Check if it's a preset vibe
    if intent.lower() in VIBE_EFFECTS:
        return jsonify(VIBE_EFFECTS[intent.lower()])

    # For custom text intents, do simple keyword matching
    intent_lower = intent.lower()

    if any(w in intent_lower for w in ['relax', 'calm', 'chill', 'soft', 'gentle']):
        return jsonify(VIBE_EFFECTS['chill'])
    elif any(w in intent_lower for w in ['party', 'dance', 'energy', 'exciting', 'fun']):
        return jsonify(VIBE_EFFECTS['party'])
    elif any(w in intent_lower for w in ['romantic', 'intimate', 'love', 'date', 'cozy']):
        return jsonify(VIBE_EFFECTS['romantic'])
    elif any(w in intent_lower for w in ['dramatic', 'intense', 'powerful', 'bold']):
        return jsonify(VIBE_EFFECTS['dramatic'])
    elif any(w in intent_lower for w in ['concert', 'show', 'performance', 'live']):
        return jsonify(VIBE_EFFECTS['concert'])
    elif any(w in intent_lower for w in ['sunset', 'warm', 'golden', 'evening']):
        return jsonify(VIBE_EFFECTS['sunset'])
    elif any(w in intent_lower for w in ['focus', 'work', 'study', 'clean', 'bright']):
        return jsonify(VIBE_EFFECTS['focus'])
    elif any(w in intent_lower for w in ['spooky', 'halloween', 'scary', 'creepy', 'horror']):
        return jsonify(VIBE_EFFECTS['spooky'])
    elif any(w in intent_lower for w in ['rainbow', 'colorful', 'color']):
        return jsonify({
            'explanation': 'Adding vibrant rainbow colors that cycle through the spectrum.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_rainbow', 'params': {'speed': 0.3, 'depth': 100}},
            ]
        })
    elif any(w in intent_lower for w in ['pulse', 'beat', 'rhythm']):
        return jsonify({
            'explanation': 'Adding rhythmic pulses to match the beat.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_pulse', 'params': {'speed': 1.0, 'depth': 80}},
            ]
        })
    else:
        # Default suggestion
        return jsonify({
            'explanation': f'Based on "{intent}", adding some dynamic color and movement.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_fade', 'params': {'speed': 0.2, 'depth': 70}},
                {'type': 'add_effect', 'effect_id': 'wave', 'params': {'speed': 0.5, 'depth': 50}},
            ]
        })


# ============================================================
# AI Fixture Advisor Suggestion Routes
# ============================================================
@ai_bp.route('/api/ai/suggestions/distribution', methods=['POST'])
def get_ai_distribution_suggestions_route():
    """
    Get AI suggestions for distribution modes.

    IMPORTANT: AI suggestions are NEVER auto-applied.
    Returns suggestions that the user must explicitly Apply or Dismiss.
    """
    data = request.get_json() or {}
    modifier_type = data.get('modifier_type', 'pulse')
    modifier_params = data.get('modifier_params', {})
    fixture_count = data.get('fixture_count', 1)

    suggestions = get_distribution_suggestions(
        modifier_type=modifier_type,
        fixture_count=fixture_count,
        modifier_params=modifier_params
    )

    return jsonify({
        'suggestions': suggestions,
        'note': 'AI suggestions require explicit user approval. Apply or Dismiss each suggestion.'
    })


@ai_bp.route('/api/ai/suggestions/apply/<suggestion_id>', methods=['POST'])
def apply_ai_suggestion_route(suggestion_id):
    """Mark an AI suggestion as applied (after user explicit approval)."""
    success = apply_ai_suggestion(suggestion_id)
    if not success:
        return jsonify({'error': 'Suggestion not found'}), 404
    return jsonify({
        'success': True,
        'suggestion_id': suggestion_id,
        'status': 'applied'
    })


@ai_bp.route('/api/ai/suggestions/dismiss/<suggestion_id>', methods=['POST'])
def dismiss_ai_suggestion_route(suggestion_id):
    """Dismiss an AI suggestion"""
    success = dismiss_ai_suggestion(suggestion_id)
    if not success:
        return jsonify({'error': 'Suggestion not found'}), 404
    return jsonify({
        'success': True,
        'suggestion_id': suggestion_id,
        'status': 'dismissed'
    })


@ai_bp.route('/api/ai/suggestions/pending', methods=['GET'])
def get_pending_ai_suggestions_route():
    """Get all pending (not applied/dismissed) AI suggestions"""
    advisor = _get_ai_advisor()
    suggestions = advisor.get_pending_suggestions()
    return jsonify({
        'suggestions': [s.to_dict() for s in suggestions]
    })


@ai_bp.route('/api/ai/suggestions/transition', methods=['POST'])
def get_ai_transition_suggestions_route():
    """
    Get AI suggestions for transition/crossfade times.
    AI suggests but NEVER auto-applies - user must explicitly apply.
    """
    from ai_fixture_advisor import get_transition_suggestions

    data = request.get_json() or {}
    effect_type = data.get('effect_type', 'wave')
    fixture_count = data.get('fixture_count', 1)
    step_duration_ms = data.get('step_duration_ms')

    suggestions = get_transition_suggestions(
        effect_type=effect_type,
        fixture_count=fixture_count,
        step_duration_ms=step_duration_ms
    )

    return jsonify({
        'effect_type': effect_type,
        'fixture_count': fixture_count,
        'suggestions': suggestions
    })


@ai_bp.route('/api/ai/transition/recommend', methods=['GET'])
def get_recommended_transition_route():
    """Quick endpoint to get recommended transition for an effect."""
    from ai_fixture_advisor import get_recommended_transition_for_effect

    effect_type = request.args.get('effect_type', 'wave')
    fixture_count = int(request.args.get('fixture_count', 1))
    smoothness = request.args.get('smoothness', 'smooth')

    result = get_recommended_transition_for_effect(
        effect_type=effect_type,
        fixture_count=fixture_count,
        smoothness=smoothness
    )

    return jsonify({
        'effect_type': effect_type,
        'fixture_count': fixture_count,
        'smoothness': smoothness,
        **result
    })


# ============================================================
# Render Pipeline Routes
# ============================================================
@ai_bp.route('/api/render/pipeline/status', methods=['GET'])
def get_render_pipeline_status_route():
    """Get status of the final render pipeline (Phase 3 fixture-centric)"""
    pipeline = _get_render_pipeline()
    return jsonify(pipeline.get_status())


@ai_bp.route('/api/render/features', methods=['GET'])
def get_render_features_route():
    """Get feature flags for the render pipeline"""
    pipeline = _get_render_pipeline()
    return jsonify({
        'fixture_centric_enabled': pipeline.features.FIXTURE_CENTRIC_ENABLED,
        'legacy_channel_fallback': pipeline.features.LEGACY_CHANNEL_FALLBACK,
        'ai_suggestions_enabled': pipeline.features.AI_SUGGESTIONS_ENABLED,
        'distribution_modes_enabled': pipeline.features.DISTRIBUTION_MODES_ENABLED,
    })


@ai_bp.route('/api/render/features', methods=['POST'])
def set_render_features_route():
    """Update feature flags for the render pipeline."""
    data = request.get_json() or {}
    pipeline = _get_render_pipeline()

    if 'fixture_centric_enabled' in data:
        pipeline.features.FIXTURE_CENTRIC_ENABLED = bool(data['fixture_centric_enabled'])
    if 'legacy_channel_fallback' in data:
        pipeline.features.LEGACY_CHANNEL_FALLBACK = bool(data['legacy_channel_fallback'])
    if 'ai_suggestions_enabled' in data:
        pipeline.features.AI_SUGGESTIONS_ENABLED = bool(data['ai_suggestions_enabled'])
    if 'distribution_modes_enabled' in data:
        pipeline.features.DISTRIBUTION_MODES_ENABLED = bool(data['distribution_modes_enabled'])

    return jsonify({
        'success': True,
        'features': {
            'fixture_centric_enabled': pipeline.features.FIXTURE_CENTRIC_ENABLED,
            'legacy_channel_fallback': pipeline.features.LEGACY_CHANNEL_FALLBACK,
            'ai_suggestions_enabled': pipeline.features.AI_SUGGESTIONS_ENABLED,
            'distribution_modes_enabled': pipeline.features.DISTRIBUTION_MODES_ENABLED,
        }
    })
