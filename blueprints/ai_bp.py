"""
AETHER Core — AI & Render Pipeline Blueprint
Routes: /api/ai/*, /api/render/pipeline/*, /api/render/features
Dependencies: ai_fixture_advisor functions, get_ai_advisor, get_render_pipeline, get_db
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import json
from ai_fixture_advisor import (
    get_distribution_suggestions,
    apply_ai_suggestion,
    dismiss_ai_suggestion,
)

ai_bp = Blueprint('ai', __name__)

_get_ai_advisor = None
_get_render_pipeline = None
_get_db = None


def init_app(get_ai_advisor_fn, get_render_pipeline_fn, get_db_fn=None):
    """Initialize blueprint with required dependencies."""
    global _get_ai_advisor, _get_render_pipeline, _get_db
    _get_ai_advisor = get_ai_advisor_fn
    _get_render_pipeline = get_render_pipeline_fn
    _get_db = get_db_fn
    # Initialize AI tables if DB available
    if _get_db:
        _init_ai_tables()


def _init_ai_tables():
    """Create AI learning tables if they don't exist."""
    conn = _get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS ai_preferences (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS ai_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        intent TEXT,
        effect TEXT,
        rating INTEGER,
        feedback TEXT,
        context TEXT,
        created_at TEXT NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS ai_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        detail TEXT,
        created_at TEXT NOT NULL
    )''')
    print("✅ AI learning tables initialized")


def _audit(action, detail=None):
    """Log an AI action to the audit trail."""
    if not _get_db:
        return
    try:
        conn = _get_db()
        conn.execute(
            'INSERT INTO ai_audit_log (action, detail, created_at) VALUES (?, ?, ?)',
            (action, json.dumps(detail) if detail else None, datetime.now().isoformat())
        )
    except Exception as e:
        print(f"⚠️ AI audit log error: {e}")


# ============================================================
# AI Learning Routes — Preferences, Outcomes, Audit, Ops
# ============================================================
@ai_bp.route('/api/ai/preferences', methods=['GET'])
def ai_get_prefs():
    """Get all stored AI preferences."""
    if not _get_db:
        return jsonify({})
    conn = _get_db()
    rows = conn.execute('SELECT key, value FROM ai_preferences').fetchall()
    prefs = {}
    for row in rows:
        try:
            prefs[row[0]] = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            prefs[row[0]] = row[1]
    return jsonify(prefs)


@ai_bp.route('/api/ai/preferences/<key>', methods=['GET', 'POST'])
def ai_pref(key):
    """Get or set a single AI preference."""
    if not _get_db:
        if request.method == 'POST':
            return jsonify({'success': True})
        return jsonify({'value': None})

    conn = _get_db()
    if request.method == 'POST':
        data = request.get_json() or {}
        value = data.get('value', data)
        conn.execute(
            'INSERT OR REPLACE INTO ai_preferences (key, value, updated_at) VALUES (?, ?, ?)',
            (key, json.dumps(value), datetime.now().isoformat())
        )
        _audit('preference_set', {'key': key, 'value': value})
        return jsonify({'success': True, 'key': key})
    else:
        row = conn.execute('SELECT value FROM ai_preferences WHERE key = ?', (key,)).fetchone()
        if row:
            try:
                return jsonify({'value': json.loads(row[0])})
            except (json.JSONDecodeError, TypeError):
                return jsonify({'value': row[0]})
        return jsonify({'value': None})


@ai_bp.route('/api/ai/budget', methods=['GET'])
def ai_budget():
    """Get AI operation budget (tracks usage for rate limiting)."""
    if not _get_db:
        return jsonify({"remaining": 999, "used": 0})
    conn = _get_db()
    # Count outcomes in last 24h as "used"
    row = conn.execute(
        "SELECT COUNT(*) FROM ai_outcomes WHERE created_at > datetime('now', '-1 day')"
    ).fetchone()
    used = row[0] if row else 0
    return jsonify({"remaining": max(0, 999 - used), "used": used})


@ai_bp.route('/api/ai/outcomes', methods=['GET', 'POST'])
def ai_outcomes():
    """Store or retrieve AI outcome feedback (learning data)."""
    if not _get_db:
        if request.method == 'POST':
            return jsonify({'success': True})
        return jsonify({'outcomes': []})

    conn = _get_db()
    if request.method == 'POST':
        d = request.get_json() or {}
        conn.execute(
            'INSERT INTO ai_outcomes (intent, effect, rating, feedback, context, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            (
                d.get('intent', ''),
                d.get('effect', ''),
                d.get('rating', 0),
                d.get('feedback', ''),
                json.dumps(d.get('context', {})),
                datetime.now().isoformat()
            )
        )
        _audit('outcome_recorded', {'intent': d.get('intent'), 'rating': d.get('rating')})
        return jsonify({'success': True})
    else:
        limit = request.args.get('limit', 50, type=int)
        rows = conn.execute(
            'SELECT id, intent, effect, rating, feedback, context, created_at FROM ai_outcomes ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
        outcomes = []
        for r in rows:
            outcomes.append({
                'id': r[0], 'intent': r[1], 'effect': r[2],
                'rating': r[3], 'feedback': r[4],
                'context': json.loads(r[5]) if r[5] else {},
                'created_at': r[6]
            })
        return jsonify({'outcomes': outcomes})


@ai_bp.route('/api/ai/audit', methods=['GET'])
def ai_audit():
    """Get AI audit log (actions taken by the AI system)."""
    if not _get_db:
        return jsonify({'log': []})
    conn = _get_db()
    limit = request.args.get('limit', 100, type=int)
    rows = conn.execute(
        'SELECT id, action, detail, created_at FROM ai_audit_log ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    log = []
    for r in rows:
        log.append({
            'id': r[0], 'action': r[1],
            'detail': json.loads(r[2]) if r[2] else None,
            'created_at': r[3]
        })
    return jsonify({'log': log})


@ai_bp.route('/api/ai/ops', methods=['GET'])
def ai_ops():
    """Get AI operational stats."""
    if not _get_db:
        return jsonify({'ops': {}})
    conn = _get_db()
    total_outcomes = conn.execute('SELECT COUNT(*) FROM ai_outcomes').fetchone()[0]
    total_prefs = conn.execute('SELECT COUNT(*) FROM ai_preferences').fetchone()[0]
    avg_rating = conn.execute('SELECT AVG(rating) FROM ai_outcomes WHERE rating > 0').fetchone()[0]
    top_intents = conn.execute(
        'SELECT intent, COUNT(*) as cnt FROM ai_outcomes GROUP BY intent ORDER BY cnt DESC LIMIT 5'
    ).fetchall()
    return jsonify({'ops': {
        'total_outcomes': total_outcomes,
        'total_preferences': total_prefs,
        'average_rating': round(avg_rating, 2) if avg_rating else None,
        'top_intents': [{'intent': r[0], 'count': r[1]} for r in top_intents],
    }})


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
        _audit('optimize_playback', {'intent': intent, 'matched': 'preset'})
        return jsonify(VIBE_EFFECTS[intent.lower()])

    # For custom text intents, do simple keyword matching
    intent_lower = intent.lower()

    if any(w in intent_lower for w in ['relax', 'calm', 'chill', 'soft', 'gentle']):
        result = VIBE_EFFECTS['chill']
    elif any(w in intent_lower for w in ['party', 'dance', 'energy', 'exciting', 'fun']):
        result = VIBE_EFFECTS['party']
    elif any(w in intent_lower for w in ['romantic', 'intimate', 'love', 'date', 'cozy']):
        result = VIBE_EFFECTS['romantic']
    elif any(w in intent_lower for w in ['dramatic', 'intense', 'powerful', 'bold']):
        result = VIBE_EFFECTS['dramatic']
    elif any(w in intent_lower for w in ['concert', 'show', 'performance', 'live']):
        result = VIBE_EFFECTS['concert']
    elif any(w in intent_lower for w in ['sunset', 'warm', 'golden', 'evening']):
        result = VIBE_EFFECTS['sunset']
    elif any(w in intent_lower for w in ['focus', 'work', 'study', 'clean', 'bright']):
        result = VIBE_EFFECTS['focus']
    elif any(w in intent_lower for w in ['spooky', 'halloween', 'scary', 'creepy', 'horror']):
        result = VIBE_EFFECTS['spooky']
    elif any(w in intent_lower for w in ['rainbow', 'colorful', 'color']):
        result = {
            'explanation': 'Adding vibrant rainbow colors that cycle through the spectrum.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_rainbow', 'params': {'speed': 0.3, 'depth': 100}},
            ]
        }
    elif any(w in intent_lower for w in ['pulse', 'beat', 'rhythm']):
        result = {
            'explanation': 'Adding rhythmic pulses to match the beat.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_pulse', 'params': {'speed': 1.0, 'depth': 80}},
            ]
        }
    else:
        result = {
            'explanation': f'Based on "{intent}", adding some dynamic color and movement.',
            'changes': [
                {'type': 'add_effect', 'effect_id': 'fixture_color_fade', 'params': {'speed': 0.2, 'depth': 70}},
                {'type': 'add_effect', 'effect_id': 'wave', 'params': {'speed': 0.5, 'depth': 50}},
            ]
        }

    _audit('optimize_playback', {'intent': intent, 'matched': 'keyword'})
    return jsonify(result)


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
    _audit('suggestion_applied', {'suggestion_id': suggestion_id})
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
    _audit('suggestion_dismissed', {'suggestion_id': suggestion_id})
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
