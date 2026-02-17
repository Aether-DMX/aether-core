"""
AETHER Core — Operator Trust Enforcement Blueprint
Routes: /api/trust/*
Dependencies: operator_trust module only (zero shared state)
"""

from flask import Blueprint, jsonify, request
from operator_trust import (
    get_trust_status, get_trust_events, check_ui_sync, clear_failure_halt
)

trust_bp = Blueprint('trust', __name__)


@trust_bp.route('/api/trust/status', methods=['GET'])
def trust_status():
    """Get current trust enforcement status.

    TRUST RULE: Operator must always see system state.
    Returns:
        - monitoring: whether background monitoring is active
        - node_health: health status for all tracked nodes
        - playback_halted_due_to_failure: whether playback was halted
        - active_alerts: list of active alert IDs
        - recent_events: last 10 trust events
    """
    return jsonify(get_trust_status())


@trust_bp.route('/api/trust/events', methods=['GET'])
def trust_events():
    """Get trust event history.

    TRUST RULE: All trust events must be visible to operators.
    Returns chronological list of trust-related events.
    """
    limit = request.args.get('limit', 100, type=int)
    return jsonify({'events': get_trust_events(limit)})


@trust_bp.route('/api/trust/ui-sync', methods=['POST'])
def trust_ui_sync_check():
    """Check if UI state matches backend reality.

    TRUST RULE: UI desync -> REALITY wins over UI.
    UI should call this periodically to verify its state matches backend.
    If mismatch detected, UI must update to match reality.
    """
    data = request.get_json() or {}
    ui_state = data.get('state', {})
    component = data.get('component', 'unknown')

    result = check_ui_sync(ui_state, component)
    return jsonify(result)


@trust_bp.route('/api/trust/clear-halt', methods=['POST'])
def trust_clear_halt():
    """Clear playback halt after operator acknowledges failure.

    TRUST RULE: Partial node failure -> SYSTEM HALTS playback + ALERTS.
    After operator acknowledges the issue, this endpoint clears the halt.
    """
    print("TRUST: Operator clearing failure halt", flush=True)
    clear_failure_halt()
    return jsonify({'success': True, 'message': 'Failure halt cleared'})
