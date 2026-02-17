"""
AETHER Core — RDM Blueprint (Remote Device Management)
Routes: /api/rdm/*, /api/nodes/<node_id>/rdm/*
Dependencies: rdm_manager, get_db
"""

from flask import Blueprint, jsonify, request

rdm_bp = Blueprint('rdm', __name__)

_rdm_manager = None
_get_db = None


def init_app(rdm_manager, get_db_fn):
    """Initialize blueprint with required dependencies."""
    global _rdm_manager, _get_db
    _rdm_manager = rdm_manager
    _get_db = get_db_fn


# ─────────────────────────────────────────────────────────
# Node-scoped RDM Routes
# ─────────────────────────────────────────────────────────

@rdm_bp.route('/api/nodes/<node_id>/rdm/discover', methods=['POST'])
def rdm_discover(node_id):
    """Start RDM discovery on a node - finds all RDM fixtures on DMX bus"""
    try:
        result = _rdm_manager.discover_devices(node_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@rdm_bp.route('/api/nodes/<node_id>/rdm/discover/status', methods=['GET'])
def rdm_discover_status(node_id):
    """Get RDM discovery status for a node"""
    status = _rdm_manager.discovery_tasks.get(node_id, {"status": "idle"})
    return jsonify(status)


@rdm_bp.route('/api/nodes/<node_id>/rdm/devices', methods=['GET'])
def rdm_devices_for_node(node_id):
    """List all RDM devices discovered on a node"""
    devices = _rdm_manager.get_devices_for_node(node_id)
    return jsonify(devices)


# ─────────────────────────────────────────────────────────
# Global RDM Routes
# ─────────────────────────────────────────────────────────

@rdm_bp.route('/api/rdm/devices/<uid>/label', methods=['POST'])
def rdm_set_label(uid):
    """Set device label (stored in database, not on fixture)"""
    data = request.get_json() or {}
    label = data.get('label', '')

    conn = _get_db()
    c = conn.cursor()
    c.execute('UPDATE rdm_devices SET device_label = ? WHERE uid = ?', (label, uid))
    conn.commit()

    return jsonify({'success': True, 'uid': uid, 'label': label})


@rdm_bp.route('/api/rdm/devices/<uid>/personality', methods=['POST'])
def rdm_set_personality(uid):
    """Set RDM device personality - not yet supported in firmware."""
    data = request.get_json() or {}
    personality = data.get('personality')
    return jsonify({
        'success': False,
        'error': 'SET_PERSONALITY not yet supported in firmware. Personality must be set on the fixture directly.',
        'uid': uid,
        'personality': personality
    }), 501


@rdm_bp.route('/api/rdm/devices/<uid>', methods=['DELETE'])
def rdm_delete_device(uid):
    """Remove a stale RDM device from database"""
    result = _rdm_manager.delete_device(uid)
    return jsonify(result)


@rdm_bp.route('/api/rdm/status', methods=['GET'])
def rdm_status():
    """Get consolidated RDM status including live_inventory."""
    return jsonify(_rdm_manager.get_status())


@rdm_bp.route('/api/rdm/discover', methods=['POST'])
def rdm_discover_all():
    """Run RDM discovery on all nodes via consolidated RDMManager."""
    print("RDM: Discovery requested via API", flush=True)
    result = _rdm_manager.discover_all()
    return jsonify(result)


@rdm_bp.route('/api/rdm/devices', methods=['GET'])
def rdm_devices():
    """Get cached RDM devices from consolidated RDMManager."""
    return jsonify({'devices': _rdm_manager.get_cached_devices()})


@rdm_bp.route('/api/rdm/devices/<uid>', methods=['GET'])
def rdm_device_info(uid):
    """Get info for a specific RDM device from cache."""
    return jsonify(_rdm_manager.get_cached_device_info(uid))


@rdm_bp.route('/api/rdm/devices/<uid>/address', methods=['GET'])
def rdm_device_get_address(uid):
    """Get current DMX address for a device via RDMManager."""
    return jsonify(_rdm_manager.get_address_by_uid(uid))


@rdm_bp.route('/api/rdm/devices/<uid>/address', methods=['POST'])
def rdm_device_set_address(uid):
    """Set DMX address for a device via RDMManager."""
    data = request.get_json() or {}
    address = data.get('address')

    if address is None:
        return jsonify({'success': False, 'error': 'Address required'}), 400

    print(f"RDM: Setting {uid} to address {address}", flush=True)
    result = _rdm_manager.set_address_by_uid(uid, address)
    return jsonify(result)


@rdm_bp.route('/api/rdm/devices/<uid>/identify', methods=['POST'])
def rdm_device_identify(uid):
    """Turn identify mode on/off for a device via RDMManager."""
    data = request.get_json() or {}
    state = data.get('state', True)

    print(f"RDM: Identify {uid} = {state}", flush=True)
    result = _rdm_manager.identify_by_uid(uid, state)
    return jsonify(result)


@rdm_bp.route('/api/rdm/address-suggestions', methods=['GET'])
def rdm_address_suggestions():
    """Get address conflict suggestions from consolidated RDMManager."""
    return jsonify(_rdm_manager.suggest_addresses())


@rdm_bp.route('/api/rdm/auto-fix', methods=['POST'])
def rdm_auto_fix():
    """Automatically fix all address conflicts via RDMManager."""
    print("RDM: Auto-fix addresses requested", flush=True)
    result = _rdm_manager.auto_fix_addresses()
    return jsonify(result)


@rdm_bp.route('/api/rdm/verify-cue', methods=['POST'])
def rdm_verify_cue():
    """Verify all fixtures for a cue are ready via RDMManager."""
    data = request.get_json() or {}
    result = _rdm_manager.verify_cue_readiness(data)
    return jsonify(result)
