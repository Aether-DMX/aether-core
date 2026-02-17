"""
AETHER Core — Fixtures Blueprint
Routes: /api/fixtures/*
Dependencies: content_manager, get_db, rdm_manager
"""

from flask import Blueprint, jsonify, request

fixtures_bp = Blueprint('fixtures', __name__)

_content_manager = None
_get_db = None
_rdm_manager = None


def init_app(content_manager, get_db_fn, rdm_manager):
    """Initialize blueprint with required dependencies."""
    global _content_manager, _get_db, _rdm_manager
    _content_manager = content_manager
    _get_db = get_db_fn
    _rdm_manager = rdm_manager


@fixtures_bp.route('/api/fixtures', methods=['GET'])
def get_fixtures():
    return jsonify(_content_manager.get_fixtures())

@fixtures_bp.route('/api/fixtures', methods=['POST'])
def create_fixture():
    return jsonify(_content_manager.create_fixture(request.get_json()))

@fixtures_bp.route('/api/fixtures/<fixture_id>', methods=['GET'])
def get_fixture(fixture_id):
    fixture = _content_manager.get_fixture(fixture_id)
    return jsonify(fixture) if fixture else (jsonify({'error': 'Fixture not found'}), 404)

@fixtures_bp.route('/api/fixtures/<fixture_id>', methods=['PUT'])
def update_fixture(fixture_id):
    result = _content_manager.update_fixture(fixture_id, request.get_json())
    if result.get('error'):
        return jsonify(result), 404
    return jsonify(result)

@fixtures_bp.route('/api/fixtures/<fixture_id>', methods=['DELETE'])
def delete_fixture(fixture_id):
    return jsonify(_content_manager.delete_fixture(fixture_id))

@fixtures_bp.route('/api/fixtures/<fixture_id>/identify', methods=['POST'])
def identify_fixture(fixture_id):
    """Identify a fixture by flashing its LED via RDM."""
    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT rdm_uid, universe FROM fixtures WHERE fixture_id = ?', (fixture_id,))
    row = c.fetchone()
    if not row:
        return jsonify({'success': False, 'error': 'Fixture not found'}), 404
    rdm_uid, universe = row
    if not rdm_uid:
        return jsonify({'success': False, 'error': 'Fixture has no RDM UID — cannot identify'}), 400
    data = request.get_json() or {}
    state = data.get('state', True)
    try:
        result = _rdm_manager.identify_by_uid(rdm_uid, state)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@fixtures_bp.route('/api/fixtures/universe/<int:universe>', methods=['GET'])
def get_fixtures_by_universe(universe):
    fixtures = _content_manager.get_fixtures()
    filtered = [f for f in fixtures if f.get('universe') == universe]
    return jsonify(filtered)

@fixtures_bp.route('/api/fixtures/channels', methods=['POST'])
def get_fixtures_for_channels():
    """Get fixtures that cover specific channel ranges"""
    data = request.get_json() or {}
    universe = data.get('universe', 1)
    channels = data.get('channels', [])
    return jsonify(_content_manager.get_fixtures_for_channels(universe, channels))
