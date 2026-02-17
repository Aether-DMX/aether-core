"""
AETHER Core — Fixture Library Blueprint
Routes: /api/fixture-library/*
Dependencies: fixture_library, content_manager, channel_mapper, looks_sequences_manager, get_db
"""

import time
from flask import Blueprint, jsonify, request
from fixture_library import FixtureProfile, FixtureMode, ChannelCapability, FixtureInstance

fixture_library_bp = Blueprint('fixture_library', __name__)

_fixture_library = None
_content_manager = None
_channel_mapper = None
_looks_sequences_manager = None
_get_db = None


def init_app(fixture_library, content_manager, channel_mapper, looks_sequences_manager, get_db_fn):
    """Initialize blueprint with required dependencies."""
    global _fixture_library, _content_manager, _channel_mapper, _looks_sequences_manager, _get_db
    _fixture_library = fixture_library
    _content_manager = content_manager
    _channel_mapper = channel_mapper
    _looks_sequences_manager = looks_sequences_manager
    _get_db = get_db_fn


@fixture_library_bp.route('/api/fixture-library/profiles', methods=['GET'])
def get_fixture_profiles():
    """Get all available fixture profiles"""
    category = request.args.get('category')
    profiles = _fixture_library.get_all_profiles(category)
    return jsonify([{
        'profile_id': p.profile_id,
        'manufacturer': p.manufacturer,
        'model': p.model,
        'category': p.category,
        'modes': [{'mode_id': m.mode_id, 'name': m.name, 'channel_count': m.channel_count} for m in p.modes],
        'source': p.source
    } for p in profiles])

@fixture_library_bp.route('/api/fixture-library/profiles/<profile_id>', methods=['GET'])
def get_fixture_profile(profile_id):
    """Get a specific fixture profile with full details"""
    profile = _fixture_library.get_profile(profile_id)
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({
        'profile_id': profile.profile_id,
        'manufacturer': profile.manufacturer,
        'model': profile.model,
        'category': profile.category,
        'modes': [{
            'mode_id': m.mode_id,
            'name': m.name,
            'channel_count': m.channel_count,
            'channels': [{'name': ch.name, 'type': ch.type, 'default': ch.default} for ch in m.channels]
        } for m in profile.modes],
        'source': profile.source,
        'ofl_key': profile.ofl_key,
        'rdm_manufacturer_id': profile.rdm_manufacturer_id,
        'rdm_device_model_id': profile.rdm_device_model_id
    })

@fixture_library_bp.route('/api/fixture-library/profiles', methods=['POST'])
def create_fixture_profile():
    """Create a new fixture profile"""
    data = request.get_json() or {}
    try:
        modes = []
        for m in data.get('modes', []):
            channels = [ChannelCapability(
                name=ch.get('name', f'Channel {i+1}'),
                type=ch.get('type', 'control'),
                default=ch.get('default', 0)
            ) for i, ch in enumerate(m.get('channels', []))]
            modes.append(FixtureMode(
                mode_id=m.get('mode_id', 'default'),
                name=m.get('name', 'Default'),
                channel_count=len(channels),
                channels=channels
            ))

        profile = FixtureProfile(
            profile_id=data.get('profile_id', f"custom-{int(time.time())}"),
            manufacturer=data.get('manufacturer', 'Custom'),
            model=data.get('model', 'Fixture'),
            category=data.get('category', 'generic'),
            modes=modes,
            source='manual'
        )
        _fixture_library.save_profile(profile)
        return jsonify({'success': True, 'profile_id': profile.profile_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@fixture_library_bp.route('/api/fixture-library/ofl/search', methods=['GET'])
def search_ofl():
    """Search Open Fixture Library for fixtures"""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify({'error': 'Query must be at least 2 characters'}), 400
    results = _fixture_library.search_ofl(query)
    return jsonify(results)

@fixture_library_bp.route('/api/fixture-library/ofl/manufacturers', methods=['GET'])
def get_ofl_manufacturers():
    """Get list of manufacturers from Open Fixture Library"""
    manufacturers = _fixture_library.get_ofl_manufacturers()
    return jsonify(manufacturers)

@fixture_library_bp.route('/api/fixture-library/ofl/import', methods=['POST'])
def import_ofl_fixture():
    """Import a fixture from Open Fixture Library"""
    data = request.get_json() or {}
    manufacturer = data.get('manufacturer')
    fixture = data.get('fixture')
    if not manufacturer or not fixture:
        return jsonify({'error': 'manufacturer and fixture required'}), 400

    profile = _fixture_library.import_from_ofl(manufacturer, fixture)
    if profile:
        return jsonify({
            'success': True,
            'profile_id': profile.profile_id,
            'manufacturer': profile.manufacturer,
            'model': profile.model,
            'modes': [{'mode_id': m.mode_id, 'name': m.name, 'channel_count': m.channel_count} for m in profile.modes]
        })
    return jsonify({'error': 'Failed to import fixture'}), 500

@fixture_library_bp.route('/api/fixture-library/rdm/auto-configure', methods=['POST'])
def auto_configure_from_rdm():
    """Auto-configure fixtures from RDM devices"""
    data = request.get_json() or {}
    rdm_uid = data.get('rdm_uid')

    conn = _get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM rdm_devices WHERE uid = ?', (rdm_uid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'RDM device not found'}), 404

    rdm_device = dict(row)

    profile = _fixture_library.find_profile_by_rdm(
        rdm_device.get('manufacturer_id', 0),
        rdm_device.get('device_model_id', 0)
    )

    instance = _fixture_library.create_fixture_from_rdm(rdm_device, profile)

    mode = None
    if profile:
        mode = profile.get_mode(instance.mode_id)

    fixture_data = {
        'fixture_id': instance.fixture_id,
        'name': instance.name,
        'type': profile.category if profile else 'generic',
        'manufacturer': profile.manufacturer if profile else _fixture_library.get_rdm_manufacturer_name(rdm_device.get('manufacturer_id', 0)),
        'model': profile.model if profile else 'Unknown',
        'universe': instance.universe,
        'start_channel': instance.start_channel,
        'channel_count': mode.channel_count if mode else rdm_device.get('dmx_footprint', 4),
        'channel_map': [ch.name for ch in mode.channels] if mode else [],
        'rdm_uid': rdm_uid
    }
    _content_manager.create_fixture(fixture_data)

    return jsonify({
        'success': True,
        'fixture': fixture_data,
        'profile_matched': profile is not None,
        'profile_id': profile.profile_id if profile else None
    })

@fixture_library_bp.route('/api/fixture-library/apply-color', methods=['POST'])
def apply_color_to_fixtures():
    """Apply a color to specified fixtures intelligently"""
    data = request.get_json() or {}
    fixture_ids = data.get('fixture_ids', [])
    color = data.get('color', {})
    fade_ms = data.get('fade_ms', 0)
    universe = data.get('universe', 1)

    if not fixture_ids:
        return jsonify({'error': 'No fixtures specified'}), 400

    fixtures = []
    for fid in fixture_ids:
        fixture_data = _content_manager.get_fixture(fid)
        if fixture_data:
            profile = _fixture_library.get_profile(fixture_data.get('profile_id', 'generic-rgbw'))
            if not profile:
                profile = _fixture_library._create_generic_profile_for_footprint(
                    fixture_data.get('channel_count', 4)
                )
            fixtures.append(FixtureInstance(
                fixture_id=fixture_data['fixture_id'],
                name=fixture_data['name'],
                profile_id=profile.profile_id,
                mode_id=profile.modes[0].mode_id if profile.modes else 'default',
                universe=fixture_data.get('universe', 1),
                start_channel=fixture_data.get('start_channel', 1)
            ))

    if not fixtures:
        return jsonify({'error': 'No valid fixtures found'}), 404

    channels = _channel_mapper.apply_color_to_fixtures(
        fixtures,
        r=color.get('r', 0),
        g=color.get('g', 0),
        b=color.get('b', 0),
        w=color.get('w', 0),
        dimmer=color.get('dimmer', 255)
    )

    _content_manager.set_channels(universe, channels, fade_ms=fade_ms)

    return jsonify({
        'success': True,
        'channels': channels,
        'fixture_count': len(fixtures)
    })

@fixture_library_bp.route('/api/fixture-library/apply-scene-to-all', methods=['POST'])
def apply_scene_to_all_fixtures():
    """Apply a scene's color pattern to ALL configured fixtures intelligently."""
    data = request.get_json() or {}
    scene_id = data.get('scene_id')
    fade_ms = data.get('fade_ms', 1000)
    universes = data.get('universes', [])

    if not scene_id:
        return jsonify({'error': 'scene_id required'}), 400

    scene = _content_manager.get_scene(scene_id)
    if not scene:
        return jsonify({'error': 'Scene not found'}), 404

    scene_channels = scene.get('channels', {})
    if not scene_channels:
        return jsonify({'error': 'Scene has no channels'}), 400

    ch_values = sorted([(int(k), v) for k, v in scene_channels.items()], key=lambda x: x[0])

    color = {'r': 0, 'g': 0, 'b': 0, 'w': 0, 'dimmer': 255}
    if len(ch_values) >= 1:
        color['r'] = ch_values[0][1]
    if len(ch_values) >= 2:
        color['g'] = ch_values[1][1]
    if len(ch_values) >= 3:
        color['b'] = ch_values[2][1]
    if len(ch_values) >= 4:
        color['w'] = ch_values[3][1]
    if len(ch_values) >= 5:
        color['dimmer'] = ch_values[4][1]

    all_fixtures = _content_manager.get_fixtures()

    if universes:
        all_fixtures = [f for f in all_fixtures if f.get('universe') in universes]

    if not all_fixtures:
        return jsonify({'error': 'No fixtures configured'}), 404

    fixtures_by_universe = {}
    for fixture_data in all_fixtures:
        u = fixture_data.get('universe', 1)
        if u not in fixtures_by_universe:
            fixtures_by_universe[u] = []

        profile = _fixture_library.get_profile(fixture_data.get('profile_id'))
        if not profile:
            profile = _fixture_library._create_generic_profile_for_footprint(
                fixture_data.get('channel_count', 4)
            )

        fixtures_by_universe[u].append(FixtureInstance(
            fixture_id=fixture_data['fixture_id'],
            name=fixture_data['name'],
            profile_id=profile.profile_id,
            mode_id=profile.modes[0].mode_id if profile.modes else 'default',
            universe=u,
            start_channel=fixture_data.get('start_channel', 1)
        ))

    total_fixtures = 0
    all_channels = {}

    for universe, fixtures in fixtures_by_universe.items():
        channels = _channel_mapper.apply_color_to_fixtures(
            fixtures,
            r=color['r'],
            g=color['g'],
            b=color['b'],
            w=color['w'],
            dimmer=color['dimmer']
        )

        _content_manager.set_channels(universe, channels, fade_ms=fade_ms)
        total_fixtures += len(fixtures)
        all_channels[universe] = channels

    return jsonify({
        'success': True,
        'scene_name': scene.get('name'),
        'color_extracted': color,
        'fixture_count': total_fixtures,
        'universes': list(fixtures_by_universe.keys()),
        'channels_by_universe': all_channels
    })

@fixture_library_bp.route('/api/fixture-library/apply-look-to-all', methods=['POST'])
def apply_look_to_all_fixtures():
    data = request.get_json() or {}
    look_id = data.get('look_id')
    fade_ms = data.get('fade_ms', 1000)
    universes_filter = data.get('universes', [])
    if not look_id:
        return jsonify({'error': 'look_id required'}), 400
    look = _looks_sequences_manager.get_look(look_id)
    if not look:
        return jsonify({'error': 'Look not found'}), 404
    ch = look.channels
    if not ch:
        return jsonify({'error': 'Look has no channels'}), 400
    by_univ = {}
    for k, v in ch.items():
        ks = str(k)
        if ':' in ks:
            u, c_num = ks.split(':',1)
            u = int(u)
        else:
            u = 1
            c_num = ks
        if universes_filter and u not in universes_filter:
            continue
        if u not in by_univ:
            by_univ[u] = {}
        by_univ[u][c_num] = v
    for u, channels in by_univ.items():
        _content_manager.set_channels(u, channels, fade_ms=fade_ms)
    return jsonify({
        'success': True,
        'look_name': look.name,
        'universes': list(by_univ.keys()),
    })

@fixture_library_bp.route('/api/fixture-library/apply-color-to-all', methods=['POST'])
def apply_color_to_all_fixtures():
    """Apply a color to ALL configured fixtures."""
    data = request.get_json() or {}
    color = data.get('color', {})
    fade_ms = data.get('fade_ms', 0)
    universes = data.get('universes', [])

    if not color:
        return jsonify({'error': 'color required'}), 400

    all_fixtures = _content_manager.get_fixtures()

    if universes:
        all_fixtures = [f for f in all_fixtures if f.get('universe') in universes]

    if not all_fixtures:
        return jsonify({'error': 'No fixtures configured'}), 404

    fixtures_by_universe = {}
    for fixture_data in all_fixtures:
        u = fixture_data.get('universe', 1)
        if u not in fixtures_by_universe:
            fixtures_by_universe[u] = []

        profile = _fixture_library.get_profile(fixture_data.get('profile_id'))
        if not profile:
            profile = _fixture_library._create_generic_profile_for_footprint(
                fixture_data.get('channel_count', 4)
            )

        fixtures_by_universe[u].append(FixtureInstance(
            fixture_id=fixture_data['fixture_id'],
            name=fixture_data['name'],
            profile_id=profile.profile_id,
            mode_id=profile.modes[0].mode_id if profile.modes else 'default',
            universe=u,
            start_channel=fixture_data.get('start_channel', 1)
        ))

    total_fixtures = 0
    for universe, fixtures in fixtures_by_universe.items():
        channels = _channel_mapper.apply_color_to_fixtures(
            fixtures,
            r=color.get('r', 0),
            g=color.get('g', 0),
            b=color.get('b', 0),
            w=color.get('w', 0),
            dimmer=color.get('dimmer', 255)
        )

        _content_manager.set_channels(universe, channels, fade_ms=fade_ms)
        total_fixtures += len(fixtures)

    return jsonify({
        'success': True,
        'color': color,
        'fixture_count': total_fixtures,
        'universes': list(fixtures_by_universe.keys())
    })
