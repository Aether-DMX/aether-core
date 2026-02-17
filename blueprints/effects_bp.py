"""
AETHER Core — Effects Blueprint
Routes: /api/effects/*
Dependencies: effects_engine, content_manager, unified_engine
"""

from flask import Blueprint, jsonify, request

effects_bp = Blueprint('effects', __name__)

_effects_engine = None
_content_manager = None
_unified_engine = None


def init_app(effects_engine, content_manager, unified_engine):
    """Initialize blueprint with required dependencies."""
    global _effects_engine, _content_manager, _unified_engine
    _effects_engine = effects_engine
    _content_manager = content_manager
    _unified_engine = unified_engine


# ─────────────────────────────────────────────────────────
# Legacy Effects Routes (backward compat)
# ─────────────────────────────────────────────────────────

@effects_bp.route('/api/effects/christmas', methods=['POST'])
def start_christmas_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.christmas_stagger(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('fade_ms', 1500),
        data.get('hold_ms', 1000),
        data.get('stagger_ms', 300)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/twinkle', methods=['POST'])
def start_twinkle_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.random_twinkle(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('min_fade_ms', 500),
        data.get('max_fade_ms', 2000)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/smooth', methods=['POST'])
def start_smooth_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.smooth_chase(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('fade_ms', 1500),
        data.get('hold_ms', 500)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/wave', methods=['POST'])
def start_wave_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.wave(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 0, 0, 0]),
        data.get('wave_speed_ms', 2000),
        data.get('tail_length', 2)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/strobe', methods=['POST'])
def start_strobe_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.strobe(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 255, 255, 0]),
        data.get('on_ms', 50),
        data.get('off_ms', 50)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/pulse', methods=['POST'])
def start_pulse_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.pulse(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('color', [255, 255, 255, 0]),
        data.get('pulse_ms', 2000),
        data.get('min_brightness', 0),
        data.get('max_brightness', 255)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/fade', methods=['POST'])
def start_fade_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.fade(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('colors'),
        data.get('cycle_ms', 10000)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/fire', methods=['POST'])
def start_fire_effect():
    data = request.get_json() or {}
    universes = data.get('universes', [2, 4])
    effect_id = _effects_engine.fire(
        universes,
        data.get('fixtures_per_universe', 2),
        data.get('channels_per_fixture', 4),
        data.get('intensity', 0.8)
    )
    return jsonify({'success': True, 'effect_id': effect_id})

@effects_bp.route('/api/effects/stop', methods=['POST'])
def stop_effects():
    data = request.get_json() or {}
    _effects_engine.stop_effect(data.get('effect_id'))
    return jsonify({'success': True})

@effects_bp.route('/api/effects', methods=['GET'])
def get_running_effects():
    return jsonify({'running': list(_effects_engine.running.keys()), 'count': len(_effects_engine.running)})


# ─────────────────────────────────────────────────────────
# Fixture-Aware Effects (Intelligent Distribution)
# ─────────────────────────────────────────────────────────

@effects_bp.route('/api/effects/fixture', methods=['POST'])
def play_fixture_effect_endpoint():
    """Play a fixture-aware effect with intelligent distribution."""
    data = request.get_json() or {}
    effect_type = data.get('effect_type', 'fixture_rainbow')
    fixture_ids = data.get('fixture_ids')
    mode = data.get('mode', 'chase')
    params = data.get('params', {})
    stack = data.get('stack', False)
    target_universes = data.get('universes')
    is_modifier = data.get('is_modifier', False)

    print(f"Fixture effect request: type={effect_type}, mode={mode}, universes={target_universes}, is_modifier={is_modifier}", flush=True)

    if mode not in ['chase', 'sync', 'wave']:
        return jsonify({'success': False, 'error': f"Invalid mode: {mode}. Must be 'chase', 'sync', or 'wave'"}), 400

    color_effects = ['fixture_rainbow', 'fixture_gradient', 'fixture_pulse', 'fixture_chase',
                     'fixture_hue_shift', 'fixture_color_temp', 'fixture_saturation_pulse',
                     'fixture_color_fade']
    motion_effects = ['strobe', 'wave', 'sweep_lr', 'sweep_rl', 'random',
                      'fixture_scanner', 'fixture_sparkle', 'fixture_lightning',
                      'fixture_heartbeat', 'fixture_tidal', 'fixture_ember', 'fixture_swell']
    valid_effects = color_effects + motion_effects

    if effect_type not in valid_effects:
        return jsonify({'success': False, 'error': f"Invalid effect_type: {effect_type}. Must be one of {valid_effects}"}), 400

    try:
        from unified_playback import play_fixture_effect

        if not is_modifier:
            if stack:
                is_motion_effect = effect_type in motion_effects
                status = _unified_engine.get_status()
                for session_info in status.get('sessions', []):
                    if session_info['id'].startswith('fixture_effect_'):
                        session_effect = session_info['id'].split('_')[2] if len(session_info['id'].split('_')) > 2 else ''
                        session_is_motion = session_effect in motion_effects
                        if is_motion_effect == session_is_motion:
                            _unified_engine.stop_session(session_info['id'])
            else:
                status = _unified_engine.get_status()
                for session_info in status.get('sessions', []):
                    if session_info['id'].startswith('fixture_effect_'):
                        _unified_engine.stop_session(session_info['id'])

        all_fixtures = _content_manager.get_fixtures()

        if target_universes and not fixture_ids:
            fixture_ids = [f['fixture_id'] for f in all_fixtures if f.get('universe') in target_universes]
            print(f"Filtered to {len(fixture_ids)} fixtures in universes {target_universes}", flush=True)
        elif fixture_ids and not target_universes:
            resolved = set()
            for f in all_fixtures:
                if f['fixture_id'] in fixture_ids:
                    u = f.get('universe')
                    if u is not None:
                        resolved.add(int(u))
            target_universes = sorted(resolved) if resolved else None
        elif not fixture_ids and not target_universes:
            resolved = set()
            for f in all_fixtures:
                u = f.get('universe')
                if u is not None:
                    resolved.add(int(u))
            target_universes = sorted(resolved) if resolved else None

        session_id = play_fixture_effect(effect_type, fixture_ids, mode, params, is_modifier, target_universes)
        print(f"Effect started: session_id={session_id}", flush=True)

        session = _unified_engine.get_session(session_id)
        fixtures_count = len(fixture_ids) if fixture_ids else len(_content_manager.get_fixtures())
        universes = session.universes if session else [1]

        return jsonify({
            'success': True,
            'session_id': session_id,
            'fixtures_count': fixtures_count,
            'universes': universes,
            'mode': mode,
            'effect_type': effect_type
        })
    except Exception as e:
        print(f"Fixture effect error: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@effects_bp.route('/api/effects/fixture/stop', methods=['POST'])
def stop_fixture_effect_endpoint():
    """Stop fixture-aware effects."""
    data = request.get_json() or {}
    session_id = data.get('session_id')

    try:
        if session_id:
            _unified_engine.stop_session(session_id)
        else:
            status = _unified_engine.get_status()
            for session_info in status.get('sessions', []):
                if session_info['id'].startswith('fixture_effect_'):
                    _unified_engine.stop_session(session_info['id'])

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@effects_bp.route('/api/effects/fixture/params', methods=['PATCH'])
def update_fixture_effect_params():
    """Update parameters of a running fixture effect without restarting it."""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    params = data.get('params', {})

    if not session_id:
        return jsonify({'success': False, 'error': 'session_id is required'}), 400

    try:
        session = _unified_engine.get_session(session_id)
        if not session:
            return jsonify({'success': False, 'error': f'Session not found: {session_id}'}), 404

        session.effect_params.update(params)

        return jsonify({'success': True, 'session_id': session_id, 'updated_params': params})
    except Exception as e:
        print(f"Fixture effect param update error: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@effects_bp.route('/api/effects/fixture/types', methods=['GET'])
def get_fixture_effect_types():
    """Get available fixture-aware effect types with their parameters."""
    return jsonify({
        'effect_types': [
            # Motion Effects (Primary)
            {
                'id': 'strobe',
                'name': 'Strobe',
                'description': 'Rapid on/off flashing',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 0.1, 'min': 0.05, 'max': 1.0, 'description': 'Flash interval (lower = faster)'},
                    'duty_cycle': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 0.9, 'description': 'On-time ratio'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Strobe color'}
                }
            },
            {
                'id': 'wave',
                'name': 'Wave',
                'description': 'Smooth wave rolling through fixtures',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0, 'description': 'Wave speed'},
                    'width': {'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 0.8, 'description': 'Wave width'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Wave color'},
                    'min_brightness': {'type': 'float', 'default': 0.0, 'min': 0, 'max': 0.5, 'description': 'Background brightness'}
                }
            },
            {
                'id': 'sweep_lr',
                'name': 'Sweep \u2192',
                'description': 'Light bar sweeps left to right',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0, 'description': 'Sweep time (seconds)'},
                    'width': {'type': 'float', 'default': 0.2, 'min': 0.05, 'max': 0.5, 'description': 'Bar width'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Sweep color'},
                    'bg_brightness': {'type': 'float', 'default': 0.0, 'min': 0, 'max': 0.3, 'description': 'Background brightness'}
                }
            },
            {
                'id': 'sweep_rl',
                'name': '\u2190 Sweep',
                'description': 'Light bar sweeps right to left',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0, 'description': 'Sweep time (seconds)'},
                    'width': {'type': 'float', 'default': 0.2, 'min': 0.05, 'max': 0.5, 'description': 'Bar width'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Sweep color'},
                    'bg_brightness': {'type': 'float', 'default': 0.0, 'min': 0, 'max': 0.3, 'description': 'Background brightness'}
                }
            },
            {
                'id': 'random',
                'name': 'Random',
                'description': 'Random fixtures light up',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 1.0, 'description': 'Change rate'},
                    'density': {'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 0.8, 'description': 'Fraction of fixtures lit'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Light color'},
                    'fade': {'type': 'bool', 'default': True, 'description': 'Smooth fade between states'}
                }
            },
            # Color Effects
            {
                'id': 'fixture_rainbow',
                'name': 'Rainbow',
                'description': 'Color-cycling rainbow effect',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.2, 'min': 0.01, 'max': 2.0, 'description': 'Color cycle speed'},
                    'saturation': {'type': 'float', 'default': 1.0, 'min': 0, 'max': 1.0, 'description': 'Color saturation'},
                    'value': {'type': 'float', 'default': 1.0, 'min': 0, 'max': 1.0, 'description': 'Brightness'}
                }
            },
            {
                'id': 'fixture_pulse',
                'name': 'Pulse',
                'description': 'Breathing/pulsing effect',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 5.0, 'description': 'Pulse speed'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Pulse color'},
                    'min_brightness': {'type': 'float', 'default': 0.1, 'min': 0, 'max': 1.0, 'description': 'Minimum brightness'},
                    'max_brightness': {'type': 'float', 'default': 1.0, 'min': 0, 'max': 1.0, 'description': 'Maximum brightness'}
                }
            },
            {
                'id': 'fixture_chase',
                'name': 'Chase',
                'description': 'Traveling chase across fixtures',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 10.0, 'description': 'Fixtures per second'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Chase color'},
                    'tail_length': {'type': 'int', 'default': 2, 'min': 0, 'max': 10, 'description': 'Trailing fixtures'},
                    'bg_color': {'type': 'rgb', 'default': [0, 0, 0], 'description': 'Background color'}
                }
            },
            # Color Modifier Effects
            {
                'id': 'fixture_hue_shift',
                'name': 'Hue Shift',
                'description': 'Rotates the hue of the active color',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.2, 'min': 0.02, 'max': 2.0, 'description': 'Rotation speed (Hz)'},
                    'range': {'type': 'float', 'default': 360, 'min': 30, 'max': 360, 'description': 'Hue range (degrees)'},
                    'offset': {'type': 'float', 'default': 0, 'min': 0, 'max': 360, 'description': 'Starting hue offset'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_color_temp',
                'name': 'Color Temp',
                'description': 'Shifts warm (amber) or cool (blue)',
                'category': 'color',
                'params': {
                    'temperature': {'type': 'float', 'default': 0, 'min': -100, 'max': 100, 'description': 'Cool (-100) to warm (+100)'},
                    'speed': {'type': 'float', 'default': 0, 'min': 0, 'max': 2.0, 'description': 'Animation speed (0 = static)'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_saturation_pulse',
                'name': 'Saturation Pulse',
                'description': 'Breathes between color and white',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 3.0, 'description': 'Pulse speed (Hz)'},
                    'min_saturation': {'type': 'float', 'default': 20, 'min': 0, 'max': 100, 'description': 'Minimum saturation (%)'},
                    'max_saturation': {'type': 'float', 'default': 100, 'min': 50, 'max': 100, 'description': 'Maximum saturation (%)'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_color_fade',
                'name': 'Color Fade',
                'description': 'Cycles through a color palette',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.2, 'min': 0.05, 'max': 2.0, 'description': 'Cycle speed (Hz)'},
                    'colors': {'type': 'color_array', 'default': [[255,0,0],[255,165,0],[255,255,0],[0,255,0],[0,0,255],[128,0,255]], 'description': 'Color palette'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_gradient',
                'name': 'Gradient',
                'description': 'Color gradient across fixtures',
                'category': 'color',
                'params': {
                    'speed': {'type': 'float', 'default': 0.2, 'min': 0.05, 'max': 2.0, 'description': 'Movement speed (Hz)'},
                    'colors': {'type': 'color_array', 'default': [[255,0,0],[0,0,255]], 'description': 'Gradient colors'},
                    'spread': {'type': 'float', 'default': 100, 'min': 10, 'max': 100, 'description': 'Spread across fixtures (%)'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            # Motion/Random Effects
            {
                'id': 'fixture_scanner',
                'name': 'Scanner',
                'description': 'Knight Rider style beam scanner',
                'category': 'motion',
                'params': {
                    'speed': {'type': 'float', 'default': 1.0, 'min': 0.5, 'max': 5.0, 'description': 'Scan speed (Hz)'},
                    'width': {'type': 'int', 'default': 2, 'min': 1, 'max': 5, 'description': 'Beam width (fixtures)'},
                    'direction': {'type': 'select', 'default': 'bounce', 'options': ['bounce', 'forward', 'backward'], 'description': 'Scan direction'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_sparkle',
                'name': 'Sparkle',
                'description': 'Random fixtures flash to white',
                'category': 'random',
                'params': {
                    'density': {'type': 'float', 'default': 30, 'min': 5, 'max': 80, 'description': 'Sparkle density (%)'},
                    'flash_duration': {'type': 'float', 'default': 150, 'min': 50, 'max': 500, 'description': 'Flash duration (ms)'},
                    'color': {'type': 'rgb', 'default': [255, 255, 255], 'description': 'Sparkle color (null = white)'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            },
            {
                'id': 'fixture_lightning',
                'name': 'Lightning',
                'description': 'Random bright flashes with decay',
                'category': 'random',
                'params': {
                    'frequency': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0, 'description': 'Flashes per second'},
                    'intensity': {'type': 'float', 'default': 100, 'min': 50, 'max': 100, 'description': 'Flash intensity (%)'},
                    'decay': {'type': 'float', 'default': 200, 'min': 50, 'max': 500, 'description': 'Fade-out time (ms)'},
                    'depth': {'type': 'float', 'default': 100, 'min': 0, 'max': 100, 'description': 'Effect depth (%)'}
                }
            }
        ],
        'modes': [
            {'id': 'chase', 'name': 'Chase', 'description': 'Each fixture gets a phase offset - effect travels across fixtures'},
            {'id': 'sync', 'name': 'Sync', 'description': 'All fixtures show the same value simultaneously'},
            {'id': 'wave', 'name': 'Wave', 'description': 'Smooth sine wave offset across fixtures'}
        ]
    })
