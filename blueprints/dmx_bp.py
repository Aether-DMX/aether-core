"""
AETHER Core — DMX Blueprint
Routes: /api/dmx/*
Dependencies: content_manager, node_manager (aliased as ssot), dmx_state,
              stop_all_playback, get_render_pipeline, get_db,
              arbitration, ArbitrationManager, effects_engine, chase_engine,
              show_engine, schedule_runner, playback_manager, NodeManager,
              AETHER_UDPJSON_PORT, WIFI_COMMAND_PORT,
              AETHER_VERSION, AETHER_COMMIT, AETHER_START_TIME, AETHER_FILE_PATH
"""

from datetime import datetime
from flask import Blueprint, jsonify, request

dmx_bp = Blueprint('dmx', __name__)

_content_manager = None
_node_manager = None
_ssot = None  # Same as node_manager, used for UDP commands
_dmx_state = None
_stop_all_playback = None
_get_render_pipeline = None
_get_db = None
_arbitration = None
_ArbitrationManager = None
_effects_engine = None
_chase_engine = None
_show_engine = None
_schedule_runner = None
_playback_manager = None
_NodeManager = None
_AETHER_UDPJSON_PORT = None
_WIFI_COMMAND_PORT = None
_AETHER_VERSION = None
_AETHER_COMMIT = None
_AETHER_START_TIME = None
_AETHER_FILE_PATH = None


def init_app(content_manager, node_manager, dmx_state, stop_all_playback_fn,
             get_render_pipeline_fn, get_db_fn, arbitration, ArbitrationManager,
             effects_engine, chase_engine, show_engine, schedule_runner,
             playback_manager, NodeManager, AETHER_UDPJSON_PORT, WIFI_COMMAND_PORT,
             AETHER_VERSION, AETHER_COMMIT, AETHER_START_TIME, AETHER_FILE_PATH):
    """Initialize blueprint with required dependencies."""
    global _content_manager, _node_manager, _ssot, _dmx_state
    global _stop_all_playback, _get_render_pipeline, _get_db
    global _arbitration, _ArbitrationManager
    global _effects_engine, _chase_engine, _show_engine, _schedule_runner
    global _playback_manager, _NodeManager
    global _AETHER_UDPJSON_PORT, _WIFI_COMMAND_PORT
    global _AETHER_VERSION, _AETHER_COMMIT, _AETHER_START_TIME, _AETHER_FILE_PATH
    _content_manager = content_manager
    _node_manager = node_manager
    _ssot = node_manager  # ssot is an alias for node_manager
    _dmx_state = dmx_state
    _stop_all_playback = stop_all_playback_fn
    _get_render_pipeline = get_render_pipeline_fn
    _get_db = get_db_fn
    _arbitration = arbitration
    _ArbitrationManager = ArbitrationManager
    _effects_engine = effects_engine
    _chase_engine = chase_engine
    _show_engine = show_engine
    _schedule_runner = schedule_runner
    _playback_manager = playback_manager
    _NodeManager = NodeManager
    _AETHER_UDPJSON_PORT = AETHER_UDPJSON_PORT
    _WIFI_COMMAND_PORT = WIFI_COMMAND_PORT
    _AETHER_VERSION = AETHER_VERSION
    _AETHER_COMMIT = AETHER_COMMIT
    _AETHER_START_TIME = AETHER_START_TIME
    _AETHER_FILE_PATH = AETHER_FILE_PATH


# ---------------------------------------------------------
# DMX Routes
# ---------------------------------------------------------
@dmx_bp.route('/api/dmx/set', methods=['POST'])
def dmx_set():
    """
    Set DMX channel values.

    Supports both legacy channel-based and new fixture-centric modes.

    Legacy (channel-based):
    {
        "universe": 2,
        "channels": {"1": 255, "2": 128},
        "fade_ms": 0
    }

    Fixture-Centric (Phase 0+):
    {
        "universe": 2,
        "fixture_id": "par_1",
        "attributes": {"intensity": 255, "color": [255, 0, 0]},
        "fade_ms": 0
    }

    Or multiple fixtures:
    {
        "universe": 2,
        "fixture_channels": {
            "par_1": {"intensity": 255, "color": [255, 0, 0]},
            "par_2": {"intensity": 200, "color": [0, 255, 0]}
        },
        "fade_ms": 0
    }
    """
    data = request.get_json()
    universe = data.get('universe', 1)

    # Universe 1 is offline - reject
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400

    fade_ms = data.get('fade_ms', 0)

    # Check for fixture-centric mode
    fixture_id = data.get('fixture_id')
    attributes = data.get('attributes')
    fixture_channels = data.get('fixture_channels')

    pipeline = _get_render_pipeline()

    # Single fixture mode
    if fixture_id and attributes and pipeline.features.FIXTURE_CENTRIC_ENABLED:
        try:
            channels = pipeline.render_fixture_values(fixture_id, attributes, universe)
            if channels:
                # Convert to string keys for content_manager
                str_channels = {str(k): v for k, v in channels.items()}
                return jsonify(_content_manager.set_channels(universe, str_channels, fade_ms))
            else:
                # Fixture not found or not registered
                return jsonify({
                    'error': f'Fixture {fixture_id} not registered in render pipeline',
                    'success': False
                }), 404
        except Exception as e:
            # Fall back to legacy mode on error
            print(f"Fixture-centric render failed, falling back: {e}")

    # Multiple fixtures mode
    if fixture_channels and pipeline.features.FIXTURE_CENTRIC_ENABLED:
        try:
            all_channels = {}
            for fid, attrs in fixture_channels.items():
                channels = pipeline.render_fixture_values(fid, attrs, universe)
                all_channels.update({str(k): v for k, v in channels.items()})
            if all_channels:
                return jsonify(_content_manager.set_channels(universe, all_channels, fade_ms))
        except Exception as e:
            print(f"Multi-fixture render failed, falling back: {e}")

    # Legacy channel-based mode (or fallback)
    return jsonify(_content_manager.set_channels(
        universe, data.get('channels', {}), fade_ms))

@dmx_bp.route('/api/dmx/fade', methods=['POST'])
def dmx_fade():
    """Fade channels over duration - sends UDPJSON fade command"""
    data = request.get_json()
    universe = data.get('universe', 2)  # Default to 2 (not 1)
    # Universe 1 is offline - reject
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400
    channels = data.get('channels', {})
    duration_ms = data.get('duration_ms', 1000)
    return jsonify(_content_manager.set_channels(universe, channels, duration_ms))

@dmx_bp.route('/api/dmx/blackout', methods=['POST'])
def dmx_blackout():
    data = request.get_json() or {}
    universe = data.get('universe')
    # Universe 1 is offline - reject if explicitly requested
    if universe == 1:
        return jsonify({'error': 'Universe 1 is offline. Use universes 2-5.', 'success': False}), 400
    # If no universe specified, blackout all online universes (2-5)
    return jsonify(_content_manager.blackout(universe, data.get('fade_ms', 1000)))


# ============================================================
# SAFETY ACTIONS - Phase 4 Hardening
# ============================================================
# These endpoints MUST work regardless of:
# - UI state
# - Playback state
# - AI layer availability
# - Backend degradation
#
# Each action logs success/failure explicitly.
# ============================================================

@dmx_bp.route('/api/dmx/panic', methods=['POST'])
def dmx_panic():
    """SAFETY ACTION: Immediate blackout with no fade.

    Bypasses all playback/effects and commands immediate zero output.
    Use this when something is wrong and you need lights OFF NOW.

    Request body:
        universe (optional): Target universe. If not specified, panics all universes.
    """
    print("🚨 SAFETY ACTION: /api/dmx/panic called", flush=True)
    data = request.get_json() or {}
    universe = data.get('universe')

    results = {'success': True, 'action': 'panic', 'universes': []}

    # Stop all playback first (bypasses normal paths)
    try:
        _stop_all_playback(blackout=False)
        print("   ✓ All playback stopped", flush=True)
    except Exception as e:
        print(f"   ⚠️ Failed to stop playback: {e}", flush=True)

    # Get all online nodes
    all_nodes = _node_manager.get_all_nodes(include_offline=False)

    if universe is not None:
        # Panic specific universe
        target_nodes = [n for n in all_nodes if n.get('universe') == universe]
        universes_to_panic = [universe]
    else:
        # Panic all universes
        target_nodes = all_nodes
        universes_to_panic = list(set(n.get('universe', 1) for n in all_nodes))

    for univ in universes_to_panic:
        univ_nodes = [n for n in target_nodes if n.get('universe') == univ]
        for node in univ_nodes:
            node_ip = node.get('ip')
            if node_ip:
                try:
                    _ssot.send_udpjson_panic(node_ip, univ)
                    results['universes'].append({'universe': univ, 'node': node_ip, 'success': True})
                except Exception as e:
                    results['universes'].append({'universe': univ, 'node': node_ip, 'success': False, 'error': str(e)})
                    results['success'] = False

    # Also clear SSOT state
    for univ in universes_to_panic:
        try:
            _dmx_state.universes[univ] = [0] * 512
            print(f"   ✓ SSOT cleared for universe {univ}", flush=True)
        except Exception as e:
            print(f"   ⚠️ Failed to clear SSOT for universe {univ}: {e}", flush=True)

    print(f"🚨 PANIC complete: {len(results['universes'])} nodes commanded", flush=True)
    return jsonify(results)


@dmx_bp.route('/api/dmx/master', methods=['POST'])
def dmx_master():
    """Master dimmer - scales all output proportionally

    SSOT COMPLIANCE: Routes through ContentManager.set_channels for unified dispatch.
    """
    data = request.get_json() or {}
    level = data.get('level', 100)
    capture = data.get('capture', False)

    print(f"🎚️ Master dimmer: level={level}%, capture={capture}", flush=True)

    # Capture current state if requested or if we don't have a base yet
    if capture or not _dmx_state.master_base:
        _dmx_state.master_base = {}
        captured_any = False

        for univ, channels in _dmx_state.universes.items():
            if any(v > 0 for v in channels):
                _dmx_state.master_base[univ] = list(channels)
                total_val = sum(channels)
                print(f"   📸 Captured universe {univ}: {total_val} total brightness", flush=True)
                captured_any = True

        if not captured_any:
            print("   ⚠️ No active channels to capture", flush=True)
            return jsonify({'success': False, 'error': 'No active lighting to dim'})

    _dmx_state.master_level = level
    scale = level / 100.0

    all_results = []
    for univ, base in _dmx_state.master_base.items():
        scaled = {}
        for ch_idx, base_val in enumerate(base):
            if base_val > 0:
                scaled[str(ch_idx + 1)] = int(base_val * scale)

        if scaled:
            print(f"   🔧 Scaling universe {univ}: {len(scaled)} channels at {level}%", flush=True)
            # SSOT FIX: Route through ContentManager.set_channels (was direct node send)
            result = _content_manager.set_channels(univ, scaled, fade_ms=0)
            all_results.append({'universe': univ, 'result': result})

    return jsonify({'success': True, 'level': level, 'results': all_results})

@dmx_bp.route('/api/dmx/master/reset', methods=['POST'])
def dmx_master_reset():
    _dmx_state.master_base = {}
    _dmx_state.master_level = 100
    return jsonify({'success': True})


@dmx_bp.route('/api/dmx/universe/<int:universe>', methods=['GET'])
def dmx_get_universe(universe):
    return jsonify({'universe': universe, 'channels': _dmx_state.get_universe(universe)})

@dmx_bp.route('/api/dmx/status', methods=['GET'])
def dmx_status():
    """Get DMX system status with online nodes and universe info"""
    conn = _get_db()
    c = conn.cursor()
    c.execute("""
        SELECT node_id, name, universe, ip, status, channel_start, channel_end, slice_mode, last_seen
        FROM nodes
        WHERE is_paired = 1 AND type = 'wifi'
    """)
    nodes = []
    for row in c.fetchall():
        nodes.append({
            'node_id': row[0],
            'name': row[1],
            'universe': row[2],
            'ip': row[3],
            'status': row[4],
            'slice_start': row[5] or 1,
            'slice_end': row[6] or 512,
            'slice_mode': row[7] or 'zero_outside',
            'last_seen': row[8]
        })
    conn.close()

    # Group by universe
    universes = {}
    for node in nodes:
        u = node['universe']
        if u not in universes:
            universes[u] = []
        universes[u].append(node)

    return jsonify({
        'transport': 'udpjson',
        'port': _AETHER_UDPJSON_PORT,
        'online_nodes': [n for n in nodes if n['status'] == 'online'],
        'all_nodes': nodes,
        'universes': universes,
        'universe_1_note': 'Universe 1 is OFFLINE - use universes 2-5',
        'stats': {
            'total_sends': _node_manager._udpjson_send_count,
            'errors': _node_manager._udpjson_errors,
            'per_universe': _node_manager._udpjson_per_universe
        }
    })

@dmx_bp.route('/api/dmx/diagnostics', methods=['GET'])
def dmx_diagnostics():
    """Diagnostics endpoint for debugging DMX output issues

    SSOT COMPLIANCE: This endpoint shows complete state of the SSOT system,
    including ownership, routing, and any rejected writes.
    """
    arb_status = _arbitration.get_status()

    # Calculate active channels per universe
    universe_stats = {}
    for univ, channels in _dmx_state.universes.items():
        non_zero = sum(1 for v in channels if v > 0)
        total_brightness = sum(channels)
        universe_stats[univ] = {
            'active_channels': non_zero,
            'total_brightness': total_brightness,
            'max_value': max(channels) if channels else 0
        }

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'packet_version': getattr(_NodeManager, 'PACKET_VERSION', 3),
        'packet_version_info': 'v3: full 512-ch frames, ESP32 firmware v1.1 has 2500-byte buffer',

        # Transport diagnostics (UDPJSON only)
        'transport': {
            'udpjson': {
                'port': _AETHER_UDPJSON_PORT,
                'last_send': _node_manager._last_udpjson_send,
                'total_sends': _node_manager._udpjson_send_count,
                'errors': _node_manager._udpjson_errors,
                'per_universe': _node_manager._udpjson_per_universe,
                'output': 'Direct UDP JSON to ESP32 nodes'
            },
            'config_udp': {
                'last_send': _node_manager._last_udp_send,
                'total_sends': _node_manager._udp_send_count,
                'port': _WIFI_COMMAND_PORT
            }
        },

        # SSOT ownership and control
        'ownership': {
            'current_owner': arb_status.get('current_owner'),
            'current_id': arb_status.get('current_id'),
            'priority': _ArbitrationManager.PRIORITY.get(arb_status.get('current_owner'), 0),
            'blackout_active': arb_status.get('blackout_active'),
            'last_change': arb_status.get('last_change'),
            'last_writer': arb_status.get('last_writer'),
            'last_scene_id': arb_status.get('last_scene_id'),
            'last_scene_time': arb_status.get('last_scene_time')
        },

        # Write statistics for detecting spammers
        'writes_per_service': arb_status.get('writes_per_service', {}),

        # Rejected writes (potential conflicts)
        'rejected_writes': arb_status.get('rejected_writes', []),

        # Arbitration history
        'arbitration_history': arb_status.get('history', []),

        # Running engines
        'engines': {
            'effects': _effects_engine.get_status(),
            'chase': {
                'running_chases': list(_chase_engine.running_chases.keys()),
                'health': _chase_engine.chase_health
            },
            'show': {
                'running': _show_engine.running,
                'current_show': _show_engine.current_show.get('name') if _show_engine.current_show else None,
                'paused': _show_engine.paused,
                'tempo': _show_engine.tempo
            },
            'schedule': {
                'running': _schedule_runner.running,
                'schedule_count': len(_schedule_runner.schedules)
            }
        },

        # Playback state per universe
        'playback': _playback_manager.get_status(),

        # SSOT state
        'ssot': {
            'universes_active': list(_dmx_state.universes.keys()),
            'universe_stats': universe_stats,
            'master_level': _dmx_state.master_level,
            'has_master_base': bool(_dmx_state.master_base)
        },

        # System info
        'system': {
            'version': _AETHER_VERSION,
            'commit': _AETHER_COMMIT,
            'uptime_seconds': int((datetime.now() - _AETHER_START_TIME).total_seconds()),
            'file_path': _AETHER_FILE_PATH
        },

        # SSOT compliance summary
        'ssot_compliance': {
            'all_services_routed': True,  # After fixes, all services route through SSOT
            'arbitration_enforced': True,
            'dispatcher_unified': True,
            'notes': [
                'Manual faders: /api/dmx/set -> ContentManager.set_channels',
                'Scenes: /api/scenes/{id}/play -> ContentManager.play_scene -> set_channels',
                'Chases: ChaseEngine._send_step -> ContentManager.set_channels',
                'Effects: DynamicEffectsEngine._send_frame -> ssot_send_frame -> set_channels',
                'Blackout: ContentManager.blackout',
                'Master dimmer: /api/dmx/master -> ContentManager.set_channels',
                'Output: NodeManager UDPJSON to ESP32 nodes on port 6455'
            ]
        }
    })
