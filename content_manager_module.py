"""
ContentManager Module - Extracted from aether-core.py

This module provides scene, chase, and fixture management functionality.
It was extracted to support modular architecture while maintaining
cross-module references through core_registry.

The ContentManager is responsible for:
- Scene creation, playback, and deletion
- Chase creation, playback, and deletion
- Fixture definition and management
- Unified playback integration via unified_playback module
"""

import json
import threading
import time
from datetime import datetime

import core_registry as reg
from unified_playback import SessionFactory, PlaybackType


# Constants
CHUNK_DELAY = 0.05
SUPABASE_AVAILABLE = False  # This should be set from parent module


def get_supabase_service():
    """Placeholder for optional Supabase service"""
    return None


def cloud_submit(func):
    """Placeholder for cloud submission decorator"""
    def decorator(f):
        return f
    return decorator


class ContentManager:
    """Manages scenes, chases, and fixtures with SSOT lock for thread-safe playback transitions"""

    def __init__(self):
        """Initialize ContentManager with SSOT lock for thread-safe playback transitions"""
        self.ssot_lock = threading.Lock()
        self.current_playback = {"type": None, "id": None, "universe": None}
        print("✓ ContentManager initialized with SSOT lock")

    def set_channels(self, universe, channels, fade_ms=0):
        """Set DMX channels - builds full 512-channel frame and sends via UDPJSON

        SSOT COMPLIANCE:
        1. Updates dmx_state with the requested channel changes
        2. Builds full 512-channel frame from SSOT
        3. Sends to each node via UDPJSON (filtered by node's slice)

        All nodes receive complete universe data for their slice.
        Missing channels default to their SSOT value (or 0 if never set).
        """
        # Update SSOT with the channel changes
        reg.dmx_state.set_channels(universe, channels, fade_ms=fade_ms)

        # Build full 512-channel frame from SSOT
        full_frame = reg.dmx_state.get_output_values(universe)

        nodes = reg.node_manager.get_nodes_in_universe(universe)

        if not nodes:
            return {'success': True, 'results': []}

        # Send full frame to each node via UDPJSON
        results = []
        for node in nodes:
            # Route through Seance if node is connected via Seance bridge
            node_ip = node.get('seance_ip') if node.get('via_seance') else node.get('ip')
            if not node_ip or node_ip == 'localhost':
                continue

            # Get node's channel slice - use fixture-aware calculation if available
            # Priority: 1) Explicit channel_start/end, 2) Calculate from fixtures, 3) Default 1-4 for RGBW
            slice_start = node.get('channel_start') or 1
            slice_end = node.get('channel_end')

            # If no explicit slice_end, default to 4 channels (single RGBW fixture)
            # This is a safe default since most Pulse nodes have 1 fixture
            if slice_end is None:
                slice_end = 4  # Default: single RGBW fixture

            # Build channels dict for this node's slice from the full frame
            node_channels = {}
            for ch in range(slice_start, slice_end + 1):
                value = full_frame[ch - 1] if ch <= 512 else 0
                node_channels[str(ch)] = value

            node_non_zero = sum(1 for v in node_channels.values() if v > 0)

            if fade_ms > 0:
                success = reg.node_manager.send_udpjson_fade(node_ip, universe, node_channels, fade_ms)
            else:
                success = reg.node_manager.send_udpjson_set(node_ip, universe, node_channels)

            results.append({'node': node['name'], 'success': success, 'channels': len(node_channels)})

        return {'success': True, 'results': results}

    def blackout(self, universe=None, fade_ms=1000):
        """Blackout all channels - if universe is None, blackout ALL universes"""
        # ARBITRATION: Blackout is highest priority - stop everything first
        reg.arbitration.set_blackout(True)
        reg.effects_engine.stop_effect()  # Stop all dynamic effects
        reg.chase_engine.stop_all()  # Stop all chases

        all_nodes = reg.node_manager.get_all_nodes(include_offline=False)
        all_universes_online = list(set(node.get('universe', 1) for node in all_nodes))
        playback_before = dict(self.current_playback) if hasattr(self, 'current_playback') else {}

        if universe is not None:
            universes_to_blackout = [universe]
        else:
            universes_to_blackout = all_universes_online

        reg.beta_log("blackout", {
            "requested_universe": universe,
            "selected_universes_at_action_time": sorted(all_universes_online),
            "expanded_target_universes": sorted(universes_to_blackout),
            "dispatch_targets_final": sorted(universes_to_blackout),
            "playback_state_before": playback_before,
            "fade_ms": fade_ms
        })

        print(f"⬛ Blackout on universes: {sorted(universes_to_blackout)}", flush=True)

        results = []
        for univ in universes_to_blackout:
            reg.dmx_state.blackout(univ, fade_ms=fade_ms)
            reg.playback_manager.stop(univ)
            nodes = reg.node_manager.get_nodes_in_universe(univ)
            for node in nodes:
                node_ip = node.get('ip')
                if node_ip and node_ip != 'localhost':
                    success = reg.node_manager.send_udpjson_blackout(node_ip, univ, fade_ms=fade_ms)
                    results.append({'node': node['name'], 'success': success})

        if hasattr(self, 'current_playback'):
            self.current_playback = {"type": None, "id": None, "universe": None}

        reg.beta_log("blackout_complete", {
            "dispatch_targets_final": sorted(universes_to_blackout),
            "playback_state_after": {"type": None, "id": None, "universe": None}
        })

        # Release blackout after sending zeros (allow future commands)
        reg.arbitration.set_blackout(False)

        return {'success': True, 'results': results}


    # ─────────────────────────────────────────────────────────
    # Scenes
    # ─────────────────────────────────────────────────────────
    def create_scene(self, data):
        """Create/update scene and sync to nodes"""
        scene_id = data.get('scene_id', f"scene_{int(time.time())}")
        universe = data.get('universe', 1)
        channels = data.get('channels', {})

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO scenes (scene_id, name, description, universe, channels,
            fade_ms, curve, color, icon, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (scene_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, json.dumps(channels), data.get('fade_ms', 500), data.get('curve', 'linear'),
             data.get('color', '#3b82f6'), data.get('icon', 'lightbulb'), False, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Sync to all nodes in this universe
        scene = self.get_scene(scene_id)
        if scene:
            nodes = reg.node_manager.get_wifi_nodes_in_universe(universe)
            for node in nodes:
                reg.node_manager.sync_scene_to_node(node, scene)
                time.sleep(CHUNK_DELAY)

            # Mark as synced
            conn = reg.get_db()
            c = conn.cursor()
            c.execute('UPDATE scenes SET synced_to_nodes = 1 WHERE scene_id = ?', (scene_id,))
            conn.commit()
            conn.close()

        if reg.socketio:
            reg.socketio.emit('scenes_update', {'scenes': self.get_scenes()})

        # Async sync to Supabase (non-blocking)
        if SUPABASE_AVAILABLE:
            supabase = get_supabase_service()
            if supabase and supabase.is_enabled():
                cloud_submit(supabase.sync_scene)(scene)

        return {'success': True, 'scene_id': scene_id}

    def get_scenes(self):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        scenes = []
        for row in rows:
            scene = dict(row)
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            scenes.append(scene)
        return scenes

    def get_scene(self, scene_id):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM scenes WHERE scene_id = ?', (scene_id,))
        row = c.fetchone()
        conn.close()
        if row:
            scene = dict(row)
            scene['channels'] = json.loads(scene['channels']) if scene['channels'] else {}
            return scene
        return None

    def replicate_scene_to_fixtures(self, channels, fixture_size=4, max_fixtures=128, distribution_mode='unified', universe=None):
        """Replicate a scene pattern across all fixtures in a universe.

        If scene has channels 1-4, replicate to 5-8, 9-12, etc.
        distribution_mode: 'unified' = replicate same pattern, 'pixel' = unique per fixture
        """
        if not channels:
            return channels

        # PIXEL MODE: distribute unique values to each fixture
        if distribution_mode == 'pixel' and universe is not None:
            fixtures = self.get_fixtures(universe)
            if fixtures:
                fixtures = sorted(fixtures, key=lambda f: f.get('start_channel', 1))
                pattern_vals = list(channels.values())
                distributed = {}
                for idx, fix in enumerate(fixtures):
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', 1)
                    val = pattern_vals[idx % len(pattern_vals)] if pattern_vals else 0
                    for ch in range(count):
                        distributed[str(start + ch)] = val
                return distributed

        # Find the base pattern - channels that define one fixture
        ch_nums = sorted(int(k) for k in channels.keys())
        if not ch_nums:
            return channels

        # Detect fixture size from the scene (could be 3 for RGB, 4 for RGBW, etc.)
        min_ch = min(ch_nums)
        max_ch = max(ch_nums)
        pattern_size = max_ch - min_ch + 1

        # If pattern is larger than typical fixture, don't replicate
        if pattern_size > 8:
            return channels

        # Use the larger of detected pattern or standard fixture size
        fixture_size = max(fixture_size, pattern_size)

        # Build the base pattern (normalized to start at channel 1)
        base_pattern = {}
        for ch_str, value in channels.items():
            ch = int(ch_str)
            offset = (ch - 1) % fixture_size  # 0-indexed offset within fixture
            base_pattern[offset] = value

        # Replicate across all fixtures
        replicated = {}
        for fixture_num in range(max_fixtures):
            fixture_start = fixture_num * fixture_size + 1
            if fixture_start > 512:
                break
            for offset, value in base_pattern.items():
                ch = fixture_start + offset
                if ch <= 512:
                    replicated[str(ch)] = value

        print(f"🔄 Replicated {len(channels)} channels -> {len(replicated)} channels (pattern size: {fixture_size})")
        return replicated

    def play_scene(self, scene_id, fade_ms=None, use_local=True, target_channels=None, universe=None, universes=None, skip_ssot=False, replicate=True):
        """Play a scene via unified engine - supports modifier stacking

        Args:
            scene_id: ID of the scene to play
            fade_ms: Fade time override
            use_local: Use local playback
            target_channels: Optional list of specific channels
            universe: Single universe (legacy, use universes instead)
            universes: List of universes to target (preferred)
            skip_ssot: Skip SSOT lock (internal use)
            replicate: Replicate scene across fixtures
        """
        print(f"▶️ play_scene called: scene_id={scene_id}", flush=True)
        scene = self.get_scene(scene_id)
        if not scene:
            return {'success': False, 'error': 'Scene not found'}

        # ARBITRATION: Acquire scene ownership
        if not reg.arbitration.acquire('scene', scene_id):
            print(f"⚠️ Cannot play scene - arbitration denied (owner: {reg.arbitration.current_owner})", flush=True)
            return {'success': False, 'error': f'Arbitration denied: {reg.arbitration.current_owner} has control'}

        # Get target universes - priority: universes array > single universe > all online paired nodes
        all_nodes = reg.node_manager.get_all_nodes(include_offline=False)
        if universes is not None and len(universes) > 0:
            universes_with_nodes = list(universes)
        elif universe is not None:
            universes_with_nodes = [universe]
        else:
            # Default: all online PAIRED universes only
            universes_with_nodes = list(set(node.get('universe', 1) for node in all_nodes if node.get('is_paired')))
            if not universes_with_nodes:
                universes_with_nodes = [1]

        fade = fade_ms if fade_ms is not None else scene.get('fade_ms', 500)
        channels_to_apply = scene['channels']

        # Replicate scene pattern across all fixtures (unless targeting specific channels)
        if replicate and not target_channels:
            channels_to_apply = self.replicate_scene_to_fixtures(channels_to_apply)

        if target_channels:
            target_set = set(target_channels)
            channels_to_apply = {k: v for k, v in channels_to_apply.items() if int(k) in target_set}

        print(f"🎬 Playing scene '{scene['name']}' on universes: {sorted(universes_with_nodes)}, fade={fade}ms", flush=True)

        # SSOT: Acquire lock and stop conflicting playback
        if not skip_ssot:
            with self.ssot_lock:
                print(f"🔒 SSOT Lock - stopping conflicting playback", flush=True)
                try:
                    if reg.show_engine and hasattr(reg.show_engine, "stop_silent"):
                        reg.show_engine.stop_silent()
                    elif reg.show_engine:
                        reg.show_engine.stop()
                except Exception as e:
                    print(f"⚠️ Show stop error: {e}", flush=True)
                reg.chase_engine.stop_all()
                reg.unified_engine.stop_type(PlaybackType.SCENE)
                reg.unified_engine.stop_type(PlaybackType.CHASE)
                reg.effects_engine.stop_effect()
                self.current_playback = {'type': 'scene', 'id': scene_id, 'universe': universe}

        # Set playback state for all universes
        for univ in universes_with_nodes:
            reg.playback_manager.set_playing(univ, 'scene', scene_id)

        # Create scene data with expanded channels for unified engine
        scene_data = {
            'name': scene.get('name', f'Scene {scene_id}'),
            'channels': {int(k): v for k, v in channels_to_apply.items()},
            'fade_ms': fade
        }

        # Create unified engine session from scene data
        session = SessionFactory.from_scene(scene_id, scene_data, universes_with_nodes, fade)

        # Get current DMX state for fade-from values
        fade_from = None
        if fade > 0 and universes_with_nodes:
            current_universe_values = reg.dmx_state.universes.get(universes_with_nodes[0], [0] * 512)
            fade_from = {i+1: v for i, v in enumerate(current_universe_values) if v > 0}

        # Start via unified engine - this allows modifier stacking!
        reg.unified_engine.play(session, fade_from)
        print(f"▶️ Scene '{scene['name']}' started via unified engine: {session.session_id}", flush=True)

        # Update play count
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('UPDATE scenes SET play_count = play_count + 1 WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()

        reg.dmx_state.save_state_now()  # [F09] Persist immediately on playback start
        return {'success': True, 'universes': universes_with_nodes, 'fade_ms': fade, 'session_id': session.session_id}

    def delete_scene(self, scene_id):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('DELETE FROM scenes WHERE scene_id = ?', (scene_id,))
        conn.commit()
        conn.close()
        if reg.socketio:
            reg.socketio.emit('scenes_update', {'scenes': self.get_scenes()})
        return {'success': True}

    # ─────────────────────────────────────────────────────────
    # Chases
    # ─────────────────────────────────────────────────────────
    def create_chase(self, data):
        """Create/update chase and sync to nodes"""
        chase_id = data.get('chase_id', f"chase_{int(time.time())}")
        universe = data.get('universe', 1)
        steps = data.get('steps', [])

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO chases (chase_id, name, description, universe, bpm, loop,
            steps, color, fade_ms, distribution_mode, synced_to_nodes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (chase_id, data.get('name', 'Untitled'), data.get('description', ''),
             universe, data.get('bpm', 120), data.get('loop', True),
             json.dumps(steps), data.get('color', '#10b981'), data.get('fade_ms', 0), data.get('distribution_mode', 'unified'), False, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Sync to nodes (non-blocking)
        try:
            chase = self.get_chase(chase_id)
            if chase:
                nodes = reg.node_manager.get_wifi_nodes_in_universe(universe)
                for node in nodes:
                    reg.node_manager.sync_chase_to_node(node, chase)
                    time.sleep(CHUNK_DELAY)
                conn = reg.get_db()
                c = conn.cursor()
                c.execute('UPDATE chases SET synced_to_nodes = 1 WHERE chase_id = ?', (chase_id,))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Chase sync error: {e}", flush=True)

        if reg.socketio:
            reg.socketio.emit('chases_update', {'chases': self.get_chases()})

        # Async sync to Supabase
        try:
            if SUPABASE_AVAILABLE:
                supabase = get_supabase_service()
                if supabase and supabase.is_enabled():
                    cloud_submit(supabase.sync_chase)(chase)
        except Exception as e:
            print(f"Supabase error: {e}", flush=True)

        return {'success': True, 'chase_id': chase_id}

    def get_chases(self):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM chases ORDER BY updated_at DESC')
        rows = c.fetchall()
        conn.close()
        chases = []
        for row in rows:
            chase = dict(row)
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            chases.append(chase)
        return chases

    def get_chase(self, chase_id):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM chases WHERE chase_id = ?', (chase_id,))
        row = c.fetchone()
        conn.close()
        if row:
            chase = dict(row)
            chase['steps'] = json.loads(chase['steps']) if chase['steps'] else []
            return chase
        return None

    def play_chase(self, chase_id, target_channels=None, universe=None, universes=None, fade_ms=None):
        """Start chase playback via unified engine - supports modifier stacking

        Args:
            chase_id: ID of the chase to play
            target_channels: Optional list of specific channels to target
            universe: Single universe (legacy, use universes instead)
            universes: List of universes to target (preferred)
            fade_ms: Fade time override
        """
        chase = self.get_chase(chase_id)
        if not chase:
            return {'success': False, 'error': 'Chase not found'}

        # Apply-time fade override: fade_ms param > chase default > 0
        effective_fade_ms = fade_ms if fade_ms is not None else chase.get('fade_ms', 0)
        print(f"🎚️ Chase fade: requested={fade_ms}, chase_default={chase.get('fade_ms')}, effective={effective_fade_ms}", flush=True)

        # Get target universes - priority: universes array > single universe > all online paired nodes
        if universes is not None and len(universes) > 0:
            universes_with_nodes = list(universes)
        elif universe is not None:
            universes_with_nodes = [universe]
        else:
            # Default: all online PAIRED universes only
            all_nodes = reg.node_manager.get_all_nodes(include_offline=False)
            universes_with_nodes = list(set(node.get('universe', 1) for node in all_nodes if node.get('is_paired')))
        print(f"🎬 Playing chase '{chase['name']}' on universes: {sorted(universes_with_nodes)}, fade={effective_fade_ms}ms", flush=True)

        # SSOT: Acquire lock and stop only conflicting playback on target universes
        with self.ssot_lock:
            print(f"🔒 SSOT Lock - stopping playback on target universes: {universes_with_nodes}", flush=True)
            try:
                if reg.show_engine:
                    reg.show_engine.stop()
            except Exception as e:
                print(f"⚠️ Show stop: {e}", flush=True)
            # Stop old chase engine (legacy)
            reg.chase_engine.stop_all()
            # Stop existing chase sessions in unified engine for these universes
            reg.unified_engine.stop_type(PlaybackType.CHASE)
            reg.effects_engine.stop_effect()
            for univ in universes_with_nodes:
                reg.playback_manager.stop(univ)
            self.current_playback = {'type': 'chase', 'id': chase_id, 'universe': universe}
            print(f"✓ SSOT: Now playing chase '{chase_id}'", flush=True)

        # Set playback state for all universes
        for univ in universes_with_nodes:
            reg.playback_manager.set_playing(univ, 'chase', chase_id)

        # Create unified engine session from chase data
        session = SessionFactory.from_chase(chase_id, chase, universes_with_nodes)

        # Replicate chase step channels across all fixtures in the universe
        # Without this, chase steps only affect one fixture (scenes do this in play_scene)
        if universes_with_nodes:
            fixtures_in_universe = self.get_fixtures(universes_with_nodes[0])
            if fixtures_in_universe and len(fixtures_in_universe) > 1:
                sorted_fx = sorted(fixtures_in_universe, key=lambda f: f.get('start_channel', 1))
                fixture_ch_count = sorted_fx[0].get('channel_count', 4)
                for step in session.steps:
                    # Extract base pattern using modulo (ch 1→offset 0, ch 6→offset 1, etc.)
                    base_pattern = {}
                    for ch, val in step.channels.items():
                        offset = (ch - 1) % fixture_ch_count
                        if val > 0 or offset not in base_pattern:
                            base_pattern[offset] = val
                    # Expand to all fixtures
                    expanded = {}
                    for fix in sorted_fx:
                        start = fix.get('start_channel', 1)
                        count = fix.get('channel_count', fixture_ch_count)
                        for offset, value in base_pattern.items():
                            if offset < count:
                                expanded[start + offset] = value
                    step.channels = expanded
                print(f"🔄 Replicated chase steps across {len(sorted_fx)} fixtures (ch_count={fixture_ch_count})", flush=True)

        # Apply fade override if specified
        if effective_fade_ms > 0:
            for step in session.steps:
                step.fade_ms = effective_fade_ms

        # Start via unified engine - this allows modifier stacking!
        reg.unified_engine.play(session)
        print(f"▶️ Chase '{chase['name']}' started via unified engine: {session.session_id}", flush=True)

        reg.dmx_state.save_state_now()  # [F09] Persist immediately on playback start
        return {'success': True, 'universes': universes_with_nodes, 'fade_ms': effective_fade_ms, 'session_id': session.session_id}

    def stop_playback(self, universe=None):
        """Stop all playback"""
        reg.chase_engine.stop_all()  # Stop chase engine
        reg.playback_manager.stop(universe)
        node_results = reg.node_manager.stop_playback_on_nodes(universe)
        reg.dmx_state.save_state_now()  # [F09] Persist immediately on playback stop
        return {'success': True, 'results': node_results}

    def delete_chase(self, chase_id):
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('DELETE FROM chases WHERE chase_id = ?', (chase_id,))
        conn.commit()
        conn.close()
        if reg.socketio:
            reg.socketio.emit('chases_update', {'chases': self.get_chases()})
        return {'success': True}

    # ─────────────────────────────────────────────────────────
    # Fixtures
    # ─────────────────────────────────────────────────────────
    def create_fixture(self, data):
        """Create or update a fixture definition"""
        fixture_id = data.get('fixture_id', f"fixture_{int(time.time())}")

        # Default channel map based on type
        default_map = self._get_default_channel_map(data.get('type', 'generic'), data.get('channel_count', 1))
        channel_map = data.get('channel_map', default_map)

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO fixtures (fixture_id, name, type, manufacturer, model,
            universe, start_channel, channel_count, channel_map, color, notes, rdm_uid, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (fixture_id, data.get('name', 'Untitled Fixture'), data.get('type', 'generic'),
             data.get('manufacturer', ''), data.get('model', ''),
             data.get('universe', 1), data.get('start_channel', 1), data.get('channel_count', 1),
             json.dumps(channel_map), data.get('color', '#8b5cf6'),
             data.get('notes', ''), data.get('rdm_uid'), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        if reg.socketio:
            reg.socketio.emit('fixtures_update', {'fixtures': self.get_fixtures()})
        return {'success': True, 'fixture_id': fixture_id}

    def _get_default_channel_map(self, fixture_type, channel_count):
        """Generate default channel names based on fixture type"""
        maps = {
            'rgb': ['Red', 'Green', 'Blue'],
            'rgbw': ['Red', 'Green', 'Blue', 'White'],
            'rgba': ['Red', 'Green', 'Blue', 'Amber'],
            'rgbwa': ['Red', 'Green', 'Blue', 'White', 'Amber'],
            'dimmer': ['Intensity'],
            'moving_head': ['Pan', 'Pan Fine', 'Tilt', 'Tilt Fine', 'Speed', 'Dimmer', 'Strobe', 'Color', 'Gobo', 'Prism'],
            'par': ['Red', 'Green', 'Blue', 'White', 'Dimmer', 'Strobe'],
            'wash': ['Red', 'Green', 'Blue', 'White', 'Dimmer', 'Pan', 'Tilt'],
        }
        default = maps.get(fixture_type.lower(), [])
        # Pad with generic channel names if needed
        while len(default) < channel_count:
            default.append(f'Channel {len(default) + 1}')
        return default[:channel_count]

    def get_fixtures(self, universe=None):
        """Get all fixtures, optionally filtered by universe"""
        conn = reg.get_db()
        c = conn.cursor()
        if universe:
            c.execute('SELECT * FROM fixtures WHERE universe = ? ORDER BY start_channel', (universe,))
        else:
            c.execute('SELECT * FROM fixtures ORDER BY universe, start_channel')
        rows = c.fetchall()
        conn.close()
        fixtures = []
        for row in rows:
            fixture = dict(row)
            fixture['channel_map'] = json.loads(fixture['channel_map']) if fixture['channel_map'] else []
            # Calculate end channel
            fixture['end_channel'] = fixture['start_channel'] + fixture['channel_count'] - 1
            fixtures.append(fixture)
        return fixtures

    def get_fixture(self, fixture_id):
        """Get single fixture by ID"""
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM fixtures WHERE fixture_id = ?', (fixture_id,))
        row = c.fetchone()
        conn.close()
        if row:
            fixture = dict(row)
            fixture['channel_map'] = json.loads(fixture['channel_map']) if fixture['channel_map'] else []
            fixture['end_channel'] = fixture['start_channel'] + fixture['channel_count'] - 1
            return fixture
        return None

    def update_fixture(self, fixture_id, data):
        """Update an existing fixture"""
        existing = self.get_fixture(fixture_id)
        if not existing:
            return {'success': False, 'error': 'Fixture not found'}

        # Merge with existing data
        merged = {**existing, **data}
        merged['fixture_id'] = fixture_id
        return self.create_fixture(merged)

    def delete_fixture(self, fixture_id):
        """Delete a fixture"""
        conn = reg.get_db()
        c = conn.cursor()
        c.execute('DELETE FROM fixtures WHERE fixture_id = ?', (fixture_id,))
        conn.commit()
        conn.close()
        if reg.socketio:
            reg.socketio.emit('fixtures_update', {'fixtures': self.get_fixtures()})
        return {'success': True}

    def get_fixtures_for_channels(self, universe, channels):
        """Find which fixtures cover the given channels"""
        fixtures = self.get_fixtures(universe)
        affected = []
        channel_nums = [int(c) for c in channels.keys()]

        for fixture in fixtures:
            start = fixture['start_channel']
            end = fixture['end_channel']
            for ch in channel_nums:
                if start <= ch <= end:
                    affected.append(fixture)
                    break
        return affected
