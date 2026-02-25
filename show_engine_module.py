"""
ShowEngine Module - Extracted from aether-core.py

Plays back timeline-based shows with timed events.

This module was extracted from aether-core.py to improve modularity while
maintaining the production accuracy of the show playback system.
All global references are accessed through core_registry.
"""

import json
import threading
import time
import logging

import core_registry as reg
from unified_playback import (
    play_look as unified_play_look,
    play_sequence as unified_play_sequence,
    stop as unified_stop,
)


class ShowEngine:
    """Plays back timeline-based shows with timed events.

    # ⚠️ AUTHORITY VIOLATION (TASK-0007) ⚠️
    # This engine MUST NOT own timing loops.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. Timeline event scheduling
    # will be preserved as utilities called BY UnifiedPlaybackEngine.
    #
    # DO NOT START SHOWS INDEPENDENTLY - See TASK_LEDGER.md

    RACE CONDITION FIX: Now integrates with merge layer for proper
    priority-based merging. Direct channel writes route through merge layer.
    """

    def __init__(self):
        self.current_show = None
        self.running = False
        self.thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.paused = False
        self.tempo = 1.0
        # MERGE LAYER: Reference and source tracking
        self._merge_layer = None
        self._merge_source_id = None
        self._last_look_session = None

    def set_merge_layer(self, merge_layer_ref):
        """Set reference to merge layer for priority-based output"""
        self._merge_layer = merge_layer_ref

    def play_show(self, show_id, universe=1):
        """
        Play a show timeline.

        # ⚠️ AUTHORITY VIOLATION WARNING ⚠️
        # This method spawns an independent timeline thread, violating
        # AETHER Hard Rule 1.1. Only UnifiedPlaybackEngine should own timing.
        # See TASK-0007 in TASK_LEDGER.md
        """
        # PHASE 1 GUARD: Log violation when show spawns independent thread
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: ShowEngine.play_show() spawning "
            "independent timing thread. This violates AETHER Hard Rule 1.1 - "
            "Only UnifiedPlaybackEngine should own playback timing. See TASK-0007"
        )

        if not reg.get_db:
            return {'success': False, 'error': 'Database connection not available'}

        conn = reg.get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM shows WHERE show_id = ?', (show_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return {'success': False, 'error': 'Show not found'}

        timeline = json.loads(row[3]) if row[3] else []
        if not timeline:
            return {'success': False, 'error': 'Show has no timeline events'}

        # Stop any current show
        self.stop()

        # MERGE LAYER: Register as a show source for priority handling
        # Shows get sequence priority (45) for timeline-based playback
        if self._merge_layer:
            self._merge_source_id = f"show_{show_id}"
            # Determine all universes that might be affected by this show
            show_universes = [universe]
            if row[6] if len(row) > 6 else False:  # distributed mode
                # In distributed mode, show might affect multiple universes
                show_universes = list(range(1, 65))  # Support all possible universes
            self._merge_layer.register_source(self._merge_source_id, 'sequence', show_universes)
            print(f"📥 Show '{row[1]}' registered as merge source (priority=45)", flush=True)

        self.current_show = {
            'show_id': show_id,
            'name': row[1],
            'timeline': timeline,
            'distributed': row[6] if len(row) > 6 else False,
            'universe': universe
        }
        self.running = True
        self.stop_flag.clear()

        self.thread = threading.Thread(
            target=self._run_timeline,
            args=(timeline, universe, True),
            daemon=True
        )
        self.thread.start()

        print(f"🎬 Playing show '{row[1]}' on universe {universe}")
        return {'success': True, 'show_id': show_id, 'name': row[1]}

    def stop(self):
        """Stop current show"""
        if self.running:
            self.stop_flag.set()
            self.pause_flag.clear()
            self.running = False
            self.paused = False
            if self.current_show:
                print(f"⏹️ Show '{self.current_show['name']}' stopped")
            self.current_show = None
        # MERGE LAYER: Unregister source
        self._unregister_merge_source()

    def stop_silent(self):
        """Stop without blackout (for SSOT transitions)"""
        print(f"🛑 stop_silent called, running={self.running}", flush=True)
        if self.running:
            self.stop_flag.set()
            self.pause_flag.clear()
            self.running = False
            self.paused = False
            self.current_show = None
        # MERGE LAYER: Unregister source
        self._unregister_merge_source()

    def _unregister_merge_source(self):
        """Unregister from merge layer when stopping"""
        if self._merge_layer and self._merge_source_id:
            self._merge_layer.unregister_source(self._merge_source_id)
            print(f"📤 Show unregistered from merge layer: {self._merge_source_id}", flush=True)
            self._merge_source_id = None

    def pause(self):
        """Pause current show"""
        if self.running and not self.paused:
            self.pause_flag.set()
            self.paused = True
            print(f"⏸️ Show paused")

    def resume(self):
        """Resume paused show"""
        if self.running and self.paused:
            self.pause_flag.clear()
            self.paused = False
            print(f"▶️ Show resumed")

    def set_tempo(self, tempo):
        """Set playback tempo (0.25 to 100.0)"""
        self.tempo = max(0.25, min(100.0, tempo))
        print(f"⏩ Tempo set to {self.tempo}x")

    def _run_timeline(self, timeline, universe, loop=True):
        """Execute timeline events in sequence, with optional looping.

        [F15] Uses time.monotonic() to prevent NTP-induced timing jumps.
        Tempo scaling is applied to logical time tracking, not just sleep duration.
        """
        sorted_events = sorted(timeline, key=lambda x: x.get('time_ms', 0))

        while self.running and not self.stop_flag.is_set():
            start_time = time.monotonic()  # [F15] monotonic prevents NTP drift

            for event in sorted_events:
                if self.stop_flag.is_set():
                    break
                # Wait until event time (scaled by tempo)
                event_time_s = event.get('time_ms', 0) / 1000.0  # Convert ms to seconds

                while not self.stop_flag.is_set():
                    # Check pause — paused time doesn't count toward elapsed
                    while self.pause_flag.is_set() and not self.stop_flag.is_set():
                        time.sleep(0.1)
                        start_time += 0.1  # Shift origin so pause doesn't eat timeline
                    if self.stop_flag.is_set():
                        break

                    # [F15] Elapsed logical time = wall time * tempo
                    elapsed_wall = time.monotonic() - start_time
                    elapsed_logical = elapsed_wall * self.tempo
                    remaining = event_time_s - elapsed_logical

                    if remaining <= 0:
                        break  # Event time reached

                    # Sleep in small chunks, converting logical remaining to wall time
                    wall_remaining = remaining / self.tempo
                    sleep_chunk = min(wall_remaining, 0.1)
                    time.sleep(sleep_chunk)
                if self.stop_flag.is_set():
                    break
                # Execute the event
                # Check if distributed mode - extract all scene IDs from timeline
                distributed = self.current_show.get('distributed', False) if self.current_show else False
                all_scenes = None
                event_index = 0
                if distributed:
                    all_scenes = [e.get('scene_id') for e in sorted_events if e.get('type') == 'scene' and e.get('scene_id')]
                    event_index = [i for i, e in enumerate(sorted_events) if e.get('type') == 'scene'].index(
                        sorted_events.index(event)) if event.get('type') == 'scene' else 0
                self._execute_event(event, universe, distributed, all_scenes, event_index)

            if not loop or self.stop_flag.is_set():
                break
            print("🔁 Show looping...")

        self.running = False
        self.current_show = None
        print("🎬 Show playback stopped")

    def _execute_event(self, event, universe, distributed=False, all_scenes=None, event_index=0):
        """Execute a single timeline event.

        RACE CONDITION FIX: Check stop flag before each operation.
        Direct channel writes now route through merge layer for proper priority handling.
        """
        # Check stop flag before executing (race condition fix)
        if self.stop_flag.is_set():
            return

        # Support both old format (type) and new format (action_type)
        event_type = event.get('action_type') or event.get('type', 'scene')

        # Get universes from event if specified, otherwise use default
        event_universes = event.get('universes', [universe])
        if isinstance(event_universes, int):
            event_universes = [event_universes]

        try:
            if event_type == 'scene':
                scene_id = event.get('scene_id') or event.get('action_id')
                fade_ms = event.get('fade_ms', 500)

                if distributed and all_scenes:
                    # Get all online universes
                    if not reg.node_manager:
                        return
                    all_nodes = reg.node_manager.get_all_nodes(include_offline=False)
                    universes = sorted(set(node.get('universe', 1) for node in all_nodes))

                    # Send offset scenes to each universe
                    for i, univ in enumerate(universes):
                        if self.stop_flag.is_set():
                            return
                        offset_index = (event_index + i) % len(all_scenes)
                        offset_scene_id = all_scenes[offset_index]
                        if reg.content_manager:
                            reg.content_manager.play_scene(offset_scene_id, fade_ms=fade_ms, universe=univ, skip_ssot=True)
                    print(f"  🌈 Distributed at {event.get('time_ms')}ms -> {len(universes)} universes")
                else:
                    if self.stop_flag.is_set():
                        return
                    if reg.content_manager:
                        reg.content_manager.play_scene(scene_id, fade_ms=fade_ms, universe=universe, skip_ssot=True)
                    print(f"  ▶️ Scene '{scene_id}' at {event.get('time_ms')}ms")

            elif event_type == 'chase':
                if self.stop_flag.is_set():
                    return
                chase_id = event.get('chase_id') or event.get('action_id')
                if reg.content_manager:
                    reg.content_manager.play_chase(chase_id, universe=universe)
                print(f"  ▶️ Chase '{chase_id}' at {event.get('time_ms')}ms")

            elif event_type == 'sequence':
                if self.stop_flag.is_set():
                    return
                sequence_id = event.get('action_id') or event.get('sequence_id')
                fade_ms = event.get('fade_ms', 0)

                # Load the sequence from database
                try:
                    if not hasattr(reg, 'looks_sequences_manager') or not reg.looks_sequences_manager:
                        print(f"  ❌ Sequence manager not available")
                        return
                    sequence = reg.looks_sequences_manager.get_sequence(sequence_id)
                    if not sequence or not sequence.steps:
                        print(f"  ❌ Sequence '{sequence_id}' not found or empty")
                    else:
                        # Build sequence_data dict for unified playback
                        steps = []
                        for step in sequence.steps:
                            step_data = {
                                'step_id': step.step_id,
                                'name': step.name,
                                'channels': step.channels or {},
                                'modifiers': step.modifiers or [],
                                'fade_ms': step.fade_ms,
                                'hold_ms': step.hold_ms,
                            }
                            if step.look_id:
                                step_data['look_id'] = step.look_id
                            steps.append(step_data)

                        sequence_data = {
                            'name': sequence.name,
                            'steps': steps,
                            'loop_mode': 'one_shot',  # Shows control timing, not sequence
                            'bpm': sequence.bpm,
                        }

                        # Play on specified universes
                        unified_play_sequence(
                            sequence_id,
                            sequence_data,
                            universes=event_universes
                        )
                        print(f"  ▶️ Sequence '{sequence.name}' at {event.get('time_ms')}ms on universes {event_universes}")
                except Exception as seq_err:
                    print(f"  ❌ Sequence play error: {seq_err}")

            elif event_type == 'look':
                if self.stop_flag.is_set():
                    return
                # Stop previous look session from this show
                if hasattr(self, '_last_look_session') and self._last_look_session:
                    unified_stop(self._last_look_session)
                look_id = event.get('look_id') or event.get('action_id')
                fade_ms = event.get('fade_ms', 500)
                try:
                    if not hasattr(reg, 'looks_sequences_manager') or not reg.looks_sequences_manager:
                        print(f"  ❌ Looks manager not available")
                        return
                    look = reg.looks_sequences_manager.get_look(look_id)
                    if not look:
                        print(f"  ❌ Look '{look_id}' not found")
                    else:
                        look_data = look.to_dict()
                        # Extract universes from channel keys (e.g. "4:1")
                        lu = set()
                        for k in look.channels:
                            if ':' in str(k):
                                lu.add(int(str(k).split(':')[0]))
                        tgt = sorted(lu) if lu else event_universes
                        sid = unified_play_look(
                            look_id,
                            look_data,
                            universes=tgt,
                            fade_ms=fade_ms
                        )
                        self._last_look_session = sid
                        print(f"  ▶️ Look '{look.name}' at {event.get('time_ms')}ms on universes {tgt} (lu={lu})")
                except Exception as look_err:
                    print(f"  ❌ Look play error: {look_err}")

            elif event_type == 'blackout':
                if self.stop_flag.is_set():
                    return
                fade_ms = event.get('fade_ms', 1000)
                if reg.content_manager:
                    reg.content_manager.blackout(universe=universe, fade_ms=fade_ms)
                print(f"  ⬛ Blackout at {event.get('time_ms')}ms")

            elif event_type == 'channels':
                if self.stop_flag.is_set():
                    return
                channels = event.get('channels', {})
                fade_ms = event.get('fade_ms', 0)

                # MERGE LAYER: Route direct channel writes through merge layer
                if self._merge_layer and self._merge_source_id:
                    # Convert channels to int keys for merge layer
                    parsed_channels = {int(k): int(v) for k, v in channels.items()}
                    self._merge_layer.set_source_channels(self._merge_source_id, universe, parsed_channels)
                    # Compute merged output
                    merged = self._merge_layer.compute_merge(universe)
                    if merged and reg.content_manager:
                        reg.content_manager.set_channels(universe, {str(k): v for k, v in merged.items()}, fade_ms)
                else:
                    # Fallback: direct write
                    if reg.content_manager:
                        reg.content_manager.set_channels(universe, channels, fade_ms)
                print(f"  🎛️ Channels at {event.get('time_ms')}ms")

        except Exception as e:
            print(f"  ❌ Event error: {e}")
