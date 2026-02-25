"""
ChaseEngine Module - Extracted from aether-core.py

Runs chases by streaming each step via UDPJSON to all universes.

This module was extracted from aether-core.py to improve modularity while
maintaining the production accuracy of the chase playback system.
All global references are accessed through core_registry.
"""

import threading
import time
import logging

import core_registry as reg


class ChaseEngine:
    """Runs chases by streaming each step via UDPJSON to all universes.

    # ⚠️ AUTHORITY VIOLATION (TASK-0006) ⚠️
    # This engine MUST NOT own timing loops.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. Chase step computation
    # will be preserved as utilities called BY UnifiedPlaybackEngine.
    #
    # DO NOT START CHASES INDEPENDENTLY - See TASK_LEDGER.md

    RACE CONDITION FIX: Now routes all output through merge layer for proper
    priority-based merging with other playback sources (effects, scenes, etc.)
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.running_chases = {}  # {chase_id: thread}
        self.stop_flags = {}  # {chase_id: Event}
        # Health tracking for debugging
        self.chase_health = {}  # {chase_id: {"step": int, "last_time": float, "status": str}}
        # Merge layer source tracking - maps chase_id to source_id
        self._merge_sources = {}  # {chase_id: source_id}
        # Reference to merge layer (set after merge_layer is created)
        self._merge_layer = None

    def set_merge_layer(self, merge_layer_ref):
        """Set reference to merge layer for priority-based output"""
        self._merge_layer = merge_layer_ref

    def start_chase(self, chase, universes, fade_ms_override=None):
        """
        Start a chase on the given universes with optional fade override.

        # ⚠️ AUTHORITY VIOLATION WARNING ⚠️
        # This method spawns an independent chase thread, violating
        # AETHER Hard Rule 1.1. Only UnifiedPlaybackEngine should own timing.
        # See TASK-0006 in TASK_LEDGER.md
        """
        # PHASE 1 GUARD: Log violation when chase spawns independent thread
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: ChaseEngine.start_chase() spawning "
            "independent timing thread. This violates AETHER Hard Rule 1.1 - "
            "Only UnifiedPlaybackEngine should own playback timing. See TASK-0006"
        )

        chase_id = chase['chase_id']

        # ARBITRATION: Acquire chase ownership — returns token for TOCTOU safety [F08]
        if not reg.arbitration:
            print(f"⚠️ Arbitration not available", flush=True)
            return False

        arb_token = reg.arbitration.acquire('chase', chase_id)
        if not arb_token:
            print(f"⚠️ Cannot start chase - arbitration denied (owner: {reg.arbitration.current_owner})", flush=True)
            return False

        # Stop any other running chases first
        self.stop_chase(chase_id)

        # MERGE LAYER: Register as a merge source for proper priority handling
        if self._merge_layer:
            source_id = f"chase_{chase_id}"
            self._merge_layer.register_source(source_id, 'chase', universes)
            with self.lock:
                self._merge_sources[chase_id] = source_id
            print(f"📥 Chase '{chase['name']}' registered as merge source (priority=40)", flush=True)

        # Create stop flag
        stop_flag = threading.Event()
        self.stop_flags[chase_id] = stop_flag

        # Start chase thread with fade override and arbitration token [F08]
        thread = threading.Thread(
            target=self._run_chase,
            args=(chase, universes, stop_flag, fade_ms_override, arb_token),
            daemon=True
        )
        self.running_chases[chase_id] = thread
        thread.start()
        print(f"🏃 Chase engine started: {chase['name']} (fade_override={fade_ms_override})", flush=True)
        return True

    def stop_chase(self, chase_id=None, wait=True):
        """Stop a chase or all chases, optionally waiting for thread to finish"""
        threads_to_join = []
        sources_to_unregister = []
        with self.lock:
            if chase_id:
                if chase_id in self.stop_flags:
                    self.stop_flags[chase_id].set()
                    self.stop_flags.pop(chase_id, None)
                    thread = self.running_chases.pop(chase_id, None)
                    if thread and wait:
                        threads_to_join.append(thread)
                # Track merge source to unregister
                source_id = self._merge_sources.pop(chase_id, None)
                if source_id:
                    sources_to_unregister.append(source_id)
            else:
                # Stop all
                for flag in self.stop_flags.values():
                    flag.set()
                if wait:
                    threads_to_join = list(self.running_chases.values())
                self.stop_flags.clear()
                self.running_chases.clear()
                # Track all merge sources to unregister
                sources_to_unregister = list(self._merge_sources.values())
                self._merge_sources.clear()

            # ARBITRATION: Release chase ownership if no more chases running
            if not self.running_chases and reg.arbitration:
                reg.arbitration.release('chase')

        # Wait for threads outside of lock to avoid deadlock
        if wait:
            for thread in threads_to_join:
                thread.join(timeout=0.5)  # Max 500ms wait per thread

        # MERGE LAYER: Unregister sources after threads have stopped
        if self._merge_layer and sources_to_unregister:
            for source_id in sources_to_unregister:
                self._merge_layer.unregister_source(source_id)
                print(f"📤 Chase unregistered from merge layer: {source_id}", flush=True)

    def stop_all(self):
        """Stop all running chases"""
        self.stop_chase(None)

    def _run_chase(self, chase, universes, stop_flag, fade_ms_override=None, arb_token=None):
        """Chase playback loop - runs in background thread"""
        chase_id = chase['chase_id']
        steps = chase.get('steps', [])
        bpm = chase.get('bpm', 120)
        loop = chase.get('loop', True)
        distribution_mode = chase.get('distribution_mode', 'unified')
        # Apply-time fade override > chase default > 0
        chase_fade_ms = fade_ms_override if fade_ms_override is not None else chase.get('fade_ms', 0)

        if not steps:
            print(f"⚠️ Chase {chase['name']} has no steps", flush=True)
            self.chase_health[chase_id] = {"step": -1, "last_time": time.time(), "status": "no_steps"}
            return

        # Default step interval from BPM (used if step doesn't have duration)
        default_interval = 60.0 / bpm
        step_index = 0
        loop_count = 0

        print(f"🎬 Chase '{chase['name']}': {len(steps)} steps, fade={chase_fade_ms}ms, universes={universes}", flush=True)
        self.chase_health[chase_id] = {"step": 0, "last_time": time.time(), "status": "running", "loop": 0}

        try:
            while not stop_flag.is_set():
                step = steps[step_index]
                channels = step.get('channels', {})
                # Use step fade or chase fade
                fade_ms = step.get('fade_ms', chase_fade_ms)
                # Calculate step duration: support both 'duration' and 'hold_ms' formats
                # hold_ms = time to hold AFTER fade completes
                # duration = total step time (legacy format)
                if 'hold_ms' in step:
                    # New format: fade_ms + hold_ms = total step time
                    step_duration_ms = fade_ms + step['hold_ms']
                elif 'duration' in step:
                    # Legacy format: duration is total step time
                    step_duration_ms = step['duration']
                else:
                    # Fallback to BPM timing
                    step_duration_ms = int(default_interval * 1000)

                # Update health heartbeat
                self.chase_health[chase_id] = {
                    "step": step_index,
                    "last_time": time.time(),
                    "status": "running",
                    "loop": loop_count,
                    "duration_ms": step_duration_ms,
                    "fade_ms": fade_ms,
                    "channel_count": len(channels)
                }

                # Log step transition (SSOT tracing)
                print(f"🔄 Chase '{chase['name']}' step {step_index}/{len(steps)-1} (loop {loop_count}): "
                      f"{len(channels)} channels, duration={step_duration_ms}ms, fade={fade_ms}ms", flush=True)

                # Send step to all universes in parallel for synchronized playback
                # TODO: Future improvement - store chase data on ESP nodes and trigger playback
                # locally for perfect sync. See: sync_chase_to_node() infrastructure already exists.
                # Would need ESP firmware to handle local chase playback with 'play_chase' command.
                def send_to_universe(univ, cid, sflag, token):
                    try:
                        # Pass chase_id, stop_flag, and arb_token for TOCTOU safety [F08]
                        self._send_step(univ, channels, fade_ms, distribution_mode, chase_id=cid, stop_flag=sflag, arb_token=token)
                    except Exception as e:
                        print(f"❌ Chase step send error (U{univ}): {e}", flush=True)

                # Check stop flag before spawning threads (race condition fix)
                if stop_flag.is_set():
                    break

                threads = [threading.Thread(target=send_to_universe, args=(univ, chase_id, stop_flag, arb_token)) for univ in universes]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()  # Wait for all sends to complete before timing next step

                # Wait for step duration (in seconds)
                stop_flag.wait(step_duration_ms / 1000.0)

                # Advance step
                step_index += 1
                if step_index >= len(steps):
                    if loop:
                        step_index = 0
                        loop_count += 1
                        print(f"🔁 Chase '{chase['name']}' loop {loop_count}", flush=True)
                    else:
                        break

        except Exception as e:
            print(f"❌ Chase '{chase['name']}' crashed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.chase_health[chase_id] = {"step": step_index, "last_time": time.time(), "status": f"error: {e}"}
        finally:
            self.chase_health[chase_id] = {"step": step_index, "last_time": time.time(), "status": "stopped"}
            if reg.get_db:
                # [N05 fix] Clean up thread-local DB connection
                try:
                    # Import close_db from aether-core where it's defined
                    from aether_core import close_db
                    close_db()
                except ImportError:
                    pass
            print(f"⏹️ Chase '{chase['name']}' stopped after {loop_count} loops", flush=True)

    def _send_step(self, universe, channels, fade_ms=0, distribution_mode='unified', chase_id=None, stop_flag=None, arb_token=None):
        """Send chase step with intelligent distribution.

        [F08] TOCTOU FIX: Validates arb_token before writing to ensure another
        engine hasn't acquired arbitration since the chase started.

        distribution_mode: 'unified' = replicate to all, 'pixel' = unique per fixture"""
        # Check stop flag BEFORE writing (race condition fix)
        if stop_flag and stop_flag.is_set():
            return
        # [F08] Validate arbitration token — reject stale writes
        if arb_token is not None and reg.arbitration and not reg.arbitration.validate_token(arb_token):
            return

        if not channels:
            return
        parsed = {}
        for key, value in channels.items():
            key_str = str(key)
            if ':' in key_str:
                parts = key_str.split(':')
                if int(parts[0]) == universe:
                    parsed[int(parts[1])] = value
            else:
                parsed[int(key_str)] = value
        if not parsed:
            return

        # Get fixtures and apply distribution mode
        if not reg.content_manager:
            return

        fixtures = reg.content_manager.get_fixtures(universe)
        if fixtures:
            fixtures = sorted(fixtures, key=lambda f: f.get('start_channel', 1))
            pattern_vals = list(parsed.values())
            expanded = {}
            if distribution_mode == 'pixel':
                # PIXEL: Each fixture gets unique sequential value
                for idx, fix in enumerate(fixtures):
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', 1)
                    val = pattern_vals[idx % len(pattern_vals)] if pattern_vals else 0
                    for ch in range(count):
                        expanded[start + ch] = val
            else:
                # UNIFIED: Replicate pattern to all fixtures
                for fix in fixtures:
                    start = fix.get('start_channel', 1)
                    count = fix.get('channel_count', len(pattern_vals))
                    for i in range(min(count, len(pattern_vals))):
                        expanded[start + i] = pattern_vals[i]
            if expanded:
                parsed = expanded

        # Check stop flag again before writing (double-check for race condition)
        if stop_flag and stop_flag.is_set():
            return

        # MERGE LAYER: Route through merge layer for proper priority handling
        if self._merge_layer and chase_id:
            source_id = self._merge_sources.get(chase_id)
            if source_id:
                # Update merge layer source channels
                self._merge_layer.set_source_channels(source_id, universe, parsed)
                # Compute merged output and send
                merged = self._merge_layer.compute_merge(universe)
                if merged and reg.content_manager:
                    # Send merged result to SSOT (content_manager handles node dispatch)
                    reg.content_manager.set_channels(universe, {str(k): v for k, v in merged.items()}, fade_ms=fade_ms)
                return

        # Fallback: direct write if merge layer not available (legacy behavior)
        # [N03] This path bypasses UnifiedPlayback — log for visibility
        logging.warning(f"⚠️ N03: ChaseEngine fallback direct write (merge_layer={self._merge_layer is not None}, chase_id={chase_id})")
        if reg.audit_log:
            reg.audit_log('chase_fallback_write', chase_id=chase_id, universe=universe, channels=len(parsed))

        # [F08] Token-validated arbitration guard
        if reg.arbitration and not reg.arbitration.can_write('chase', token=arb_token):
            return

        if reg.content_manager:
            reg.content_manager.set_channels(universe, parsed, fade_ms=fade_ms)
