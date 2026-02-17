"""
Dynamic Effects Engine — DEPRECATED (F06 consolidation)

New effects should use /api/effects/fixture which routes through
UnifiedPlaybackEngine. This module is retained for backward compatibility
with the legacy /api/effects/* routes (christmas, twinkle, smooth, wave,
strobe, pulse, fade, fire). These legacy routes will be removed once the
frontend fully migrates to fixture-aware effects.

The effect computation LOGIC (frame interpolation, timing curves, etc.)
is preserved as reference — UnifiedPlaybackEngine reimplements these
patterns natively in its own render loop.

See F06 in AETHER architecture remediation plan.
"""
import logging
import threading
import time
import random


class DynamicEffectsEngine:
    """Creates dynamic lighting effects with frame-by-frame interpolation for smooth fades.

    # ⚠️ AUTHORITY VIOLATION (TASK-0005) ⚠️
    # This engine MUST NOT own timing loops.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. Effect computation logic
    # will be preserved as utilities called BY UnifiedPlaybackEngine.
    #
    # DO NOT START EFFECTS INDEPENDENTLY - See TASK_LEDGER.md

    All output routes through SSOT (dmx_state + node_manager.send_via_ola).
    Properly integrates with ArbitrationManager for priority-based control.
    """

    def __init__(self):
        self.running = {}  # {effect_id: stop_flag}
        self.threads = {}
        self.fps = 40  # [F17] Aligned with DMX refresh (was 30)
        self.current_effect = None  # Track what's playing
        self._dmx_state = None  # Will be set by main module
        self._send_callback = None  # Will be set by main module
        self._arbitration = None  # Will be set by main module

    def set_ssot_hooks(self, dmx_state, send_callback, arbitration=None):
        """Set SSOT hooks from main module for state updates and output"""
        self._dmx_state = dmx_state
        self._send_callback = send_callback
        self._arbitration = arbitration

    def _acquire_arbitration(self, effect_id):
        """Acquire effect ownership from arbitration manager"""
        if self._arbitration:
            if not self._arbitration.acquire('effect', effect_id):
                print(f"⚠️ Cannot start effect - arbitration denied", flush=True)
                return False
        return True

    def _release_arbitration(self):
        """Release effect ownership if no effects running"""
        if self._arbitration and not self.running:
            self._arbitration.release('effect')

    def stop_effect(self, effect_id=None):
        """Stop an effect or all effects"""
        if effect_id:
            if effect_id in self.running:
                self.running[effect_id].set()
                if self.current_effect == effect_id:
                    self.current_effect = None
        else:
            for flag in self.running.values():
                flag.set()
            self.running.clear()
            self.threads.clear()
            self.current_effect = None
        self._release_arbitration()
        print(f"⏹️ Effects stopped: {effect_id or 'all'}")

    def _send_frame(self, universe, channels):
        """Send a single DMX frame through SSOT pipeline - NO FALLBACK, must use dispatcher"""
        try:
            # Check arbitration - skip if we don't own output
            if self._arbitration and not self._arbitration.can_write('effect'):
                return

            # Use callback (SSOT dispatcher) - this is REQUIRED, no fallback
            if self._send_callback:
                self._send_callback(universe, channels)
            else:
                # No fallback - effects engine MUST be hooked to SSOT
                print(f"❌ Effects engine not hooked to SSOT - skipping frame for U{universe}", flush=True)
        except Exception as e:
            print(f"❌ Effects frame error U{universe}: {e}")

    def get_status(self):
        """Get current effect status for diagnostics"""
        return {
            'running': list(self.running.keys()),
            'current_effect': self.current_effect,
            'count': len(self.running),
            'fps': self.fps
        }

    def christmas_stagger(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
                          fade_ms=1500, hold_ms=1000, stagger_ms=300):
        """Christmas colors with staggered fixture timing for wave effects"""
        effect_id = f"christmas_stagger_{int(time.time())}"

        # ARBITRATION: Acquire effect ownership
        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        # Christmas colors (RGBW)
        colors = [
            [255, 0, 0, 0],    # Red
            [0, 255, 0, 0],    # Green
            [255, 255, 255, 255],  # White
        ]

        def run():
            color_index = 0
            # Track current color for each fixture
            fixture_colors = {}
            for univ in universes:
                for fix in range(fixtures_per_universe):
                    fixture_colors[(univ, fix)] = list(colors[0])

            frame_interval = 1.0 / self.fps

            while not stop_flag.is_set():
                # Stagger through each fixture
                total_fixtures = len(universes) * fixtures_per_universe
                for fixture_num in range(total_fixtures):
                    if stop_flag.is_set():
                        break

                    univ_idx = fixture_num // fixtures_per_universe
                    fix_idx = fixture_num % fixtures_per_universe
                    universe = universes[univ_idx]
                    key = (universe, fix_idx)

                    from_color = fixture_colors[key]
                    to_color = colors[(color_index + fixture_num + 1) % len(colors)]

                    # Fade this fixture over fade_ms
                    fade_frames = max(1, int((fade_ms / 1000.0) * self.fps))
                    fade_start = time.monotonic()

                    for f in range(fade_frames):
                        if stop_flag.is_set():
                            break

                        frame_start = time.monotonic()
                        progress = f / fade_frames

                        # Build frame for this universe
                        frame = [0] * 512
                        for fix in range(fixtures_per_universe):
                            start_ch = fix * channels_per_fixture
                            fkey = (universe, fix)
                            if fix == fix_idx:
                                # This fixture is fading
                                for ch in range(channels_per_fixture):
                                    fr = from_color[ch] if ch < len(from_color) else 0
                                    to = to_color[ch] if ch < len(to_color) else 0
                                    frame[start_ch + ch] = int(fr + (to - fr) * progress)
                            else:
                                # Other fixtures hold their color
                                curr = fixture_colors[fkey]
                                for ch in range(channels_per_fixture):
                                    frame[start_ch + ch] = curr[ch] if ch < len(curr) else 0

                        self._send_frame(universe, frame)

                        # Precise timing using monotonic clock
                        elapsed = time.monotonic() - frame_start
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                    # Update fixture color
                    fixture_colors[key] = list(to_color)

                    # Stagger delay
                    if stagger_ms > 0:
                        stop_flag.wait(stagger_ms / 1000.0)

                # Hold before next cycle
                stop_flag.wait(hold_ms / 1000.0)
                color_index = (color_index + 1) % len(colors)

            print(f"⏹️ Effect christmas_stagger stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        # PHASE 1 GUARD: Log violation when effect spawns independent thread
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: DynamicEffectsEngine spawning independent "
            "timing thread. This violates AETHER Hard Rule 1.1 - Only "
            "UnifiedPlaybackEngine should own playback timing. See TASK-0005"
        )

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"🎄 Christmas stagger effect started on universes {universes}")
        return effect_id

    def random_twinkle(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
                       colors=None, min_fade_ms=500, max_fade_ms=2000):
        """Random twinkling effect - fixtures fade to random colors at random times"""
        effect_id = f"twinkle_{int(time.time())}"

        # ARBITRATION: Acquire effect ownership
        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        if colors is None:
            colors = [
                [255, 0, 0, 0],      # Red
                [0, 255, 0, 0],      # Green
                [255, 50, 0, 0],     # Orange
                [255, 255, 255, 255], # White
            ]

        def run():
            # Track current state of each fixture
            fixture_states = {}
            for univ in universes:
                for fix in range(fixtures_per_universe):
                    key = (univ, fix)
                    fixture_states[key] = {
                        'current': [0] * channels_per_fixture,
                        'target': list(random.choice(colors)[:channels_per_fixture]),
                        'fade_time': random.uniform(min_fade_ms, max_fade_ms) / 1000.0,
                        'start_time': time.monotonic()
                    }

            frame_interval = 1.0 / self.fps

            while not stop_flag.is_set():
                frame_start = time.monotonic()

                for univ in universes:
                    frame = [0] * 512
                    for fix in range(fixtures_per_universe):
                        key = (univ, fix)
                        state = fixture_states[key]
                        start_ch = fix * channels_per_fixture

                        elapsed = frame_start - state['start_time']
                        progress = min(1.0, elapsed / state['fade_time'])

                        for ch in range(channels_per_fixture):
                            curr = state['current'][ch]
                            tgt = state['target'][ch] if ch < len(state['target']) else 0
                            frame[start_ch + ch] = int(curr + (tgt - curr) * progress)

                        if progress >= 1.0:
                            state['current'] = list(state['target'])
                            state['target'] = list(random.choice(colors)[:channels_per_fixture])
                            state['fade_time'] = random.uniform(min_fade_ms, max_fade_ms) / 1000.0
                            state['start_time'] = frame_start

                    self._send_frame(univ, frame)

                # Precise timing
                elapsed = time.monotonic() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"⏹️ Effect twinkle stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"✨ Random twinkle effect started on universes {universes}")
        return effect_id

    def smooth_chase(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
                     colors=None, fade_ms=1500, hold_ms=500):
        """Smooth synchronized color fade across all fixtures"""
        effect_id = f"smooth_chase_{int(time.time())}"

        # ARBITRATION: Acquire effect ownership
        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        if colors is None:
            colors = [
                [255, 0, 0, 0],    # Red
                [0, 255, 0, 0],    # Green
            ]

        def run():
            color_index = 0
            current_color = list(colors[0])
            frame_interval = 1.0 / self.fps

            while not stop_flag.is_set():
                target_color = colors[(color_index + 1) % len(colors)]
                fade_frames = max(1, int((fade_ms / 1000.0) * self.fps))

                for f in range(fade_frames):
                    if stop_flag.is_set():
                        break

                    frame_start = time.monotonic()
                    progress = f / fade_frames

                    interp = []
                    for ch in range(channels_per_fixture):
                        curr = current_color[ch] if ch < len(current_color) else 0
                        tgt = target_color[ch] if ch < len(target_color) else 0
                        interp.append(int(curr + (tgt - curr) * progress))

                    for univ in universes:
                        frame = [0] * 512
                        for fix in range(fixtures_per_universe):
                            start_ch = fix * channels_per_fixture
                            for ch in range(channels_per_fixture):
                                frame[start_ch + ch] = interp[ch]
                        self._send_frame(univ, frame)

                    # Precise timing
                    elapsed = time.monotonic() - frame_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                current_color = list(target_color)
                color_index = (color_index + 1) % len(colors)
                stop_flag.wait(hold_ms / 1000.0)

            print(f"⏹️ Effect smooth_chase stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"🌈 Smooth chase started on universes {universes}")
        return effect_id

    def wave(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
             color=[255, 0, 0, 0], wave_speed_ms=2000, tail_length=2):
        """Wave effect - color travels across fixtures like a wave"""
        effect_id = f"wave_{int(time.time())}"

        # ARBITRATION: Acquire effect ownership
        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        def run():
            total_fixtures = len(universes) * fixtures_per_universe
            position = 0.0
            frame_interval = 1.0 / self.fps

            while not stop_flag.is_set():
                frame_start = time.monotonic()

                for univ_idx, univ in enumerate(universes):
                    frame = [0] * 512
                    for fix in range(fixtures_per_universe):
                        fixture_num = univ_idx * fixtures_per_universe + fix
                        start_ch = fix * channels_per_fixture

                        # Calculate brightness based on distance from wave position
                        distance = abs(fixture_num - position)
                        if distance < tail_length:
                            brightness = 1.0 - (distance / tail_length)
                        else:
                            brightness = 0.0

                        for ch in range(channels_per_fixture):
                            val = color[ch] if ch < len(color) else 0
                            frame[start_ch + ch] = int(val * brightness)

                    self._send_frame(univ, frame)

                # Move wave position
                position += (total_fixtures / (wave_speed_ms / 1000.0)) / self.fps
                if position >= total_fixtures + tail_length:
                    position = -tail_length

                # Precise timing
                elapsed = time.monotonic() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"⏹️ Effect wave stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"🌊 Wave effect started on universes {universes}")
        return effect_id

    def strobe(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
               color=[255, 255, 255, 0], on_ms=50, off_ms=50):
        """Strobe effect - fast on/off flashing"""
        effect_id = f"strobe_{int(time.time())}"

        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        def run():
            on = True
            while not stop_flag.is_set():
                for univ in universes:
                    frame = [0] * 512
                    if on:
                        for fix in range(fixtures_per_universe):
                            start_ch = fix * channels_per_fixture
                            for ch in range(channels_per_fixture):
                                frame[start_ch + ch] = color[ch] if ch < len(color) else 0
                    self._send_frame(univ, frame)

                delay = on_ms if on else off_ms
                on = not on
                stop_flag.wait(delay / 1000.0)

            print(f"⏹️ Effect strobe stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"⚡ Strobe effect started on universes {universes}")
        return effect_id

    def pulse(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
              color=[255, 255, 255, 0], pulse_ms=2000, min_brightness=0, max_brightness=255):
        """Pulse/breathing effect - smooth fade up and down"""
        effect_id = f"pulse_{int(time.time())}"

        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        def run():
            import math
            frame_interval = 1.0 / self.fps
            phase = 0.0

            while not stop_flag.is_set():
                frame_start = time.monotonic()

                # Sine wave for smooth breathing
                brightness = (math.sin(phase) + 1) / 2  # 0 to 1
                brightness = min_brightness + brightness * (max_brightness - min_brightness)
                brightness = brightness / 255.0  # Normalize

                for univ in universes:
                    frame = [0] * 512
                    for fix in range(fixtures_per_universe):
                        start_ch = fix * channels_per_fixture
                        for ch in range(channels_per_fixture):
                            val = color[ch] if ch < len(color) else 0
                            frame[start_ch + ch] = int(val * brightness)
                    self._send_frame(univ, frame)

                # Advance phase (full cycle = 2*pi)
                phase += (2 * math.pi) / ((pulse_ms / 1000.0) * self.fps)
                if phase >= 2 * math.pi:
                    phase -= 2 * math.pi

                elapsed = time.monotonic() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"⏹️ Effect pulse stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"💫 Pulse effect started on universes {universes}")
        return effect_id

    def fade(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
             colors=None, cycle_ms=10000):
        """Color fade - slowly cycle through colors"""
        effect_id = f"fade_{int(time.time())}"

        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        if colors is None:
            colors = [
                [255, 0, 0, 0],    # Red
                [255, 128, 0, 0],  # Orange
                [255, 255, 0, 0],  # Yellow
                [0, 255, 0, 0],    # Green
                [0, 255, 255, 0],  # Cyan
                [0, 0, 255, 0],    # Blue
                [128, 0, 255, 0],  # Purple
                [255, 0, 255, 0],  # Magenta
            ]

        def run():
            frame_interval = 1.0 / self.fps
            color_index = 0
            current_color = list(colors[0])

            # Time per color transition
            transition_ms = cycle_ms / len(colors)
            fade_frames = max(1, int((transition_ms / 1000.0) * self.fps))

            while not stop_flag.is_set():
                target_color = colors[(color_index + 1) % len(colors)]

                for f in range(fade_frames):
                    if stop_flag.is_set():
                        break

                    frame_start = time.monotonic()
                    progress = f / fade_frames

                    interp = []
                    for ch in range(channels_per_fixture):
                        curr = current_color[ch] if ch < len(current_color) else 0
                        tgt = target_color[ch] if ch < len(target_color) else 0
                        interp.append(int(curr + (tgt - curr) * progress))

                    for univ in universes:
                        frame = [0] * 512
                        for fix in range(fixtures_per_universe):
                            start_ch = fix * channels_per_fixture
                            for ch in range(channels_per_fixture):
                                frame[start_ch + ch] = interp[ch] if ch < len(interp) else 0
                        self._send_frame(univ, frame)

                    elapsed = time.monotonic() - frame_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                current_color = list(target_color)
                color_index = (color_index + 1) % len(colors)

            print(f"⏹️ Effect fade stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"🌙 Color fade effect started on universes {universes}")
        return effect_id

    def fire(self, universes, fixtures_per_universe=2, channels_per_fixture=4,
             intensity=0.8):
        """Fire flicker effect - realistic fire simulation"""
        effect_id = f"fire_{int(time.time())}"

        if not self._acquire_arbitration(effect_id):
            return None

        stop_flag = threading.Event()
        self.running[effect_id] = stop_flag
        self.current_effect = effect_id

        def run():
            frame_interval = 1.0 / self.fps

            # Fire colors (red/orange/yellow)
            fire_colors = [
                [255, 30, 0, 0],   # Deep red
                [255, 60, 0, 0],   # Red-orange
                [255, 100, 0, 0],  # Orange
                [255, 150, 10, 0], # Yellow-orange
            ]

            # Track each fixture's current state
            fixture_states = {}
            for univ in universes:
                for fix in range(fixtures_per_universe):
                    fixture_states[(univ, fix)] = {
                        'brightness': random.uniform(0.5, 1.0),
                        'color_idx': random.randint(0, len(fire_colors) - 1),
                        'flicker_timer': 0,
                        'next_flicker': random.uniform(0.05, 0.2),
                    }

            while not stop_flag.is_set():
                frame_start = time.monotonic()

                for univ in universes:
                    frame = [0] * 512
                    for fix in range(fixtures_per_universe):
                        key = (univ, fix)
                        state = fixture_states[key]
                        start_ch = fix * channels_per_fixture

                        # Random flicker
                        state['flicker_timer'] += frame_interval
                        if state['flicker_timer'] >= state['next_flicker']:
                            state['flicker_timer'] = 0
                            state['next_flicker'] = random.uniform(0.03, 0.15)
                            # Random brightness change
                            state['brightness'] += random.uniform(-0.3, 0.3)
                            state['brightness'] = max(0.2, min(1.0, state['brightness']))
                            # Occasionally change color
                            if random.random() < 0.3:
                                state['color_idx'] = random.randint(0, len(fire_colors) - 1)

                        color = fire_colors[state['color_idx']]
                        bright = state['brightness'] * intensity

                        for ch in range(channels_per_fixture):
                            val = color[ch] if ch < len(color) else 0
                            frame[start_ch + ch] = int(val * bright)

                    self._send_frame(univ, frame)

                elapsed = time.monotonic() - frame_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"⏹️ Effect fire stopped")
            self.running.pop(effect_id, None)
            if self.current_effect == effect_id:
                self.current_effect = None

        thread = threading.Thread(target=run, daemon=True)
        self.threads[effect_id] = thread
        thread.start()
        print(f"🔥 Fire flicker effect started on universes {universes}")
        return effect_id
