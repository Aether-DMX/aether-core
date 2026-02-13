"""
Render Engine - Real-time modifier composition for Looks and Sequences

# ============================================================================
# ⚠️  AUTHORITY VIOLATION WARNING (TASK-0004)
# ============================================================================
#
# This module contains RenderEngine, which ILLEGALLY owns a render loop.
#
# Per AETHER Hard Rule 1.1:
#   "Flask UnifiedPlaybackEngine is the ONLY authority allowed to generate
#    final DMX output. No other system may run an independent render loop."
#
# CURRENT VIOLATION:
#   - RenderEngine.start() spawns its own thread
#   - RenderEngine._render_loop() runs at 30 FPS independently
#   - /api/looks/{id}/play calls RenderEngine directly, bypassing
#     UnifiedPlaybackEngine (see TASK-0018)
#
# ALLOWED USAGE:
#   - ModifierRenderer class may be used as a UTILITY by UnifiedPlaybackEngine
#   - TimeContext, MergeMode, ModifierState are safe data classes
#
# PROHIBITED USAGE:
#   - RenderEngine.start() - DO NOT CALL INDEPENDENTLY
#   - RenderEngine._render_loop() - MUST NOT OWN TIMING
#
# This will be fixed in Phase 2. Until then, any use of RenderEngine.start()
# should log a warning. See TASK_LEDGER.md for full details.
#
# TODO: TASK-0004 - Retire RenderEngine loop, keep ModifierRenderer as utility
# ============================================================================

This module provides:
- ModifierRenderer: Computes modifier effects on base channels (SAFE - utility)
- RenderEngine: Single scheduler loop for rendering at target FPS (VIOLATION)
- Deterministic output with seeded random for reproducibility
- Composition rules for stacking multiple modifiers

Architecture:
- Input: base channels + list of modifiers + time context
- Output: final DMX channel values (0-255)
- All rendering is deterministic given same seed + time

Version: 1.0.0
"""

import math
import time
import threading
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Merge Modes - How modifiers combine with base values
# ============================================================

class MergeMode(Enum):
    """How a modifier's output combines with the current value"""
    MULTIPLY = "multiply"    # value * modifier (for brightness scaling)
    ADD = "add"              # value + modifier (clamped)
    REPLACE = "replace"      # modifier replaces value entirely
    MAX = "max"              # max(value, modifier) - HTP style
    MIN = "min"              # min(value, modifier)
    BLEND = "blend"          # lerp based on modifier intensity
    DEPTH_BLEND = "depth_blend"  # lerp(base, modifier_output, depth%) - for color modifiers


# Modifier type -> default merge mode
DEFAULT_MERGE_MODES = {
    "pulse": MergeMode.MULTIPLY,
    "strobe": MergeMode.MULTIPLY,
    "flicker": MergeMode.MULTIPLY,
    "wave": MergeMode.MULTIPLY,
    "rainbow": MergeMode.REPLACE,
    "twinkle": MergeMode.MULTIPLY,
    "hue_shift": MergeMode.DEPTH_BLEND,
    "color_temp": MergeMode.DEPTH_BLEND,
    "saturation_pulse": MergeMode.DEPTH_BLEND,
    "color_fade": MergeMode.DEPTH_BLEND,
    "gradient": MergeMode.DEPTH_BLEND,
    "scanner": MergeMode.MULTIPLY,
    "sparkle": MergeMode.DEPTH_BLEND,
    "lightning": MergeMode.DEPTH_BLEND,
}


# ============================================================
# Time Context - Passed to each render frame
# ============================================================

@dataclass
class TimeContext:
    """Time information for a single render frame"""
    absolute_time: float      # Time since epoch (for determinism)
    delta_time: float         # Time since last frame
    elapsed_time: float       # Time since render started
    frame_number: int         # Current frame count
    seed: int                 # Random seed for deterministic output


# ============================================================
# Render State - Per-modifier persistent state
# ============================================================

@dataclass
class ModifierState:
    """Persistent state for a modifier instance across frames"""
    modifier_id: str
    modifier_type: str
    random_state: random.Random  # Seeded RNG for this modifier
    phase: float = 0.0           # Current phase (0-1) for oscillators
    last_trigger: float = 0.0    # Last trigger time (for strobe, twinkle)
    custom_data: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Modifier Renderer - Computes single modifier effect
# ============================================================

class ModifierRenderer:
    """
    Renders a single modifier's effect on base channel values.

    Each modifier type has a dedicated render function that:
    - Takes base channels, modifier params, and time context
    - Returns a multiplier/value per channel
    - Is deterministic given same inputs

    Phase 1 Addition:
    - render_with_distribution(): Applies modifier with distribution across fixtures
    - render_fixture_frame(): Renders modifier on a RenderedFixtureFrame
    """

    def __init__(self):
        self._render_funcs = {
            "pulse": self._render_pulse,
            "strobe": self._render_strobe,
            "flicker": self._render_flicker,
            "wave": self._render_wave,
            "rainbow": self._render_rainbow,
            "twinkle": self._render_twinkle,
            "hue_shift": self._render_hue_shift,
            "color_temp": self._render_color_temp,
            "saturation_pulse": self._render_saturation_pulse,
            "color_fade": self._render_color_fade,
            "gradient": self._render_gradient,
            "scanner": self._render_scanner,
            "sparkle": self._render_sparkle,
            "lightning": self._render_lightning,
        }
        # Distribution calculator instance
        self._distribution_calculator = None

    def render(
        self,
        modifier_type: str,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int = 0,
        total_fixtures: int = 1,
    ) -> Dict[int, float]:
        """
        Render a modifier's effect.

        Returns dict of channel -> multiplier (0.0-1.0 for multiply mode)
        or channel -> value (0-255 for replace mode)
        """
        render_func = self._render_funcs.get(modifier_type)
        if not render_func:
            # Unknown modifier, return neutral (no change)
            return {ch: 1.0 for ch in base_channels}

        return render_func(params, base_channels, time_ctx, state, fixture_index, total_fixtures)

    # ─────────────────────────────────────────────────────────
    # DISTRIBUTION-AWARE RENDERING (Phase 1)
    # ─────────────────────────────────────────────────────────

    def render_with_distribution(
        self,
        modifier_type: str,
        params: Dict[str, Any],
        distribution_config: Any,  # DistributionConfig
        fixture_states: Dict[str, Dict[str, Any]],
        fixture_order: List[str],
        time_ctx: TimeContext,
        modifier_states: Dict[str, ModifierState],
        modifier_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Render a modifier with distribution across multiple fixtures.

        This is the main entry point for fixture-centric rendering with
        distribution modes (SYNCED, PHASED, INDEXED, etc.)

        Args:
            modifier_type: The modifier type (pulse, wave, etc.)
            params: Modifier parameters
            distribution_config: DistributionConfig controlling distribution
            fixture_states: Dict of fixture_id -> {"intensity": 255, "color": [...], ...}
            fixture_order: Ordered list of fixture IDs
            time_ctx: Time context for this frame
            modifier_states: Dict of modifier_id -> ModifierState
            modifier_id: ID of this modifier instance

        Returns:
            Dict of fixture_id -> modified attributes
        """
        from distribution_modes import DistributionMode, DistributionCalculator

        # Lazy init distribution calculator
        if self._distribution_calculator is None:
            self._distribution_calculator = DistributionCalculator(time_ctx.seed)

        total_fixtures = len(fixture_order)
        result = {}

        # Get or create modifier state
        state = modifier_states.get(modifier_id)
        if not state:
            state = ModifierState(
                modifier_id=modifier_id,
                modifier_type=modifier_type,
                random_state=random.Random(time_ctx.seed + hash(modifier_id))
            )
            modifier_states[modifier_id] = state

        for fixture_index, fixture_id in enumerate(fixture_order):
            fixture_attrs = fixture_states.get(fixture_id, {}).copy()

            # Calculate distribution-adjusted phase
            if distribution_config:
                dist_phase = self._distribution_calculator.get_fixture_phase(
                    distribution_config,
                    fixture_index,
                    total_fixtures,
                    base_phase=0.0
                )
                # Inject distribution phase into params
                adjusted_params = params.copy()
                current_phase_offset = adjusted_params.get("phase_offset", 0.0)
                adjusted_params["phase_offset"] = current_phase_offset + dist_phase
            else:
                adjusted_params = params

            # Convert fixture attributes to channel-like format for rendering
            base_channels = self._fixture_attrs_to_channels(fixture_attrs)

            # Create per-fixture state if needed for PIXELATED/RANDOM modes
            if distribution_config and distribution_config.mode in (
                DistributionMode.PIXELATED, DistributionMode.RANDOM
            ):
                # Use fixture-specific seed
                fixture_seed = self._distribution_calculator.get_fixture_seed(
                    distribution_config, fixture_index
                )
                fixture_state = ModifierState(
                    modifier_id=f"{modifier_id}_{fixture_id}",
                    modifier_type=modifier_type,
                    random_state=random.Random(fixture_seed)
                )
            else:
                fixture_state = state

            # Render modifier for this fixture
            mod_output = self.render(
                modifier_type=modifier_type,
                params=adjusted_params,
                base_channels=base_channels,
                time_ctx=time_ctx,
                state=fixture_state,
                fixture_index=fixture_index,
                total_fixtures=total_fixtures,
            )

            # Apply modifier output to fixture attributes
            depth = params.get("depth", 100) / 100.0
            result[fixture_id] = self._apply_modifier_to_attrs(
                fixture_attrs, mod_output, modifier_type, depth=depth
            )

        return result

    def _fixture_attrs_to_channels(
        self,
        attrs: Dict[str, Any]
    ) -> Dict[int, int]:
        """
        Convert fixture attributes to channel-like format for modifier rendering.

        Maps:
        - intensity -> channel 0
        - color[0] (R) -> channel 1
        - color[1] (G) -> channel 2
        - color[2] (B) -> channel 3
        - color[3] (W) -> channel 4
        - etc.
        """
        channels = {}

        # Intensity as channel 0
        if "intensity" in attrs:
            channels[0] = attrs["intensity"]

        # Color channels
        color = attrs.get("color", [])
        for i, val in enumerate(color):
            channels[i + 1] = val

        return channels if channels else {0: 255}

    def _apply_modifier_to_attrs(
        self,
        original_attrs: Dict[str, Any],
        mod_output: Dict[int, float],
        modifier_type: str,
        depth: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Apply modifier output back to fixture attributes.

        Args:
            original_attrs: Original fixture attributes
            mod_output: Modifier output (channel -> multiplier/value)
            modifier_type: Type of modifier (for merge mode determination)
            depth: 0.0-1.0 blend factor (0=no effect, 1=full effect)

        Returns:
            Modified fixture attributes
        """
        result = original_attrs.copy()
        merge_mode = DEFAULT_MERGE_MODES.get(modifier_type, MergeMode.MULTIPLY)

        # Apply intensity (channel 0)
        if 0 in mod_output:
            original_intensity = original_attrs.get("intensity", 255)
            if merge_mode == MergeMode.MULTIPLY:
                effected = int(original_intensity * mod_output[0])
                result["intensity"] = int(original_intensity + (effected - original_intensity) * depth)
            elif merge_mode == MergeMode.REPLACE:
                effected = int(mod_output[0])
                result["intensity"] = int(original_intensity + (effected - original_intensity) * depth)
            elif merge_mode == MergeMode.ADD:
                effected = min(255, int(original_intensity + mod_output[0]))
                result["intensity"] = int(original_intensity + (effected - original_intensity) * depth)
            elif merge_mode == MergeMode.DEPTH_BLEND:
                # For depth_blend, mod_output is the target value
                effected = int(mod_output[0])
                result["intensity"] = int(original_intensity + (effected - original_intensity) * depth)
            result["intensity"] = max(0, min(255, result["intensity"]))

        # Apply color channels
        original_color = original_attrs.get("color", [255, 255, 255])
        if len(original_color) > 0:
            new_color = list(original_color)
            for i in range(len(new_color)):
                ch = i + 1
                if ch in mod_output:
                    orig_val = new_color[i]
                    if merge_mode == MergeMode.MULTIPLY:
                        effected = int(orig_val * mod_output[ch])
                    elif merge_mode == MergeMode.REPLACE:
                        effected = int(mod_output[ch])
                    elif merge_mode == MergeMode.ADD:
                        effected = min(255, int(orig_val + mod_output[ch]))
                    elif merge_mode == MergeMode.DEPTH_BLEND:
                        effected = int(mod_output[ch])
                    else:
                        effected = orig_val
                    # Apply depth blend
                    new_color[i] = max(0, min(255, int(orig_val + (effected - orig_val) * depth)))
            result["color"] = new_color

        return result

    def render_fixture_frame(
        self,
        modifier_type: str,
        params: Dict[str, Any],
        distribution_config: Any,
        frame: Any,  # RenderedFixtureFrame
        time_ctx: TimeContext,
        modifier_states: Dict[str, ModifierState],
        modifier_id: str,
    ) -> Any:  # RenderedFixtureFrame
        """
        Render a modifier on a RenderedFixtureFrame with distribution.

        This is a convenience method that works directly with RenderedFixtureFrame.

        Args:
            modifier_type: The modifier type
            params: Modifier parameters
            distribution_config: DistributionConfig
            frame: RenderedFixtureFrame to modify
            time_ctx: Time context
            modifier_states: Modifier state storage
            modifier_id: ID of this modifier

        Returns:
            Modified RenderedFixtureFrame (copy)
        """
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        # Extract fixture states and order
        fixture_order = list(frame.fixtures.keys())
        fixture_states = {
            fid: state.attributes.copy()
            for fid, state in frame.fixtures.items()
        }

        # Render with distribution
        modified_attrs = self.render_with_distribution(
            modifier_type=modifier_type,
            params=params,
            distribution_config=distribution_config,
            fixture_states=fixture_states,
            fixture_order=fixture_order,
            time_ctx=time_ctx,
            modifier_states=modifier_states,
            modifier_id=modifier_id,
        )

        # Build new frame with modified states
        new_frame = RenderedFixtureFrame(
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            seed=frame.seed,
        )

        for fixture_id in fixture_order:
            original_state = frame.fixtures[fixture_id]
            new_attrs = modified_attrs.get(fixture_id, original_state.attributes)

            new_frame.fixtures[fixture_id] = RenderedFixtureState(
                fixture_id=fixture_id,
                attributes=new_attrs,
                source=original_state.source,
                modified_by=original_state.modified_by + [modifier_id],
            )

        return new_frame

    # ─────────────────────────────────────────────────────────
    # PULSE - Breathing brightness effect
    # ─────────────────────────────────────────────────────────
    def _render_pulse(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Smooth sinusoidal brightness pulsing"""
        speed = params.get("speed", 1.0)  # Hz
        min_bright = params.get("min_brightness", 20) / 100.0
        max_bright = params.get("max_brightness", 100) / 100.0
        curve = params.get("curve", "sine")
        phase_offset = params.get("phase_offset", 0.0)

        # Calculate phase based on time and speed
        phase = (time_ctx.elapsed_time * speed + phase_offset) % 1.0

        # Apply curve
        if curve == "sine":
            # Sine wave: smooth oscillation
            value = (math.sin(phase * 2 * math.pi) + 1) / 2
        elif curve == "linear":
            # Triangle wave: linear up/down
            value = 1 - abs(2 * phase - 1)
        elif curve == "ease-in":
            # Quadratic ease in
            value = phase * phase if phase < 0.5 else 1 - (1 - phase) ** 2
            value = 1 - abs(2 * value - 1)
        elif curve == "ease-out":
            # Quadratic ease out
            value = 1 - (1 - phase) ** 2 if phase < 0.5 else phase ** 2
            value = 1 - abs(2 * value - 1)
        elif curve == "ease-in-out":
            # Smooth ease in/out
            if phase < 0.5:
                value = 2 * phase * phase
            else:
                value = 1 - (-2 * phase + 2) ** 2 / 2
            value = 1 - abs(2 * value - 1)
        else:
            value = (math.sin(phase * 2 * math.pi) + 1) / 2

        # Scale to brightness range
        multiplier = min_bright + value * (max_bright - min_bright)

        return {ch: multiplier for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # STROBE - On/off flash effect
    # ─────────────────────────────────────────────────────────
    def _render_strobe(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Sharp on/off strobe with optional attack/decay"""
        rate = params.get("rate", 10.0)  # Hz
        duty_cycle = params.get("duty_cycle", 50) / 100.0
        attack = params.get("attack", 0) / 1000.0  # ms to seconds
        decay = params.get("decay", 0) / 1000.0

        # Calculate period and position within cycle
        period = 1.0 / rate
        cycle_pos = (time_ctx.elapsed_time % period) / period

        # Determine if we're in ON or OFF phase
        if cycle_pos < duty_cycle:
            # ON phase
            on_time = cycle_pos * period
            if attack > 0 and on_time < attack:
                # Attack ramp
                multiplier = on_time / attack
            else:
                multiplier = 1.0
        else:
            # OFF phase
            off_time = (cycle_pos - duty_cycle) * period
            if decay > 0 and off_time < decay:
                # Decay ramp
                multiplier = 1.0 - (off_time / decay)
            else:
                multiplier = 0.0

        return {ch: multiplier for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # FLICKER - Random brightness variance (deterministic)
    # ─────────────────────────────────────────────────────────
    def _render_flicker(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Random flickering with smoothing - deterministic via seeded RNG"""
        intensity = params.get("intensity", 20) / 100.0
        speed = params.get("speed", 3.0)  # Changes per second
        smoothing = params.get("smoothing", 30) / 100.0
        min_bright = params.get("min_brightness", 20) / 100.0

        # Use seeded random for deterministic output
        rng = state.random_state

        # Generate new target at intervals based on speed
        change_interval = 1.0 / speed

        # Initialize custom data if needed
        if "current_value" not in state.custom_data:
            state.custom_data["current_value"] = 1.0
            state.custom_data["target_value"] = 1.0
            state.custom_data["last_change"] = time_ctx.elapsed_time

        # Check if it's time for a new target
        time_since_change = time_ctx.elapsed_time - state.custom_data["last_change"]
        if time_since_change >= change_interval:
            # Generate new random target
            # Re-seed based on frame for determinism
            rng.seed(time_ctx.seed + int(time_ctx.elapsed_time * speed))
            variation = rng.uniform(-intensity, intensity)
            state.custom_data["target_value"] = max(min_bright, min(1.0, 1.0 + variation))
            state.custom_data["last_change"] = time_ctx.elapsed_time

        # Smooth interpolation to target
        current = state.custom_data["current_value"]
        target = state.custom_data["target_value"]

        # Smoothing factor based on delta time
        smooth_factor = 1.0 - smoothing
        lerp_amount = min(1.0, smooth_factor * time_ctx.delta_time * speed * 10)
        new_value = current + (target - current) * lerp_amount
        state.custom_data["current_value"] = new_value

        return {ch: max(min_bright, new_value) for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # WAVE - Position-based brightness wave
    # ─────────────────────────────────────────────────────────
    def _render_wave(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Brightness wave traveling across fixtures"""
        speed = params.get("speed", 1.0)  # Hz (full wave cycles per second)
        width = params.get("width", 3)
        direction = params.get("direction", "forward")
        shape = params.get("shape", "sine")
        min_bright = params.get("min_brightness", 0) / 100.0

        # Calculate wave position (0 to total_fixtures)
        if direction == "forward":
            wave_pos = (time_ctx.elapsed_time * speed * total_fixtures) % (total_fixtures + width)
        elif direction == "backward":
            wave_pos = total_fixtures - (time_ctx.elapsed_time * speed * total_fixtures) % (total_fixtures + width)
        elif direction == "bounce":
            # Ping-pong
            cycle = (time_ctx.elapsed_time * speed) % 2.0
            if cycle < 1.0:
                wave_pos = cycle * (total_fixtures + width)
            else:
                wave_pos = (2.0 - cycle) * (total_fixtures + width)
        elif direction == "center-out":
            # Expand from center
            center = total_fixtures / 2
            expansion = (time_ctx.elapsed_time * speed * total_fixtures) % (total_fixtures / 2 + width)
            dist_from_center = abs(fixture_index - center)
            wave_pos = center + expansion if fixture_index >= center else center - expansion
        elif direction == "edges-in":
            # Contract to center
            center = total_fixtures / 2
            contraction = (total_fixtures / 2) - (time_ctx.elapsed_time * speed * total_fixtures) % (total_fixtures / 2 + width)
            wave_pos = contraction if fixture_index < center else total_fixtures - contraction
        else:
            wave_pos = (time_ctx.elapsed_time * speed * total_fixtures) % (total_fixtures + width)

        # Calculate distance from wave peak
        distance = abs(fixture_index - wave_pos)

        # Calculate brightness based on distance and width
        if distance >= width:
            value = 0.0
        else:
            # Normalize distance (0 at peak, 1 at edge)
            norm_dist = distance / width

            if shape == "sine":
                value = (math.cos(norm_dist * math.pi) + 1) / 2
            elif shape == "triangle":
                value = 1.0 - norm_dist
            elif shape == "square":
                value = 1.0 if norm_dist < 0.5 else 0.0
            elif shape == "sawtooth":
                value = 1.0 - norm_dist
            else:
                value = (math.cos(norm_dist * math.pi) + 1) / 2

        # Apply min brightness
        multiplier = min_bright + value * (1.0 - min_bright)

        return {ch: multiplier for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # RAINBOW - Hue rotation effect
    # ─────────────────────────────────────────────────────────
    def _render_rainbow(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """
        Rainbow cycles through hue - REPLACES color channels.
        Returns RGB values (0-255) instead of multipliers.
        """
        speed = params.get("speed", 0.2)  # Hz
        saturation = params.get("saturation", 100) / 100.0
        spread = params.get("spread", 0) / 100.0
        hue_range = params.get("hue_range", 360)
        hue_offset = params.get("hue_offset", 0)

        # Calculate hue based on time and fixture position
        base_hue = (time_ctx.elapsed_time * speed * 360) % 360

        # Apply fixture spread (phase offset between fixtures)
        if spread > 0 and total_fixtures > 1:
            fixture_offset = (fixture_index / total_fixtures) * 360 * spread
        else:
            fixture_offset = 0

        # Final hue within range
        hue = (base_hue + fixture_offset + hue_offset) % hue_range

        # Convert HSV to RGB (full value/brightness - modifier only affects color)
        r, g, b = self._hsv_to_rgb(hue / 360.0, saturation, 1.0)

        # Find RGB channels in base_channels (assume first 3 are RGB)
        sorted_channels = sorted(base_channels.keys())
        result = {}

        for i, ch in enumerate(sorted_channels):
            if i == 0:
                result[ch] = r * 255
            elif i == 1:
                result[ch] = g * 255
            elif i == 2:
                result[ch] = b * 255
            else:
                # Non-RGB channels pass through
                result[ch] = base_channels[ch]

        return result

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV (0-1) to RGB (0-1)"""
        if s == 0:
            return v, v, v

        h = h * 6
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

    # ─────────────────────────────────────────────────────────
    # TWINKLE - Random sparkle effect (deterministic)
    # ─────────────────────────────────────────────────────────
    def _render_twinkle(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Random twinkling - deterministic via seeded RNG"""
        density = params.get("density", 30) / 100.0
        fade_time = params.get("fade_time", 500) / 1000.0  # seconds
        min_bright = params.get("min_brightness", 20) / 100.0
        max_bright = params.get("max_brightness", 100) / 100.0
        hold_time = params.get("hold_time", 100) / 1000.0  # seconds

        # Use seeded random for deterministic output
        rng = state.random_state

        # Initialize twinkle state if needed
        if "twinkle_phase" not in state.custom_data:
            state.custom_data["twinkle_phase"] = "idle"  # idle, rising, hold, falling
            state.custom_data["phase_start"] = 0.0
            state.custom_data["current_bright"] = min_bright

        phase = state.custom_data["twinkle_phase"]
        phase_start = state.custom_data["phase_start"]
        current_bright = state.custom_data["current_bright"]

        # State machine for twinkle
        elapsed_in_phase = time_ctx.elapsed_time - phase_start

        if phase == "idle":
            # Check if we should start a new twinkle (density-based probability)
            # Seed based on time slot for determinism
            time_slot = int(time_ctx.elapsed_time * 10)  # 10 checks per second
            rng.seed(time_ctx.seed + fixture_index * 1000 + time_slot)

            if rng.random() < density * time_ctx.delta_time * 2:
                state.custom_data["twinkle_phase"] = "rising"
                state.custom_data["phase_start"] = time_ctx.elapsed_time
                current_bright = min_bright
            else:
                current_bright = min_bright

        elif phase == "rising":
            if elapsed_in_phase < fade_time:
                # Fade up
                progress = elapsed_in_phase / fade_time
                current_bright = min_bright + progress * (max_bright - min_bright)
            else:
                state.custom_data["twinkle_phase"] = "hold"
                state.custom_data["phase_start"] = time_ctx.elapsed_time
                current_bright = max_bright

        elif phase == "hold":
            if elapsed_in_phase >= hold_time:
                state.custom_data["twinkle_phase"] = "falling"
                state.custom_data["phase_start"] = time_ctx.elapsed_time
            current_bright = max_bright

        elif phase == "falling":
            if elapsed_in_phase < fade_time:
                # Fade down
                progress = elapsed_in_phase / fade_time
                current_bright = max_bright - progress * (max_bright - min_bright)
            else:
                state.custom_data["twinkle_phase"] = "idle"
                state.custom_data["phase_start"] = time_ctx.elapsed_time
                current_bright = min_bright

        state.custom_data["current_bright"] = current_bright

        return {ch: current_bright for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # HUE SHIFT - Rotates hue of the active color
    # ─────────────────────────────────────────────────────────
    def _render_hue_shift(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Rotate the hue of the base color. Returns RGB values (0-255)."""
        speed = params.get("speed", 0.2)
        hue_range = params.get("range", 360)
        hue_offset = params.get("offset", 0)
        phase_offset = params.get("phase_offset", 0.0)

        # Extract base RGB (first 3 channels)
        sorted_chs = sorted(base_channels.keys())
        if len(sorted_chs) < 3:
            return {ch: base_channels[ch] for ch in base_channels}

        r = base_channels[sorted_chs[0]] / 255.0
        g = base_channels[sorted_chs[1]] / 255.0
        b = base_channels[sorted_chs[2]] / 255.0

        # Convert to HSV
        h, s, v = self._rgb_to_hsv(r, g, b)

        # Rotate hue
        shift = ((time_ctx.elapsed_time * speed + phase_offset) % 1.0) * (hue_range / 360.0)
        h = (h + shift + hue_offset / 360.0) % 1.0

        # Convert back to RGB
        nr, ng, nb = self._hsv_to_rgb(h, s, v)

        result = {}
        for i, ch in enumerate(sorted_chs):
            if i == 0:
                result[ch] = nr * 255
            elif i == 1:
                result[ch] = ng * 255
            elif i == 2:
                result[ch] = nb * 255
            else:
                result[ch] = base_channels[ch]
        return result

    # ─────────────────────────────────────────────────────────
    # COLOR TEMP - Warm/cool tint shift
    # ─────────────────────────────────────────────────────────
    def _render_color_temp(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Shift color temperature warm (amber) or cool (blue). Returns RGB 0-255."""
        temperature = params.get("temperature", 0)  # -100 to +100
        speed = params.get("speed", 0.0)
        phase_offset = params.get("phase_offset", 0.0)

        # Animate temperature if speed > 0
        if speed > 0:
            phase = (time_ctx.elapsed_time * speed + phase_offset) % 1.0
            temperature = temperature * math.sin(phase * 2 * math.pi)

        temp_factor = temperature / 100.0  # -1 to +1

        sorted_chs = sorted(base_channels.keys())
        if len(sorted_chs) < 3:
            return {ch: base_channels[ch] for ch in base_channels}

        r = base_channels[sorted_chs[0]] / 255.0
        g = base_channels[sorted_chs[1]] / 255.0
        b = base_channels[sorted_chs[2]] / 255.0

        if temp_factor > 0:
            # Warm: boost red/green, reduce blue
            r = min(1.0, r + 0.3 * temp_factor)
            g = min(1.0, g + 0.1 * temp_factor)
            b = max(0.0, b - 0.3 * temp_factor)
        else:
            # Cool: boost blue, reduce red
            r = max(0.0, r + 0.3 * temp_factor)
            g = min(1.0, g + 0.05 * abs(temp_factor))
            b = min(1.0, b + 0.3 * abs(temp_factor))

        result = {}
        for i, ch in enumerate(sorted_chs):
            if i == 0:
                result[ch] = r * 255
            elif i == 1:
                result[ch] = g * 255
            elif i == 2:
                result[ch] = b * 255
            else:
                result[ch] = base_channels[ch]
        return result

    # ─────────────────────────────────────────────────────────
    # SATURATION PULSE - Breathes between color and white
    # ─────────────────────────────────────────────────────────
    def _render_saturation_pulse(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Oscillate saturation of the base color. Returns RGB 0-255."""
        speed = params.get("speed", 0.5)
        min_sat = params.get("min_saturation", 0) / 100.0
        max_sat = params.get("max_saturation", 100) / 100.0
        phase_offset = params.get("phase_offset", 0.0)

        sorted_chs = sorted(base_channels.keys())
        if len(sorted_chs) < 3:
            return {ch: base_channels[ch] for ch in base_channels}

        r = base_channels[sorted_chs[0]] / 255.0
        g = base_channels[sorted_chs[1]] / 255.0
        b = base_channels[sorted_chs[2]] / 255.0

        h, s, v = self._rgb_to_hsv(r, g, b)

        # Oscillate saturation
        phase = (time_ctx.elapsed_time * speed + phase_offset) % 1.0
        sat_value = (math.sin(phase * 2 * math.pi) + 1) / 2
        new_s = min_sat + sat_value * (max_sat - min_sat)

        nr, ng, nb = self._hsv_to_rgb(h, new_s, v)

        result = {}
        for i, ch in enumerate(sorted_chs):
            if i == 0:
                result[ch] = nr * 255
            elif i == 1:
                result[ch] = ng * 255
            elif i == 2:
                result[ch] = nb * 255
            else:
                result[ch] = base_channels[ch]
        return result

    # ─────────────────────────────────────────────────────────
    # COLOR FADE - Cycle through a color palette
    # ─────────────────────────────────────────────────────────
    def _render_color_fade(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Smoothly cycle through a list of colors. Returns RGB 0-255."""
        speed = params.get("speed", 0.2)
        colors = params.get("colors", [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        phase_offset = params.get("phase_offset", 0.0)

        if not colors or len(colors) < 2:
            return {ch: base_channels[ch] for ch in base_channels}

        num_colors = len(colors)
        phase = (time_ctx.elapsed_time * speed + phase_offset) % 1.0
        scaled = phase * num_colors
        idx = int(scaled) % num_colors
        next_idx = (idx + 1) % num_colors
        blend = scaled - int(scaled)

        c1 = colors[idx]
        c2 = colors[next_idx]
        r = c1[0] + (c2[0] - c1[0]) * blend
        g = c1[1] + (c2[1] - c1[1]) * blend
        b = c1[2] + (c2[2] - c1[2]) * blend

        sorted_chs = sorted(base_channels.keys())
        result = {}
        for i, ch in enumerate(sorted_chs):
            if i == 0:
                result[ch] = r
            elif i == 1:
                result[ch] = g
            elif i == 2:
                result[ch] = b
            else:
                result[ch] = base_channels[ch]
        return result

    # ─────────────────────────────────────────────────────────
    # GRADIENT - Position-based color from palette
    # ─────────────────────────────────────────────────────────
    def _render_gradient(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Position-based color gradient across fixtures. Returns RGB 0-255."""
        speed = params.get("speed", 0.2)
        colors = params.get("colors", [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        spread = params.get("spread", 100) / 100.0
        phase_offset = params.get("phase_offset", 0.0)

        if not colors or len(colors) < 2:
            return {ch: base_channels[ch] for ch in base_channels}

        num_colors = len(colors)
        # Base phase advances with time
        base_phase = (time_ctx.elapsed_time * speed + phase_offset) % 1.0
        # Fixture position adds spatial offset
        if total_fixtures > 1 and spread > 0:
            fixture_phase = (fixture_index / total_fixtures) * spread
        else:
            fixture_phase = 0

        phase = (base_phase + fixture_phase) % 1.0
        scaled = phase * num_colors
        idx = int(scaled) % num_colors
        next_idx = (idx + 1) % num_colors
        blend = scaled - int(scaled)

        c1 = colors[idx]
        c2 = colors[next_idx]
        r = c1[0] + (c2[0] - c1[0]) * blend
        g = c1[1] + (c2[1] - c1[1]) * blend
        b = c1[2] + (c2[2] - c1[2]) * blend

        sorted_chs = sorted(base_channels.keys())
        result = {}
        for i, ch in enumerate(sorted_chs):
            if i == 0:
                result[ch] = r
            elif i == 1:
                result[ch] = g
            elif i == 2:
                result[ch] = b
            else:
                result[ch] = base_channels[ch]
        return result

    # ─────────────────────────────────────────────────────────
    # SCANNER - Knight Rider style beam
    # ─────────────────────────────────────────────────────────
    def _render_scanner(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Focused beam scanning across fixtures. Returns multiplier."""
        speed = params.get("speed", 1.5)
        width = params.get("width", 2)
        direction = params.get("direction", "bounce")

        if total_fixtures <= 1:
            return {ch: 1.0 for ch in base_channels}

        # Calculate beam position
        if direction == "bounce":
            cycle = (time_ctx.elapsed_time * speed) % 2.0
            if cycle < 1.0:
                beam_pos = cycle * (total_fixtures - 1)
            else:
                beam_pos = (2.0 - cycle) * (total_fixtures - 1)
        elif direction == "forward":
            beam_pos = (time_ctx.elapsed_time * speed * total_fixtures) % total_fixtures
        else:  # backward
            beam_pos = total_fixtures - (time_ctx.elapsed_time * speed * total_fixtures) % total_fixtures

        distance = abs(fixture_index - beam_pos)
        if distance >= width:
            multiplier = 0.0
        else:
            multiplier = (math.cos((distance / width) * math.pi) + 1) / 2

        return {ch: multiplier for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # SPARKLE - Random white flashes
    # ─────────────────────────────────────────────────────────
    def _render_sparkle(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Random fixtures flash to white/color briefly. Returns RGB 0-255."""
        density = params.get("density", 30) / 100.0
        flash_duration = params.get("flash_duration", 200) / 1000.0
        sparkle_color = params.get("color", [255, 255, 255])
        phase_offset = params.get("phase_offset", 0.0)

        rng = state.random_state

        # Initialize state
        if "sparkle_active" not in state.custom_data:
            state.custom_data["sparkle_active"] = False
            state.custom_data["sparkle_start"] = 0.0
            state.custom_data["next_check"] = 0.0

        # Check if we should start a new sparkle
        if not state.custom_data["sparkle_active"]:
            if time_ctx.elapsed_time >= state.custom_data["next_check"]:
                time_slot = int(time_ctx.elapsed_time * 20)
                rng.seed(time_ctx.seed + fixture_index * 1000 + time_slot)
                if rng.random() < density * time_ctx.delta_time * 3:
                    state.custom_data["sparkle_active"] = True
                    state.custom_data["sparkle_start"] = time_ctx.elapsed_time
                state.custom_data["next_check"] = time_ctx.elapsed_time + 0.05

        # If sparkling, return sparkle color with fade
        if state.custom_data["sparkle_active"]:
            elapsed_in_flash = time_ctx.elapsed_time - state.custom_data["sparkle_start"]
            if elapsed_in_flash >= flash_duration:
                state.custom_data["sparkle_active"] = False
                return {ch: base_channels[ch] for ch in base_channels}

            # Fade out over flash duration
            brightness = 1.0 - (elapsed_in_flash / flash_duration)
            sorted_chs = sorted(base_channels.keys())
            result = {}
            for i, ch in enumerate(sorted_chs):
                if i < 3 and i < len(sparkle_color):
                    # Blend toward sparkle color
                    base_val = base_channels[ch]
                    target = sparkle_color[i]
                    result[ch] = base_val + (target - base_val) * brightness
                else:
                    result[ch] = base_channels[ch]
            return result

        return {ch: base_channels[ch] for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # LIGHTNING - Sudden random bright flashes
    # ─────────────────────────────────────────────────────────
    def _render_lightning(
        self,
        params: Dict[str, Any],
        base_channels: Dict[int, int],
        time_ctx: TimeContext,
        state: ModifierState,
        fixture_index: int,
        total_fixtures: int,
    ) -> Dict[int, float]:
        """Random bright flashes with decay. Returns RGB 0-255."""
        frequency = params.get("frequency", 0.5)
        intensity = params.get("intensity", 100) / 100.0
        decay = params.get("decay", 200) / 1000.0
        phase_offset = params.get("phase_offset", 0.0)

        rng = state.random_state

        if "flash_active" not in state.custom_data:
            state.custom_data["flash_active"] = False
            state.custom_data["flash_start"] = 0.0
            state.custom_data["next_check"] = 0.0

        # Random flash trigger
        if not state.custom_data["flash_active"]:
            if time_ctx.elapsed_time >= state.custom_data["next_check"]:
                time_slot = int(time_ctx.elapsed_time * 10)
                rng.seed(time_ctx.seed + time_slot)
                if rng.random() < frequency * time_ctx.delta_time:
                    state.custom_data["flash_active"] = True
                    state.custom_data["flash_start"] = time_ctx.elapsed_time
                state.custom_data["next_check"] = time_ctx.elapsed_time + 0.1

        if state.custom_data["flash_active"]:
            elapsed_in_flash = time_ctx.elapsed_time - state.custom_data["flash_start"]
            if elapsed_in_flash >= decay:
                state.custom_data["flash_active"] = False
                return {ch: base_channels[ch] for ch in base_channels}

            # Exponential decay flash
            flash_brightness = intensity * math.exp(-3 * elapsed_in_flash / decay)
            sorted_chs = sorted(base_channels.keys())
            result = {}
            for i, ch in enumerate(sorted_chs):
                if i < 3:
                    # Flash to white
                    base_val = base_channels[ch]
                    result[ch] = base_val + (255 - base_val) * flash_brightness
                else:
                    result[ch] = base_channels[ch]
            return result

        return {ch: base_channels[ch] for ch in base_channels}

    # ─────────────────────────────────────────────────────────
    # Color space helpers
    # ─────────────────────────────────────────────────────────
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Convert RGB (0-1) to HSV (0-1)"""
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c

        # Value
        v = max_c

        # Saturation
        if max_c == 0:
            s = 0.0
        else:
            s = delta / max_c

        # Hue
        if delta == 0:
            h = 0.0
        elif max_c == r:
            h = ((g - b) / delta) % 6 / 6.0
        elif max_c == g:
            h = ((b - r) / delta + 2) / 6.0
        else:
            h = ((r - g) / delta + 4) / 6.0

        if h < 0:
            h += 1.0

        return h, s, v


# ============================================================
# Render Engine - Main scheduler loop
# ============================================================

class RenderEngine:
    """
    Single scheduler loop for rendering Looks with modifiers.

    # ⚠️ AUTHORITY VIOLATION (TASK-0004) ⚠️
    # This engine MUST NOT own a render loop.
    # Playback timing is owned by UnifiedPlaybackEngine.
    #
    # This class will be retired in Phase 2. ModifierRenderer (above)
    # will be preserved as a utility called BY UnifiedPlaybackEngine.
    #
    # DO NOT CALL start() INDEPENDENTLY - See TASK_LEDGER.md

    Features:
    - Target FPS with frame timing
    - Modifier state management
    - Composition of multiple modifiers
    - SSOT output integration
    """

    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._running = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._renderer = ModifierRenderer()
        self._modifier_states: Dict[str, ModifierState] = {}

        # Current render job
        self._current_job: Optional[Dict] = None
        self._job_lock = threading.Lock()

        # Output callback
        self._send_callback: Optional[Callable] = None

        # Stats
        self._frame_count = 0
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._actual_fps = 0.0

    def set_output_callback(self, callback: Callable[[int, Dict[int, int]], None]):
        """Set callback for sending rendered frames: callback(universe, channels)"""
        self._send_callback = callback

    def start(self):
        """
        Start the render loop.

        # ⚠️ AUTHORITY VIOLATION WARNING ⚠️
        # This method spawns an independent render thread, violating
        # AETHER Hard Rule 1.1. Only UnifiedPlaybackEngine should own timing.
        # See TASK-0004 in TASK_LEDGER.md
        """
        # PHASE 1 GUARD: Log violation when this is called independently
        import logging
        logging.warning(
            "⚠️ AUTHORITY VIOLATION: RenderEngine.start() called independently. "
            "This violates AETHER Hard Rule 1.1 - Only UnifiedPlaybackEngine "
            "should own playback timing. See TASK-0004 in TASK_LEDGER.md"
        )

        if self._running:
            return

        self._running = True
        self._stop_flag.clear()
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._last_frame_time = self._start_time

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        print(f"🎬 RenderEngine started at {self.target_fps} FPS")

    def stop(self):
        """Stop the render loop"""
        if not self._running:
            return

        self._stop_flag.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._current_job = None
        self._modifier_states.clear()
        print("⏹️ RenderEngine stopped")

    def render_look(
        self,
        look_id: str,
        channels: Dict[str, int],
        modifiers: List[Dict],
        universes: List[int],
        seed: Optional[int] = None,
    ):
        """
        Start rendering a Look with modifiers.

        Args:
            look_id: Unique identifier for this render job
            channels: Base channel values {"1": 255, "2": 128, ...}
            modifiers: List of modifier dicts with type, params, enabled
            universes: Target universes to output to
            seed: Random seed for deterministic output (default: based on look_id)
        """
        if seed is None:
            seed = hash(look_id) & 0xFFFFFFFF

        # Convert channel keys to int, handling universe:channel format
        int_channels = {}
        for k, v in channels.items():
            ks = str(k)
            if ':' in ks:
                _, ch = ks.split(':',1)
                int_channels[int(ch)] = v
            else:
                int_channels[int(ks)] = v

        # Create modifier states
        new_states = {}
        for mod in modifiers:
            if not mod.get("enabled", True):
                continue
            mod_id = mod.get("id", f"mod_{id(mod)}")
            mod_type = mod.get("type", "")

            # Create seeded RNG for this modifier
            mod_seed = seed + hash(mod_id) & 0xFFFFFFFF
            rng = random.Random(mod_seed)

            new_states[mod_id] = ModifierState(
                modifier_id=mod_id,
                modifier_type=mod_type,
                random_state=rng,
            )

        with self._job_lock:
            self._current_job = {
                "look_id": look_id,
                "channels": int_channels,
                "modifiers": modifiers,
                "universes": universes,
                "seed": seed,
            }
            self._modifier_states = new_states
            self._frame_count = 0
            self._start_time = time.monotonic()

    def stop_rendering(self):
        """Stop current render job"""
        with self._job_lock:
            self._current_job = None
            self._modifier_states.clear()

    def get_status(self) -> Dict:
        """Get current render status"""
        with self._job_lock:
            job = self._current_job

        return {
            "running": self._running,
            "rendering": job is not None,
            "look_id": job.get("look_id") if job else None,
            "frame_count": self._frame_count,
            "target_fps": self.target_fps,
            "actual_fps": round(self._actual_fps, 1),
            "modifier_count": len(self._modifier_states),
        }

    def _render_loop(self):
        """Main render loop - runs at target FPS"""
        while not self._stop_flag.is_set():
            frame_start = time.monotonic()

            # Get current job
            with self._job_lock:
                job = self._current_job
                states = self._modifier_states

            if job:
                try:
                    self._render_frame(job, states, frame_start)
                except Exception as e:
                    print(f"❌ Render error: {e}")

            # Frame timing
            self._frame_count += 1
            frame_end = time.monotonic()
            frame_duration = frame_end - frame_start

            # Calculate actual FPS
            total_elapsed = frame_end - self._start_time
            if total_elapsed > 0:
                self._actual_fps = self._frame_count / total_elapsed

            # Sleep to maintain target FPS
            sleep_time = self.frame_interval - frame_duration
            if sleep_time > 0:
                self._stop_flag.wait(sleep_time)

            self._last_frame_time = frame_end

    def _render_frame(
        self,
        job: Dict,
        states: Dict[str, ModifierState],
        frame_time: float,
    ):
        """Render a single frame"""
        channels = job["channels"]
        modifiers = job["modifiers"]
        universes = job["universes"]
        seed = job["seed"]

        # Create time context
        elapsed = frame_time - self._start_time
        delta = frame_time - self._last_frame_time if self._last_frame_time > 0 else self.frame_interval

        time_ctx = TimeContext(
            absolute_time=frame_time,
            delta_time=delta,
            elapsed_time=elapsed,
            frame_number=self._frame_count,
            seed=seed,
        )

        # Start with base channels
        result_channels = dict(channels)

        # Apply each modifier in sequence
        for mod in modifiers:
            if not mod.get("enabled", True):
                continue

            mod_id = mod.get("id", f"mod_{id(mod)}")
            mod_type = mod.get("type", "")
            params = mod.get("params", {})

            state = states.get(mod_id)
            if not state:
                continue

            # Get merge mode for this modifier type
            merge_mode = DEFAULT_MERGE_MODES.get(mod_type, MergeMode.MULTIPLY)

            # Render modifier effect
            # For now, treat all channels as one fixture (index 0)
            # TODO: Support per-fixture rendering based on channel groups
            mod_result = self._renderer.render(
                modifier_type=mod_type,
                params=params,
                base_channels=result_channels,
                time_ctx=time_ctx,
                state=state,
                fixture_index=0,
                total_fixtures=1,
            )

            # Apply merge mode
            result_channels = self._apply_merge(
                result_channels, mod_result, merge_mode, mod_type
            )

        # Clamp all values to 0-255
        final_channels = {
            ch: max(0, min(255, int(val)))
            for ch, val in result_channels.items()
        }

        # Output to all target universes
        if self._send_callback:
            for universe in universes:
                self._send_callback(universe, final_channels)

    def _apply_merge(
        self,
        base: Dict[int, int],
        modifier_output: Dict[int, float],
        merge_mode: MergeMode,
        mod_type: str,
    ) -> Dict[int, float]:
        """Apply merge mode to combine modifier output with base"""
        result = {}

        for ch, base_val in base.items():
            mod_val = modifier_output.get(ch, 1.0 if merge_mode == MergeMode.MULTIPLY else base_val)

            if merge_mode == MergeMode.MULTIPLY:
                # Multiply: mod_val is 0.0-1.0 multiplier
                result[ch] = base_val * mod_val

            elif merge_mode == MergeMode.ADD:
                # Add: mod_val is added to base (clamped later)
                result[ch] = base_val + mod_val

            elif merge_mode == MergeMode.REPLACE:
                # Replace: mod_val completely replaces base
                # For rainbow, mod_val is already 0-255
                result[ch] = mod_val

            elif merge_mode == MergeMode.MAX:
                # HTP: take maximum
                result[ch] = max(base_val, mod_val)

            elif merge_mode == MergeMode.MIN:
                # Take minimum
                result[ch] = min(base_val, mod_val)

            elif merge_mode == MergeMode.BLEND:
                # Blend: lerp between base and mod based on mod intensity
                blend_factor = 0.5
                result[ch] = base_val + (mod_val - base_val) * blend_factor

            elif merge_mode == MergeMode.DEPTH_BLEND:
                # Depth blend: mod_val is target 0-255, blend by depth param
                # depth is applied in _apply_modifier_to_attrs for fixture rendering
                # For channel-based rendering, treat like replace
                result[ch] = mod_val

            else:
                result[ch] = base_val

        return result


# ============================================================
# Convenience Functions
# ============================================================

def render_look_frame(
    channels: Dict[str, int],
    modifiers: List[Dict],
    elapsed_time: float,
    seed: int = 0,
    fixture_index: int = 0,
    total_fixtures: int = 1,
) -> Dict[int, int]:
    """
    Render a single frame for a Look - stateless convenience function.

    For one-shot rendering without the full engine.
    Note: For continuous rendering, use RenderEngine for proper state management.
    """
    renderer = ModifierRenderer()

    # Convert channel keys to int
    int_channels = {int(k): v for k, v in channels.items()}
    result = dict(int_channels)

    # Create time context
    time_ctx = TimeContext(
        absolute_time=time.time(),
        delta_time=1.0/30,
        elapsed_time=elapsed_time,
        frame_number=int(elapsed_time * 30),
        seed=seed,
    )

    # Apply each modifier
    for mod in modifiers:
        if not mod.get("enabled", True):
            continue

        mod_id = mod.get("id", f"mod_{id(mod)}")
        mod_type = mod.get("type", "")
        params = mod.get("params", {})

        # Create temporary state
        mod_seed = seed + hash(mod_id) & 0xFFFFFFFF
        state = ModifierState(
            modifier_id=mod_id,
            modifier_type=mod_type,
            random_state=random.Random(mod_seed),
        )

        merge_mode = DEFAULT_MERGE_MODES.get(mod_type, MergeMode.MULTIPLY)

        mod_result = renderer.render(
            modifier_type=mod_type,
            params=params,
            base_channels=result,
            time_ctx=time_ctx,
            state=state,
            fixture_index=fixture_index,
            total_fixtures=total_fixtures,
        )

        # Apply merge
        for ch in result:
            mod_val = mod_result.get(ch, 1.0 if merge_mode == MergeMode.MULTIPLY else result[ch])

            if merge_mode == MergeMode.MULTIPLY:
                result[ch] = result[ch] * mod_val
            elif merge_mode == MergeMode.REPLACE:
                result[ch] = mod_val
            elif merge_mode == MergeMode.ADD:
                result[ch] = result[ch] + mod_val
            elif merge_mode == MergeMode.MAX:
                result[ch] = max(result[ch], mod_val)

    # Clamp to 0-255
    return {ch: max(0, min(255, int(val))) for ch, val in result.items()}


# ============================================================
# Global Engine Instance
# ============================================================

render_engine = RenderEngine(target_fps=30)
