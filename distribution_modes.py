"""
Distribution Modes Module - Per-Fixture Effect Distribution

This module implements distribution modes for AETHER's fixture-centric architecture.
Distribution modes control how modifiers are applied across multiple fixtures.

Key Concepts:
- SYNCED: All fixtures receive identical modifier values (default)
- INDEXED: Modifier values scaled by fixture index (0 to N-1)
- PHASED: Time offset applied per fixture (wave effects)
- PIXELATED: Unique values per fixture (individual control)
- RANDOM: Deterministic random variation per fixture
- GROUPED: Same value per group (future)

The distribution system enables effects like:
- Rainbow chase across multiple fixtures (PHASED)
- Wave patterns traveling through fixtures (INDEXED)
- Individual twinkle/sparkle per fixture (PIXELATED)
- Synchronized breathing across all fixtures (SYNCED)

Version: 1.0.0
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum


# ============================================================
# Distribution Mode Enum
# ============================================================

class DistributionMode(Enum):
    """
    How a modifier's effect is distributed across fixtures.

    Each mode changes how the modifier's output varies between fixtures:
    - SYNCED: Identical output for all fixtures
    - INDEXED: Output scaled by fixture index
    - PHASED: Time offset per fixture
    - PIXELATED: Unique output per fixture
    - RANDOM: Random variation per fixture (deterministic)
    - GROUPED: Same per group (requires group configuration)
    """
    SYNCED = "synced"         # All fixtures identical (default)
    INDEXED = "indexed"       # Scaled by fixture index (0..1)
    PHASED = "phased"         # Time offset per fixture
    PIXELATED = "pixelated"   # Unique per fixture (chases, waves)
    RANDOM = "random"         # Deterministic random per fixture
    GROUPED = "grouped"       # Same per group (stub for future)


# ============================================================
# Distribution Configuration
# ============================================================

@dataclass
class DistributionConfig:
    """
    Configuration for how a modifier distributes across fixtures.

    This is attached to each Modifier and controls how the modifier's
    effect varies across the fixture set.
    """
    mode: DistributionMode = DistributionMode.SYNCED

    # Phase offset between fixtures (for PHASED mode)
    # Range: 0.0 to 1.0 (fraction of modifier's cycle per fixture)
    phase_offset: float = 0.0

    # Random seed for RANDOM mode (deterministic randomness)
    # If None, uses the session seed
    random_seed: Optional[int] = None

    # Index scaling for INDEXED mode
    # Scales the 0..1 index range
    index_scale: float = 1.0

    # Index offset for INDEXED mode
    # Shifts the starting index
    index_offset: float = 0.0

    # Reverse direction for PHASED/INDEXED
    reverse: bool = False

    # Group ID for GROUPED mode (future)
    group_id: Optional[str] = None

    # Custom mapping function name (for advanced use)
    custom_mapper: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mode": self.mode.value,
            "phase_offset": self.phase_offset,
            "random_seed": self.random_seed,
            "index_scale": self.index_scale,
            "index_offset": self.index_offset,
            "reverse": self.reverse,
            "group_id": self.group_id,
            "custom_mapper": self.custom_mapper,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributionConfig":
        """Create from dictionary"""
        mode_str = data.get("mode", "synced")
        try:
            mode = DistributionMode(mode_str)
        except ValueError:
            mode = DistributionMode.SYNCED

        return cls(
            mode=mode,
            phase_offset=data.get("phase_offset", 0.0),
            random_seed=data.get("random_seed"),
            index_scale=data.get("index_scale", 1.0),
            index_offset=data.get("index_offset", 0.0),
            reverse=data.get("reverse", False),
            group_id=data.get("group_id"),
            custom_mapper=data.get("custom_mapper"),
        )


# ============================================================
# Distribution Calculator
# ============================================================

class DistributionCalculator:
    """
    Calculates distribution parameters for each fixture.

    Given a DistributionConfig, fixture index, and total fixtures,
    computes the modifier parameters that should be applied.
    """

    def __init__(self, session_seed: int = 0):
        self.session_seed = session_seed
        self._random_cache: Dict[str, float] = {}

    def get_fixture_phase(
        self,
        config: DistributionConfig,
        fixture_index: int,
        total_fixtures: int,
        base_phase: float = 0.0
    ) -> float:
        """
        Calculate the phase for a specific fixture.

        Returns a phase value (0.0 to 1.0) that can be added to
        the modifier's time-based phase calculation.

        Args:
            config: Distribution configuration
            fixture_index: Index of this fixture (0 to total-1)
            total_fixtures: Total number of fixtures
            base_phase: Base phase from modifier (optional)

        Returns:
            Phase value (0.0 to 1.0)
        """
        if total_fixtures <= 1:
            return base_phase

        # Normalize index to 0-1 range
        norm_index = fixture_index / (total_fixtures - 1) if total_fixtures > 1 else 0.0

        # Apply reverse if configured
        if config.reverse:
            norm_index = 1.0 - norm_index

        # Apply scaling and offset for INDEXED mode
        norm_index = (norm_index * config.index_scale + config.index_offset) % 1.0

        if config.mode == DistributionMode.SYNCED:
            # All fixtures get the same phase
            return base_phase

        elif config.mode == DistributionMode.INDEXED:
            # Phase varies linearly with index
            return (base_phase + norm_index * config.phase_offset) % 1.0

        elif config.mode == DistributionMode.PHASED:
            # Each fixture offset by phase_offset
            fixture_phase = fixture_index * config.phase_offset
            return (base_phase + fixture_phase) % 1.0

        elif config.mode == DistributionMode.PIXELATED:
            # Each fixture has unique phase based on index
            # Creates distinct "pixels" that don't blend
            return (base_phase + norm_index) % 1.0

        elif config.mode == DistributionMode.RANDOM:
            # Deterministic random phase per fixture
            seed = (config.random_seed or self.session_seed) + fixture_index
            rng = random.Random(seed)
            random_offset = rng.random()
            return (base_phase + random_offset) % 1.0

        elif config.mode == DistributionMode.GROUPED:
            # Group-based phase (stub - needs group system)
            # For now, behaves like SYNCED
            return base_phase

        return base_phase

    def get_fixture_multiplier(
        self,
        config: DistributionConfig,
        fixture_index: int,
        total_fixtures: int,
    ) -> float:
        """
        Calculate a multiplier for a specific fixture.

        This can be used to scale modifier intensity per fixture.

        Returns:
            Multiplier value (0.0 to 1.0)
        """
        if total_fixtures <= 1:
            return 1.0

        # Normalize index to 0-1 range
        norm_index = fixture_index / (total_fixtures - 1) if total_fixtures > 1 else 0.0

        if config.reverse:
            norm_index = 1.0 - norm_index

        if config.mode == DistributionMode.SYNCED:
            return 1.0

        elif config.mode == DistributionMode.INDEXED:
            # Linear scaling by index
            return (norm_index * config.index_scale + config.index_offset) % 1.0

        elif config.mode == DistributionMode.RANDOM:
            # Deterministic random multiplier
            seed = (config.random_seed or self.session_seed) + fixture_index + 1000
            rng = random.Random(seed)
            return rng.random()

        return 1.0

    def get_fixture_seed(
        self,
        config: DistributionConfig,
        fixture_index: int,
    ) -> int:
        """
        Calculate a unique seed for a fixture.

        Used for per-fixture random effects (twinkle, flicker).
        """
        base_seed = config.random_seed or self.session_seed
        return (base_seed + fixture_index * 31337) & 0xFFFFFFFF


# ============================================================
# Distribution Presets
# ============================================================

DISTRIBUTION_PRESETS: Dict[str, DistributionConfig] = {
    # Basic modes
    "synced": DistributionConfig(mode=DistributionMode.SYNCED),

    "indexed": DistributionConfig(mode=DistributionMode.INDEXED, phase_offset=1.0),

    "phased": DistributionConfig(mode=DistributionMode.PHASED, phase_offset=0.1),

    "pixelated": DistributionConfig(mode=DistributionMode.PIXELATED),

    "random": DistributionConfig(mode=DistributionMode.RANDOM),

    # Chase presets
    "chase_forward": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.15,
        reverse=False
    ),

    "chase_backward": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.15,
        reverse=True
    ),

    "chase_fast": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.25
    ),

    "chase_slow": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.05
    ),

    # Rainbow presets
    "rainbow_spread": DistributionConfig(
        mode=DistributionMode.INDEXED,
        phase_offset=1.0,  # Full hue spread across fixtures
        index_scale=1.0
    ),

    "rainbow_half": DistributionConfig(
        mode=DistributionMode.INDEXED,
        phase_offset=0.5,  # Half hue spread
        index_scale=0.5
    ),

    # Wave presets
    "wave_gentle": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.08
    ),

    "wave_tight": DistributionConfig(
        mode=DistributionMode.PHASED,
        phase_offset=0.3
    ),

    # Twinkle presets
    "twinkle_random": DistributionConfig(
        mode=DistributionMode.RANDOM
    ),

    "twinkle_sequential": DistributionConfig(
        mode=DistributionMode.PIXELATED
    ),
}


def get_distribution_preset(name: str) -> Optional[DistributionConfig]:
    """Get a distribution preset by name"""
    preset = DISTRIBUTION_PRESETS.get(name)
    if preset:
        # Return a copy to prevent modification
        return DistributionConfig.from_dict(preset.to_dict())
    return None


def list_distribution_presets() -> List[Dict[str, Any]]:
    """List all available distribution presets"""
    return [
        {"name": name, "config": config.to_dict()}
        for name, config in DISTRIBUTION_PRESETS.items()
    ]


# ============================================================
# Distribution Mode Compatibility
# ============================================================

# Which modifier types support which distribution modes
MODIFIER_DISTRIBUTION_SUPPORT: Dict[str, List[DistributionMode]] = {
    "pulse": [
        DistributionMode.SYNCED,
        DistributionMode.PHASED,
        DistributionMode.RANDOM,
    ],
    "strobe": [
        DistributionMode.SYNCED,
        DistributionMode.PHASED,
        DistributionMode.RANDOM,
    ],
    "flicker": [
        DistributionMode.SYNCED,
        DistributionMode.RANDOM,
        DistributionMode.PIXELATED,
    ],
    "wave": [
        DistributionMode.SYNCED,
        DistributionMode.INDEXED,
        DistributionMode.PHASED,
        DistributionMode.PIXELATED,
    ],
    "rainbow": [
        DistributionMode.SYNCED,
        DistributionMode.INDEXED,
        DistributionMode.PHASED,
    ],
    "twinkle": [
        DistributionMode.SYNCED,
        DistributionMode.RANDOM,
        DistributionMode.PIXELATED,
    ],
}


def get_supported_distributions(modifier_type: str) -> List[DistributionMode]:
    """Get list of distribution modes supported by a modifier type"""
    return MODIFIER_DISTRIBUTION_SUPPORT.get(modifier_type, [DistributionMode.SYNCED])


def is_distribution_supported(modifier_type: str, mode: DistributionMode) -> bool:
    """Check if a distribution mode is supported for a modifier type"""
    supported = get_supported_distributions(modifier_type)
    return mode in supported


# ============================================================
# Distribution Suggestions (for AI Advisor)
# ============================================================

def suggest_distribution_for_effect(
    modifier_type: str,
    fixture_count: int,
    effect_intent: str = None
) -> DistributionConfig:
    """
    Suggest an appropriate distribution mode based on modifier and fixtures.

    This provides intelligent defaults for common scenarios.

    Args:
        modifier_type: The modifier type (pulse, wave, rainbow, etc.)
        fixture_count: Number of fixtures in the selection
        effect_intent: Optional hint about desired effect (chase, sync, spread)

    Returns:
        Suggested DistributionConfig
    """
    # Default to SYNCED for single fixtures or unknown types
    if fixture_count <= 1:
        return DistributionConfig(mode=DistributionMode.SYNCED)

    # Intent-based suggestions
    if effect_intent:
        intent_lower = effect_intent.lower()
        if "chase" in intent_lower:
            return get_distribution_preset("chase_forward") or DistributionConfig(
                mode=DistributionMode.PHASED,
                phase_offset=0.15
            )
        elif "spread" in intent_lower or "rainbow" in intent_lower:
            return get_distribution_preset("rainbow_spread") or DistributionConfig(
                mode=DistributionMode.INDEXED,
                phase_offset=1.0
            )
        elif "random" in intent_lower:
            return DistributionConfig(mode=DistributionMode.RANDOM)

    # Type-based suggestions for multiple fixtures
    if modifier_type == "wave":
        # Waves naturally look better phased
        phase_offset = min(0.5, 2.0 / fixture_count)
        return DistributionConfig(mode=DistributionMode.PHASED, phase_offset=phase_offset)

    elif modifier_type == "rainbow":
        # Rainbow looks great spread across fixtures
        return DistributionConfig(mode=DistributionMode.INDEXED, phase_offset=1.0)

    elif modifier_type == "twinkle":
        # Twinkle should be random per fixture
        return DistributionConfig(mode=DistributionMode.RANDOM)

    elif modifier_type == "flicker":
        # Flicker can be per-fixture for realistic fire
        if fixture_count >= 3:
            return DistributionConfig(mode=DistributionMode.RANDOM)

    elif modifier_type == "pulse":
        # Pulse usually looks better synchronized
        return DistributionConfig(mode=DistributionMode.SYNCED)

    elif modifier_type == "strobe":
        # Strobe usually synchronized, but can chase
        return DistributionConfig(mode=DistributionMode.SYNCED)

    # Default fallback
    return DistributionConfig(mode=DistributionMode.SYNCED)


# ============================================================
# Utility Functions
# ============================================================

def calculate_phase_array(
    config: DistributionConfig,
    total_fixtures: int,
    base_time: float,
    cycle_duration: float = 1.0
) -> List[float]:
    """
    Calculate phase values for all fixtures at once.

    Returns a list of phase values (0.0 to 1.0) for each fixture.

    Args:
        config: Distribution configuration
        total_fixtures: Number of fixtures
        base_time: Current time in seconds
        cycle_duration: Duration of one effect cycle

    Returns:
        List of phase values, one per fixture
    """
    calculator = DistributionCalculator()
    base_phase = (base_time / cycle_duration) % 1.0

    return [
        calculator.get_fixture_phase(config, i, total_fixtures, base_phase)
        for i in range(total_fixtures)
    ]


def apply_distribution_to_value(
    base_value: float,
    config: DistributionConfig,
    fixture_index: int,
    total_fixtures: int,
    time_ctx: Any = None
) -> float:
    """
    Apply distribution transformation to a single value.

    Used for simple value scaling based on distribution mode.

    Args:
        base_value: The original value to transform
        config: Distribution configuration
        fixture_index: Index of this fixture
        total_fixtures: Total fixtures

    Returns:
        Transformed value
    """
    if config.mode == DistributionMode.SYNCED:
        return base_value

    calculator = DistributionCalculator()
    multiplier = calculator.get_fixture_multiplier(config, fixture_index, total_fixtures)

    if config.mode == DistributionMode.INDEXED:
        return base_value * multiplier

    elif config.mode == DistributionMode.RANDOM:
        # Random variation around base value
        return base_value * (0.5 + multiplier)

    return base_value
