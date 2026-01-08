"""
Modifier Registry - Single Source of Truth for Effect Modifiers

This module provides:
- Schema definitions for all modifier types
- Parameter validation with detailed error messages
- Preset system with named parameter configurations
- Normalization (apply defaults for missing params)
- Unique ID generation for modifier instances

All modifiers are validated against their schemas before storage.
The UI can fetch schemas to dynamically generate controls.

Version: 1.0.0
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# ============================================================
# Modifier Type Enumeration
# ============================================================

class ModifierType(Enum):
    """Available modifier types"""
    PULSE = "pulse"
    STROBE = "strobe"
    FLICKER = "flicker"
    WAVE = "wave"
    RAINBOW = "rainbow"
    TWINKLE = "twinkle"


# ============================================================
# Parameter Schema Definitions
# ============================================================

@dataclass
class ParamSchema:
    """Schema for a single modifier parameter"""
    name: str
    type: str  # "float", "int", "bool", "enum"
    default: Any
    label: str
    description: str = ""
    min: Optional[float] = None  # For numeric types
    max: Optional[float] = None  # For numeric types
    step: Optional[float] = None  # UI step increment
    unit: Optional[str] = None  # Display unit (Hz, %, ms, etc.)
    options: Optional[List[str]] = None  # For enum types

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "default": self.default,
            "label": self.label,
        }
        if self.description:
            d["description"] = self.description
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        if self.unit:
            d["unit"] = self.unit
        if self.options:
            d["options"] = self.options
        return d


@dataclass
class ModifierSchema:
    """Complete schema for a modifier type"""
    type: str
    name: str
    description: str
    icon: str
    category: str  # "brightness", "color", "motion", "random"
    params: Dict[str, ParamSchema]
    presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "category": self.category,
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "presets": self.presets,
        }


# ============================================================
# Modifier Registry - Single Source of Truth
# ============================================================

class ModifierRegistry:
    """
    Central registry for all modifier types, schemas, and presets.

    This is the SSOT for modifier definitions. All validation,
    normalization, and preset application goes through this class.
    """

    def __init__(self):
        self._schemas: Dict[str, ModifierSchema] = {}
        self._register_builtin_modifiers()

    def _register_builtin_modifiers(self):
        """Register all built-in modifier types with their schemas and presets"""

        # ─────────────────────────────────────────────────────────
        # PULSE - Breathing brightness effect
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="pulse",
            name="Pulse",
            description="Smooth breathing brightness animation",
            icon="activity",
            category="brightness",
            params={
                "speed": ParamSchema(
                    name="speed",
                    type="float",
                    default=1.0,
                    label="Speed",
                    description="Pulse frequency in Hz",
                    min=0.1, max=5.0, step=0.1,
                    unit="Hz"
                ),
                "min_brightness": ParamSchema(
                    name="min_brightness",
                    type="int",
                    default=20,
                    label="Min Brightness",
                    description="Minimum brightness level",
                    min=0, max=100, step=5,
                    unit="%"
                ),
                "max_brightness": ParamSchema(
                    name="max_brightness",
                    type="int",
                    default=100,
                    label="Max Brightness",
                    description="Maximum brightness level",
                    min=0, max=100, step=5,
                    unit="%"
                ),
                "curve": ParamSchema(
                    name="curve",
                    type="enum",
                    default="sine",
                    label="Curve",
                    description="Animation curve type",
                    options=["sine", "linear", "ease-in", "ease-out", "ease-in-out"]
                ),
                "phase_offset": ParamSchema(
                    name="phase_offset",
                    type="float",
                    default=0.0,
                    label="Phase Offset",
                    description="Starting phase offset (0-1)",
                    min=0.0, max=1.0, step=0.1,
                    unit=""
                ),
            },
            presets={
                "gentle": {
                    "name": "Gentle Breath",
                    "description": "Slow, subtle breathing",
                    "params": {"speed": 0.3, "min_brightness": 60, "max_brightness": 100, "curve": "sine"}
                },
                "heartbeat": {
                    "name": "Heartbeat",
                    "description": "Quick double-pulse like a heartbeat",
                    "params": {"speed": 1.2, "min_brightness": 30, "max_brightness": 100, "curve": "ease-out"}
                },
                "dramatic": {
                    "name": "Dramatic",
                    "description": "Deep slow pulse for dramatic effect",
                    "params": {"speed": 0.5, "min_brightness": 5, "max_brightness": 100, "curve": "ease-in-out"}
                },
                "fast": {
                    "name": "Fast Pulse",
                    "description": "Rapid pulsing energy",
                    "params": {"speed": 3.0, "min_brightness": 40, "max_brightness": 100, "curve": "linear"}
                },
            }
        ))

        # ─────────────────────────────────────────────────────────
        # STROBE - On/off flash effect
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="strobe",
            name="Strobe",
            description="Sharp on/off flashing effect",
            icon="zap",
            category="brightness",
            params={
                "rate": ParamSchema(
                    name="rate",
                    type="float",
                    default=10.0,
                    label="Rate",
                    description="Flash frequency in Hz",
                    min=1.0, max=25.0, step=0.5,
                    unit="Hz"
                ),
                "duty_cycle": ParamSchema(
                    name="duty_cycle",
                    type="int",
                    default=50,
                    label="Duty Cycle",
                    description="Percentage of time light is on",
                    min=5, max=95, step=5,
                    unit="%"
                ),
                "attack": ParamSchema(
                    name="attack",
                    type="int",
                    default=0,
                    label="Attack",
                    description="Fade-in time for each flash",
                    min=0, max=50, step=5,
                    unit="ms"
                ),
                "decay": ParamSchema(
                    name="decay",
                    type="int",
                    default=0,
                    label="Decay",
                    description="Fade-out time for each flash",
                    min=0, max=50, step=5,
                    unit="ms"
                ),
            },
            presets={
                "slow": {
                    "name": "Slow Flash",
                    "description": "Slow deliberate flashing",
                    "params": {"rate": 2.0, "duty_cycle": 50, "attack": 0, "decay": 0}
                },
                "party": {
                    "name": "Party Strobe",
                    "description": "Classic party strobe effect",
                    "params": {"rate": 10.0, "duty_cycle": 30, "attack": 0, "decay": 0}
                },
                "lightning": {
                    "name": "Lightning",
                    "description": "Quick bright flashes like lightning",
                    "params": {"rate": 15.0, "duty_cycle": 15, "attack": 0, "decay": 10}
                },
                "警察": {
                    "name": "Police",
                    "description": "Fast emergency-style strobe",
                    "params": {"rate": 8.0, "duty_cycle": 40, "attack": 0, "decay": 0}
                },
            }
        ))

        # ─────────────────────────────────────────────────────────
        # FLICKER - Random brightness variance
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="flicker",
            name="Flicker",
            description="Random brightness variation for fire/candle effects",
            icon="flame",
            category="random",
            params={
                "intensity": ParamSchema(
                    name="intensity",
                    type="int",
                    default=20,
                    label="Intensity",
                    description="Amount of brightness variation",
                    min=5, max=50, step=5,
                    unit="%"
                ),
                "speed": ParamSchema(
                    name="speed",
                    type="float",
                    default=3.0,
                    label="Speed",
                    description="How fast the flickering occurs",
                    min=0.5, max=15.0, step=0.5,
                    unit="Hz"
                ),
                "smoothing": ParamSchema(
                    name="smoothing",
                    type="int",
                    default=30,
                    label="Smoothing",
                    description="How smooth the transitions are",
                    min=0, max=100, step=10,
                    unit="%"
                ),
                "min_brightness": ParamSchema(
                    name="min_brightness",
                    type="int",
                    default=20,
                    label="Min Brightness",
                    description="Floor brightness level",
                    min=0, max=80, step=5,
                    unit="%"
                ),
            },
            presets={
                "candle": {
                    "name": "Candle",
                    "description": "Gentle candle flame flicker",
                    "params": {"intensity": 15, "speed": 2.5, "smoothing": 50, "min_brightness": 40}
                },
                "fire": {
                    "name": "Fire",
                    "description": "More intense fire effect",
                    "params": {"intensity": 35, "speed": 5.0, "smoothing": 20, "min_brightness": 20}
                },
                "torch": {
                    "name": "Torch",
                    "description": "Strong torch flame",
                    "params": {"intensity": 25, "speed": 4.0, "smoothing": 30, "min_brightness": 50}
                },
                "dying_ember": {
                    "name": "Dying Ember",
                    "description": "Low smoldering ember effect",
                    "params": {"intensity": 20, "speed": 1.5, "smoothing": 60, "min_brightness": 10}
                },
            }
        ))

        # ─────────────────────────────────────────────────────────
        # WAVE - Position-based brightness wave
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="wave",
            name="Wave",
            description="Brightness wave traveling across fixtures",
            icon="waves",
            category="motion",
            params={
                "speed": ParamSchema(
                    name="speed",
                    type="float",
                    default=1.0,
                    label="Speed",
                    description="Wave travel speed",
                    min=0.1, max=5.0, step=0.1,
                    unit="Hz"
                ),
                "width": ParamSchema(
                    name="width",
                    type="int",
                    default=3,
                    label="Width",
                    description="Number of fixtures in wave peak",
                    min=1, max=10, step=1,
                    unit="fixtures"
                ),
                "direction": ParamSchema(
                    name="direction",
                    type="enum",
                    default="forward",
                    label="Direction",
                    description="Wave travel direction",
                    options=["forward", "backward", "bounce", "center-out", "edges-in"]
                ),
                "shape": ParamSchema(
                    name="shape",
                    type="enum",
                    default="sine",
                    label="Shape",
                    description="Wave shape curve",
                    options=["sine", "triangle", "square", "sawtooth"]
                ),
                "min_brightness": ParamSchema(
                    name="min_brightness",
                    type="int",
                    default=0,
                    label="Min Brightness",
                    description="Brightness in wave trough",
                    min=0, max=50, step=5,
                    unit="%"
                ),
            },
            presets={
                "gentle_wave": {
                    "name": "Gentle Wave",
                    "description": "Slow rolling wave",
                    "params": {"speed": 0.5, "width": 4, "direction": "forward", "shape": "sine", "min_brightness": 20}
                },
                "chase": {
                    "name": "Chase",
                    "description": "Fast chasing lights",
                    "params": {"speed": 2.0, "width": 2, "direction": "forward", "shape": "triangle", "min_brightness": 0}
                },
                "breathe": {
                    "name": "Center Breathe",
                    "description": "Wave from center outward",
                    "params": {"speed": 0.8, "width": 3, "direction": "center-out", "shape": "sine", "min_brightness": 10}
                },
                "scanner": {
                    "name": "Scanner",
                    "description": "Knight Rider scanner effect",
                    "params": {"speed": 1.5, "width": 2, "direction": "bounce", "shape": "triangle", "min_brightness": 0}
                },
            }
        ))

        # ─────────────────────────────────────────────────────────
        # RAINBOW - Hue rotation effect
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="rainbow",
            name="Rainbow",
            description="Smooth color cycling through the spectrum",
            icon="palette",
            category="color",
            params={
                "speed": ParamSchema(
                    name="speed",
                    type="float",
                    default=0.2,
                    label="Speed",
                    description="Color cycle frequency",
                    min=0.02, max=2.0, step=0.02,
                    unit="Hz"
                ),
                "saturation": ParamSchema(
                    name="saturation",
                    type="int",
                    default=100,
                    label="Saturation",
                    description="Color saturation level",
                    min=50, max=100, step=5,
                    unit="%"
                ),
                "spread": ParamSchema(
                    name="spread",
                    type="int",
                    default=0,
                    label="Spread",
                    description="Phase offset between fixtures (0=sync)",
                    min=0, max=100, step=10,
                    unit="%"
                ),
                "hue_range": ParamSchema(
                    name="hue_range",
                    type="int",
                    default=360,
                    label="Hue Range",
                    description="Degrees of hue to cycle through",
                    min=60, max=360, step=30,
                    unit="°"
                ),
                "hue_offset": ParamSchema(
                    name="hue_offset",
                    type="int",
                    default=0,
                    label="Hue Offset",
                    description="Starting hue position",
                    min=0, max=360, step=15,
                    unit="°"
                ),
            },
            presets={
                "classic": {
                    "name": "Classic Rainbow",
                    "description": "Full spectrum color cycle",
                    "params": {"speed": 0.15, "saturation": 100, "spread": 0, "hue_range": 360, "hue_offset": 0}
                },
                "pastel": {
                    "name": "Pastel Rainbow",
                    "description": "Softer pastel colors",
                    "params": {"speed": 0.1, "saturation": 60, "spread": 0, "hue_range": 360, "hue_offset": 0}
                },
                "warm_cycle": {
                    "name": "Warm Cycle",
                    "description": "Red/orange/yellow only",
                    "params": {"speed": 0.2, "saturation": 100, "spread": 0, "hue_range": 90, "hue_offset": 0}
                },
                "cool_cycle": {
                    "name": "Cool Cycle",
                    "description": "Blue/cyan/purple only",
                    "params": {"speed": 0.2, "saturation": 100, "spread": 0, "hue_range": 120, "hue_offset": 180}
                },
                "chase_rainbow": {
                    "name": "Rainbow Chase",
                    "description": "Rainbow spread across fixtures",
                    "params": {"speed": 0.3, "saturation": 100, "spread": 50, "hue_range": 360, "hue_offset": 0}
                },
            }
        ))

        # ─────────────────────────────────────────────────────────
        # TWINKLE - Random sparkle effect
        # ─────────────────────────────────────────────────────────
        self.register(ModifierSchema(
            type="twinkle",
            name="Twinkle",
            description="Random twinkling sparkle effect",
            icon="sparkles",
            category="random",
            params={
                "density": ParamSchema(
                    name="density",
                    type="int",
                    default=30,
                    label="Density",
                    description="Percentage of fixtures twinkling at once",
                    min=5, max=80, step=5,
                    unit="%"
                ),
                "fade_time": ParamSchema(
                    name="fade_time",
                    type="int",
                    default=500,
                    label="Fade Time",
                    description="Duration of each twinkle fade",
                    min=50, max=2000, step=50,
                    unit="ms"
                ),
                "min_brightness": ParamSchema(
                    name="min_brightness",
                    type="int",
                    default=20,
                    label="Min Brightness",
                    description="Base brightness between twinkles",
                    min=0, max=50, step=5,
                    unit="%"
                ),
                "max_brightness": ParamSchema(
                    name="max_brightness",
                    type="int",
                    default=100,
                    label="Max Brightness",
                    description="Peak twinkle brightness",
                    min=50, max=100, step=5,
                    unit="%"
                ),
                "hold_time": ParamSchema(
                    name="hold_time",
                    type="int",
                    default=100,
                    label="Hold Time",
                    description="How long twinkle stays at peak",
                    min=0, max=500, step=25,
                    unit="ms"
                ),
            },
            presets={
                "stars": {
                    "name": "Starry Night",
                    "description": "Gentle starlight twinkle",
                    "params": {"density": 20, "fade_time": 800, "min_brightness": 30, "max_brightness": 100, "hold_time": 200}
                },
                "fairy": {
                    "name": "Fairy Lights",
                    "description": "Quick magical sparkles",
                    "params": {"density": 40, "fade_time": 300, "min_brightness": 40, "max_brightness": 100, "hold_time": 50}
                },
                "snow": {
                    "name": "Snow Sparkle",
                    "description": "Crisp white sparkles",
                    "params": {"density": 25, "fade_time": 400, "min_brightness": 50, "max_brightness": 100, "hold_time": 100}
                },
                "fireflies": {
                    "name": "Fireflies",
                    "description": "Slow organic glow",
                    "params": {"density": 15, "fade_time": 1500, "min_brightness": 5, "max_brightness": 80, "hold_time": 300}
                },
            }
        ))

    # ─────────────────────────────────────────────────────────
    # Registry Operations
    # ─────────────────────────────────────────────────────────

    def register(self, schema: ModifierSchema) -> None:
        """Register a modifier schema"""
        self._schemas[schema.type] = schema

    def get_schema(self, modifier_type: str) -> Optional[ModifierSchema]:
        """Get schema for a modifier type"""
        return self._schemas.get(modifier_type)

    def get_all_schemas(self) -> Dict[str, ModifierSchema]:
        """Get all registered schemas"""
        return self._schemas.copy()

    def get_types(self) -> List[str]:
        """Get list of all registered modifier types"""
        return list(self._schemas.keys())

    def get_categories(self) -> Dict[str, List[str]]:
        """Get modifier types grouped by category"""
        categories: Dict[str, List[str]] = {}
        for mod_type, schema in self._schemas.items():
            cat = schema.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(mod_type)
        return categories

    # ─────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────

    def validate(self, modifier: dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a modifier against its schema.
        Returns (is_valid, error_message).
        """
        # Check type field
        mod_type = modifier.get("type")
        if not mod_type:
            return False, "Modifier missing 'type' field"

        schema = self._schemas.get(mod_type)
        if not schema:
            valid_types = ", ".join(self._schemas.keys())
            return False, f"Unknown modifier type: '{mod_type}'. Valid types: {valid_types}"

        # Check id field (optional but if present must be string)
        mod_id = modifier.get("id")
        if mod_id is not None and not isinstance(mod_id, str):
            return False, "Modifier 'id' must be a string"

        # Check enabled field (optional but if present must be bool)
        enabled = modifier.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            return False, "Modifier 'enabled' must be a boolean"

        # Validate params
        params = modifier.get("params", {})
        if not isinstance(params, dict):
            return False, "Modifier 'params' must be an object"

        # Check for unknown params
        valid_params = set(schema.params.keys())
        provided_params = set(params.keys())
        unknown_params = provided_params - valid_params
        if unknown_params:
            return False, f"Unknown parameters for {mod_type}: {', '.join(unknown_params)}"

        # Validate each parameter
        for param_name, param_schema in schema.params.items():
            value = params.get(param_name)

            # Skip if not provided (will use default)
            if value is None:
                continue

            # Type-specific validation
            error = self._validate_param(param_name, value, param_schema)
            if error:
                return False, error

        return True, None

    def _validate_param(self, name: str, value: Any, schema: ParamSchema) -> Optional[str]:
        """Validate a single parameter value. Returns error message or None."""
        param_type = schema.type

        if param_type == "float":
            if not isinstance(value, (int, float)):
                return f"Parameter '{name}' must be a number, got {type(value).__name__}"
            if schema.min is not None and value < schema.min:
                return f"Parameter '{name}' must be >= {schema.min}, got {value}"
            if schema.max is not None and value > schema.max:
                return f"Parameter '{name}' must be <= {schema.max}, got {value}"

        elif param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"Parameter '{name}' must be an integer, got {type(value).__name__}"
            if schema.min is not None and value < schema.min:
                return f"Parameter '{name}' must be >= {int(schema.min)}, got {value}"
            if schema.max is not None and value > schema.max:
                return f"Parameter '{name}' must be <= {int(schema.max)}, got {value}"

        elif param_type == "bool":
            if not isinstance(value, bool):
                return f"Parameter '{name}' must be a boolean, got {type(value).__name__}"

        elif param_type == "enum":
            if not schema.options:
                return f"Parameter '{name}' has no valid options defined"
            if value not in schema.options:
                return f"Parameter '{name}' must be one of: {', '.join(schema.options)}. Got: '{value}'"

        return None

    # ─────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────

    def normalize(self, modifier: dict) -> dict:
        """
        Normalize a modifier by:
        - Adding missing 'id' field
        - Applying defaults for missing parameters
        - Setting 'enabled' to True if not present
        - Resolving preset_id to actual params

        Returns a complete modifier with all fields populated.
        """
        mod_type = modifier.get("type")
        schema = self._schemas.get(mod_type)

        if not schema:
            return modifier  # Can't normalize unknown type

        # Generate ID if not present
        mod_id = modifier.get("id") or self.generate_id()

        # Check for preset
        preset_id = modifier.get("preset_id")
        if preset_id and preset_id in schema.presets:
            # Start with preset params
            base_params = schema.presets[preset_id]["params"].copy()
            # Override with any explicitly provided params
            provided_params = modifier.get("params", {})
            base_params.update(provided_params)
            params = base_params
        else:
            params = modifier.get("params", {}).copy()

        # Apply defaults for missing params
        for param_name, param_schema in schema.params.items():
            if param_name not in params:
                params[param_name] = param_schema.default

        return {
            "id": mod_id,
            "type": mod_type,
            "enabled": modifier.get("enabled", True),
            "params": params,
            "preset_id": preset_id,  # Keep track of which preset was used
        }

    def generate_id(self) -> str:
        """Generate a unique modifier ID"""
        return f"mod_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    # ─────────────────────────────────────────────────────────
    # Presets
    # ─────────────────────────────────────────────────────────

    def get_presets(self, modifier_type: str) -> Dict[str, dict]:
        """Get all presets for a modifier type"""
        schema = self._schemas.get(modifier_type)
        if not schema:
            return {}
        return schema.presets

    def get_preset(self, modifier_type: str, preset_id: str) -> Optional[dict]:
        """Get a specific preset"""
        schema = self._schemas.get(modifier_type)
        if not schema:
            return None
        return schema.presets.get(preset_id)

    def apply_preset(self, modifier: dict, preset_id: str) -> dict:
        """Apply a preset to a modifier, returning updated modifier"""
        mod_type = modifier.get("type")
        preset = self.get_preset(mod_type, preset_id)

        if not preset:
            return modifier

        # Merge preset params with modifier
        new_modifier = modifier.copy()
        new_modifier["preset_id"] = preset_id
        new_modifier["params"] = {
            **modifier.get("params", {}),
            **preset["params"]
        }
        return self.normalize(new_modifier)

    def create_from_preset(self, modifier_type: str, preset_id: str) -> Optional[dict]:
        """Create a new modifier from a preset"""
        preset = self.get_preset(modifier_type, preset_id)
        if not preset:
            return None

        return self.normalize({
            "type": modifier_type,
            "preset_id": preset_id,
            "params": preset["params"].copy()
        })

    # ─────────────────────────────────────────────────────────
    # API Output
    # ─────────────────────────────────────────────────────────

    def to_api_response(self) -> dict:
        """
        Generate complete API response with all schemas and presets.
        Used by GET /api/modifiers/schemas endpoint.
        """
        return {
            "version": "1.0.0",
            "types": self.get_types(),
            "categories": self.get_categories(),
            "schemas": {k: v.to_dict() for k, v in self._schemas.items()},
        }


# ============================================================
# Global Registry Instance
# ============================================================

# Singleton instance - import this in other modules
modifier_registry = ModifierRegistry()


# ============================================================
# Convenience Functions (for backward compatibility)
# ============================================================

def validate_modifier(modifier: dict) -> Tuple[bool, Optional[str]]:
    """Validate a modifier using the global registry"""
    return modifier_registry.validate(modifier)


def normalize_modifier(modifier: dict) -> dict:
    """Normalize a modifier using the global registry"""
    return modifier_registry.normalize(modifier)


def get_modifier_schemas() -> dict:
    """Get all modifier schemas for API response"""
    return modifier_registry.to_api_response()


def get_modifier_presets(modifier_type: str) -> Dict[str, dict]:
    """Get presets for a modifier type"""
    return modifier_registry.get_presets(modifier_type)
