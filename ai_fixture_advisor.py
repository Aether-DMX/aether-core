"""
AI Fixture Advisor - Intelligent Suggestions for Fixture-Centric Playback

This module provides AI-powered suggestions for distribution modes and
fixture configurations. Key principle: AI SUGGESTS, never auto-applies.

Key Features:
- Rule-based distribution mode suggestions
- Fixture selection recommendations
- Effect parameter optimization hints
- Confidence scores for all suggestions

Prime Directive:
    AI may suggest, never silently override.
    All suggestions require explicit user approval via Apply/Dismiss UI.

Version: 1.0.0
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


# ============================================================
# Suggestion Types
# ============================================================

class SuggestionType(Enum):
    """Types of AI suggestions"""
    DISTRIBUTION_MODE = "distribution_mode"
    FIXTURE_SELECTION = "fixture_selection"
    MODIFIER_PARAMS = "modifier_params"
    EFFECT_COMBINATION = "effect_combination"
    TIMING_ADJUSTMENT = "timing_adjustment"
    COLOR_PALETTE = "color_palette"
    TRANSITION_SMOOTHNESS = "transition_smoothness"


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# AI Suggestion Data Classes
# ============================================================

@dataclass
class AISuggestion:
    """
    A single AI-generated suggestion.

    IMPORTANT: Suggestions are never auto-applied.
    The 'applied' field tracks whether the user explicitly applied this.
    """
    # Unique ID for this suggestion
    suggestion_id: str

    # What kind of suggestion this is
    suggestion_type: SuggestionType

    # The actual suggestion content
    suggestion: Dict[str, Any]

    # Human-readable explanation
    reason: str

    # How confident the AI is (0.0 to 1.0)
    confidence: float

    # Priority level
    priority: SuggestionPriority = SuggestionPriority.MEDIUM

    # Whether user has explicitly applied this
    applied: bool = False

    # Whether user has dismissed this
    dismissed: bool = False

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "suggestion_id": self.suggestion_id,
            "suggestion_type": self.suggestion_type.value,
            "suggestion": self.suggestion,
            "reason": self.reason,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "applied": self.applied,
            "dismissed": self.dismissed,
            "created_at": self.created_at,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AISuggestion":
        """Create from dictionary"""
        return cls(
            suggestion_id=data["suggestion_id"],
            suggestion_type=SuggestionType(data["suggestion_type"]),
            suggestion=data["suggestion"],
            reason=data["reason"],
            confidence=data["confidence"],
            priority=SuggestionPriority(data.get("priority", "medium")),
            applied=data.get("applied", False),
            dismissed=data.get("dismissed", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
            context=data.get("context", {}),
        )


# ============================================================
# Fixture Context for AI Decisions
# ============================================================

@dataclass
class FixtureContext:
    """
    Context about fixtures for AI decision making.

    Provides fixture counts, types, groups, and other metadata
    needed for intelligent suggestions.
    """
    total_fixtures: int = 0
    fixture_types: Dict[str, int] = field(default_factory=dict)  # category -> count
    fixture_groups: Dict[str, List[str]] = field(default_factory=dict)  # group -> fixture_ids
    fixture_positions: List[str] = field(default_factory=list)  # Ordered fixture IDs
    has_moving_heads: bool = False
    has_rgb_fixtures: bool = False
    has_dimmer_only: bool = False

    @classmethod
    def from_fixtures(
        cls,
        fixtures: List[Any],  # List of FixtureInstance
        profiles: Dict[str, Any] = None  # profile_id -> FixtureProfile
    ) -> "FixtureContext":
        """Build context from fixture instances"""
        ctx = cls()
        ctx.total_fixtures = len(fixtures)
        ctx.fixture_positions = [f.fixture_id for f in fixtures]

        for fixture in fixtures:
            # Count by category if profile available
            if profiles and fixture.profile_id in profiles:
                profile = profiles[fixture.profile_id]
                category = profile.category
                ctx.fixture_types[category] = ctx.fixture_types.get(category, 0) + 1

                if category == "moving_head":
                    ctx.has_moving_heads = True
                elif category in ("par", "wash", "rgbw"):
                    ctx.has_rgb_fixtures = True
                elif category == "dimmer":
                    ctx.has_dimmer_only = True

            # Track groups
            if fixture.group:
                if fixture.group not in ctx.fixture_groups:
                    ctx.fixture_groups[fixture.group] = []
                ctx.fixture_groups[fixture.group].append(fixture.fixture_id)

        return ctx

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI prompts"""
        return {
            "total_fixtures": self.total_fixtures,
            "fixture_types": self.fixture_types,
            "group_count": len(self.fixture_groups),
            "groups": list(self.fixture_groups.keys()),
            "has_moving_heads": self.has_moving_heads,
            "has_rgb_fixtures": self.has_rgb_fixtures,
            "has_dimmer_only": self.has_dimmer_only,
        }


# ============================================================
# AI Fixture Advisor
# ============================================================

class AIFixtureAdvisor:
    """
    Provides intelligent suggestions for fixture-centric playback.

    Uses rule-based heuristics to suggest:
    - Distribution modes for modifiers
    - Fixture selections for effects
    - Parameter optimizations

    IMPORTANT: All suggestions require explicit user approval.
    This class NEVER auto-applies any changes.
    """

    def __init__(self):
        self._suggestion_counter = 0
        self._pending_suggestions: Dict[str, AISuggestion] = {}

    def _generate_id(self) -> str:
        """Generate unique suggestion ID"""
        self._suggestion_counter += 1
        return f"ai_suggestion_{self._suggestion_counter}_{int(datetime.now().timestamp())}"

    # ─────────────────────────────────────────────────────────
    # Context Methods
    # ─────────────────────────────────────────────────────────

    def get_fixture_context(
        self,
        fixtures: List[Any] = None,
        profiles: Dict[str, Any] = None
    ) -> FixtureContext:
        """
        Build fixture context for AI suggestions.

        Args:
            fixtures: List of FixtureInstance objects
            profiles: Dict of profile_id -> FixtureProfile

        Returns:
            FixtureContext with fixture metadata
        """
        if not fixtures:
            return FixtureContext()

        return FixtureContext.from_fixtures(fixtures, profiles)

    # ─────────────────────────────────────────────────────────
    # Distribution Suggestions
    # ─────────────────────────────────────────────────────────

    def suggest_distribution(
        self,
        modifier_type: str,
        modifier_params: Dict[str, Any],
        fixture_count: int,
        fixture_context: FixtureContext = None
    ) -> List[AISuggestion]:
        """
        Suggest distribution modes for a modifier based on fixtures.

        Rule-based suggestions:
        - Wave + 3+ fixtures -> PHASED (for traveling wave effect)
        - Rainbow + multiple fixtures -> INDEXED (spread colors across)
        - Twinkle + fixtures -> RANDOM (individual sparkle)
        - Pulse + fixtures -> SYNCED or PHASED depending on count

        Args:
            modifier_type: The modifier type (pulse, wave, rainbow, etc.)
            modifier_params: Current modifier parameters
            fixture_count: Number of fixtures selected
            fixture_context: Optional fixture context for smarter suggestions

        Returns:
            List of AISuggestion objects (never auto-applied)
        """
        suggestions = []

        # No suggestions for single fixtures
        if fixture_count <= 1:
            return suggestions

        # Wave modifier suggestions
        if modifier_type == "wave":
            if fixture_count >= 3:
                phase_offset = min(0.5, 2.0 / fixture_count)
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "phased",
                        "phase_offset": phase_offset,
                        "reason_detail": f"Traveling wave across {fixture_count} fixtures"
                    },
                    reason=f"Wave effects look best with PHASED distribution for {fixture_count}+ fixtures, creating a traveling wave pattern.",
                    confidence=0.85,
                    priority=SuggestionPriority.HIGH,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Rainbow modifier suggestions
        elif modifier_type == "rainbow":
            if fixture_count >= 2:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "indexed",
                        "phase_offset": 1.0,
                        "reason_detail": "Full rainbow spread across all fixtures"
                    },
                    reason=f"Rainbow with INDEXED distribution spreads the full color spectrum across your {fixture_count} fixtures.",
                    confidence=0.90,
                    priority=SuggestionPriority.HIGH,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

            if fixture_count >= 4:
                # Also suggest phased for rainbow chase
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "phased",
                        "phase_offset": 0.1,
                        "reason_detail": "Rainbow chase effect"
                    },
                    reason="Alternative: PHASED creates a rainbow chase effect where colors flow through fixtures.",
                    confidence=0.70,
                    priority=SuggestionPriority.MEDIUM,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Twinkle modifier suggestions
        elif modifier_type == "twinkle":
            suggestions.append(AISuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                suggestion={
                    "mode": "random",
                    "reason_detail": "Independent sparkle per fixture"
                },
                reason=f"Twinkle with RANDOM distribution creates independent sparkle timing per fixture for a natural starfield effect.",
                confidence=0.88,
                priority=SuggestionPriority.HIGH,
                context={"modifier_type": modifier_type, "fixture_count": fixture_count}
            ))

            if fixture_count >= 5:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "pixelated",
                        "reason_detail": "Unique twinkle pattern per fixture"
                    },
                    reason="Alternative: PIXELATED gives each fixture its own unique twinkle pattern.",
                    confidence=0.65,
                    priority=SuggestionPriority.LOW,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Flicker modifier suggestions
        elif modifier_type == "flicker":
            if fixture_count >= 3:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "random",
                        "reason_detail": "Realistic fire/candle simulation"
                    },
                    reason="Flicker with RANDOM distribution creates realistic fire simulation with independent variation per fixture.",
                    confidence=0.82,
                    priority=SuggestionPriority.MEDIUM,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Pulse modifier suggestions
        elif modifier_type == "pulse":
            if fixture_count >= 4:
                phase_offset = 0.15
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "phased",
                        "phase_offset": phase_offset,
                        "reason_detail": "Breathing wave effect"
                    },
                    reason=f"Pulse with PHASED distribution creates a breathing wave that travels across your {fixture_count} fixtures.",
                    confidence=0.72,
                    priority=SuggestionPriority.MEDIUM,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Strobe modifier suggestions
        elif modifier_type == "strobe":
            if fixture_count >= 3:
                phase_offset = 1.0 / fixture_count
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.DISTRIBUTION_MODE,
                    suggestion={
                        "mode": "phased",
                        "phase_offset": phase_offset,
                        "reason_detail": "Sequential strobe chase"
                    },
                    reason="Strobe with PHASED distribution creates a chase effect where each fixture strobes in sequence.",
                    confidence=0.75,
                    priority=SuggestionPriority.MEDIUM,
                    context={"modifier_type": modifier_type, "fixture_count": fixture_count}
                ))

        # Store suggestions for tracking
        for suggestion in suggestions:
            self._pending_suggestions[suggestion.suggestion_id] = suggestion

        return suggestions

    # ─────────────────────────────────────────────────────────
    # Fixture Selection Suggestions
    # ─────────────────────────────────────────────────────────

    def suggest_fixture_selection(
        self,
        intent: str,
        available_fixtures: List[Any],
        fixture_context: FixtureContext
    ) -> List[AISuggestion]:
        """
        Suggest fixture selections based on effect intent.

        Args:
            intent: User's intent (e.g., "stage wash", "chase", "accent")
            available_fixtures: List of available fixtures
            fixture_context: Context about fixtures

        Returns:
            List of AISuggestion for fixture selection
        """
        suggestions = []
        intent_lower = intent.lower()

        # Chase intent
        if "chase" in intent_lower:
            if fixture_context.total_fixtures >= 3:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.FIXTURE_SELECTION,
                    suggestion={
                        "selection": "all",
                        "order": "position",
                        "reason_detail": "All fixtures in position order for chase"
                    },
                    reason="For a chase effect, select all fixtures in position order for the best visual flow.",
                    confidence=0.85,
                    priority=SuggestionPriority.MEDIUM,
                    context={"intent": intent}
                ))

        # Stage wash intent
        elif "wash" in intent_lower or "fill" in intent_lower:
            if fixture_context.has_rgb_fixtures:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.FIXTURE_SELECTION,
                    suggestion={
                        "selection": "rgb_fixtures",
                        "filter": {"category": ["par", "wash", "rgbw"]},
                        "reason_detail": "RGB fixtures for color wash"
                    },
                    reason="Select RGB-capable fixtures (pars, washes) for a color stage wash.",
                    confidence=0.80,
                    priority=SuggestionPriority.MEDIUM,
                    context={"intent": intent}
                ))

        # Accent/spot intent
        elif "accent" in intent_lower or "spot" in intent_lower:
            if fixture_context.has_moving_heads:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.FIXTURE_SELECTION,
                    suggestion={
                        "selection": "moving_heads",
                        "filter": {"category": ["moving_head"]},
                        "reason_detail": "Moving heads for accent lighting"
                    },
                    reason="Moving heads are ideal for accent/spot lighting with their beam focus.",
                    confidence=0.82,
                    priority=SuggestionPriority.MEDIUM,
                    context={"intent": intent}
                ))

        # Group-based suggestions
        if fixture_context.fixture_groups:
            for group_name, fixture_ids in fixture_context.fixture_groups.items():
                if len(fixture_ids) >= 2:
                    suggestions.append(AISuggestion(
                        suggestion_id=self._generate_id(),
                        suggestion_type=SuggestionType.FIXTURE_SELECTION,
                        suggestion={
                            "selection": "group",
                            "group_name": group_name,
                            "fixture_ids": fixture_ids,
                            "reason_detail": f"Use '{group_name}' group ({len(fixture_ids)} fixtures)"
                        },
                        reason=f"Consider using the '{group_name}' group with {len(fixture_ids)} fixtures.",
                        confidence=0.60,
                        priority=SuggestionPriority.LOW,
                        context={"intent": intent, "group": group_name}
                    ))

        for suggestion in suggestions:
            self._pending_suggestions[suggestion.suggestion_id] = suggestion

        return suggestions

    # ─────────────────────────────────────────────────────────
    # Modifier Parameter Suggestions
    # ─────────────────────────────────────────────────────────

    def suggest_modifier_params(
        self,
        modifier_type: str,
        current_params: Dict[str, Any],
        fixture_count: int,
        effect_intent: str = None
    ) -> List[AISuggestion]:
        """
        Suggest optimized modifier parameters.

        Args:
            modifier_type: The modifier type
            current_params: Current parameter values
            fixture_count: Number of fixtures
            effect_intent: Optional intent hint

        Returns:
            List of AISuggestion for parameter changes
        """
        suggestions = []

        # Wave width optimization
        if modifier_type == "wave":
            current_width = current_params.get("width", 3)
            optimal_width = max(2, min(5, fixture_count // 2))

            if abs(current_width - optimal_width) > 1:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.MODIFIER_PARAMS,
                    suggestion={
                        "param": "width",
                        "current": current_width,
                        "suggested": optimal_width,
                        "reason_detail": f"Optimized for {fixture_count} fixtures"
                    },
                    reason=f"Wave width of {optimal_width} works better for {fixture_count} fixtures (currently {current_width}).",
                    confidence=0.70,
                    priority=SuggestionPriority.LOW,
                    context={"modifier_type": modifier_type}
                ))

        # Pulse speed for larger fixture counts
        if modifier_type == "pulse":
            current_speed = current_params.get("speed", 1.0)
            if fixture_count >= 6 and current_speed > 1.5:
                suggestions.append(AISuggestion(
                    suggestion_id=self._generate_id(),
                    suggestion_type=SuggestionType.MODIFIER_PARAMS,
                    suggestion={
                        "param": "speed",
                        "current": current_speed,
                        "suggested": 0.8,
                        "reason_detail": "Slower pulse for larger fixture groups"
                    },
                    reason=f"With {fixture_count} fixtures, a slower pulse (0.8 Hz) creates a more dramatic effect.",
                    confidence=0.65,
                    priority=SuggestionPriority.LOW,
                    context={"modifier_type": modifier_type}
                ))

        for suggestion in suggestions:
            self._pending_suggestions[suggestion.suggestion_id] = suggestion

        return suggestions

    # ─────────────────────────────────────────────────────────
    # Transition Smoothness Suggestions
    # ─────────────────────────────────────────────────────────

    def suggest_transition_time(
        self,
        effect_type: str,
        fixture_count: int,
        step_duration_ms: int = None,
        effect_intent: str = None
    ) -> List[AISuggestion]:
        """
        Suggest appropriate transition/crossfade times for effects.

        Different effects benefit from different transition smoothness:
        - Fast effects (strobe): instant transitions (0ms)
        - Waves/chases: smooth transitions (200-400ms)
        - Ambient/mood: very smooth (400-800ms)
        - Dramatic reveals: medium (200-300ms)

        Args:
            effect_type: Type of effect (wave, chase, pulse, etc.)
            fixture_count: Number of fixtures involved
            step_duration_ms: Duration of each step (if applicable)
            effect_intent: Optional hint about desired feel

        Returns:
            List of AISuggestion for transition times
        """
        suggestions = []

        # Base transition recommendations by effect type
        effect_transitions = {
            "wave": {"default": 300, "smooth": 400, "ultra_smooth": 600, "snappy": 100},
            "chase": {"default": 200, "smooth": 350, "ultra_smooth": 500, "snappy": 50},
            "rainbow": {"default": 300, "smooth": 500, "ultra_smooth": 700, "snappy": 100},
            "pulse": {"default": 400, "smooth": 600, "ultra_smooth": 800, "snappy": 150},
            "twinkle": {"default": 150, "smooth": 250, "ultra_smooth": 400, "snappy": 50},
            "flicker": {"default": 50, "smooth": 100, "ultra_smooth": 200, "snappy": 0},
            "strobe": {"default": 0, "smooth": 0, "ultra_smooth": 25, "snappy": 0},
        }

        timings = effect_transitions.get(effect_type, {"default": 200, "smooth": 350, "snappy": 50})

        # Adjust for fixture count (more fixtures = slightly longer transitions look better)
        fixture_factor = 1.0 + (fixture_count - 1) * 0.05 if fixture_count > 1 else 1.0
        fixture_factor = min(1.5, fixture_factor)  # Cap at 1.5x

        default_ms = int(timings["default"] * fixture_factor)
        smooth_ms = int(timings["smooth"] * fixture_factor)
        ultra_smooth_ms = int(timings.get("ultra_smooth", timings["smooth"] * 1.5) * fixture_factor)

        # If step duration provided, ensure transition doesn't exceed it
        if step_duration_ms:
            max_transition = int(step_duration_ms * 0.8)  # Max 80% of step duration
            default_ms = min(default_ms, max_transition)
            smooth_ms = min(smooth_ms, max_transition)
            ultra_smooth_ms = min(ultra_smooth_ms, max_transition)

        # Main suggestion: smooth transition
        suggestions.append(AISuggestion(
            suggestion_id=self._generate_id(),
            suggestion_type=SuggestionType.TRANSITION_SMOOTHNESS,
            suggestion={
                "transition_ms": smooth_ms,
                "transition_easing": "ease-in-out",
                "preset": "smooth",
                "reason_detail": f"Smooth crossfade for {effect_type} effect"
            },
            reason=f"A {smooth_ms}ms transition creates smooth crossfades between steps, eliminating the snappy/jumpy appearance.",
            confidence=0.85,
            priority=SuggestionPriority.HIGH,
            context={"effect_type": effect_type, "fixture_count": fixture_count}
        ))

        # Alternative: ultra smooth for ambient effects
        if effect_type in ("wave", "pulse", "rainbow"):
            suggestions.append(AISuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.TRANSITION_SMOOTHNESS,
                suggestion={
                    "transition_ms": ultra_smooth_ms,
                    "transition_easing": "ease-in-out",
                    "preset": "ultra_smooth",
                    "reason_detail": "Ultra-smooth for ambient mood"
                },
                reason=f"For a more ambient/dreamy feel, {ultra_smooth_ms}ms transitions create buttery-smooth movement.",
                confidence=0.70,
                priority=SuggestionPriority.MEDIUM,
                context={"effect_type": effect_type, "fixture_count": fixture_count}
            ))

        # Alternative: default (balanced)
        if default_ms != smooth_ms:
            suggestions.append(AISuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.TRANSITION_SMOOTHNESS,
                suggestion={
                    "transition_ms": default_ms,
                    "transition_easing": "linear",
                    "preset": "default",
                    "reason_detail": "Balanced transition timing"
                },
                reason=f"Balanced option: {default_ms}ms provides smooth transitions while maintaining effect energy.",
                confidence=0.75,
                priority=SuggestionPriority.MEDIUM,
                context={"effect_type": effect_type, "fixture_count": fixture_count}
            ))

        # Snappy option for certain effects
        if effect_type in ("chase", "strobe", "flicker"):
            snappy_ms = timings["snappy"]
            suggestions.append(AISuggestion(
                suggestion_id=self._generate_id(),
                suggestion_type=SuggestionType.TRANSITION_SMOOTHNESS,
                suggestion={
                    "transition_ms": snappy_ms,
                    "transition_easing": "linear",
                    "preset": "snappy",
                    "reason_detail": "Sharp transitions for punch"
                },
                reason=f"For punchy, high-energy effects, {snappy_ms}ms transitions keep the sharp edges.",
                confidence=0.60,
                priority=SuggestionPriority.LOW,
                context={"effect_type": effect_type, "fixture_count": fixture_count}
            ))

        for suggestion in suggestions:
            self._pending_suggestions[suggestion.suggestion_id] = suggestion

        return suggestions

    def get_recommended_transition(
        self,
        effect_type: str,
        fixture_count: int,
        smoothness: str = "smooth"
    ) -> Dict[str, Any]:
        """
        Get recommended transition settings without creating a suggestion.

        Quick helper for getting transition values programmatically.

        Args:
            effect_type: Type of effect
            fixture_count: Number of fixtures
            smoothness: "snappy", "default", "smooth", or "ultra_smooth"

        Returns:
            Dict with transition_ms and transition_easing
        """
        effect_transitions = {
            "wave": {"default": 300, "smooth": 400, "ultra_smooth": 600, "snappy": 100},
            "chase": {"default": 200, "smooth": 350, "ultra_smooth": 500, "snappy": 50},
            "rainbow": {"default": 300, "smooth": 500, "ultra_smooth": 700, "snappy": 100},
            "pulse": {"default": 400, "smooth": 600, "ultra_smooth": 800, "snappy": 150},
            "twinkle": {"default": 150, "smooth": 250, "ultra_smooth": 400, "snappy": 50},
            "flicker": {"default": 50, "smooth": 100, "ultra_smooth": 200, "snappy": 0},
            "strobe": {"default": 0, "smooth": 0, "ultra_smooth": 25, "snappy": 0},
        }

        timings = effect_transitions.get(effect_type, {"default": 200, "smooth": 350, "snappy": 50})

        # Fixture count adjustment
        fixture_factor = 1.0 + (fixture_count - 1) * 0.05 if fixture_count > 1 else 1.0
        fixture_factor = min(1.5, fixture_factor)

        transition_ms = int(timings.get(smoothness, timings["default"]) * fixture_factor)
        easing = "ease-in-out" if smoothness in ("smooth", "ultra_smooth") else "linear"

        return {
            "transition_ms": transition_ms,
            "transition_easing": easing
        }

    # ─────────────────────────────────────────────────────────
    # Suggestion Management
    # ─────────────────────────────────────────────────────────

    def apply_suggestion(self, suggestion_id: str) -> bool:
        """
        Mark a suggestion as applied (by user action).

        This does NOT apply the suggestion - it only marks it as applied
        after the user has explicitly chosen to apply it.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            True if found and marked, False otherwise
        """
        if suggestion_id in self._pending_suggestions:
            self._pending_suggestions[suggestion_id].applied = True
            return True
        return False

    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """
        Mark a suggestion as dismissed.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            True if found and marked, False otherwise
        """
        if suggestion_id in self._pending_suggestions:
            self._pending_suggestions[suggestion_id].dismissed = True
            return True
        return False

    def get_pending_suggestions(self) -> List[AISuggestion]:
        """Get all pending (not applied/dismissed) suggestions"""
        return [
            s for s in self._pending_suggestions.values()
            if not s.applied and not s.dismissed
        ]

    def clear_suggestions(self):
        """Clear all pending suggestions"""
        self._pending_suggestions.clear()


# ============================================================
# Global Instance
# ============================================================

_advisor: Optional[AIFixtureAdvisor] = None


def get_ai_advisor() -> AIFixtureAdvisor:
    """Get or create the global AI advisor instance"""
    global _advisor
    if _advisor is None:
        _advisor = AIFixtureAdvisor()
    return _advisor


# ============================================================
# Convenience Functions
# ============================================================

def get_distribution_suggestions(
    modifier_type: str,
    fixture_count: int,
    modifier_params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Get distribution suggestions for a modifier.

    Convenience function that returns serialized suggestions.

    Args:
        modifier_type: The modifier type
        fixture_count: Number of fixtures
        modifier_params: Optional current parameters

    Returns:
        List of suggestion dictionaries
    """
    advisor = get_ai_advisor()
    suggestions = advisor.suggest_distribution(
        modifier_type=modifier_type,
        modifier_params=modifier_params or {},
        fixture_count=fixture_count
    )
    return [s.to_dict() for s in suggestions]


def apply_ai_suggestion(suggestion_id: str) -> bool:
    """Mark a suggestion as applied"""
    return get_ai_advisor().apply_suggestion(suggestion_id)


def dismiss_ai_suggestion(suggestion_id: str) -> bool:
    """Mark a suggestion as dismissed"""
    return get_ai_advisor().dismiss_suggestion(suggestion_id)


def get_transition_suggestions(
    effect_type: str,
    fixture_count: int,
    step_duration_ms: int = None
) -> List[Dict[str, Any]]:
    """
    Get transition smoothness suggestions for an effect.

    Convenience function for the frontend to get AI recommendations
    on appropriate crossfade/transition times.

    Args:
        effect_type: Type of effect (wave, chase, pulse, etc.)
        fixture_count: Number of fixtures
        step_duration_ms: Optional step duration

    Returns:
        List of suggestion dictionaries
    """
    advisor = get_ai_advisor()
    suggestions = advisor.suggest_transition_time(
        effect_type=effect_type,
        fixture_count=fixture_count,
        step_duration_ms=step_duration_ms
    )
    return [s.to_dict() for s in suggestions]


def get_recommended_transition_for_effect(
    effect_type: str,
    fixture_count: int,
    smoothness: str = "smooth"
) -> Dict[str, Any]:
    """
    Quick helper to get recommended transition settings.

    Args:
        effect_type: Type of effect
        fixture_count: Number of fixtures
        smoothness: "snappy", "default", "smooth", or "ultra_smooth"

    Returns:
        Dict with transition_ms and transition_easing
    """
    return get_ai_advisor().get_recommended_transition(
        effect_type=effect_type,
        fixture_count=fixture_count,
        smoothness=smoothness
    )
