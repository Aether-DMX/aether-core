"""
Final Render Pipeline - Single Authority for DMX Output

This module provides the unified render pipeline that converts fixture-centric
playback to final DMX channel output. It serves as the single authority for
all DMX output in AETHER.

Key Responsibilities:
1. Build RenderedFixtureFrame from PlaybackSession.fixture_channels
2. Apply modifiers with distribution modes
3. Convert fixture frames to raw DMX channels via ChannelMapper
4. Route through MergeLayer for priority mixing
5. Output to SSOT

Architecture:
    PlaybackSession (fixture_channels)
           ↓
    RenderedFixtureFrame (fixture-semantic)
           ↓
    Apply Modifiers (with distribution)
           ↓
    Convert to Channels (ChannelMapper)
           ↓
    MergeLayer (priority mixing)
           ↓
    SSOT Output (UDP JSON)

Feature Flags:
- FIXTURE_CENTRIC_ENABLED: Enable fixture-centric path (default: True)
- LEGACY_CHANNEL_FALLBACK: Fall back to channel path when no fixtures (default: True)

Version: 1.0.0
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum


# ============================================================
# Feature Flags
# ============================================================

class FeatureFlags:
    """Feature flags for gradual rollout"""
    FIXTURE_CENTRIC_ENABLED = True
    LEGACY_CHANNEL_FALLBACK = True
    AI_SUGGESTIONS_ENABLED = True
    DISTRIBUTION_MODES_ENABLED = True


# ============================================================
# Time Context for Rendering
# ============================================================

@dataclass
class RenderTimeContext:
    """
    Time context for a single render frame.

    Extended version of render_engine.TimeContext with additional fields
    for fixture-centric rendering.
    """
    # Core timing
    absolute_time: float       # Time since epoch
    delta_time: float          # Time since last frame
    elapsed_time: float        # Time since render started
    frame_number: int          # Current frame count

    # Determinism
    seed: int                  # Random seed for reproducible effects

    # BPM sync (optional)
    bpm: float = 120.0         # Beats per minute
    beat_phase: float = 0.0    # Current position in beat (0.0 to 1.0)
    bar_phase: float = 0.0     # Current position in bar (0.0 to 1.0)

    def to_render_engine_ctx(self):
        """Convert to render_engine.TimeContext"""
        from render_engine import TimeContext
        return TimeContext(
            absolute_time=self.absolute_time,
            delta_time=self.delta_time,
            elapsed_time=self.elapsed_time,
            frame_number=self.frame_number,
            seed=self.seed,
        )


# ============================================================
# Render Job - What to render
# ============================================================

@dataclass
class RenderJob:
    """
    A render job encapsulates everything needed to render a frame.

    Can be built from PlaybackSession or created directly.
    """
    job_id: str

    # Fixture-centric data
    fixture_ids: List[str] = field(default_factory=list)
    fixture_channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Legacy channel data (backward compat)
    channels: Dict[int, int] = field(default_factory=dict)

    # Modifiers to apply
    modifiers: List[Any] = field(default_factory=list)  # List of Modifier

    # Output configuration
    universes: List[int] = field(default_factory=lambda: [1])

    # Timing
    seed: int = 0

    @classmethod
    def from_playback_session(cls, session: Any) -> "RenderJob":
        """Build RenderJob from PlaybackSession"""
        return cls(
            job_id=session.session_id,
            fixture_ids=list(session.fixture_ids),
            fixture_channels=dict(session.fixture_channels),
            channels=dict(session.channels),
            modifiers=list(session.modifiers),
            universes=list(session.universes),
            seed=session.seed,
        )

    def has_fixture_data(self) -> bool:
        """Check if this job has fixture-centric data"""
        return bool(self.fixture_ids) or bool(self.fixture_channels)


# ============================================================
# Final Render Pipeline
# ============================================================

class FinalRenderPipeline:
    """
    Single authority for converting fixture frames to DMX output.

    This is the final stage in AETHER's render pipeline, responsible for:
    1. Building fixture frames from playback data
    2. Applying modifiers with distribution
    3. Converting to raw DMX channels
    4. Routing through merge layer
    5. Outputting to SSOT

    All DMX output should flow through this pipeline.
    """

    def __init__(self):
        # Renderer instances
        self._modifier_renderer = None
        self._channel_mapper = None
        self._fixture_library = None

        # State management
        self._modifier_states: Dict[str, Dict[str, Any]] = {}  # job_id -> modifier_id -> state
        self._lock = threading.RLock()

        # Output callback
        self._output_callback: Optional[Callable[[int, Dict[int, int], int], None]] = None

        # Fixture instance cache (fixture_id -> FixtureInstance)
        self._fixture_cache: Dict[str, Any] = {}

        # Feature flags
        self.features = FeatureFlags()

        # Stats
        self._render_count = 0
        self._last_render_time = 0.0

    # ─────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────

    def set_output_callback(self, callback: Callable[[int, Dict[int, int], int], None]):
        """
        Set callback for DMX output.

        Args:
            callback: Function(universe, channels, fade_ms) -> None
        """
        self._output_callback = callback

    def set_fixture_library(self, library: Any):
        """Set the fixture library for channel mapping"""
        self._fixture_library = library
        from fixture_library import get_channel_mapper
        self._channel_mapper = get_channel_mapper()

    def set_modifier_renderer(self, renderer: Any):
        """Set the modifier renderer"""
        self._modifier_renderer = renderer

    def register_fixture(self, fixture_id: str, fixture_instance: Any):
        """Register a fixture instance for rendering"""
        self._fixture_cache[fixture_id] = fixture_instance

    def register_fixtures(self, fixtures: Dict[str, Any]):
        """Register multiple fixture instances"""
        self._fixture_cache.update(fixtures)

    # ─────────────────────────────────────────────────────────
    # Main Render Entry Point
    # ─────────────────────────────────────────────────────────

    def render_session(
        self,
        session: Any,  # PlaybackSession
        time_ctx: RenderTimeContext
    ) -> Dict[int, Dict[int, int]]:
        """
        Render a PlaybackSession to DMX channels.

        This is the main entry point for rendering. It:
        1. Checks for fixture-centric data
        2. Builds RenderedFixtureFrame if available
        3. Applies modifiers with distribution
        4. Converts to channels
        5. Outputs to callback

        Args:
            session: PlaybackSession to render
            time_ctx: Time context for this frame

        Returns:
            Dict of {universe: {channel: value}}
        """
        job = RenderJob.from_playback_session(session)
        return self.render_job(job, time_ctx)

    def render_job(
        self,
        job: RenderJob,
        time_ctx: RenderTimeContext
    ) -> Dict[int, Dict[int, int]]:
        """
        Render a RenderJob to DMX channels.

        Args:
            job: RenderJob containing all render data
            time_ctx: Time context for this frame

        Returns:
            Dict of {universe: {channel: value}}
        """
        with self._lock:
            self._render_count += 1
            self._last_render_time = time.monotonic()

            # Determine render path
            if self.features.FIXTURE_CENTRIC_ENABLED and job.has_fixture_data():
                return self._render_fixture_centric(job, time_ctx)
            elif self.features.LEGACY_CHANNEL_FALLBACK and job.channels:
                return self._render_legacy_channels(job, time_ctx)
            else:
                return {}

    # ─────────────────────────────────────────────────────────
    # Fixture-Centric Render Path
    # ─────────────────────────────────────────────────────────

    def _render_fixture_centric(
        self,
        job: RenderJob,
        time_ctx: RenderTimeContext
    ) -> Dict[int, Dict[int, int]]:
        """
        Fixture-centric render path.

        1. Build RenderedFixtureFrame from fixture_channels
        2. Apply modifiers with distribution
        3. Convert to raw channels
        4. Output
        """
        from fixture_render import (
            RenderedFixtureFrame,
            RenderedFixtureState,
            create_frame_from_fixture_channels
        )

        # Step 1: Build fixture frame
        frame = create_frame_from_fixture_channels(
            fixture_channels=job.fixture_channels,
            frame_number=time_ctx.frame_number,
            timestamp=time_ctx.absolute_time,
            seed=job.seed
        )

        # Ensure all fixture_ids are represented
        for fixture_id in job.fixture_ids:
            if fixture_id not in frame.fixtures:
                # Create default state for missing fixtures
                frame.fixtures[fixture_id] = RenderedFixtureState(
                    fixture_id=fixture_id,
                    attributes={"intensity": 255, "color": [255, 255, 255]},
                    source="default"
                )

        # Step 2: Apply modifiers with distribution
        if job.modifiers and self._modifier_renderer:
            frame = self._apply_modifiers_to_frame(job, frame, time_ctx)

        # Step 3: Convert to raw channels
        channel_frame = self._frame_to_channels(frame, job)

        # Step 4: Output
        if self._output_callback:
            for universe in job.universes:
                channels = channel_frame.get(universe, {})
                if channels:
                    self._output_callback(universe, channels, 0)

        return channel_frame

    def _apply_modifiers_to_frame(
        self,
        job: RenderJob,
        frame: Any,  # RenderedFixtureFrame
        time_ctx: RenderTimeContext
    ) -> Any:  # RenderedFixtureFrame
        """Apply all modifiers to a fixture frame with distribution"""
        # Get or create modifier states for this job
        if job.job_id not in self._modifier_states:
            self._modifier_states[job.job_id] = {}

        mod_states = self._modifier_states[job.job_id]
        render_ctx = time_ctx.to_render_engine_ctx()

        current_frame = frame

        for modifier in job.modifiers:
            if not modifier.enabled:
                continue

            # Get distribution config
            dist_config = modifier.get_distribution_config()

            # Render modifier on frame
            if self.features.DISTRIBUTION_MODES_ENABLED and dist_config:
                current_frame = self._modifier_renderer.render_fixture_frame(
                    modifier_type=modifier.type,
                    params=modifier.params,
                    distribution_config=dist_config,
                    frame=current_frame,
                    time_ctx=render_ctx,
                    modifier_states=mod_states,
                    modifier_id=modifier.id,
                )
            else:
                # Render without distribution (legacy path)
                current_frame = self._modifier_renderer.render_fixture_frame(
                    modifier_type=modifier.type,
                    params=modifier.params,
                    distribution_config=None,
                    frame=current_frame,
                    time_ctx=render_ctx,
                    modifier_states=mod_states,
                    modifier_id=modifier.id,
                )

        return current_frame

    def _frame_to_channels(
        self,
        frame: Any,  # RenderedFixtureFrame
        job: RenderJob
    ) -> Dict[int, Dict[int, int]]:
        """Convert RenderedFixtureFrame to raw DMX channels"""
        if not self._fixture_library:
            return {}

        # Build instances dict from cache
        instances = {}
        for fixture_id in frame.fixtures.keys():
            if fixture_id in self._fixture_cache:
                instances[fixture_id] = self._fixture_cache[fixture_id]

        # Convert using frame's method
        return frame.to_channel_frame(self._fixture_library, instances)

    # ─────────────────────────────────────────────────────────
    # Legacy Channel Render Path
    # ─────────────────────────────────────────────────────────

    def _render_legacy_channels(
        self,
        job: RenderJob,
        time_ctx: RenderTimeContext
    ) -> Dict[int, Dict[int, int]]:
        """
        Legacy channel-based render path.

        For backward compatibility with channel-only content.
        """
        result = {}

        # Start with base channels
        channels = dict(job.channels)

        # Apply modifiers (channel-based)
        if job.modifiers and self._modifier_renderer:
            channels = self._apply_modifiers_to_channels(job, channels, time_ctx)

        # Build output per universe
        for universe in job.universes:
            result[universe] = channels

        # Output
        if self._output_callback:
            for universe in job.universes:
                self._output_callback(universe, channels, 0)

        return result

    def _apply_modifiers_to_channels(
        self,
        job: RenderJob,
        channels: Dict[int, int],
        time_ctx: RenderTimeContext
    ) -> Dict[int, int]:
        """Apply modifiers to raw channels (legacy path)"""
        from render_engine import ModifierState, MergeMode, DEFAULT_MERGE_MODES
        import random

        if job.job_id not in self._modifier_states:
            self._modifier_states[job.job_id] = {}

        mod_states = self._modifier_states[job.job_id]
        render_ctx = time_ctx.to_render_engine_ctx()
        result = dict(channels)

        for modifier in job.modifiers:
            if not modifier.enabled:
                continue

            # Get or create modifier state
            if modifier.id not in mod_states:
                mod_states[modifier.id] = ModifierState(
                    modifier_id=modifier.id,
                    modifier_type=modifier.type,
                    random_state=random.Random(job.seed + hash(modifier.id))
                )

            state = mod_states[modifier.id]

            # Render modifier
            mod_output = self._modifier_renderer.render(
                modifier_type=modifier.type,
                params=modifier.params,
                base_channels=result,
                time_ctx=render_ctx,
                state=state,
                fixture_index=0,
                total_fixtures=1,
            )

            # Apply merge mode
            merge_mode = DEFAULT_MERGE_MODES.get(modifier.type, MergeMode.MULTIPLY)
            for ch, mod_val in mod_output.items():
                if ch in result:
                    if merge_mode == MergeMode.MULTIPLY:
                        result[ch] = int(result[ch] * mod_val)
                    elif merge_mode == MergeMode.REPLACE:
                        result[ch] = int(mod_val)
                    elif merge_mode == MergeMode.ADD:
                        result[ch] = min(255, result[ch] + int(mod_val))
                    result[ch] = max(0, min(255, result[ch]))

        return result

    # ─────────────────────────────────────────────────────────
    # Direct Fixture Render (for /api/dmx/set)
    # ─────────────────────────────────────────────────────────

    def render_fixture_values(
        self,
        fixture_id: str,
        attributes: Dict[str, Any],
        universe: int = 1
    ) -> Dict[int, int]:
        """
        Render a single fixture's values to DMX channels.

        Used for direct fixture control (e.g., /api/dmx/set with fixture_id).

        Args:
            fixture_id: ID of the fixture
            attributes: Attribute dict {"intensity": 255, "color": [...], ...}
            universe: Target universe

        Returns:
            Dict of channel -> value for this fixture
        """
        from fixture_render import RenderedFixtureState

        if not self._channel_mapper or fixture_id not in self._fixture_cache:
            return {}

        instance = self._fixture_cache[fixture_id]

        # Convert attributes to channel values format
        values = self._attributes_to_mapper_values(attributes)

        # Get channels from mapper
        channels = self._channel_mapper.get_channels_for_fixture(instance, values)

        # Convert string keys to int
        return {int(k): v for k, v in channels.items()}

    def _attributes_to_mapper_values(self, attrs: Dict[str, Any]) -> Dict[str, int]:
        """Convert fixture attributes to ChannelMapper values format"""
        values = {}

        if "intensity" in attrs:
            values["dimmer"] = attrs["intensity"]
            values["intensity"] = attrs["intensity"]

        color = attrs.get("color", [])
        if len(color) >= 1:
            values["r"] = color[0]
        if len(color) >= 2:
            values["g"] = color[1]
        if len(color) >= 3:
            values["b"] = color[2]
        if len(color) >= 4:
            values["w"] = color[3]
        if len(color) >= 5:
            values["a"] = color[4]

        if "pan" in attrs:
            values["pan"] = attrs["pan"] >> 8 if attrs["pan"] > 255 else attrs["pan"]
        if "tilt" in attrs:
            values["tilt"] = attrs["tilt"] >> 8 if attrs["tilt"] > 255 else attrs["tilt"]

        return values

    # ─────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "render_count": self._render_count,
            "last_render_time": self._last_render_time,
            "cached_fixtures": len(self._fixture_cache),
            "active_jobs": len(self._modifier_states),
            "features": {
                "fixture_centric": self.features.FIXTURE_CENTRIC_ENABLED,
                "legacy_fallback": self.features.LEGACY_CHANNEL_FALLBACK,
                "ai_suggestions": self.features.AI_SUGGESTIONS_ENABLED,
                "distribution_modes": self.features.DISTRIBUTION_MODES_ENABLED,
            }
        }

    def clear_job_state(self, job_id: str):
        """Clear modifier states for a job"""
        with self._lock:
            self._modifier_states.pop(job_id, None)

    def clear_all_state(self):
        """Clear all modifier states"""
        with self._lock:
            self._modifier_states.clear()


# ============================================================
# Global Instance
# ============================================================

_pipeline: Optional[FinalRenderPipeline] = None


def get_render_pipeline() -> FinalRenderPipeline:
    """Get or create the global render pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = FinalRenderPipeline()
    return _pipeline


def init_render_pipeline(
    fixture_library: Any = None,
    modifier_renderer: Any = None,
    output_callback: Callable = None
) -> FinalRenderPipeline:
    """
    Initialize the global render pipeline.

    Args:
        fixture_library: FixtureLibrary instance
        modifier_renderer: ModifierRenderer instance
        output_callback: DMX output callback

    Returns:
        Configured FinalRenderPipeline
    """
    pipeline = get_render_pipeline()

    if fixture_library:
        pipeline.set_fixture_library(fixture_library)

    if modifier_renderer:
        pipeline.set_modifier_renderer(modifier_renderer)

    if output_callback:
        pipeline.set_output_callback(output_callback)

    return pipeline
