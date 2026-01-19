"""
Fixture Render Module - Fixture-Semantic State Representation

This module provides the canonical fixture frame representation for AETHER's
fixture-centric playback architecture.

Key Concepts:
- RenderedFixtureState: Per-fixture attribute values (intensity, color, etc.)
- RenderedFixtureFrame: Collection of all fixture states for a single frame
- Conversion to/from raw DMX channels via ChannelMapper integration

The fixture frame is the SSOT for "what should fixtures look like" before
conversion to raw DMX channels. This enables:
- Fixture-aware modifiers (apply effects to fixture attributes, not raw channels)
- Distribution modes (SYNCED, PHASED, PIXELATED) for multi-fixture effects
- AI suggestions based on fixture semantics

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from fixture_library import FixtureLibrary, FixtureInstance


# ============================================================
# Fixture Attribute Types
# ============================================================

class AttributeType(Enum):
    """Semantic attribute types for fixture control"""
    INTENSITY = "intensity"      # Overall brightness (0-255)
    COLOR_RGB = "color_rgb"      # RGB tuple (r, g, b)
    COLOR_RGBW = "color_rgbw"    # RGBW tuple (r, g, b, w)
    COLOR_RGBWA = "color_rgbwa"  # RGBWA tuple (r, g, b, w, a)
    HUE = "hue"                  # Color hue (0-360)
    SATURATION = "saturation"   # Color saturation (0-100)
    PAN = "pan"                  # Pan position (0-540 degrees typically)
    TILT = "tilt"                # Tilt position (0-270 degrees typically)
    GOBO = "gobo"                # Gobo selection
    STROBE = "strobe"            # Strobe rate/mode
    ZOOM = "zoom"                # Zoom/beam angle
    FOCUS = "focus"              # Focus
    PRISM = "prism"              # Prism effect
    CUSTOM = "custom"            # Custom/control channels


# ============================================================
# Rendered Fixture State
# ============================================================

@dataclass
class RenderedFixtureState:
    """
    State representation for a single fixture at a point in time.

    This captures fixture-semantic values (intensity, color, position)
    rather than raw DMX channel values. The state can be converted to
    DMX channels via the ChannelMapper.

    Attributes:
        fixture_id: Unique identifier for the fixture instance
        attributes: Dict of attribute_name -> value
            - intensity: 0-255
            - color: [R, G, B] or [R, G, B, W] or [R, G, B, W, A]
            - pan: 0-65535 (16-bit position)
            - tilt: 0-65535 (16-bit position)
            - Other attributes as defined by fixture profile
    """
    fixture_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Metadata for debugging/tracking
    source: str = "direct"  # direct, modifier, ai_suggestion
    modified_by: List[str] = field(default_factory=list)  # Modifier IDs that touched this

    def get_intensity(self) -> int:
        """Get intensity value, defaulting to 255 (full on)"""
        return self.attributes.get("intensity", 255)

    def set_intensity(self, value: int):
        """Set intensity value (clamped to 0-255)"""
        self.attributes["intensity"] = max(0, min(255, int(value)))

    def get_color(self) -> List[int]:
        """Get color as RGB list, defaulting to white"""
        return self.attributes.get("color", [255, 255, 255])

    def set_color(self, r: int, g: int, b: int, w: int = None, a: int = None):
        """Set color values"""
        color = [
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
        ]
        if w is not None:
            color.append(max(0, min(255, int(w))))
        if a is not None:
            color.append(max(0, min(255, int(a))))
        self.attributes["color"] = color

    def get_position(self) -> Tuple[int, int]:
        """Get pan/tilt as (pan, tilt) tuple"""
        return (
            self.attributes.get("pan", 32768),  # Center
            self.attributes.get("tilt", 32768)   # Center
        )

    def set_position(self, pan: int, tilt: int):
        """Set pan/tilt position (16-bit values)"""
        self.attributes["pan"] = max(0, min(65535, int(pan)))
        self.attributes["tilt"] = max(0, min(65535, int(tilt)))

    def apply_intensity_multiplier(self, multiplier: float):
        """Apply a brightness multiplier (for modifiers)"""
        current = self.get_intensity()
        self.set_intensity(int(current * multiplier))

    def apply_color_multiplier(self, multiplier: float):
        """Apply a brightness multiplier to color channels"""
        color = self.get_color()
        self.attributes["color"] = [
            max(0, min(255, int(c * multiplier))) for c in color
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "fixture_id": self.fixture_id,
            "attributes": self.attributes.copy(),
            "source": self.source,
            "modified_by": list(self.modified_by),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderedFixtureState":
        """Create from dictionary"""
        return cls(
            fixture_id=data["fixture_id"],
            attributes=data.get("attributes", {}).copy(),
            source=data.get("source", "direct"),
            modified_by=list(data.get("modified_by", [])),
        )

    def copy(self) -> "RenderedFixtureState":
        """Create a deep copy"""
        return RenderedFixtureState(
            fixture_id=self.fixture_id,
            attributes={k: (list(v) if isinstance(v, list) else v)
                       for k, v in self.attributes.items()},
            source=self.source,
            modified_by=list(self.modified_by),
        )


# ============================================================
# Rendered Fixture Frame
# ============================================================

@dataclass
class RenderedFixtureFrame:
    """
    Complete frame state for all fixtures.

    This is the canonical representation of "what all fixtures should look like"
    at a given point in time. It serves as the intermediate representation
    between:
    - Playback content (Looks, Sequences) with fixture_channels
    - Raw DMX output (channel -> value dictionaries)

    The frame can be:
    1. Built from PlaybackSession.fixture_channels
    2. Modified by fixture-aware modifiers
    3. Converted to raw channels via to_channel_frame()
    """
    fixtures: Dict[str, RenderedFixtureState] = field(default_factory=dict)

    # Frame metadata
    frame_number: int = 0
    timestamp: float = 0.0
    seed: int = 0

    def get_fixture(self, fixture_id: str) -> Optional[RenderedFixtureState]:
        """Get state for a specific fixture"""
        return self.fixtures.get(fixture_id)

    def set_fixture(self, state: RenderedFixtureState):
        """Set or update fixture state"""
        self.fixtures[state.fixture_id] = state

    def get_all_fixture_ids(self) -> List[str]:
        """Get list of all fixture IDs in this frame"""
        return list(self.fixtures.keys())

    def apply_intensity_to_all(self, multiplier: float):
        """Apply intensity multiplier to all fixtures"""
        for state in self.fixtures.values():
            state.apply_intensity_multiplier(multiplier)

    def to_channel_frame(
        self,
        library: "FixtureLibrary",
        instances: Dict[str, "FixtureInstance"] = None
    ) -> Dict[int, Dict[int, int]]:
        """
        Convert fixture frame to raw DMX channels.

        Args:
            library: FixtureLibrary for profile lookup
            instances: Optional dict of fixture_id -> FixtureInstance
                      If not provided, looks up from library

        Returns:
            Dict of {universe: {channel: value}}
        """
        from fixture_library import get_channel_mapper

        mapper = get_channel_mapper()
        if not mapper:
            return {}

        result: Dict[int, Dict[int, int]] = {}

        for fixture_id, state in self.fixtures.items():
            # Get fixture instance
            if instances:
                instance = instances.get(fixture_id)
            else:
                # Would need to look up from library's instance storage
                instance = None

            if not instance:
                continue

            # Convert attributes to channel-compatible format
            values = self._attributes_to_channel_values(state.attributes)

            # Get DMX channels for this fixture
            channels = mapper.get_channels_for_fixture(instance, values)

            # Add to universe result
            universe = instance.universe
            if universe not in result:
                result[universe] = {}

            for ch_str, value in channels.items():
                result[universe][int(ch_str)] = value

        return result

    def to_flat_channels(
        self,
        library: "FixtureLibrary",
        instances: Dict[str, "FixtureInstance"] = None,
        universe: int = 1
    ) -> Dict[int, int]:
        """
        Convert fixture frame to flat channel dict for a single universe.

        This is a convenience method for single-universe setups.
        """
        channel_frame = self.to_channel_frame(library, instances)
        return channel_frame.get(universe, {})

    def _attributes_to_channel_values(self, attributes: Dict[str, Any]) -> Dict[str, int]:
        """
        Convert fixture attributes to ChannelMapper-compatible values dict.

        Maps semantic attributes to channel-level controls:
        - intensity -> dimmer
        - color[0,1,2] -> r, g, b
        - color[3] -> w (white)
        - color[4] -> a (amber)
        - pan, tilt -> position values
        """
        values = {}

        # Intensity -> dimmer
        if "intensity" in attributes:
            values["dimmer"] = attributes["intensity"]
            values["intensity"] = attributes["intensity"]

        # Color -> RGB(WA)
        color = attributes.get("color", [])
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

        # Position
        if "pan" in attributes:
            # Convert 16-bit to 8-bit for basic fixtures
            values["pan"] = attributes["pan"] >> 8
        if "tilt" in attributes:
            values["tilt"] = attributes["tilt"] >> 8

        # Pass through other attributes directly
        for key in ["gobo", "strobe", "zoom", "focus", "prism"]:
            if key in attributes:
                values[key] = attributes[key]

        return values

    @classmethod
    def from_channel_frame(
        cls,
        channel_frame: Dict[int, Dict[int, int]],
        library: "FixtureLibrary",
        instances: List["FixtureInstance"],
        frame_number: int = 0,
        timestamp: float = 0.0,
        seed: int = 0
    ) -> "RenderedFixtureFrame":
        """
        Create a RenderedFixtureFrame from raw channel data.

        This is the reverse operation - given raw channels, reconstruct
        fixture-semantic state. Used for backward compatibility with
        existing channel-based content.

        Args:
            channel_frame: {universe: {channel: value}}
            library: FixtureLibrary for profile lookup
            instances: List of fixture instances to map
        """
        frame = cls(frame_number=frame_number, timestamp=timestamp, seed=seed)

        for instance in instances:
            profile = library.get_profile(instance.profile_id)
            if not profile:
                continue

            mode = profile.get_mode(instance.mode_id)
            if not mode:
                continue

            universe_channels = channel_frame.get(instance.universe, {})

            # Extract fixture attributes from raw channels
            attributes = {}
            color = [0, 0, 0]
            has_color = False

            for i, channel_def in enumerate(mode.channels):
                dmx_channel = instance.start_channel + i
                value = universe_channels.get(dmx_channel, channel_def.default)

                name_lower = channel_def.name.lower()

                # Map channel to attribute
                if channel_def.type == "dimmer" or "dimmer" in name_lower:
                    attributes["intensity"] = value
                elif "red" in name_lower:
                    color[0] = value
                    has_color = True
                elif "green" in name_lower:
                    color[1] = value
                    has_color = True
                elif "blue" in name_lower:
                    color[2] = value
                    has_color = True
                elif "white" in name_lower:
                    if len(color) == 3:
                        color.append(value)
                    else:
                        color[3] = value
                    has_color = True
                elif "amber" in name_lower:
                    if len(color) == 4:
                        color.append(value)
                    elif len(color) == 3:
                        color.extend([0, value])
                    has_color = True
                elif "pan" in name_lower and "fine" not in name_lower:
                    attributes["pan"] = value << 8
                elif "tilt" in name_lower and "fine" not in name_lower:
                    attributes["tilt"] = value << 8

            if has_color:
                attributes["color"] = color

            # Default intensity to 255 if not found
            if "intensity" not in attributes:
                attributes["intensity"] = 255

            frame.fixtures[instance.fixture_id] = RenderedFixtureState(
                fixture_id=instance.fixture_id,
                attributes=attributes,
                source="channel_import"
            )

        return frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "fixtures": {
                fid: state.to_dict() for fid, state in self.fixtures.items()
            },
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderedFixtureFrame":
        """Create from dictionary"""
        frame = cls(
            frame_number=data.get("frame_number", 0),
            timestamp=data.get("timestamp", 0.0),
            seed=data.get("seed", 0),
        )

        fixtures_data = data.get("fixtures", {})
        for fid, state_data in fixtures_data.items():
            frame.fixtures[fid] = RenderedFixtureState.from_dict(state_data)

        return frame

    def copy(self) -> "RenderedFixtureFrame":
        """Create a deep copy of this frame"""
        frame = RenderedFixtureFrame(
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            seed=self.seed,
        )
        for fid, state in self.fixtures.items():
            frame.fixtures[fid] = state.copy()
        return frame


# ============================================================
# Frame Builder - Convenience for building frames
# ============================================================

class FixtureFrameBuilder:
    """
    Builder pattern for constructing RenderedFixtureFrame.

    Example:
        frame = (FixtureFrameBuilder()
            .add_fixture("par1", intensity=255, color=[255, 0, 0])
            .add_fixture("par2", intensity=200, color=[0, 255, 0])
            .set_metadata(frame_number=1, seed=12345)
            .build())
    """

    def __init__(self):
        self._fixtures: Dict[str, RenderedFixtureState] = {}
        self._frame_number = 0
        self._timestamp = 0.0
        self._seed = 0

    def add_fixture(
        self,
        fixture_id: str,
        intensity: int = 255,
        color: List[int] = None,
        pan: int = None,
        tilt: int = None,
        **extra_attributes
    ) -> "FixtureFrameBuilder":
        """Add a fixture with given attributes"""
        attributes = {"intensity": intensity}

        if color is not None:
            attributes["color"] = list(color)

        if pan is not None:
            attributes["pan"] = pan

        if tilt is not None:
            attributes["tilt"] = tilt

        attributes.update(extra_attributes)

        self._fixtures[fixture_id] = RenderedFixtureState(
            fixture_id=fixture_id,
            attributes=attributes
        )

        return self

    def set_all_intensity(self, intensity: int) -> "FixtureFrameBuilder":
        """Set intensity for all fixtures"""
        for state in self._fixtures.values():
            state.set_intensity(intensity)
        return self

    def set_all_color(self, r: int, g: int, b: int) -> "FixtureFrameBuilder":
        """Set color for all fixtures"""
        for state in self._fixtures.values():
            state.set_color(r, g, b)
        return self

    def set_metadata(
        self,
        frame_number: int = None,
        timestamp: float = None,
        seed: int = None
    ) -> "FixtureFrameBuilder":
        """Set frame metadata"""
        if frame_number is not None:
            self._frame_number = frame_number
        if timestamp is not None:
            self._timestamp = timestamp
        if seed is not None:
            self._seed = seed
        return self

    def build(self) -> RenderedFixtureFrame:
        """Build the RenderedFixtureFrame"""
        frame = RenderedFixtureFrame(
            fixtures=self._fixtures.copy(),
            frame_number=self._frame_number,
            timestamp=self._timestamp,
            seed=self._seed,
        )
        return frame


# ============================================================
# Utility Functions
# ============================================================

def create_frame_from_fixture_channels(
    fixture_channels: Dict[str, Dict[str, Any]],
    frame_number: int = 0,
    timestamp: float = 0.0,
    seed: int = 0
) -> RenderedFixtureFrame:
    """
    Create a RenderedFixtureFrame from PlaybackSession.fixture_channels format.

    Args:
        fixture_channels: Dict of fixture_id -> {"intensity": 255, "color": [255,0,0], ...}

    Returns:
        RenderedFixtureFrame with fixture states
    """
    frame = RenderedFixtureFrame(
        frame_number=frame_number,
        timestamp=timestamp,
        seed=seed,
    )

    for fixture_id, attributes in fixture_channels.items():
        frame.fixtures[fixture_id] = RenderedFixtureState(
            fixture_id=fixture_id,
            attributes=attributes.copy() if attributes else {},
            source="fixture_channels"
        )

    return frame


def merge_frames(
    frames: List[RenderedFixtureFrame],
    mode: str = "htp"
) -> RenderedFixtureFrame:
    """
    Merge multiple fixture frames using specified mode.

    Args:
        frames: List of frames to merge
        mode: Merge mode - "htp" (highest takes precedence), "ltp" (latest), "average"

    Returns:
        Merged RenderedFixtureFrame
    """
    if not frames:
        return RenderedFixtureFrame()

    if len(frames) == 1:
        return frames[0].copy()

    result = RenderedFixtureFrame()

    # Collect all fixture IDs
    all_fixture_ids = set()
    for frame in frames:
        all_fixture_ids.update(frame.fixtures.keys())

    for fixture_id in all_fixture_ids:
        states = [f.fixtures.get(fixture_id) for f in frames if fixture_id in f.fixtures]

        if not states:
            continue

        if mode == "ltp":
            # Latest takes precedence (last frame wins)
            result.fixtures[fixture_id] = states[-1].copy()

        elif mode == "htp":
            # Highest takes precedence for intensity
            merged = states[0].copy()
            for state in states[1:]:
                if state.get_intensity() > merged.get_intensity():
                    merged.attributes["intensity"] = state.get_intensity()
                # For color, take the one with highest average value
                state_color = state.get_color()
                merged_color = merged.get_color()
                if sum(state_color) > sum(merged_color):
                    merged.attributes["color"] = state_color.copy()
            result.fixtures[fixture_id] = merged

        elif mode == "average":
            # Average all values
            merged = RenderedFixtureState(fixture_id=fixture_id)

            # Average intensity
            intensities = [s.get_intensity() for s in states]
            merged.set_intensity(sum(intensities) // len(intensities))

            # Average color
            colors = [s.get_color() for s in states]
            max_len = max(len(c) for c in colors)
            avg_color = []
            for i in range(max_len):
                values = [c[i] if i < len(c) else 0 for c in colors]
                avg_color.append(sum(values) // len(values))
            merged.attributes["color"] = avg_color

            result.fixtures[fixture_id] = merged

    return result
