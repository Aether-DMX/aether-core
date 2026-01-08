"""
Merge Layer - Priority-based channel merging for multiple playback sources

This module provides:
- MergeLayer: Combines multiple DMX sources with HTP/LTP rules
- ChannelClassifier: Determines merge strategy per channel
- LayeredOutput: Final output stage before SSOT

Architecture:
- Each playback source registers with a priority level
- Channels are classified as dimmer (HTP) or non-dimmer (LTP)
- Output is computed per-frame by merging all active sources
- Blackout/emergency override bypasses normal merge

HTP (Highest Takes Precedence): For dimmer/intensity channels
- Multiple sources can contribute, highest value wins
- Natural for brightness - additive feel

LTP (Latest Takes Precedence): For color/position channels
- Most recent write wins based on priority
- Prevents color mixing artifacts

Priority Levels (higher = wins):
- blackout: 100 (emergency override)
- manual: 80 (fader/slider input)
- effect: 60 (dynamic effects)
- look: 50 (look playback)
- sequence: 45 (sequence playback)
- chase: 40 (legacy chase)
- scene: 20 (static scene)
- background: 10 (ambient/default)
- idle: 0 (no output)

Version: 1.0.0
"""

import time
import threading
from typing import Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Channel Classification
# ============================================================

class ChannelType(Enum):
    """How a channel should be merged"""
    DIMMER = "dimmer"      # HTP merge (brightness/intensity)
    COLOR = "color"        # LTP merge (RGB/color)
    POSITION = "position"  # LTP merge (pan/tilt)
    CONTROL = "control"    # LTP merge (strobe/gobo/etc)
    UNKNOWN = "unknown"    # Default to LTP


# Channel name patterns that indicate dimmer/intensity
DIMMER_PATTERNS = {
    'dimmer', 'intensity', 'dim', 'master', 'brightness', 'level',
    'white', 'warm', 'cool', 'amber', 'uv',  # White/amber channels often act as dimmers
}

# Channel name patterns that indicate color
COLOR_PATTERNS = {
    'red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
    'color', 'hue', 'saturation',
}

# Channel name patterns that indicate position
POSITION_PATTERNS = {
    'pan', 'tilt', 'pan fine', 'tilt fine', 'rotation',
}


class ChannelClassifier:
    """
    Classifies channels as dimmer (HTP) or non-dimmer (LTP) based on:
    1. Fixture profile metadata (preferred)
    2. Channel name heuristics
    3. Position-based defaults (channel 1 often dimmer in simple fixtures)
    """

    def __init__(self):
        self._fixture_cache: Dict[str, Dict[int, ChannelType]] = {}  # universe -> {channel: type}
        self._lock = threading.Lock()

    def load_fixtures(self, fixtures: List[Dict]):
        """Load fixture definitions and cache channel classifications"""
        with self._lock:
            self._fixture_cache.clear()

            for fixture in fixtures:
                universe = fixture.get('universe', 1)
                start_channel = fixture.get('start_channel', 1)
                channel_map = fixture.get('channel_map', [])
                fixture_type = fixture.get('type', 'generic')

                if universe not in self._fixture_cache:
                    self._fixture_cache[universe] = {}

                for offset, channel_name in enumerate(channel_map):
                    channel_num = start_channel + offset
                    channel_type = self._classify_by_name(channel_name, fixture_type)
                    self._fixture_cache[universe][channel_num] = channel_type

    def classify(self, universe: int, channel: int) -> ChannelType:
        """Get the classification for a specific channel"""
        with self._lock:
            # Check fixture cache first
            if universe in self._fixture_cache:
                if channel in self._fixture_cache[universe]:
                    return self._fixture_cache[universe][channel]

            # Default heuristics for unknown channels
            return self._default_classification(channel)

    def is_dimmer(self, universe: int, channel: int) -> bool:
        """Quick check if channel should use HTP merge"""
        return self.classify(universe, channel) == ChannelType.DIMMER

    def _classify_by_name(self, name: str, fixture_type: str) -> ChannelType:
        """Classify based on channel name from fixture profile"""
        name_lower = name.lower().strip()

        # Check dimmer patterns
        for pattern in DIMMER_PATTERNS:
            if pattern in name_lower:
                return ChannelType.DIMMER

        # Check color patterns
        for pattern in COLOR_PATTERNS:
            if pattern in name_lower:
                return ChannelType.COLOR

        # Check position patterns
        for pattern in POSITION_PATTERNS:
            if pattern in name_lower:
                return ChannelType.POSITION

        # Special case: simple dimmer fixtures
        if fixture_type == 'dimmer':
            return ChannelType.DIMMER

        return ChannelType.CONTROL

    def _default_classification(self, channel: int) -> ChannelType:
        """Default classification when no fixture profile exists"""
        # Common convention: in RGBW fixtures, channel pattern is R,G,B,W or R,G,B,D
        # We'll be conservative and treat only explicitly named channels as dimmers
        # For unknown fixtures, default to LTP (safer for color channels)
        return ChannelType.UNKNOWN

    def get_dimmer_channels(self, universe: int) -> Set[int]:
        """Get all channels classified as dimmers in a universe"""
        with self._lock:
            if universe not in self._fixture_cache:
                return set()
            return {ch for ch, ct in self._fixture_cache[universe].items()
                    if ct == ChannelType.DIMMER}


# ============================================================
# Priority Levels
# ============================================================

PRIORITY_LEVELS = {
    'blackout': 100,
    'emergency': 100,
    'manual': 80,
    'fader': 80,
    'effect': 60,
    'look': 50,
    'sequence': 45,
    'chase': 40,
    'scene': 20,
    'background': 10,
    'ambient': 10,
    'idle': 0,
}


def get_priority(source_type: str) -> int:
    """Get priority level for a source type"""
    return PRIORITY_LEVELS.get(source_type.lower(), 0)


# ============================================================
# Source Registration
# ============================================================

@dataclass
class MergeSource:
    """A registered source contributing to the merge"""
    source_id: str
    source_type: str  # look, sequence, effect, manual, etc.
    priority: int
    universes: List[int]
    channels: Dict[int, Dict[int, int]]  # universe -> {channel: value}
    last_update: float = 0.0
    active: bool = True

    def set_channels(self, universe: int, channels: Dict[int, int]):
        """Update channels for a universe"""
        self.channels[universe] = channels
        self.last_update = time.monotonic()


# ============================================================
# Merge Layer
# ============================================================

class MergeLayer:
    """
    Combines multiple DMX sources with HTP/LTP merge rules.

    Each source registers with:
    - source_id: Unique identifier
    - source_type: Category (look, sequence, effect, manual)
    - universes: Which universes this source outputs to

    Per-frame, the merge layer:
    1. Collects all active sources
    2. For each universe/channel:
       - Dimmer channels: HTP (highest value wins)
       - Other channels: LTP (highest priority wins)
    3. Returns merged output
    """

    def __init__(self, classifier: Optional[ChannelClassifier] = None):
        self._classifier = classifier or ChannelClassifier()
        self._sources: Dict[str, MergeSource] = {}
        self._lock = threading.Lock()

        # Blackout state
        self._blackout_active = False
        self._blackout_universes: Set[int] = set()

        # Output callback
        self._output_callback: Optional[Callable] = None

        # Stats
        self._merge_count = 0
        self._last_merge_time = 0.0

    @property
    def classifier(self) -> ChannelClassifier:
        return self._classifier

    def set_output_callback(self, callback: Callable[[int, Dict[int, int]], None]):
        """Set callback for sending merged output: callback(universe, channels)"""
        self._output_callback = callback

    # ─────────────────────────────────────────────────────────
    # Source Management
    # ─────────────────────────────────────────────────────────

    def register_source(
        self,
        source_id: str,
        source_type: str,
        universes: List[int],
    ) -> MergeSource:
        """Register a new merge source"""
        priority = get_priority(source_type)

        source = MergeSource(
            source_id=source_id,
            source_type=source_type,
            priority=priority,
            universes=universes,
            channels={},
            last_update=time.monotonic(),
            active=True,
        )

        with self._lock:
            self._sources[source_id] = source

        print(f"📥 MergeLayer: Registered source '{source_id}' (type={source_type}, priority={priority})")
        return source

    def unregister_source(self, source_id: str):
        """Remove a source from the merge"""
        with self._lock:
            if source_id in self._sources:
                del self._sources[source_id]
                print(f"📤 MergeLayer: Unregistered source '{source_id}'")

    def get_source(self, source_id: str) -> Optional[MergeSource]:
        """Get a registered source by ID"""
        with self._lock:
            return self._sources.get(source_id)

    def set_source_channels(self, source_id: str, universe: int, channels: Dict[int, int]):
        """Update channels for a source"""
        with self._lock:
            source = self._sources.get(source_id)
            if source:
                source.set_channels(universe, channels)

    def deactivate_source(self, source_id: str):
        """Temporarily deactivate a source without unregistering"""
        with self._lock:
            source = self._sources.get(source_id)
            if source:
                source.active = False

    def activate_source(self, source_id: str):
        """Reactivate a deactivated source"""
        with self._lock:
            source = self._sources.get(source_id)
            if source:
                source.active = True

    # ─────────────────────────────────────────────────────────
    # Blackout / Emergency
    # ─────────────────────────────────────────────────────────

    def set_blackout(self, active: bool, universes: Optional[List[int]] = None):
        """Set blackout state - overrides all other sources"""
        with self._lock:
            self._blackout_active = active
            if active:
                self._blackout_universes = set(universes) if universes else set()
                print(f"⬛ MergeLayer: Blackout ACTIVE" +
                      (f" on universes {sorted(universes)}" if universes else " (all)"))
            else:
                self._blackout_universes.clear()
                print("⬜ MergeLayer: Blackout RELEASED")

    def is_blackout(self, universe: Optional[int] = None) -> bool:
        """Check if blackout is active (optionally for specific universe)"""
        with self._lock:
            if not self._blackout_active:
                return False
            if universe is None:
                return True
            # If no specific universes set, blackout applies to all
            if not self._blackout_universes:
                return True
            return universe in self._blackout_universes

    # ─────────────────────────────────────────────────────────
    # Merge Computation
    # ─────────────────────────────────────────────────────────

    def compute_merge(self, universe: int) -> Dict[int, int]:
        """
        Compute merged output for a universe.

        Returns dict of channel -> value with all sources merged.
        """
        with self._lock:
            # Blackout override
            if self._blackout_active:
                if not self._blackout_universes or universe in self._blackout_universes:
                    return {}  # Empty = all zeros

            # Collect active sources for this universe
            active_sources = [
                s for s in self._sources.values()
                if s.active and universe in s.universes and universe in s.channels
            ]

            if not active_sources:
                return {}

            # Sort by priority (highest first) for LTP
            active_sources.sort(key=lambda s: s.priority, reverse=True)

            # Merge channels
            merged: Dict[int, int] = {}
            ltp_set: Set[int] = set()  # Channels already set by higher priority (LTP)

            for source in active_sources:
                source_channels = source.channels.get(universe, {})

                for channel, value in source_channels.items():
                    if self._classifier.is_dimmer(universe, channel):
                        # HTP: Highest value wins
                        if channel in merged:
                            merged[channel] = max(merged[channel], value)
                        else:
                            merged[channel] = value
                    else:
                        # LTP: First (highest priority) write wins
                        if channel not in ltp_set:
                            merged[channel] = value
                            ltp_set.add(channel)

            self._merge_count += 1
            self._last_merge_time = time.monotonic()

            return merged

    def compute_all_universes(self) -> Dict[int, Dict[int, int]]:
        """Compute merged output for all universes with active sources"""
        with self._lock:
            # Get all universes that have sources
            all_universes = set()
            for source in self._sources.values():
                if source.active:
                    all_universes.update(source.universes)

        # Compute merge for each universe
        result = {}
        for universe in all_universes:
            merged = self.compute_merge(universe)
            if merged:
                result[universe] = merged

        return result

    def output_all(self):
        """Compute merge and send to output callback for all universes"""
        if not self._output_callback:
            return

        merged = self.compute_all_universes()
        for universe, channels in merged.items():
            self._output_callback(universe, channels)

    # ─────────────────────────────────────────────────────────
    # Status / Debug
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get merge layer status"""
        with self._lock:
            sources_info = []
            for source in self._sources.values():
                sources_info.append({
                    'source_id': source.source_id,
                    'source_type': source.source_type,
                    'priority': source.priority,
                    'active': source.active,
                    'universes': source.universes,
                    'channel_count': sum(len(ch) for ch in source.channels.values()),
                    'age_ms': int((time.monotonic() - source.last_update) * 1000),
                })

            return {
                'source_count': len(self._sources),
                'active_count': sum(1 for s in self._sources.values() if s.active),
                'blackout_active': self._blackout_active,
                'blackout_universes': sorted(self._blackout_universes) if self._blackout_universes else None,
                'merge_count': self._merge_count,
                'sources': sources_info,
            }

    def get_source_breakdown(self, universe: int, channel: int) -> Dict:
        """Debug: Show which sources are contributing to a channel"""
        with self._lock:
            contributions = []
            for source in self._sources.values():
                if not source.active or universe not in source.channels:
                    continue
                ch_val = source.channels[universe].get(channel)
                if ch_val is not None:
                    contributions.append({
                        'source_id': source.source_id,
                        'source_type': source.source_type,
                        'priority': source.priority,
                        'value': ch_val,
                    })

            # Sort by priority
            contributions.sort(key=lambda c: c['priority'], reverse=True)

            channel_type = self._classifier.classify(universe, channel)
            is_htp = channel_type == ChannelType.DIMMER

            # Compute winning value
            if is_htp:
                winning_value = max((c['value'] for c in contributions), default=0)
            else:
                winning_value = contributions[0]['value'] if contributions else 0

            return {
                'universe': universe,
                'channel': channel,
                'channel_type': channel_type.value,
                'merge_mode': 'HTP' if is_htp else 'LTP',
                'contributions': contributions,
                'final_value': winning_value,
            }


# ============================================================
# Global Instance
# ============================================================

channel_classifier = ChannelClassifier()
merge_layer = MergeLayer(classifier=channel_classifier)
