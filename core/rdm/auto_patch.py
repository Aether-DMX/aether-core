"""
RDM Auto-Patch - Automatic Fixture Patching from RDM Devices

This module handles automatic fixture patching by matching
discovered RDM devices to fixture profiles.

Classes:
    ProfileMatcher: Finds matching FixtureProfile for RDM device
    AutoPatcher: Generates and applies patch suggestions (legacy)
    AutoPatchEngine: Modern auto-patch with conflict resolution

Usage:
    # Modern usage with AutoPatchEngine
    engine = AutoPatchEngine(patch_manager, fixture_db)
    suggestions = engine.suggest_patch(discovered_fixtures)
    if engine.validate_patch(suggestions):
        results = await engine.apply_suggestions(suggestions, rdm_manager)

    # Legacy usage with AutoPatcher
    patcher = AutoPatcher(fixture_library)
    suggestion = patcher.suggest_patch(device, universe=1, existing=[])
    if not suggestion.has_conflicts():
        fixture = patcher.apply_patch(suggestion)
"""

from typing import List, Optional, Dict, Any, Tuple, Set, Protocol
import logging
import asyncio

from .types import (
    DiscoveredDevice,
    DiscoveredFixture,
    PatchSuggestion,
    AutoPatchSuggestion,
    PatchConfidence,
    RdmPersonality,
)

logger = logging.getLogger(__name__)


# Type hints for external classes (not imported to avoid circular deps)
# These will be duck-typed at runtime
FixtureProfile = Any
FixtureInstance = Any
FixtureLibrary = Any


class ProfileMatcher:
    """
    Matches RDM devices to FixtureProfiles.

    Uses RDM manufacturer and model IDs to find matching profiles,
    with fallback to generic profiles based on DMX footprint.

    Attributes:
        fixture_library: Reference to FixtureLibrary for profile lookup
    """

    # Generic profile mappings by footprint
    GENERIC_PROFILES = {
        1: "generic-dimmer",
        3: "generic-rgb",
        4: "generic-rgbw",
        5: "generic-rgbwa",
        6: "generic-rgbwa",
        7: "generic-wash",
        8: "generic-wash",
        16: "generic-moving-head",
    }

    def __init__(self, fixture_library: FixtureLibrary):
        """
        Initialize profile matcher.

        Args:
            fixture_library: FixtureLibrary instance for profile lookup
        """
        self.fixture_library = fixture_library

    def find_match(
        self,
        manufacturer_id: int,
        device_model_id: int,
        dmx_footprint: int,
        manufacturer_label: str = "",
        device_model: str = ""
    ) -> Tuple[Optional[str], Optional[str], PatchConfidence]:
        """
        Find matching profile for RDM device.

        Tries to match by:
        1. Exact RDM ID match
        2. OFL lookup by manufacturer/model strings
        3. Generic profile by footprint

        Args:
            manufacturer_id: RDM manufacturer ID
            device_model_id: RDM device model ID
            dmx_footprint: DMX channel count
            manufacturer_label: Manufacturer name string
            device_model: Model name string

        Returns:
            Tuple of (profile_id, mode_id, confidence)
        """
        # Try exact RDM ID match
        profile, mode = self._match_by_rdm_ids(manufacturer_id, device_model_id)
        if profile:
            return profile, mode, PatchConfidence.HIGH

        # Try OFL lookup by name
        profile, mode = self._match_by_name(manufacturer_label, device_model)
        if profile:
            return profile, mode, PatchConfidence.MEDIUM

        # Fall back to generic profile
        profile, mode = self._match_generic(dmx_footprint)
        if profile:
            return profile, mode, PatchConfidence.LOW

        return None, None, PatchConfidence.UNKNOWN

    def _match_by_rdm_ids(
        self,
        manufacturer_id: int,
        device_model_id: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Match by RDM manufacturer and device model IDs.

        Args:
            manufacturer_id: RDM manufacturer ID
            device_model_id: RDM device model ID

        Returns:
            Tuple of (profile_id, mode_id) or (None, None)
        """
        # TODO: Implement - call fixture_library.find_profile_by_rdm()
        # profile = self.fixture_library.find_profile_by_rdm(manufacturer_id, device_model_id)
        # if profile:
        #     return profile.profile_id, profile.modes[0].mode_id if profile.modes else None
        return None, None

    def _match_by_name(
        self,
        manufacturer: str,
        model: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Match by manufacturer and model name strings.

        Searches OFL and local profiles by name.

        Args:
            manufacturer: Manufacturer name
            model: Model name

        Returns:
            Tuple of (profile_id, mode_id) or (None, None)
        """
        if not manufacturer or not model:
            return None, None

        # TODO: Implement - search OFL by name
        # results = self.fixture_library.search_ofl(f"{manufacturer} {model}")
        # if results:
        #     profile = self.fixture_library.import_from_ofl(results[0])
        #     return profile.profile_id, profile.modes[0].mode_id
        return None, None

    def _match_generic(
        self,
        footprint: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Match to generic profile by DMX footprint.

        Args:
            footprint: DMX channel count

        Returns:
            Tuple of (profile_id, mode_id) or (None, None)
        """
        # Find closest matching generic profile
        profile_id = self.GENERIC_PROFILES.get(footprint)

        if not profile_id:
            # Find closest footprint
            for fp in sorted(self.GENERIC_PROFILES.keys()):
                if fp >= footprint:
                    profile_id = self.GENERIC_PROFILES[fp]
                    break

        if profile_id:
            return profile_id, "default"

        return None, None

    def find_mode_by_footprint(
        self,
        profile_id: str,
        footprint: int
    ) -> Optional[str]:
        """
        Find profile mode matching DMX footprint.

        Args:
            profile_id: Profile ID to search
            footprint: Target DMX footprint

        Returns:
            Mode ID or None
        """
        # TODO: Implement - get profile and find matching mode
        # profile = self.fixture_library.get_profile(profile_id)
        # if profile:
        #     for mode in profile.modes:
        #         if mode.channel_count == footprint:
        #             return mode.mode_id
        #     # Return first mode as fallback
        #     return profile.modes[0].mode_id if profile.modes else None
        return None


class AutoPatcher:
    """
    Generates and applies auto-patch suggestions.

    Takes discovered RDM devices and generates patch suggestions
    by matching to profiles and checking for conflicts.

    Attributes:
        fixture_library: FixtureLibrary for profile/fixture operations
        profile_matcher: ProfileMatcher for finding matching profiles
    """

    def __init__(self, fixture_library: FixtureLibrary):
        """
        Initialize auto-patcher.

        Args:
            fixture_library: FixtureLibrary instance
        """
        self.fixture_library = fixture_library
        self.profile_matcher = ProfileMatcher(fixture_library)

    def suggest_patch(
        self,
        device: DiscoveredDevice,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> PatchSuggestion:
        """
        Generate patch suggestion for a discovered device.

        Args:
            device: Discovered RDM device
            universe: Target universe
            existing_fixtures: Existing fixtures to check for conflicts

        Returns:
            PatchSuggestion with profile match and conflict info
        """
        # Find matching profile
        profile_id, mode_id, confidence = self.profile_matcher.find_match(
            manufacturer_id=device.manufacturer_id,
            device_model_id=device.device_model_id,
            dmx_footprint=device.dmx_footprint,
            manufacturer_label=device.manufacturer_label,
            device_model=device.device_model
        )

        # Determine profile and mode names
        profile_name = profile_id or "Unknown"
        mode_name = mode_id or "Default"
        is_generic = confidence == PatchConfidence.LOW

        # Use device's RDM address as start channel
        start_channel = device.dmx_address
        channel_count = device.dmx_footprint

        # Check for conflicts
        conflicts = self.find_conflicts(
            start_channel=start_channel,
            footprint=channel_count,
            universe=universe,
            existing_fixtures=existing_fixtures
        )

        # Generate notes
        notes = self._generate_notes(device, profile_id, confidence, conflicts)

        return PatchSuggestion(
            device=device,
            profile_id=profile_id or "generic-dimmer",
            profile_name=profile_name,
            mode_id=mode_id or "default",
            mode_name=mode_name,
            start_channel=start_channel,
            channel_count=channel_count,
            universe=universe,
            confidence=confidence,
            conflicts=[f.fixture_id for f in conflicts],
            notes=notes,
            is_generic=is_generic
        )

    def find_conflicts(
        self,
        start_channel: int,
        footprint: int,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> List[FixtureInstance]:
        """
        Find fixtures that would conflict with proposed patch.

        A conflict occurs when channel ranges overlap in the same universe.

        Args:
            start_channel: Proposed start address
            footprint: DMX channel count
            universe: Target universe
            existing_fixtures: Existing fixtures to check

        Returns:
            List of conflicting fixtures
        """
        end_channel = start_channel + footprint - 1
        conflicts = []

        for fixture in existing_fixtures:
            # Skip fixtures in different universes
            if fixture.universe != universe:
                continue

            # Check for overlap
            fixture_end = fixture.start_channel + fixture.channel_count - 1
            if not (end_channel < fixture.start_channel or
                    start_channel > fixture_end):
                conflicts.append(fixture)

        return conflicts

    def suggest_next_address(
        self,
        footprint: int,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> int:
        """
        Suggest next available DMX address for given footprint.

        Finds the first gap large enough for the footprint.

        Args:
            footprint: Required DMX channel count
            universe: Target universe
            existing_fixtures: Existing fixtures

        Returns:
            Suggested start address (1-512)
        """
        # Get fixtures in this universe, sorted by start channel
        fixtures = sorted(
            [f for f in existing_fixtures if f.universe == universe],
            key=lambda f: f.start_channel
        )

        if not fixtures:
            return 1

        # Check for gap at start
        if fixtures[0].start_channel > footprint:
            return 1

        # Find first gap between fixtures
        for i in range(len(fixtures) - 1):
            current_end = fixtures[i].start_channel + fixtures[i].channel_count
            next_start = fixtures[i + 1].start_channel
            gap = next_start - current_end

            if gap >= footprint:
                return current_end

        # Place after last fixture
        last = fixtures[-1]
        next_address = last.start_channel + last.channel_count

        if next_address + footprint <= 513:
            return next_address

        # No space available
        return -1

    def apply_patch(
        self,
        suggestion: PatchSuggestion,
        fixture_name: Optional[str] = None
    ) -> FixtureInstance:
        """
        Apply a patch suggestion, creating fixture instance.

        Args:
            suggestion: PatchSuggestion to apply
            fixture_name: Optional custom name (uses device label if not provided)

        Returns:
            Created FixtureInstance

        Raises:
            ValueError: If suggestion has unresolved conflicts
        """
        if suggestion.has_conflicts():
            raise ValueError(
                f"Cannot apply patch with conflicts: {suggestion.conflicts}"
            )

        # Generate fixture name
        name = fixture_name or suggestion.device.device_label
        if not name:
            name = f"{suggestion.profile_name} @ {suggestion.start_channel}"

        # TODO: Implement - create fixture instance via fixture_library
        # fixture = self.fixture_library.create_fixture_instance(
        #     name=name,
        #     profile_id=suggestion.profile_id,
        #     mode_id=suggestion.mode_id,
        #     universe=suggestion.universe,
        #     start_channel=suggestion.start_channel,
        #     rdm_uid=str(suggestion.device.uid)
        # )
        # return fixture

        raise NotImplementedError("apply_patch not yet implemented")

    def _generate_notes(
        self,
        device: DiscoveredDevice,
        profile_id: Optional[str],
        confidence: PatchConfidence,
        conflicts: List[FixtureInstance]
    ) -> str:
        """Generate human-readable notes about the match."""
        notes = []

        if confidence == PatchConfidence.HIGH:
            notes.append(f"Exact match found: {profile_id}")
        elif confidence == PatchConfidence.MEDIUM:
            notes.append(f"Matched by name to: {profile_id}")
        elif confidence == PatchConfidence.LOW:
            notes.append(f"Using generic profile for {device.dmx_footprint} channels")
        else:
            notes.append("No matching profile found")

        if conflicts:
            notes.append(f"Warning: Conflicts with {len(conflicts)} existing fixture(s)")

        if device.personalities and len(device.personalities) > 1:
            notes.append(f"Device has {len(device.personalities)} personalities available")

        return "; ".join(notes)


# =============================================================================
# Modern Auto-Patch Engine with Conflict Resolution
# =============================================================================


class PatchManager(Protocol):
    """Protocol for PatchManager interface."""

    def get_patch(self) -> Dict[str, Any]:
        """Get current patch state."""
        ...


class RdmManager(Protocol):
    """Protocol for RdmManager interface."""

    async def set_personality(
        self, node_ip: str, uid: str, personality_index: int
    ) -> bool:
        """Set fixture personality via RDM."""
        ...

    async def set_start_address(
        self, node_ip: str, uid: str, address: int
    ) -> bool:
        """Set fixture DMX start address via RDM."""
        ...


class AutoPatchEngine:
    """
    Modern auto-patch engine with conflict resolution.

    Analyzes discovered fixtures against current patch state and
    generates intelligent suggestions for address assignment.

    Features:
    - Conflict detection with existing fixtures
    - Contiguous address preference for easier debugging
    - Personality optimization (suggest smaller mode if space tight)
    - Multi-universe support
    - Confidence scoring

    Attributes:
        patch_manager: PatchManager for current patch state
        fixture_db: Optional fixture database for enrichment
    """

    # Confidence levels
    CONFIDENCE_NO_CHANGE = 0.99
    CONFIDENCE_SAME_UNIVERSE_CONTIGUOUS = 0.95
    CONFIDENCE_SAME_UNIVERSE_GAP = 0.85
    CONFIDENCE_PERSONALITY_REDUCTION = 0.70
    CONFIDENCE_DIFFERENT_UNIVERSE = 0.50
    CONFIDENCE_NO_SPACE = 0.0

    def __init__(
        self,
        patch_manager: Any,
        fixture_db: Optional[Any] = None
    ):
        """
        Initialize auto-patch engine.

        Args:
            patch_manager: PatchManager for current patch state
            fixture_db: Optional fixture database for enrichment
        """
        self.patch_manager = patch_manager
        self.fixture_db = fixture_db

    def suggest_patch(
        self,
        discovered: Dict[str, DiscoveredFixture],
        max_universes: int = 4,
        prefer_contiguous: bool = True
    ) -> List[AutoPatchSuggestion]:
        """
        Generate patch suggestions for discovered fixtures.

        Analyzes discovered fixtures against current patch state
        and suggests optimal address assignments.

        Args:
            discovered: Dictionary of discovered fixtures by UID
            max_universes: Maximum number of universes to use
            prefer_contiguous: Prefer contiguous addressing

        Returns:
            List of suggestions sorted by confidence (highest first)
        """
        suggestions: List[AutoPatchSuggestion] = []

        # Get current occupied ranges
        occupied = self._get_current_occupied_ranges(max_universes)

        # Also track what discovered fixtures will occupy
        pending_occupied: Dict[int, Set[int]] = {u: set() for u in range(1, max_universes + 1)}

        # Sort fixtures by universe and address for contiguous placement
        sorted_fixtures = sorted(
            discovered.values(),
            key=lambda f: (f.universe, f.start_address)
        )

        for fixture in sorted_fixtures:
            suggestion = self._suggest_for_fixture(
                fixture=fixture,
                occupied=occupied,
                pending_occupied=pending_occupied,
                max_universes=max_universes,
                prefer_contiguous=prefer_contiguous
            )
            suggestions.append(suggestion)

            # Mark suggested range as pending
            if suggestion.confidence > 0:
                universe = suggestion.suggested_universe
                start = suggestion.suggested_start_address
                count = fixture.channel_count
                if suggestion.personality_recommended:
                    # Estimate channel count for new personality
                    count = self._estimate_channels_for_personality(
                        fixture, suggestion.personality_recommended
                    )
                for ch in range(start, start + count):
                    if universe in pending_occupied:
                        pending_occupied[universe].add(ch)

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    async def apply_suggestions(
        self,
        suggestions: List[AutoPatchSuggestion],
        rdm_manager: Any,
        node_ip: str
    ) -> Dict[str, bool]:
        """
        Apply patch suggestions via RDM commands.

        For each suggestion requiring changes:
        1. Send personality change if needed
        2. Send address change if needed

        Args:
            suggestions: List of suggestions to apply
            rdm_manager: RdmManager for sending RDM commands
            node_ip: IP address of the ESP32 node

        Returns:
            Dictionary mapping UID to success status
        """
        results: Dict[str, bool] = {}

        for suggestion in suggestions:
            uid = suggestion.fixture.uid

            if not suggestion.requires_readdressing:
                results[uid] = True
                continue

            try:
                success = True

                # Apply personality change first if needed
                if suggestion.personality_recommended is not None:
                    pers_result = await rdm_manager.set_personality(
                        node_ip, uid, suggestion.personality_recommended
                    )
                    if not pers_result:
                        logger.warning(
                            f"Failed to set personality for {uid} "
                            f"to {suggestion.personality_recommended}"
                        )
                        success = False

                # Apply address change
                if success and suggestion.needs_address_change():
                    addr_result = await rdm_manager.set_start_address(
                        node_ip, uid, suggestion.suggested_start_address
                    )
                    if not addr_result:
                        logger.warning(
                            f"Failed to set address for {uid} "
                            f"to {suggestion.suggested_start_address}"
                        )
                        success = False

                results[uid] = success

            except Exception as e:
                logger.error(f"Error applying suggestion for {uid}: {e}")
                results[uid] = False

        return results

    def validate_patch(self, suggestions: List[AutoPatchSuggestion]) -> bool:
        """
        Validate that suggestions create a conflict-free patch.

        Checks:
        - No overlapping channel ranges
        - All addresses in valid range (1-512)
        - All universes valid

        Args:
            suggestions: List of suggestions to validate

        Returns:
            True if patch is valid, False otherwise
        """
        # Track occupied channels per universe
        occupied: Dict[int, Set[int]] = {}

        for suggestion in suggestions:
            if suggestion.confidence <= 0:
                continue  # Skip no-space suggestions

            universe = suggestion.suggested_universe
            start = suggestion.suggested_start_address
            count = suggestion.fixture.channel_count

            if suggestion.personality_recommended:
                count = self._estimate_channels_for_personality(
                    suggestion.fixture, suggestion.personality_recommended
                )

            # Validate address range
            if start < 1 or start > 512:
                logger.warning(f"Invalid start address: {start}")
                return False

            end = start + count - 1
            if end > 512:
                logger.warning(f"Fixture exceeds universe bounds: {start}-{end}")
                return False

            # Check for overlaps
            if universe not in occupied:
                occupied[universe] = set()

            for ch in range(start, start + count):
                if ch in occupied[universe]:
                    logger.warning(
                        f"Channel conflict at universe {universe}, channel {ch}"
                    )
                    return False
                occupied[universe].add(ch)

        return True

    def _suggest_for_fixture(
        self,
        fixture: DiscoveredFixture,
        occupied: Dict[int, Set[int]],
        pending_occupied: Dict[int, Set[int]],
        max_universes: int,
        prefer_contiguous: bool
    ) -> AutoPatchSuggestion:
        """
        Generate suggestion for a single fixture.

        Args:
            fixture: Fixture to generate suggestion for
            occupied: Currently occupied channels per universe
            pending_occupied: Channels occupied by pending suggestions
            max_universes: Maximum universes to consider
            prefer_contiguous: Prefer contiguous addressing

        Returns:
            AutoPatchSuggestion for this fixture
        """
        universe = fixture.universe
        start = fixture.start_address
        count = fixture.channel_count

        # Combine occupied and pending
        all_occupied: Dict[int, Set[int]] = {}
        for u in range(1, max_universes + 1):
            all_occupied[u] = occupied.get(u, set()) | pending_occupied.get(u, set())

        # Check if current address is conflict-free
        current_range = set(range(start, start + count))
        has_conflict = bool(current_range & all_occupied.get(universe, set()))

        if not has_conflict:
            # No conflict - keep current address
            return AutoPatchSuggestion(
                fixture=fixture,
                suggested_universe=universe,
                suggested_start_address=start,
                personality_recommended=None,
                rationale="Current address is conflict-free",
                confidence=self.CONFIDENCE_NO_CHANGE,
                requires_readdressing=False
            )

        # Conflict exists - need to find new address
        return self._find_new_address(
            fixture=fixture,
            all_occupied=all_occupied,
            max_universes=max_universes,
            prefer_contiguous=prefer_contiguous
        )

    def _find_new_address(
        self,
        fixture: DiscoveredFixture,
        all_occupied: Dict[int, Set[int]],
        max_universes: int,
        prefer_contiguous: bool
    ) -> AutoPatchSuggestion:
        """
        Find new address for fixture with conflict.

        Tries in order:
        1. Same universe, contiguous placement
        2. Same universe, any available gap
        3. Reduced personality in same universe
        4. Different universe
        5. No space available

        Args:
            fixture: Fixture needing new address
            all_occupied: All occupied channels
            max_universes: Maximum universes to consider
            prefer_contiguous: Prefer contiguous addressing

        Returns:
            AutoPatchSuggestion with new address or no-space
        """
        universe = fixture.universe
        count = fixture.channel_count

        # Try same universe first
        new_addr = self._find_free_address(
            universe=universe,
            channel_count=count,
            occupied=all_occupied,
            prefer_contiguous=prefer_contiguous
        )

        if new_addr:
            is_contiguous = self._is_contiguous_placement(
                new_addr, all_occupied.get(universe, set())
            )
            confidence = (
                self.CONFIDENCE_SAME_UNIVERSE_CONTIGUOUS if is_contiguous
                else self.CONFIDENCE_SAME_UNIVERSE_GAP
            )
            return AutoPatchSuggestion(
                fixture=fixture,
                suggested_universe=universe,
                suggested_start_address=new_addr,
                personality_recommended=None,
                rationale=f"Moved to address {new_addr} to avoid conflict",
                confidence=confidence,
                requires_readdressing=True
            )

        # Try reduced personality if available
        # (Note: DiscoveredFixture doesn't have personalities list,
        # but we can check if fixture_db has personality info)
        reduced_pers = self._try_reduced_personality(
            fixture, universe, all_occupied
        )
        if reduced_pers:
            return reduced_pers

        # Try different universe
        for alt_universe in range(1, max_universes + 1):
            if alt_universe == universe:
                continue

            new_addr = self._find_free_address(
                universe=alt_universe,
                channel_count=count,
                occupied=all_occupied,
                prefer_contiguous=prefer_contiguous
            )

            if new_addr:
                return AutoPatchSuggestion(
                    fixture=fixture,
                    suggested_universe=alt_universe,
                    suggested_start_address=new_addr,
                    personality_recommended=None,
                    rationale=f"Moved to universe {alt_universe} address {new_addr}",
                    confidence=self.CONFIDENCE_DIFFERENT_UNIVERSE,
                    requires_readdressing=True
                )

        # No space available
        return AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=universe,
            suggested_start_address=fixture.start_address,
            personality_recommended=None,
            rationale="No space available in any universe",
            confidence=self.CONFIDENCE_NO_SPACE,
            requires_readdressing=False
        )

    def _find_free_address(
        self,
        universe: int,
        channel_count: int,
        occupied: Dict[int, Set[int]],
        prefer_contiguous: bool = True
    ) -> Optional[int]:
        """
        Find first free address range in universe.

        Args:
            universe: Universe to search
            channel_count: Required channel count
            occupied: Occupied channels per universe
            prefer_contiguous: Prefer contiguous placement

        Returns:
            Start address or None if no space
        """
        universe_occupied = occupied.get(universe, set())

        if prefer_contiguous and universe_occupied:
            # Try to place right after existing fixtures
            max_occupied = max(universe_occupied) if universe_occupied else 0
            start = max_occupied + 1
            if start + channel_count <= 513:
                return start

        # Scan for first available range
        for start in range(1, 513 - channel_count + 1):
            needed = set(range(start, start + channel_count))
            if not (needed & universe_occupied):
                return start

        return None

    def _is_contiguous_placement(
        self,
        start: int,
        occupied: Set[int]
    ) -> bool:
        """Check if placement is contiguous with existing fixtures."""
        if not occupied:
            return start == 1
        max_occupied = max(occupied)
        return start == max_occupied + 1

    def _try_reduced_personality(
        self,
        fixture: DiscoveredFixture,
        universe: int,
        all_occupied: Dict[int, Set[int]]
    ) -> Optional[AutoPatchSuggestion]:
        """
        Try to fit fixture with reduced personality.

        Args:
            fixture: Fixture to check
            universe: Target universe
            all_occupied: Occupied channels

        Returns:
            Suggestion with reduced personality or None
        """
        # Without personality info, we can only try common reductions
        # Try halving the channel count as a heuristic
        reduced_counts = [
            fixture.channel_count // 2,
            fixture.channel_count // 4,
            min(8, fixture.channel_count),
            min(4, fixture.channel_count),
            1
        ]

        for reduced_count in reduced_counts:
            if reduced_count <= 0 or reduced_count >= fixture.channel_count:
                continue

            new_addr = self._find_free_address(
                universe=universe,
                channel_count=reduced_count,
                occupied=all_occupied
            )

            if new_addr:
                # Estimate personality index (this is approximate)
                pers_index = self._estimate_personality_for_channels(
                    fixture, reduced_count
                )

                return AutoPatchSuggestion(
                    fixture=fixture,
                    suggested_universe=universe,
                    suggested_start_address=new_addr,
                    personality_recommended=pers_index,
                    rationale=(
                        f"Reduced to ~{reduced_count} channels (personality {pers_index}) "
                        f"to fit at address {new_addr}"
                    ),
                    confidence=self.CONFIDENCE_PERSONALITY_REDUCTION,
                    requires_readdressing=True
                )

        return None

    def _estimate_channels_for_personality(
        self,
        fixture: DiscoveredFixture,
        personality_index: int
    ) -> int:
        """
        Estimate channel count for a personality index.

        This is a heuristic - actual count depends on fixture.

        Args:
            fixture: Fixture to estimate for
            personality_index: Target personality index

        Returns:
            Estimated channel count
        """
        # Common patterns: lower personality = fewer channels
        base = fixture.channel_count
        if personality_index == 1:
            return max(1, base // 4)
        elif personality_index == 2:
            return max(1, base // 2)
        else:
            return base

    def _estimate_personality_for_channels(
        self,
        fixture: DiscoveredFixture,
        target_channels: int
    ) -> int:
        """
        Estimate personality index for target channel count.

        Args:
            fixture: Fixture to estimate for
            target_channels: Target channel count

        Returns:
            Estimated personality index
        """
        # Heuristic: map channel reduction to personality
        ratio = target_channels / max(fixture.channel_count, 1)
        if ratio <= 0.25:
            return 1
        elif ratio <= 0.5:
            return 2
        elif ratio <= 0.75:
            return 3
        return fixture.personality_index

    def _get_current_occupied_ranges(
        self,
        max_universes: int
    ) -> Dict[int, Set[int]]:
        """
        Get currently occupied channels from PatchManager.

        Args:
            max_universes: Maximum universes to check

        Returns:
            Dictionary mapping universe to set of occupied channels
        """
        occupied: Dict[int, Set[int]] = {u: set() for u in range(1, max_universes + 1)}

        if not self.patch_manager:
            return occupied

        try:
            patch = self.patch_manager.get_patch()
            fixtures = patch.get('fixtures', [])

            for fixture in fixtures:
                universe = fixture.get('universe', 1)
                start = fixture.get('start_channel', 1)
                count = fixture.get('channel_count', 1)

                if universe in occupied:
                    for ch in range(start, start + count):
                        if 1 <= ch <= 512:
                            occupied[universe].add(ch)

        except Exception as e:
            logger.warning(f"Error getting current patch: {e}")

        return occupied

    def _calculate_confidence(
        self,
        fixture: DiscoveredFixture,
        suggestion: AutoPatchSuggestion
    ) -> float:
        """
        Calculate confidence score for a suggestion.

        Args:
            fixture: Original fixture
            suggestion: Generated suggestion

        Returns:
            Confidence score (0.0-1.0)
        """
        if not suggestion.requires_readdressing:
            return self.CONFIDENCE_NO_CHANGE

        if suggestion.personality_recommended is not None:
            return self.CONFIDENCE_PERSONALITY_REDUCTION

        if suggestion.suggested_universe != fixture.universe:
            return self.CONFIDENCE_DIFFERENT_UNIVERSE

        return self.CONFIDENCE_SAME_UNIVERSE_GAP
