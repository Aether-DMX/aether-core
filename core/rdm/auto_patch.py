"""
RDM Auto-Patch - Automatic Fixture Patching from RDM Devices

This module handles automatic fixture patching by matching
discovered RDM devices to fixture profiles.

Classes:
    ProfileMatcher: Finds matching FixtureProfile for RDM device
    AutoPatcher: Generates and applies patch suggestions

Usage:
    patcher = AutoPatcher(fixture_library)
    suggestion = patcher.suggest_patch(device, universe=1, existing=[])
    if not suggestion.has_conflicts():
        fixture = patcher.apply_patch(suggestion)
"""

from typing import List, Optional, Dict, Any, Tuple
import logging

from .types import (
    DiscoveredDevice,
    PatchSuggestion,
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
