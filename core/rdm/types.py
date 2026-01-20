"""
RDM Type Definitions - Dataclasses for RDM Data Structures

This module contains all dataclasses used by the RDM subsystem.
These are pure data containers with no business logic.

Classes:
    RdmUid: Unique RDM device identifier
    DiscoveredDevice: Device info from RDM discovery
    RdmPersonality: DMX personality/mode information
    RdmParameter: Single RDM parameter definition
    RdmDeviceInfo: Complete device information
    PatchSuggestion: Auto-patch recommendation
    DiscoveryStatus: Discovery session state
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class DiscoveryState(Enum):
    """State of an RDM discovery session."""
    IDLE = "idle"
    DISCOVERING = "discovering"
    QUERYING = "querying"
    COMPLETE = "complete"
    ERROR = "error"


class PatchConfidence(Enum):
    """Confidence level for auto-patch suggestions."""
    HIGH = "high"       # Exact RDM ID match to profile
    MEDIUM = "medium"   # Partial match or OFL lookup
    LOW = "low"         # Generic profile based on footprint
    UNKNOWN = "unknown" # Could not determine


@dataclass
class RdmUid:
    """
    Unique RDM device identifier.

    Format: MANUFACTURER_ID:DEVICE_ID (e.g., "02CA:12345678")

    Attributes:
        manufacturer_id: 16-bit ESTA manufacturer ID
        device_id: 32-bit unique device serial
    """
    manufacturer_id: int
    device_id: int

    def __str__(self) -> str:
        """Return string representation: XXXX:XXXXXXXX"""
        return f"{self.manufacturer_id:04X}:{self.device_id:08X}"

    @classmethod
    def from_string(cls, uid_str: str) -> "RdmUid":
        """Parse UID from string format."""
        parts = uid_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid RDM UID format: {uid_str}")
        return cls(
            manufacturer_id=int(parts[0], 16),
            device_id=int(parts[1], 16)
        )

    def to_string(self) -> str:
        """Return string representation."""
        return str(self)


@dataclass
class RdmPersonality:
    """
    DMX personality/mode for an RDM device.

    Attributes:
        id: Personality index (1-based)
        name: Human-readable name
        footprint: DMX channel count for this mode
    """
    id: int
    name: str
    footprint: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "footprint": self.footprint
        }


@dataclass
class RdmParameter:
    """
    Single RDM parameter definition.

    Attributes:
        pid: Parameter ID (RDM PID)
        name: Parameter name
        value: Current value
        writable: Whether parameter can be set
    """
    pid: int
    name: str
    value: Any
    writable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pid": self.pid,
            "name": self.name,
            "value": self.value,
            "writable": self.writable
        }


@dataclass
class RdmDeviceInfo:
    """
    Complete RDM device information.

    Contains all information retrieved via RDM from a device.

    Attributes:
        uid: Unique device identifier
        manufacturer_id: ESTA manufacturer ID
        device_model_id: Device model ID
        manufacturer_label: Manufacturer name string
        device_model: Model name string
        device_label: User-assigned label
        dmx_address: Current DMX start address
        dmx_footprint: Current DMX channel count
        current_personality: Active personality index
        personalities: Available personalities/modes
        software_version: Firmware version string
        rdm_protocol_version: RDM protocol version
        parameters: Additional RDM parameters
    """
    uid: RdmUid
    manufacturer_id: int
    device_model_id: int
    manufacturer_label: str = ""
    device_model: str = ""
    device_label: str = ""
    dmx_address: int = 1
    dmx_footprint: int = 1
    current_personality: int = 1
    personalities: List[RdmPersonality] = field(default_factory=list)
    software_version: str = ""
    rdm_protocol_version: str = "1.0"
    parameters: List[RdmParameter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "uid": str(self.uid),
            "manufacturer_id": self.manufacturer_id,
            "device_model_id": self.device_model_id,
            "manufacturer_label": self.manufacturer_label,
            "device_model": self.device_model,
            "device_label": self.device_label,
            "dmx_address": self.dmx_address,
            "dmx_footprint": self.dmx_footprint,
            "current_personality": self.current_personality,
            "personalities": [p.to_dict() for p in self.personalities],
            "software_version": self.software_version,
            "rdm_protocol_version": self.rdm_protocol_version,
            "parameters": [p.to_dict() for p in self.parameters]
        }


@dataclass
class DiscoveredDevice:
    """
    RDM device discovered during scan.

    This is the primary data structure for devices found via RDM discovery.
    It contains enough information for UI display and auto-patching.

    Attributes:
        uid: Unique device identifier
        node_id: ID of ESP32 node that found this device
        universe: DMX universe
        manufacturer_id: ESTA manufacturer ID
        device_model_id: Device model ID
        manufacturer_label: Manufacturer name
        device_model: Model name
        device_label: User-assigned label
        dmx_address: Current DMX start address
        dmx_footprint: Current DMX channel count
        personalities: Available DMX modes
        current_personality: Active mode index
        last_seen: Timestamp of last discovery
        is_patched: Whether linked to a FixtureInstance
        fixture_id: Linked FixtureInstance ID (if patched)
    """
    uid: RdmUid
    node_id: str
    universe: int
    manufacturer_id: int
    device_model_id: int
    manufacturer_label: str = ""
    device_model: str = ""
    device_label: str = ""
    dmx_address: int = 1
    dmx_footprint: int = 1
    personalities: List[RdmPersonality] = field(default_factory=list)
    current_personality: int = 1
    last_seen: datetime = field(default_factory=datetime.now)
    is_patched: bool = False
    fixture_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "uid": str(self.uid),
            "node_id": self.node_id,
            "universe": self.universe,
            "manufacturer_id": self.manufacturer_id,
            "device_model_id": self.device_model_id,
            "manufacturer_label": self.manufacturer_label,
            "device_model": self.device_model,
            "device_label": self.device_label,
            "dmx_address": self.dmx_address,
            "dmx_footprint": self.dmx_footprint,
            "personalities": [p.to_dict() for p in self.personalities],
            "current_personality": self.current_personality,
            "last_seen": self.last_seen.isoformat(),
            "is_patched": self.is_patched,
            "fixture_id": self.fixture_id
        }

    @classmethod
    def from_device_info(
        cls,
        info: RdmDeviceInfo,
        node_id: str,
        universe: int
    ) -> "DiscoveredDevice":
        """Create DiscoveredDevice from RdmDeviceInfo."""
        return cls(
            uid=info.uid,
            node_id=node_id,
            universe=universe,
            manufacturer_id=info.manufacturer_id,
            device_model_id=info.device_model_id,
            manufacturer_label=info.manufacturer_label,
            device_model=info.device_model,
            device_label=info.device_label,
            dmx_address=info.dmx_address,
            dmx_footprint=info.dmx_footprint,
            personalities=info.personalities,
            current_personality=info.current_personality
        )


@dataclass
class PatchSuggestion:
    """
    Auto-patch suggestion for an RDM device.

    Generated by AutoPatcher when matching a discovered device
    to a fixture profile.

    Attributes:
        device: The discovered RDM device
        profile_id: Suggested FixtureProfile ID
        profile_name: Profile display name
        mode_id: Suggested mode within profile
        mode_name: Mode display name
        start_channel: Suggested DMX start address
        channel_count: DMX footprint
        universe: Target universe
        confidence: Match confidence level
        conflicts: Fixtures that would overlap
        notes: Human-readable notes about the match
        is_generic: Whether using generic profile
    """
    device: DiscoveredDevice
    profile_id: str
    profile_name: str
    mode_id: str
    mode_name: str
    start_channel: int
    channel_count: int
    universe: int
    confidence: PatchConfidence = PatchConfidence.UNKNOWN
    conflicts: List[str] = field(default_factory=list)  # Fixture IDs
    notes: str = ""
    is_generic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "device": self.device.to_dict(),
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "mode_id": self.mode_id,
            "mode_name": self.mode_name,
            "start_channel": self.start_channel,
            "channel_count": self.channel_count,
            "universe": self.universe,
            "confidence": self.confidence.value,
            "conflicts": self.conflicts,
            "notes": self.notes,
            "is_generic": self.is_generic
        }

    def has_conflicts(self) -> bool:
        """Check if suggestion has address conflicts."""
        return len(self.conflicts) > 0


@dataclass
class DiscoveryStatus:
    """
    Status of an RDM discovery session.

    Attributes:
        node_id: Node being scanned
        state: Current discovery state
        devices_found: Number of devices discovered
        devices_queried: Number of devices with info retrieved
        started_at: Discovery start time
        completed_at: Discovery completion time
        error: Error message if failed
    """
    node_id: str
    state: DiscoveryState = DiscoveryState.IDLE
    devices_found: int = 0
    devices_queried: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "devices_found": self.devices_found,
            "devices_queried": self.devices_queried,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "progress": self._calculate_progress()
        }

    def _calculate_progress(self) -> float:
        """Calculate discovery progress 0.0 to 1.0."""
        if self.state == DiscoveryState.IDLE:
            return 0.0
        if self.state == DiscoveryState.COMPLETE:
            return 1.0
        if self.state == DiscoveryState.ERROR:
            return 0.0
        if self.devices_found == 0:
            return 0.1  # Discovering
        return 0.1 + (0.9 * self.devices_queried / self.devices_found)
