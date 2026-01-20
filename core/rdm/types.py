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
    RdmCommand: UDP JSON RDM command wrapper
    RdmResponse: UDP JSON RDM response wrapper

Constants:
    RDM_PID_*: Standard RDM Parameter IDs
    RdmCommandType: Command type enum
    RdmResponseType: Response type enum
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum, IntEnum
from datetime import datetime


# ============================================================
# RDM Constants - Standard Parameter IDs (PIDs)
# ============================================================

class RdmPid(IntEnum):
    """Standard RDM Parameter IDs (E1.20)."""
    # Discovery
    DISC_UNIQUE_BRANCH = 0x0001
    DISC_MUTE = 0x0002
    DISC_UN_MUTE = 0x0003

    # Device Info
    DEVICE_INFO = 0x0060
    PRODUCT_DETAIL_ID_LIST = 0x0070
    DEVICE_MODEL_DESCRIPTION = 0x0080
    MANUFACTURER_LABEL = 0x0081
    DEVICE_LABEL = 0x0082
    FACTORY_DEFAULTS = 0x0090
    SOFTWARE_VERSION_LABEL = 0x00C0
    BOOT_SOFTWARE_VERSION_ID = 0x00C1
    BOOT_SOFTWARE_VERSION_LABEL = 0x00C2

    # DMX512 Setup
    DMX_PERSONALITY = 0x00E0
    DMX_PERSONALITY_DESCRIPTION = 0x00E1
    DMX_START_ADDRESS = 0x00F0
    SLOT_INFO = 0x0120
    SLOT_DESCRIPTION = 0x0121

    # Sensors
    SENSOR_DEFINITION = 0x0200
    SENSOR_VALUE = 0x0201

    # Power/Lamp
    DEVICE_HOURS = 0x0400
    LAMP_HOURS = 0x0401
    LAMP_STRIKES = 0x0402
    LAMP_STATE = 0x0403
    LAMP_ON_MODE = 0x0404
    DEVICE_POWER_CYCLES = 0x0405

    # Display
    DISPLAY_INVERT = 0x0500
    DISPLAY_LEVEL = 0x0501

    # Control
    PAN_INVERT = 0x0600
    TILT_INVERT = 0x0601
    PAN_TILT_SWAP = 0x0602

    # Identity
    IDENTIFY_DEVICE = 0x1000
    RESET_DEVICE = 0x1001


class RdmCommandType(Enum):
    """RDM command types for UDP JSON messages."""
    DISCOVER = "discover"
    GET_INFO = "get_info"
    GET_PARAM = "get_param"
    SET_PARAM = "set_param"
    IDENTIFY = "identify"
    SET_ADDRESS = "set_address"
    SET_LABEL = "set_label"
    SET_PERSONALITY = "set_personality"


class RdmResponseType(Enum):
    """RDM response types."""
    ACK = "ack"
    ACK_TIMER = "ack_timer"
    NACK = "nack"
    TIMEOUT = "timeout"
    ERROR = "error"


class RdmNackReason(IntEnum):
    """RDM NACK reason codes."""
    UNKNOWN_PID = 0x0000
    FORMAT_ERROR = 0x0001
    HARDWARE_FAULT = 0x0002
    PROXY_REJECT = 0x0003
    WRITE_PROTECT = 0x0004
    UNSUPPORTED_COMMAND_CLASS = 0x0005
    DATA_OUT_OF_RANGE = 0x0006
    BUFFER_FULL = 0x0007
    PACKET_SIZE_UNSUPPORTED = 0x0008
    SUB_DEVICE_OUT_OF_RANGE = 0x0009
    PROXY_BUFFER_FULL = 0x000A


# ============================================================
# UDP JSON RDM Command/Response
# ============================================================

@dataclass
class RdmCommand:
    """
    RDM command for UDP JSON transport.

    This wraps an RDM command in the UDP JSON v2 format
    for sending to ESP32 nodes.

    Attributes:
        action: Command type (discover, get_info, etc.)
        universe: Target DMX universe
        uid: Target device UID (None for broadcast)
        pid: Parameter ID for get/set commands
        data: Command data payload
        seq: Sequence number for request tracking
    """
    action: RdmCommandType
    universe: int
    uid: Optional[str] = None
    pid: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    seq: int = 0

    def to_udp_json(self) -> Dict[str, Any]:
        """Convert to UDP JSON v2 message format."""
        msg: Dict[str, Any] = {
            "v": 2,
            "type": "rdm",
            "action": self.action.value,
            "universe": self.universe,
            "seq": self.seq,
        }

        if self.uid:
            msg["uid"] = self.uid

        if self.pid is not None:
            msg["pid"] = self.pid

        if self.data:
            msg.update(self.data)

        return msg


@dataclass
class RdmResponse:
    """
    RDM response from UDP JSON transport.

    Wraps an RDM response received from ESP32 nodes.

    Attributes:
        action: Original command action
        response_type: Response type (ack, nack, timeout, error)
        uid: Responding device UID
        data: Response data payload
        nack_reason: NACK reason code if applicable
        error: Error message if applicable
        seq: Sequence number matching request
    """
    action: RdmCommandType
    response_type: RdmResponseType
    uid: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    nack_reason: Optional[RdmNackReason] = None
    error: Optional[str] = None
    seq: int = 0

    @property
    def success(self) -> bool:
        """Check if response indicates success."""
        return self.response_type == RdmResponseType.ACK

    @classmethod
    def from_udp_json(cls, msg: Dict[str, Any]) -> "RdmResponse":
        """Parse from UDP JSON v2 response message."""
        action_str = msg.get("action", "")
        try:
            action = RdmCommandType(action_str)
        except ValueError:
            action = RdmCommandType.GET_INFO  # Default

        # Determine response type
        if msg.get("error"):
            response_type = RdmResponseType.ERROR
        elif msg.get("timeout"):
            response_type = RdmResponseType.TIMEOUT
        elif msg.get("nack"):
            response_type = RdmResponseType.NACK
        else:
            response_type = RdmResponseType.ACK

        # Extract NACK reason if present
        nack_reason = None
        if response_type == RdmResponseType.NACK:
            nack_code = msg.get("nack_reason")
            if nack_code is not None:
                try:
                    nack_reason = RdmNackReason(nack_code)
                except ValueError:
                    pass

        # Extract data (everything except meta fields)
        meta_fields = {"v", "type", "action", "seq", "error", "timeout", "nack", "nack_reason", "uid"}
        data = {k: v for k, v in msg.items() if k not in meta_fields}

        return cls(
            action=action,
            response_type=response_type,
            uid=msg.get("uid"),
            data=data if data else None,
            nack_reason=nack_reason,
            error=msg.get("error"),
            seq=msg.get("seq", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "action": self.action.value,
            "response_type": self.response_type.value,
            "success": self.success,
            "seq": self.seq,
        }
        if self.uid:
            result["uid"] = self.uid
        if self.data:
            result["data"] = self.data
        if self.nack_reason:
            result["nack_reason"] = self.nack_reason.value
        if self.error:
            result["error"] = self.error
        return result


class DiscoveryState(Enum):
    """State of an RDM discovery session."""
    IDLE = "idle"
    BROADCASTING = "broadcasting"
    QUERYING = "querying"
    ENRICHING = "enriching"
    CONFLICT_CHECK = "conflict_check"
    COMPLETE = "complete"
    ERROR = "error"

    # Aliases for backward compatibility
    DISCOVERING = "broadcasting"


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


@dataclass
class DiscoveredFixture:
    """
    Fixture discovered via RDM with full information for patching.

    This is the primary data structure returned by the DiscoveryEngine.
    It contains all RDM-reported information plus enrichment from
    the fixture database.

    Attributes:
        uid: Unique RDM device identifier (e.g., '6574:ABCD1234')
        universe: DMX universe number
        start_address: DMX start address (1-512)
        channel_count: Number of DMX channels used
        personality_index: Current personality/mode index
        personality_label: Human-readable personality name
        manufacturer: Manufacturer name from RDM
        model: Model name from RDM
        device_id: RDM device model ID
        serial_number: Device serial number
        software_version: Firmware version string
        capabilities: List of device capabilities (e.g., ['color_wheel', 'gobo'])
        fixture_type: Guessed fixture type from model (e.g., 'moving_head')
        conflicts: List of conflict descriptions (overlapping addresses)
    """
    uid: str
    universe: int
    start_address: int
    channel_count: int
    personality_index: int = 1
    personality_label: str = ""
    manufacturer: str = ""
    model: str = ""
    device_id: int = 0
    serial_number: str = ""
    software_version: str = ""
    capabilities: List[str] = field(default_factory=list)
    fixture_type: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)

    def channel_range(self) -> range:
        """
        Get the DMX channel range for this fixture.

        Returns:
            range object from start_address to start_address + channel_count
        """
        return range(self.start_address, self.start_address + self.channel_count)

    def has_conflicts(self) -> bool:
        """Check if fixture has address conflicts."""
        return len(self.conflicts) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "uid": self.uid,
            "universe": self.universe,
            "start_address": self.start_address,
            "channel_count": self.channel_count,
            "personality_index": self.personality_index,
            "personality_label": self.personality_label,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "device_id": self.device_id,
            "serial_number": self.serial_number,
            "software_version": self.software_version,
            "capabilities": self.capabilities,
            "fixture_type": self.fixture_type,
            "conflicts": self.conflicts,
            "channel_range": [self.start_address, self.start_address + self.channel_count - 1],
            "has_conflicts": self.has_conflicts(),
        }

    @classmethod
    def from_device_info(
        cls,
        info: "RdmDeviceInfo",
        universe: int
    ) -> "DiscoveredFixture":
        """
        Create DiscoveredFixture from RdmDeviceInfo.

        Args:
            info: RDM device information
            universe: DMX universe number

        Returns:
            New DiscoveredFixture instance
        """
        # Get current personality label if available
        personality_label = ""
        if info.personalities:
            for p in info.personalities:
                if p.id == info.current_personality:
                    personality_label = p.name
                    break

        return cls(
            uid=str(info.uid),
            universe=universe,
            start_address=info.dmx_address,
            channel_count=info.dmx_footprint,
            personality_index=info.current_personality,
            personality_label=personality_label,
            manufacturer=info.manufacturer_label,
            model=info.device_model,
            device_id=info.device_model_id,
            serial_number=str(info.uid.device_id),  # Use device_id as serial
            software_version=info.software_version,
        )


@dataclass
class AutoPatchSuggestion:
    """
    Auto-patch suggestion for address resolution.

    Generated by AutoPatchEngine when analyzing discovered fixtures
    against current patch state. Contains suggested address/universe
    changes with rationale and confidence scoring.

    Attributes:
        fixture: The discovered fixture to patch
        suggested_universe: Recommended universe number
        suggested_start_address: Recommended DMX start address (1-512)
        personality_recommended: Suggested personality index if change needed
        rationale: Human-readable explanation for the suggestion
        confidence: Confidence score (0.0-1.0)
        requires_readdressing: Whether fixture needs address change
    """
    fixture: DiscoveredFixture
    suggested_universe: int
    suggested_start_address: int
    personality_recommended: Optional[int] = None
    rationale: str = ""
    confidence: float = 0.0
    requires_readdressing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "fixture": self.fixture.to_dict(),
            "suggested_universe": self.suggested_universe,
            "suggested_start_address": self.suggested_start_address,
            "personality_recommended": self.personality_recommended,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "requires_readdressing": self.requires_readdressing,
        }

    def needs_personality_change(self) -> bool:
        """Check if suggestion includes personality change."""
        return self.personality_recommended is not None

    def needs_address_change(self) -> bool:
        """Check if suggested address differs from current."""
        return (
            self.suggested_start_address != self.fixture.start_address or
            self.suggested_universe != self.fixture.universe
        )
