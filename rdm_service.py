"""
RDM Service - Remote Device Management
AETHER ARCHITECTURE PROGRAM - Phase 4 Lane 4

# ============================================================================
# PURPOSE
# ============================================================================
#
# This service provides RDM (Remote Device Management) functionality:
#
# 1. DISCOVERY: Find all RDM-capable fixtures on the DMX bus
# 2. ADDRESSING: Read and set DMX start addresses remotely
# 3. FIXTURE INTELLIGENCE: Get device info (manufacturer, model, footprint)
# 4. ASSISTED SETUP: Auto-suggest addressing based on footprints
# 5. PLAYBACK SAFETY: Verify configuration before cue execution
#
# RDM commands are sent to ESP32 nodes which execute them via the DMX bus.
# This is a pass-through architecture - backend orchestrates, firmware executes.
#
# ============================================================================
# ACCEPTANCE CRITERIA (Phase 4 Lane 4)
# ============================================================================
#
# AC1: Discovery
#   - Trigger RDM discovery via API
#   - Return list of discovered devices with UIDs
#   - Cache results for quick access
#
# AC2: Addressing & Verification
#   - Read current DMX address for any discovered device
#   - Set new DMX address remotely
#   - Verify address change was accepted
#
# AC3: Fixture Intelligence
#   - Get device info (manufacturer ID, model ID, footprint)
#   - Map to known fixture profiles if possible
#   - Store in fixture library
#
# AC4: Assisted Setup Logic
#   - Given a list of devices, suggest non-overlapping addresses
#   - Detect address conflicts before they happen
#   - Provide "Fix All" option to auto-resolve conflicts
#
# AC5: Playback Safety
#   - Before executing a cue, verify all required fixtures are:
#     a) Discovered and responding
#     b) Addressed correctly
#     c) Have expected footprint
#   - If verification fails, warn operator before proceeding
#
# ============================================================================
"""

import json
import socket
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Configure logging
rdm_logger = logging.getLogger('aether.rdm')
rdm_logger.setLevel(logging.INFO)
if not rdm_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [RDM] %(levelname)s: %(message)s'
    ))
    rdm_logger.addHandler(handler)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RDMDevice:
    """Represents an RDM-capable device discovered on the bus."""
    uid: str                           # Unique ID: "XXXX:XXXXXXXX" (manufacturer:device)
    manufacturer_id: int = 0
    device_model_id: int = 0
    dmx_address: int = 0
    dmx_footprint: int = 0
    personality_id: int = 1
    personality_count: int = 1
    software_version: int = 0
    sensor_count: int = 0
    label: str = ""                    # User-assigned label
    discovered_via: str = ""           # Node ID that discovered this device
    discovered_at: str = ""            # ISO timestamp
    last_seen: str = ""               # ISO timestamp
    manufacturer_name: str = ""        # Resolved from manufacturer ID
    model_name: str = ""              # Resolved from device info

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RDMDevice':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AddressSuggestion:
    """Suggested address assignment for a device."""
    uid: str
    current_address: int
    suggested_address: int
    footprint: int
    reason: str  # "conflict", "gap", "optimal"
    conflicts_with: List[str] = field(default_factory=list)


class RDMOperationResult(Enum):
    """Result codes for RDM operations."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    NO_RESPONSE = "no_response"
    INVALID_RESPONSE = "invalid_response"
    NODE_OFFLINE = "node_offline"
    DEVICE_NOT_FOUND = "device_not_found"
    ADDRESS_CONFLICT = "address_conflict"
    NACK = "nack"


# ============================================================================
# KNOWN MANUFACTURERS (common RDM manufacturer IDs)
# ============================================================================

KNOWN_MANUFACTURERS = {
    0x0000: "PLASA (Development)",
    0x0001: "ESTA (Standards)",
    0x414C: "Avolites",
    0x4144: "ADJ",
    0x4348: "Chauvet",
    0x434D: "City Theatrical",
    0x454C: "ETC",
    0x4D41: "Martin",
    0x5052: "PR Lighting",
    0x524F: "Robe",
    0x534C: "Signify (Philips)",
    0x5354: "Strong Entertainment",
    0x5641: "Varilite",
}


# ============================================================================
# RDM SERVICE
# ============================================================================

class RDMService:
    """
    Service for RDM operations across all nodes.

    Architecture:
    - Backend sends RDM commands to ESP32 nodes via UDP JSON
    - Nodes execute RDM on DMX bus and return results
    - Backend caches and aggregates results from all nodes
    """

    # Configuration
    RDM_TIMEOUT_MS = 5000          # Timeout for single RDM operation
    DISCOVERY_TIMEOUT_MS = 30000   # Timeout for full discovery
    UDP_PORT = 6455                # Same port as DMX commands

    def __init__(self):
        self._devices: Dict[str, RDMDevice] = {}  # UID -> Device
        self._lock = threading.RLock()
        self._last_discovery: Optional[datetime] = None
        self._discovery_in_progress = False

        # Node manager reference (set externally)
        self._node_manager = None

        # UDP socket for RDM commands
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(self.RDM_TIMEOUT_MS / 1000.0)

    def set_node_manager(self, node_manager):
        """Set reference to node manager for getting node list."""
        self._node_manager = node_manager

    # ─────────────────────────────────────────────────────────────────
    # AC1: Discovery
    # ─────────────────────────────────────────────────────────────────

    def discover_all(self) -> Dict[str, Any]:
        """
        Run RDM discovery on all online nodes.

        Returns:
            Dict with discovery results
        """
        if self._discovery_in_progress:
            return {
                'success': False,
                'error': 'Discovery already in progress',
                'devices': []
            }

        self._discovery_in_progress = True
        print("RDM: Starting discovery on all nodes...", flush=True)

        try:
            nodes = self._get_rdm_capable_nodes()
            if not nodes:
                return {
                    'success': False,
                    'error': 'No RDM-capable nodes online',
                    'devices': []
                }

            all_devices = []

            for node in nodes:
                node_id = node.get('node_id')
                node_ip = node.get('ip')

                if not node_ip:
                    continue

                print(f"RDM: Discovering on {node_id} ({node_ip})...", flush=True)

                result = self._send_rdm_command(node_ip, {
                    'v': 2,
                    'type': 'rdm',
                    'action': 'discover'
                })

                if result and result.get('success'):
                    devices = result.get('devices', [])
                    count = result.get('count', 0)
                    print(f"RDM: Found {count} devices on {node_id}", flush=True)

                    # Store devices
                    for uid_str in devices:
                        if uid_str not in self._devices:
                            self._devices[uid_str] = RDMDevice(
                                uid=uid_str,
                                discovered_via=node_id,
                                discovered_at=datetime.now().isoformat(),
                                last_seen=datetime.now().isoformat()
                            )
                        else:
                            self._devices[uid_str].last_seen = datetime.now().isoformat()

                        all_devices.append(uid_str)

                        # Fetch device info
                        self._fetch_device_info(node_ip, uid_str)

            self._last_discovery = datetime.now()

            return {
                'success': True,
                'devices': [self._devices[uid].to_dict() for uid in all_devices],
                'count': len(all_devices),
                'timestamp': self._last_discovery.isoformat()
            }

        finally:
            self._discovery_in_progress = False

    def get_cached_devices(self) -> List[Dict]:
        """Get list of cached RDM devices without new discovery."""
        with self._lock:
            return [d.to_dict() for d in self._devices.values()]

    def _fetch_device_info(self, node_ip: str, uid: str) -> bool:
        """Fetch detailed info for a device."""
        result = self._send_rdm_command(node_ip, {
            'v': 2,
            'type': 'rdm',
            'action': 'get_info',
            'uid': uid
        })

        if result and result.get('success'):
            with self._lock:
                if uid in self._devices:
                    dev = self._devices[uid]
                    dev.manufacturer_id = result.get('manufacturer_id', 0)
                    dev.device_model_id = result.get('device_model_id', 0)
                    dev.dmx_address = result.get('dmx_address', 0)
                    dev.dmx_footprint = result.get('dmx_footprint', 0)
                    dev.personality_id = result.get('personality_id', 1)
                    dev.personality_count = result.get('personality_count', 1)
                    dev.software_version = result.get('software_version', 0)
                    dev.sensor_count = result.get('sensor_count', 0)

                    # Resolve manufacturer name
                    if dev.manufacturer_id in KNOWN_MANUFACTURERS:
                        dev.manufacturer_name = KNOWN_MANUFACTURERS[dev.manufacturer_id]

                    return True
        return False

    # ─────────────────────────────────────────────────────────────────
    # AC2: Addressing & Verification
    # ─────────────────────────────────────────────────────────────────

    def get_address(self, uid: str) -> Dict[str, Any]:
        """
        Get current DMX address for a device.

        Args:
            uid: Device UID string

        Returns:
            Dict with address or error
        """
        device = self._devices.get(uid)
        if not device:
            return {'success': False, 'error': 'Device not found in cache'}

        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}

        result = self._send_rdm_command(node.get('ip'), {
            'v': 2,
            'type': 'rdm',
            'action': 'get_address',
            'uid': uid
        })

        if result and result.get('success'):
            address = result.get('address', 0)
            with self._lock:
                self._devices[uid].dmx_address = address
            return {
                'success': True,
                'uid': uid,
                'address': address
            }

        return {
            'success': False,
            'error': result.get('error', 'Unknown error') if result else 'No response'
        }

    def set_address(self, uid: str, address: int) -> Dict[str, Any]:
        """
        Set DMX address for a device.

        Args:
            uid: Device UID string
            address: New DMX address (1-512)

        Returns:
            Dict with result
        """
        if address < 1 or address > 512:
            return {'success': False, 'error': 'Invalid address (must be 1-512)'}

        device = self._devices.get(uid)
        if not device:
            return {'success': False, 'error': 'Device not found in cache'}

        # Check for conflicts before setting
        conflicts = self._check_address_conflict(uid, address, device.dmx_footprint)
        if conflicts:
            return {
                'success': False,
                'error': 'Address conflict',
                'conflicts': conflicts
            }

        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}

        result = self._send_rdm_command(node.get('ip'), {
            'v': 2,
            'type': 'rdm',
            'action': 'set_address',
            'uid': uid,
            'address': address
        })

        if result and result.get('success'):
            with self._lock:
                self._devices[uid].dmx_address = address
            print(f"RDM: Set {uid} address to {address}", flush=True)
            return {
                'success': True,
                'uid': uid,
                'address': address
            }

        return {
            'success': False,
            'error': result.get('error', 'Unknown error') if result else 'No response'
        }

    def verify_address(self, uid: str, expected_address: int) -> Dict[str, Any]:
        """
        Verify a device is at the expected address.

        Args:
            uid: Device UID string
            expected_address: Expected DMX address

        Returns:
            Dict with verification result
        """
        result = self.get_address(uid)
        if not result.get('success'):
            return result

        actual_address = result.get('address', 0)
        matches = actual_address == expected_address

        return {
            'success': True,
            'uid': uid,
            'expected_address': expected_address,
            'actual_address': actual_address,
            'verified': matches
        }

    # ─────────────────────────────────────────────────────────────────
    # AC3: Fixture Intelligence
    # ─────────────────────────────────────────────────────────────────

    def get_device_info(self, uid: str) -> Dict[str, Any]:
        """
        Get full device info.

        Args:
            uid: Device UID string

        Returns:
            Dict with device info
        """
        device = self._devices.get(uid)
        if not device:
            return {'success': False, 'error': 'Device not found in cache'}

        return {
            'success': True,
            'device': device.to_dict()
        }

    def identify_device(self, uid: str, state: bool = True) -> Dict[str, Any]:
        """
        Turn identify mode on/off for a device.

        Args:
            uid: Device UID string
            state: True to turn on identify, False to turn off

        Returns:
            Dict with result
        """
        device = self._devices.get(uid)
        if not device:
            return {'success': False, 'error': 'Device not found in cache'}

        node = self._get_node_for_device(uid)
        if not node:
            return {'success': False, 'error': 'No node available for device'}

        result = self._send_rdm_command(node.get('ip'), {
            'v': 2,
            'type': 'rdm',
            'action': 'identify',
            'uid': uid,
            'state': state
        })

        if result and result.get('success'):
            return {
                'success': True,
                'uid': uid,
                'identify': state
            }

        return {
            'success': False,
            'error': result.get('error', 'Unknown error') if result else 'No response'
        }

    # ─────────────────────────────────────────────────────────────────
    # AC4: Assisted Setup Logic
    # ─────────────────────────────────────────────────────────────────

    def suggest_addresses(self) -> Dict[str, Any]:
        """
        Analyze current addressing and suggest optimal assignments.

        Returns:
            Dict with suggestions for each device
        """
        with self._lock:
            devices = list(self._devices.values())

        if not devices:
            return {'success': True, 'suggestions': [], 'conflicts': []}

        suggestions = []
        conflicts = []

        # Sort by current address
        devices.sort(key=lambda d: d.dmx_address)

        # Find conflicts
        used_ranges = []  # List of (start, end, uid)

        for dev in devices:
            if dev.dmx_address == 0:
                continue

            start = dev.dmx_address
            end = start + dev.dmx_footprint - 1

            # Check for conflicts with existing ranges
            for other_start, other_end, other_uid in used_ranges:
                if not (end < other_start or start > other_end):
                    # Overlap detected
                    conflicts.append({
                        'device1': dev.uid,
                        'device2': other_uid,
                        'range1': [start, end],
                        'range2': [other_start, other_end]
                    })

            used_ranges.append((start, end, dev.uid))

        # Generate suggestions to fix conflicts
        next_available = 1

        for dev in devices:
            footprint = max(dev.dmx_footprint, 1)

            # Check if device has a conflict
            has_conflict = any(
                c['device1'] == dev.uid or c['device2'] == dev.uid
                for c in conflicts
            )

            if has_conflict or dev.dmx_address == 0:
                # Find next available slot
                while True:
                    end = next_available + footprint - 1
                    if end > 512:
                        # Can't fit
                        suggestions.append(AddressSuggestion(
                            uid=dev.uid,
                            current_address=dev.dmx_address,
                            suggested_address=0,
                            footprint=footprint,
                            reason='no_space'
                        ))
                        break

                    # Check if slot is free
                    slot_free = True
                    for other in devices:
                        if other.uid == dev.uid:
                            continue
                        o_start = other.dmx_address
                        o_end = o_start + other.dmx_footprint - 1
                        if not (end < o_start or next_available > o_end):
                            slot_free = False
                            break

                    if slot_free:
                        suggestions.append(AddressSuggestion(
                            uid=dev.uid,
                            current_address=dev.dmx_address,
                            suggested_address=next_available,
                            footprint=footprint,
                            reason='conflict' if has_conflict else 'unaddressed'
                        ))
                        next_available = end + 1
                        break

                    next_available += 1
            else:
                # No conflict, update next_available
                next_available = max(next_available, dev.dmx_address + footprint)

        return {
            'success': True,
            'suggestions': [asdict(s) for s in suggestions],
            'conflicts': conflicts,
            'total_devices': len(devices),
            'conflicting_devices': len(set(
                c['device1'] for c in conflicts
            ).union(c['device2'] for c in conflicts))
        }

    def auto_fix_addresses(self) -> Dict[str, Any]:
        """
        Automatically fix all address conflicts.

        Returns:
            Dict with results for each device
        """
        analysis = self.suggest_addresses()
        if not analysis.get('success'):
            return analysis

        suggestions = analysis.get('suggestions', [])
        results = []

        for suggestion in suggestions:
            uid = suggestion['uid']
            new_address = suggestion['suggested_address']

            if new_address == 0:
                results.append({
                    'uid': uid,
                    'success': False,
                    'error': 'No space available'
                })
                continue

            result = self.set_address(uid, new_address)
            results.append({
                'uid': uid,
                'success': result.get('success', False),
                'address': new_address if result.get('success') else None,
                'error': result.get('error')
            })

        return {
            'success': all(r['success'] for r in results),
            'results': results
        }

    # ─────────────────────────────────────────────────────────────────
    # AC5: Playback Safety
    # ─────────────────────────────────────────────────────────────────

    def verify_cue_readiness(self, cue_data: Dict) -> Dict[str, Any]:
        """
        Verify all fixtures required for a cue are ready.

        Args:
            cue_data: Cue definition including fixture references

        Returns:
            Dict with verification results
        """
        issues = []
        warnings = []

        # Extract fixture requirements from cue
        required_fixtures = cue_data.get('fixtures', [])
        required_channels = cue_data.get('channels', {})

        # Check if we have any devices
        if not self._devices:
            warnings.append({
                'type': 'no_rdm_devices',
                'message': 'No RDM devices discovered - manual verification recommended'
            })

        # For each required fixture, verify it's available and addressed
        for fixture in required_fixtures:
            fixture_uid = fixture.get('rdm_uid')
            expected_address = fixture.get('dmx_address')
            expected_footprint = fixture.get('footprint')

            if not fixture_uid:
                continue

            device = self._devices.get(fixture_uid)

            if not device:
                issues.append({
                    'type': 'device_not_found',
                    'uid': fixture_uid,
                    'message': f'Device {fixture_uid} not found in RDM cache'
                })
                continue

            # Verify address
            if expected_address and device.dmx_address != expected_address:
                issues.append({
                    'type': 'address_mismatch',
                    'uid': fixture_uid,
                    'expected': expected_address,
                    'actual': device.dmx_address,
                    'message': f'Device at wrong address: expected {expected_address}, found {device.dmx_address}'
                })

            # Verify footprint
            if expected_footprint and device.dmx_footprint != expected_footprint:
                warnings.append({
                    'type': 'footprint_mismatch',
                    'uid': fixture_uid,
                    'expected': expected_footprint,
                    'actual': device.dmx_footprint,
                    'message': f'Footprint mismatch: expected {expected_footprint}, device reports {device.dmx_footprint}'
                })

        # Check for address conflicts
        conflicts = self._find_all_conflicts()
        if conflicts:
            issues.append({
                'type': 'address_conflicts',
                'conflicts': conflicts,
                'message': f'Found {len(conflicts)} address conflicts'
            })

        ready = len(issues) == 0

        return {
            'ready': ready,
            'issues': issues,
            'warnings': warnings,
            'can_proceed': ready or len(issues) == 0,  # Can proceed if no blocking issues
            'recommendation': 'OK to proceed' if ready else 'Review issues before proceeding'
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────

    def _get_rdm_capable_nodes(self) -> List[Dict]:
        """Get list of online nodes that support RDM."""
        if not self._node_manager:
            return []

        all_nodes = self._node_manager.get_all_nodes(include_offline=False)

        # Filter for nodes with RDM capability
        rdm_nodes = []
        for node in all_nodes:
            caps = node.get('caps', [])
            if 'rdm' in caps:
                rdm_nodes.append(node)

        return rdm_nodes

    def _get_node_for_device(self, uid: str) -> Optional[Dict]:
        """Get the node that can communicate with a device."""
        device = self._devices.get(uid)
        if not device or not device.discovered_via:
            # Try any RDM-capable node
            nodes = self._get_rdm_capable_nodes()
            return nodes[0] if nodes else None

        # Get the node that discovered it
        if self._node_manager:
            return self._node_manager.get_node(device.discovered_via)

        return None

    def _send_rdm_command(self, node_ip: str, command: Dict) -> Optional[Dict]:
        """Send RDM command to a node and wait for response."""
        try:
            message = json.dumps(command).encode()
            self._socket.sendto(message, (node_ip, self.UDP_PORT))

            # Wait for response
            try:
                data, addr = self._socket.recvfrom(4096)
                response = json.loads(data.decode())
                return response
            except socket.timeout:
                rdm_logger.warning(f"RDM timeout from {node_ip}")
                return None

        except Exception as e:
            rdm_logger.error(f"RDM command error: {e}")
            return None

    def _check_address_conflict(self, uid: str, new_address: int, footprint: int) -> List[Dict]:
        """Check if setting an address would cause a conflict."""
        conflicts = []
        new_end = new_address + footprint - 1

        with self._lock:
            for other_uid, other in self._devices.items():
                if other_uid == uid:
                    continue
                if other.dmx_address == 0:
                    continue

                other_end = other.dmx_address + other.dmx_footprint - 1

                # Check overlap
                if not (new_end < other.dmx_address or new_address > other_end):
                    conflicts.append({
                        'uid': other_uid,
                        'address': other.dmx_address,
                        'footprint': other.dmx_footprint
                    })

        return conflicts

    def _find_all_conflicts(self) -> List[Dict]:
        """Find all address conflicts among cached devices."""
        conflicts = []
        devices = list(self._devices.values())

        for i, dev1 in enumerate(devices):
            if dev1.dmx_address == 0:
                continue
            end1 = dev1.dmx_address + dev1.dmx_footprint - 1

            for dev2 in devices[i+1:]:
                if dev2.dmx_address == 0:
                    continue
                end2 = dev2.dmx_address + dev2.dmx_footprint - 1

                if not (end1 < dev2.dmx_address or dev1.dmx_address > end2):
                    conflicts.append({
                        'device1': dev1.uid,
                        'device2': dev2.uid,
                        'overlap': True
                    })

        return conflicts

    # ─────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get RDM service status."""
        with self._lock:
            return {
                'enabled': True,
                'device_count': len(self._devices),
                'last_discovery': self._last_discovery.isoformat() if self._last_discovery else None,
                'discovery_in_progress': self._discovery_in_progress,
                'devices': [d.to_dict() for d in self._devices.values()]
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

rdm_service = RDMService()


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

def discover_rdm_devices() -> Dict[str, Any]:
    """Run RDM discovery on all nodes."""
    return rdm_service.discover_all()


def get_rdm_devices() -> List[Dict]:
    """Get cached RDM devices."""
    return rdm_service.get_cached_devices()


def get_rdm_device_address(uid: str) -> Dict[str, Any]:
    """Get address for a specific device."""
    return rdm_service.get_address(uid)


def set_rdm_device_address(uid: str, address: int) -> Dict[str, Any]:
    """Set address for a specific device."""
    return rdm_service.set_address(uid, address)


def identify_rdm_device(uid: str, state: bool = True) -> Dict[str, Any]:
    """Turn identify on/off for a device."""
    return rdm_service.identify_device(uid, state)


def get_rdm_address_suggestions() -> Dict[str, Any]:
    """Get address suggestions to fix conflicts."""
    return rdm_service.suggest_addresses()


def auto_fix_rdm_addresses() -> Dict[str, Any]:
    """Automatically fix all address conflicts."""
    return rdm_service.auto_fix_addresses()


def verify_cue_rdm_readiness(cue_data: Dict) -> Dict[str, Any]:
    """Verify all fixtures for a cue are ready."""
    return rdm_service.verify_cue_readiness(cue_data)


def get_rdm_status() -> Dict[str, Any]:
    """Get RDM service status."""
    return rdm_service.get_status()
