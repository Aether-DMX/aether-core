# AETHER RDM Module - Design Document

## Version: 0.1.0 (Phase 0 - Structure Design)
## Date: January 19, 2026

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AETHER CORE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ RdmManager   │───▶│ RdmDiscovery │───▶│ DiscoveredDevice     │  │
│  │ (Facade)     │    │              │    │ (dataclass)          │  │
│  └──────┬───────┘    └──────────────┘    └──────────────────────┘  │
│         │                                                           │
│         │            ┌──────────────┐    ┌──────────────────────┐  │
│         └───────────▶│ RdmTransport │───▶│ ESP32 Nodes          │  │
│                      │ (UDP JSON)   │    │ (Port 6455)          │  │
│                      └──────────────┘    └──────────────────────┘  │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ AutoPatcher  │───▶│FixtureLibrary│───▶│ FixtureInstance      │  │
│  │              │    │ (existing)   │    │ (existing)           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ UDP JSON v2
┌─────────────────────────────────────────────────────────────────────┐
│                      ESP32 NODES (aether-pulse)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ PULSE-xxxx  │  │ PULSE-yyyy  │  │ PULSE-zzzz  │                 │
│  │ Universe 1  │  │ Universe 2  │  │ Universe 4  │                 │
│  │ RDM Gateway │  │ RDM Gateway │  │ RDM Gateway │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities

### `core/rdm/types.py`
Dataclasses for RDM data structures. No logic, just data containers.

- `RdmUid` - Unique device identifier (manufacturer:device)
- `DiscoveredDevice` - RDM device info from discovery
- `RdmPersonality` - DMX personality/mode info
- `RdmParameter` - Single RDM parameter definition
- `RdmDeviceInfo` - Complete device information
- `PatchSuggestion` - Auto-patch recommendation

### `core/rdm/transport.py`
UDP JSON communication with ESP32 RDM gateways.

- `RdmTransport` - Abstract base for RDM communication
- `UdpJsonRdmTransport` - UDP JSON v2 implementation
- Handles: discovery, get/set parameters, identify

### `core/rdm/discovery.py`
Device discovery and information gathering.

- `RdmDiscovery` - Manages discovery process
- `DiscoverySession` - Tracks in-progress discovery
- Handles: full scan, incremental scan, device info queries

### `core/rdm/auto_patch.py`
Automatic fixture patching from RDM devices.

- `AutoPatcher` - Generates patch suggestions
- `ProfileMatcher` - Finds matching FixtureProfile
- Handles: profile matching, address conflicts, suggestions

### `core/rdm/manager.py`
High-level facade coordinating all RDM operations.

- `RdmManager` - Main entry point for RDM operations
- Coordinates: transport, discovery, auto-patch
- Emits: events for UI updates

---

## 3. Integration Points

### A. Existing Code (DO NOT MODIFY)

| File | What We Use | How |
|------|-------------|-----|
| `fixture_library.py` | `FixtureProfile`, `FixtureInstance` | Import dataclasses |
| `fixture_library.py` | `FixtureLibrary.find_profile_by_rdm()` | Call for profile matching |
| `aether-core.py` | `NodeManager.get_node()` | Get node IP for transport |
| `aether-core.py` | `rdm_devices` table | Store discovered devices |

### B. New Integration (Via RdmManager Facade)

```python
# In aether-core.py - replace inline RDM code with:
from core.rdm import RdmManager

rdm_manager = RdmManager(node_manager, fixture_library, db)

# API endpoints delegate to rdm_manager
@app.route('/api/nodes/<node_id>/rdm/discover', methods=['POST'])
def rdm_discover(node_id):
    return jsonify(rdm_manager.start_discovery(node_id))
```

### C. Event System

```python
# RdmManager emits events for UI updates
rdm_manager.on('device_discovered', callback)
rdm_manager.on('discovery_complete', callback)
rdm_manager.on('device_updated', callback)
rdm_manager.on('patch_suggestion', callback)
```

---

## 4. Data Flow

### Discovery Flow
```
User clicks "Discover" in UI
        │
        ▼
RdmManager.start_discovery(node_id)
        │
        ▼
RdmTransport.send_discover(node_ip)
        │
        ▼ UDP JSON: {"v":2, "type":"rdm", "action":"discover", "universe":1}
        │
ESP32 performs RDM discovery on DMX bus
        │
        ▼ UDP JSON: {"v":2, "type":"rdm_response", "uids":["02CA:12345678",...]}
        │
        ▼
RdmDiscovery.process_response(uids)
        │
        ├──▶ For each UID: RdmTransport.get_device_info(uid)
        │
        ▼
DiscoveredDevice objects created
        │
        ▼
RdmManager emits 'discovery_complete' event
        │
        ▼
UI updates device list
```

### Auto-Patch Flow
```
User clicks "Auto-Patch" on discovered device
        │
        ▼
AutoPatcher.suggest_patch(discovered_device)
        │
        ├──▶ ProfileMatcher.find_match(manufacturer_id, model_id)
        │           │
        │           ▼
        │    Search FixtureLibrary by RDM IDs
        │           │
        │           ├── Exact match found → Use profile
        │           └── No match → Create generic profile
        │
        ▼
PatchSuggestion returned
        │
        ├── profile_id: matched or generic
        ├── mode_id: based on DMX footprint
        ├── start_channel: from RDM device
        ├── conflicts: any address overlaps
        │
        ▼
UI displays suggestion for user approval
        │
        ▼
User clicks "Apply"
        │
        ▼
FixtureLibrary.create_fixture_instance(suggestion)
        │
        ▼
Fixture patched and ready to control
```

---

## 5. API Contracts

### RdmTransport (Abstract)

```python
class RdmTransport(ABC):
    @abstractmethod
    async def discover(self, node_ip: str, universe: int) -> List[RdmUid]:
        """Start RDM discovery on a universe. Returns list of UIDs found."""
        pass

    @abstractmethod
    async def get_device_info(self, node_ip: str, uid: RdmUid) -> RdmDeviceInfo:
        """Get complete device information via RDM."""
        pass

    @abstractmethod
    async def identify(self, node_ip: str, uid: RdmUid, state: bool) -> bool:
        """Turn device identify mode on/off."""
        pass

    @abstractmethod
    async def set_dmx_address(self, node_ip: str, uid: RdmUid, address: int) -> bool:
        """Set device DMX start address."""
        pass

    @abstractmethod
    async def get_personalities(self, node_ip: str, uid: RdmUid) -> List[RdmPersonality]:
        """Get available DMX personalities/modes."""
        pass

    @abstractmethod
    async def set_personality(self, node_ip: str, uid: RdmUid, personality: int) -> bool:
        """Set device DMX personality/mode."""
        pass
```

### RdmDiscovery

```python
class RdmDiscovery:
    def __init__(self, transport: RdmTransport):
        pass

    async def discover_node(self, node_ip: str, universe: int) -> List[DiscoveredDevice]:
        """Discover all RDM devices on a node/universe."""
        pass

    async def refresh_device(self, node_ip: str, uid: RdmUid) -> DiscoveredDevice:
        """Refresh info for a single device."""
        pass

    def get_cached_devices(self, node_id: str = None) -> List[DiscoveredDevice]:
        """Get cached discovered devices, optionally filtered by node."""
        pass
```

### AutoPatcher

```python
class AutoPatcher:
    def __init__(self, fixture_library: FixtureLibrary):
        pass

    def suggest_patch(
        self,
        device: DiscoveredDevice,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> PatchSuggestion:
        """Generate patch suggestion for a discovered device."""
        pass

    def find_conflicts(
        self,
        start_channel: int,
        footprint: int,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> List[FixtureInstance]:
        """Find fixtures that would conflict with proposed patch."""
        pass

    def suggest_next_address(
        self,
        footprint: int,
        universe: int,
        existing_fixtures: List[FixtureInstance]
    ) -> int:
        """Suggest next available DMX address for given footprint."""
        pass
```

### RdmManager (Facade)

```python
class RdmManager:
    def __init__(
        self,
        node_manager: NodeManager,
        fixture_library: FixtureLibrary,
        db_connection: sqlite3.Connection
    ):
        pass

    # Discovery
    async def start_discovery(self, node_id: str) -> Dict[str, Any]:
        """Start discovery on a node. Returns discovery session info."""
        pass

    def get_discovery_status(self, node_id: str) -> Dict[str, Any]:
        """Get status of in-progress discovery."""
        pass

    def get_devices(self, node_id: str = None) -> List[DiscoveredDevice]:
        """Get discovered devices, optionally filtered by node."""
        pass

    # Device Control
    async def identify_device(self, uid: str, state: bool) -> bool:
        """Flash device identify LED."""
        pass

    async def set_device_address(self, uid: str, address: int) -> bool:
        """Change device DMX address."""
        pass

    # Auto-Patch
    def get_patch_suggestion(self, uid: str, universe: int) -> PatchSuggestion:
        """Get auto-patch suggestion for a device."""
        pass

    def apply_patch(self, suggestion: PatchSuggestion) -> FixtureInstance:
        """Apply a patch suggestion, creating fixture instance."""
        pass

    # Events
    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        pass

    def off(self, event: str, callback: Callable) -> None:
        """Unregister event callback."""
        pass
```

---

## 6. Non-Breaking Guarantees

### What Will NOT Change

1. **Existing Database Schema**
   - `fixtures` table unchanged
   - `rdm_devices` table unchanged
   - `fixture_profiles` table unchanged

2. **Existing API Endpoints**
   - All `/api/fixtures/*` endpoints unchanged
   - All `/api/fixture-library/*` endpoints unchanged
   - Response formats unchanged

3. **Existing Classes**
   - `FixtureProfile` dataclass unchanged
   - `FixtureInstance` dataclass unchanged
   - `FixtureLibrary` class unchanged
   - `ChannelMapper` class unchanged

4. **UDP JSON Protocol**
   - Same v2 protocol to ESP32 nodes
   - Same port (6455)
   - Same message format

### What Will Be Added

1. **New Module**: `core/rdm/` with new files
2. **New Manager**: `RdmManager` facade (replaces inline code)
3. **New Dataclasses**: RDM-specific types in `types.py`
4. **New Events**: For UI real-time updates

---

## 7. UDP JSON Protocol (RDM Commands)

### Discovery Request
```json
{
  "v": 2,
  "type": "rdm",
  "action": "discover",
  "universe": 1
}
```

### Discovery Response (from ESP32)
```json
{
  "v": 2,
  "type": "rdm_response",
  "action": "discover",
  "universe": 1,
  "uids": ["02CA:12345678", "02CA:87654321"]
}
```

### Get Device Info Request
```json
{
  "v": 2,
  "type": "rdm",
  "action": "get_info",
  "uid": "02CA:12345678"
}
```

### Get Device Info Response
```json
{
  "v": 2,
  "type": "rdm_response",
  "action": "get_info",
  "uid": "02CA:12345678",
  "manufacturer_id": 714,
  "device_model_id": 1234,
  "dmx_footprint": 8,
  "dmx_address": 1,
  "device_label": "My Fixture",
  "personalities": [
    {"id": 1, "name": "8-Channel", "footprint": 8},
    {"id": 2, "name": "16-Channel", "footprint": 16}
  ]
}
```

### Identify Device
```json
{
  "v": 2,
  "type": "rdm",
  "action": "identify",
  "uid": "02CA:12345678",
  "state": true
}
```

### Set DMX Address
```json
{
  "v": 2,
  "type": "rdm",
  "action": "set_address",
  "uid": "02CA:12345678",
  "address": 25
}
```

---

## 8. File Structure

```
aether-core/
├── core/
│   └── rdm/
│       ├── __init__.py          # Public exports
│       ├── DESIGN.md            # This document
│       ├── types.py             # Dataclasses
│       ├── transport.py         # UDP JSON communication
│       ├── discovery.py         # Device discovery
│       ├── auto_patch.py        # Auto-patching logic
│       └── manager.py           # Facade/coordinator
└── tests/
    └── unit/
        ├── test_rdm_transport.py
        ├── test_rdm_discovery.py
        └── test_rdm_auto_patch.py
```

---

## 9. Dependencies

### Internal (Aether)
- `fixture_library.FixtureProfile`
- `fixture_library.FixtureInstance`
- `fixture_library.FixtureLibrary`

### External (Python stdlib)
- `dataclasses`
- `typing`
- `asyncio`
- `socket` (for UDP)
- `json`
- `sqlite3`

### No New External Dependencies Required

---

## 10. Testing Strategy

### Unit Tests
- `test_rdm_transport.py` - Mock UDP, test message formatting
- `test_rdm_discovery.py` - Mock transport, test discovery logic
- `test_rdm_auto_patch.py` - Test profile matching, conflict detection

### Integration Tests (Future)
- Real ESP32 communication
- Database persistence
- End-to-end discovery flow

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-19 | Initial design document |
