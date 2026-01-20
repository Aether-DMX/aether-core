"""
RDM Discovery Engine - Device Discovery with Conflict Detection

This module handles RDM device discovery on ESP32 nodes via UDP JSON.
It manages discovery sessions, queries device info, enriches with
fixture database data, and detects address conflicts.

Classes:
    DiscoverySession: Tracks a single discovery operation
    DiscoveryEngine: Main discovery coordinator with enrichment and conflict detection

Usage:
    engine = DiscoveryEngine(transport, fixture_db)
    engine.on_progress(lambda pct, msg: print(f"{pct}%: {msg}"))
    fixtures = await engine.discover_universes([1, 2])
"""

from typing import List, Dict, Optional, Callable, Any, Protocol
from datetime import datetime
import asyncio
import logging

from .types import (
    RdmUid,
    DiscoveredDevice,
    DiscoveredFixture,
    RdmDeviceInfo,
    DiscoveryStatus,
    DiscoveryState,
)
from .transport import RdmTransport, RdmTimeoutError, RdmTransportError

logger = logging.getLogger(__name__)

# Constants
FIXTURE_QUERY_TIMEOUT_MS = 1000
TOTAL_DISCOVERY_TIMEOUT_S = 30.0
PROGRESS_INTERVAL_PERCENT = 10


class FixtureDatabase(Protocol):
    """Protocol for fixture database interface."""

    def get_profile_by_rdm_ids(
        self, manufacturer_id: int, model_id: int
    ) -> Optional[Dict[str, Any]]:
        """Look up profile by RDM IDs."""
        ...

    def get_profile_by_name(
        self, manufacturer: str, model: str
    ) -> Optional[Dict[str, Any]]:
        """Look up profile by manufacturer and model name."""
        ...


class DiscoverySession:
    """
    Tracks a single RDM discovery operation.

    A session is created for each discovery request and tracks
    progress, found devices, and completion status.

    Attributes:
        node_id: ID of node being scanned
        node_ip: IP address of node
        universe: DMX universe being scanned
        status: Current discovery status
        devices: Discovered devices
    """

    def __init__(self, node_id: str, node_ip: str, universe: int):
        """
        Initialize discovery session.

        Args:
            node_id: Node identifier
            node_ip: Node IP address
            universe: DMX universe to scan
        """
        self.node_id = node_id
        self.node_ip = node_ip
        self.universe = universe
        self.status = DiscoveryStatus(
            node_id=node_id,
            state=DiscoveryState.IDLE
        )
        self.devices: Dict[str, DiscoveredDevice] = {}
        self._uids: List[RdmUid] = []
        self._task: Optional[asyncio.Task[Any]] = None

    def start(self) -> None:
        """Mark session as started."""
        self.status.state = DiscoveryState.BROADCASTING
        self.status.started_at = datetime.now()
        self.status.error = None

    def set_uids(self, uids: List[RdmUid]) -> None:
        """Set discovered UIDs, move to querying phase."""
        self._uids = uids
        self.status.devices_found = len(uids)
        self.status.state = DiscoveryState.QUERYING

    def add_device(self, device: DiscoveredDevice) -> None:
        """Add a discovered device."""
        self.devices[str(device.uid)] = device
        self.status.devices_queried = len(self.devices)

    def complete(self) -> None:
        """Mark session as complete."""
        self.status.state = DiscoveryState.COMPLETE
        self.status.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark session as failed."""
        self.status.state = DiscoveryState.ERROR
        self.status.error = error
        self.status.completed_at = datetime.now()

    def is_active(self) -> bool:
        """Check if session is still in progress."""
        return self.status.state in (
            DiscoveryState.BROADCASTING,
            DiscoveryState.QUERYING,
            DiscoveryState.ENRICHING,
            DiscoveryState.CONFLICT_CHECK,
        )

    def get_devices(self) -> List[DiscoveredDevice]:
        """Get list of discovered devices."""
        return list(self.devices.values())


class DiscoveryEngine:
    """
    RDM Discovery Engine with enrichment and conflict detection.

    Coordinates discovery across multiple universes, queries device
    information, enriches with fixture database data, and detects
    address conflicts.

    Attributes:
        transport: RDM transport layer
        fixture_db: Fixture database for enrichment (optional)
    """

    def __init__(
        self,
        rdm_transport: RdmTransport,
        fixture_db: Optional[Any] = None
    ):
        """
        Initialize discovery engine.

        Args:
            rdm_transport: RDM transport layer for communication
            fixture_db: Fixture database for enrichment (optional)
        """
        self.transport = rdm_transport
        self.fixture_db = fixture_db
        self._progress_callbacks: List[Callable[[int, str], None]] = []
        self._state = DiscoveryState.IDLE
        self._last_progress_percent = 0

    def on_progress(self, callback: Callable[[int, str], None]) -> None:
        """
        Register progress callback.

        Callback receives (percent: int, message: str) at ~10% intervals.

        Args:
            callback: Progress callback function
        """
        self._progress_callbacks.append(callback)

    def off_progress(self, callback: Callable[[int, str], None]) -> None:
        """
        Unregister progress callback.

        Args:
            callback: Progress callback function to remove
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    async def discover_universes(
        self,
        node_ip: str,
        universe_ids: List[int]
    ) -> Dict[str, DiscoveredFixture]:
        """
        Discover all RDM fixtures on specified universes.

        This is the main entry point for discovery. It:
        1. Broadcasts discovery on each universe
        2. Queries each responding fixture for info
        3. Enriches with fixture database
        4. Detects conflicts
        5. Returns all discovered fixtures

        Args:
            node_ip: IP address of ESP32 node
            universe_ids: List of universe numbers to scan

        Returns:
            Dictionary of discovered fixtures keyed by UID string

        Raises:
            asyncio.TimeoutError: If total discovery exceeds 30 seconds
        """
        self._state = DiscoveryState.IDLE
        self._last_progress_percent = 0
        all_fixtures: Dict[str, DiscoveredFixture] = {}

        try:
            # Apply total timeout
            async with asyncio.timeout(TOTAL_DISCOVERY_TIMEOUT_S):
                self._emit_progress(0, "Starting discovery...")
                self._state = DiscoveryState.BROADCASTING

                # Discover each universe
                total_universes = len(universe_ids)
                for i, universe in enumerate(universe_ids):
                    universe_progress_base = int((i / total_universes) * 70)
                    self._emit_progress(
                        universe_progress_base,
                        f"Scanning universe {universe}..."
                    )

                    try:
                        fixtures = await self._discover_universe(
                            node_ip, universe, universe_progress_base, total_universes
                        )
                        all_fixtures.update(fixtures)
                    except Exception as e:
                        logger.warning(f"Failed to discover universe {universe}: {e}")

                # Enrichment phase
                self._state = DiscoveryState.ENRICHING
                self._emit_progress(75, "Enriching fixture data...")
                for fixture in all_fixtures.values():
                    self._enrich_fixture(fixture)

                # Conflict detection phase
                self._state = DiscoveryState.CONFLICT_CHECK
                self._emit_progress(85, "Checking for conflicts...")
                self._detect_conflicts(all_fixtures)

                # Sort by universe, then start address
                sorted_fixtures: Dict[str, DiscoveredFixture] = {}
                for uid in sorted(
                    all_fixtures.keys(),
                    key=lambda u: (all_fixtures[u].universe, all_fixtures[u].start_address)
                ):
                    sorted_fixtures[uid] = all_fixtures[uid]

                self._state = DiscoveryState.COMPLETE
                self._emit_progress(100, f"Discovery complete. Found {len(sorted_fixtures)} fixtures.")

                logger.info(
                    f"Discovery complete: {len(sorted_fixtures)} fixtures on "
                    f"{len(universe_ids)} universes"
                )

                return sorted_fixtures

        except asyncio.TimeoutError:
            self._state = DiscoveryState.ERROR
            self._emit_progress(0, "Discovery timed out")
            logger.error("Discovery timed out after 30 seconds")
            raise

        except Exception as e:
            self._state = DiscoveryState.ERROR
            self._emit_progress(0, f"Discovery error: {e}")
            logger.error(f"Discovery error: {e}")
            raise

    async def _discover_universe(
        self,
        node_ip: str,
        universe: int,
        progress_base: int,
        total_universes: int
    ) -> Dict[str, DiscoveredFixture]:
        """
        Discover fixtures on a single universe.

        Args:
            node_ip: ESP32 node IP address
            universe: Universe number
            progress_base: Base progress percentage for this universe
            total_universes: Total number of universes being scanned

        Returns:
            Dictionary of discovered fixtures keyed by UID
        """
        fixtures: Dict[str, DiscoveredFixture] = {}

        # Broadcast discovery
        logger.debug(f"Broadcasting discovery on universe {universe}")
        try:
            uids = await self.transport.discover(node_ip, universe)
        except RdmTimeoutError:
            logger.debug(f"No devices responded on universe {universe}")
            return fixtures
        except RdmTransportError as e:
            logger.warning(f"Transport error on universe {universe}: {e}")
            return fixtures

        if not uids:
            logger.debug(f"No devices found on universe {universe}")
            return fixtures

        logger.info(f"Found {len(uids)} devices on universe {universe}")
        self._state = DiscoveryState.QUERYING

        # Calculate progress increment per fixture
        universe_progress_range = 70 // total_universes
        progress_per_fixture = universe_progress_range // max(len(uids), 1)

        # Query each fixture
        for i, uid in enumerate(uids):
            fixture_progress = progress_base + int(i * progress_per_fixture)
            if fixture_progress - self._last_progress_percent >= PROGRESS_INTERVAL_PERCENT:
                self._emit_progress(
                    fixture_progress,
                    f"Querying {uid} ({i + 1}/{len(uids)})..."
                )

            try:
                fixture = await self._query_fixture(node_ip, uid, universe)
                if fixture:
                    fixtures[fixture.uid] = fixture
                    logger.debug(f"Discovered: {uid} - {fixture.manufacturer} {fixture.model} @ {fixture.start_address}")
            except Exception as e:
                logger.warning(f"Failed to query {uid}: {e}")
                # Continue with other fixtures

        return fixtures

    async def _query_fixture(
        self,
        node_ip: str,
        uid: RdmUid,
        universe: int
    ) -> Optional[DiscoveredFixture]:
        """
        Query a single fixture for its information.

        Args:
            node_ip: ESP32 node IP address
            uid: RDM UID of the fixture
            universe: Universe number

        Returns:
            DiscoveredFixture or None if query failed
        """
        try:
            # Apply per-fixture timeout
            async with asyncio.timeout(FIXTURE_QUERY_TIMEOUT_MS / 1000.0):
                info = await self.transport.get_device_info(node_ip, uid)
                return DiscoveredFixture.from_device_info(info, universe)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout querying {uid}")
            return None
        except RdmTimeoutError:
            logger.warning(f"RDM timeout querying {uid}")
            return None
        except RdmTransportError as e:
            logger.warning(f"Transport error querying {uid}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying {uid}: {e}")
            return None

    def _enrich_fixture(self, fixture: DiscoveredFixture) -> None:
        """
        Enrich fixture with data from fixture database.

        Cross-references the fixture with the fixture database to add:
        - Capabilities
        - Fixture type
        - Additional metadata

        Args:
            fixture: Fixture to enrich (modified in place)
        """
        if not self.fixture_db:
            # No database, try to guess fixture type from model name
            fixture.fixture_type = self._guess_fixture_type(fixture.model)
            return

        # Try lookup by RDM IDs first (most reliable)
        profile = None
        if hasattr(self.fixture_db, 'get_profile_by_rdm_ids'):
            try:
                # Parse manufacturer_id from UID (first 4 hex digits)
                uid_parts = fixture.uid.split(":")
                if len(uid_parts) == 2:
                    mfg_id = int(uid_parts[0], 16)
                    profile = self.fixture_db.get_profile_by_rdm_ids(
                        mfg_id, fixture.device_id
                    )
            except (ValueError, AttributeError):
                pass

        # Fall back to name-based lookup
        if not profile and hasattr(self.fixture_db, 'get_profile_by_name'):
            try:
                profile = self.fixture_db.get_profile_by_name(
                    fixture.manufacturer, fixture.model
                )
            except (ValueError, AttributeError):
                pass

        if profile:
            # Merge capabilities from profile
            if 'capabilities' in profile:
                existing_caps = set(fixture.capabilities)
                for cap in profile.get('capabilities', []):
                    if cap not in existing_caps:
                        fixture.capabilities.append(cap)

            # Set fixture type from profile
            if 'fixture_type' in profile:
                fixture.fixture_type = profile['fixture_type']
            elif 'type' in profile:
                fixture.fixture_type = profile['type']
        else:
            # No profile match, guess from model name
            fixture.fixture_type = self._guess_fixture_type(fixture.model)

    def _guess_fixture_type(self, model: str) -> Optional[str]:
        """
        Guess fixture type from model name.

        Args:
            model: Model name string

        Returns:
            Guessed fixture type or None
        """
        model_lower = model.lower()

        # Check moving head first (before spot/wash/beam which might be part of the name)
        if any(kw in model_lower for kw in ['moving head', 'mh', 'moving-head']):
            return 'moving_head'
        if any(kw in model_lower for kw in ['spot', 'profile', 'leko']):
            return 'spot'
        if any(kw in model_lower for kw in ['wash', 'flood']):
            return 'wash'
        if any(kw in model_lower for kw in ['beam', 'sharpy']):
            return 'beam'
        if any(kw in model_lower for kw in ['par', 'can']):
            return 'par'
        if any(kw in model_lower for kw in ['strip', 'bar', 'batten']):
            return 'strip'
        if any(kw in model_lower for kw in ['strobe', 'atomic']):
            return 'strobe'
        if any(kw in model_lower for kw in ['laser']):
            return 'laser'
        if any(kw in model_lower for kw in ['hazer', 'haze', 'fog', 'smoke']):
            return 'atmospheric'
        if any(kw in model_lower for kw in ['scanner', 'scan']):
            return 'scanner'
        if any(kw in model_lower for kw in ['led', 'pixel']):
            return 'led'

        return None

    def _detect_conflicts(self, fixtures: Dict[str, DiscoveredFixture]) -> None:
        """
        Detect address conflicts between fixtures.

        For each pair of fixtures on the same universe, checks if their
        channel ranges overlap. If so, adds conflict notes to both.

        Args:
            fixtures: Dictionary of fixtures to check (modified in place)
        """
        # Group fixtures by universe
        by_universe: Dict[int, List[DiscoveredFixture]] = {}
        for fixture in fixtures.values():
            if fixture.universe not in by_universe:
                by_universe[fixture.universe] = []
            by_universe[fixture.universe].append(fixture)

        # Check each universe for conflicts
        for universe, universe_fixtures in by_universe.items():
            n = len(universe_fixtures)
            for i in range(n):
                for j in range(i + 1, n):
                    f1 = universe_fixtures[i]
                    f2 = universe_fixtures[j]

                    # Check if channel ranges overlap
                    range1 = f1.channel_range()
                    range2 = f2.channel_range()

                    overlap_start = max(range1.start, range2.start)
                    overlap_end = min(range1.stop, range2.stop)

                    if overlap_start < overlap_end:
                        # Conflict detected
                        overlap_channels = list(range(overlap_start, overlap_end))
                        conflict_msg_1 = (
                            f"Overlaps with {f2.manufacturer} {f2.model} "
                            f"({f2.uid}) on channels {overlap_channels[0]}-{overlap_channels[-1]}"
                        )
                        conflict_msg_2 = (
                            f"Overlaps with {f1.manufacturer} {f1.model} "
                            f"({f1.uid}) on channels {overlap_channels[0]}-{overlap_channels[-1]}"
                        )

                        f1.conflicts.append(conflict_msg_1)
                        f2.conflicts.append(conflict_msg_2)

                        logger.warning(
                            f"Conflict on universe {universe}: "
                            f"{f1.uid} ({f1.start_address}-{f1.start_address + f1.channel_count - 1}) "
                            f"overlaps {f2.uid} ({f2.start_address}-{f2.start_address + f2.channel_count - 1})"
                        )

    def _emit_progress(self, percent: int, message: str) -> None:
        """
        Emit progress to registered callbacks.

        Only emits if progress has changed by at least PROGRESS_INTERVAL_PERCENT
        or if this is the first/last update.

        Args:
            percent: Progress percentage (0-100)
            message: Progress message
        """
        # Always emit 0% and 100%
        should_emit = percent == 0 or percent == 100

        # Emit if we've advanced by at least 10%
        if percent - self._last_progress_percent >= PROGRESS_INTERVAL_PERCENT:
            should_emit = True

        if should_emit:
            self._last_progress_percent = percent
            for callback in self._progress_callbacks:
                try:
                    callback(percent, message)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")


# Legacy RdmDiscovery class for backward compatibility
class RdmDiscovery:
    """
    Legacy RDM Device Discovery Coordinator.

    This class is maintained for backward compatibility.
    New code should use DiscoveryEngine instead.

    Attributes:
        transport: RDM transport layer
        sessions: Active discovery sessions by node_id
        device_cache: Cached discovered devices by UID
    """

    def __init__(self, transport: RdmTransport):
        """
        Initialize RDM discovery.

        Args:
            transport: RDM transport layer for communication
        """
        self.transport = transport
        self.sessions: Dict[str, DiscoverySession] = {}
        self.device_cache: Dict[str, DiscoveredDevice] = {}
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {}

    async def discover_node(
        self,
        node_id: str,
        node_ip: str,
        universe: int
    ) -> List[DiscoveredDevice]:
        """
        Discover all RDM devices on a node/universe.

        Creates a discovery session, scans for UIDs, then queries
        each device for its information.

        Args:
            node_id: Node identifier
            node_ip: Node IP address
            universe: DMX universe to scan

        Returns:
            List of discovered devices

        Raises:
            RuntimeError: If discovery already in progress for node
        """
        # Check for existing active session
        if node_id in self.sessions and self.sessions[node_id].is_active():
            raise RuntimeError(f"Discovery already in progress for node {node_id}")

        # Create new session
        session = DiscoverySession(node_id, node_ip, universe)
        self.sessions[node_id] = session

        try:
            # Start discovery
            session.start()
            self._emit("discovery_started", session.status)

            # Get UIDs from transport
            uids = await self.transport.discover(node_ip, universe)
            session.set_uids(uids)
            self._emit("uids_found", {"node_id": node_id, "count": len(uids)})

            # Query each device
            for uid in uids:
                try:
                    info = await self.transport.get_device_info(node_ip, uid)
                    device = DiscoveredDevice.from_device_info(info, node_id, universe)
                    session.add_device(device)
                    self.device_cache[str(uid)] = device
                    self._emit("device_discovered", device)
                except Exception as e:
                    logger.warning(f"Failed to get info for {uid}: {e}")

            # Complete session
            session.complete()
            self._emit("discovery_complete", session.status)

            return session.get_devices()

        except Exception as e:
            session.fail(str(e))
            self._emit("discovery_error", {"node_id": node_id, "error": str(e)})
            raise

    async def refresh_device(
        self,
        node_ip: str,
        uid: RdmUid,
        node_id: Optional[str] = None,
        universe: Optional[int] = None
    ) -> DiscoveredDevice:
        """
        Refresh information for a single device.

        Args:
            node_ip: Node IP address
            uid: Device RDM UID
            node_id: Node identifier (uses cached if not provided)
            universe: DMX universe (uses cached if not provided)

        Returns:
            Updated device information

        Raises:
            ValueError: If device not in cache and node_id/universe not provided
        """
        uid_str = str(uid)

        # Get node_id and universe from cache if not provided
        if uid_str in self.device_cache:
            cached = self.device_cache[uid_str]
            node_id = node_id or cached.node_id
            universe = universe if universe is not None else cached.universe
        elif node_id is None or universe is None:
            raise ValueError("Device not in cache; node_id and universe required")

        # Query device
        info = await self.transport.get_device_info(node_ip, uid)
        device = DiscoveredDevice.from_device_info(info, node_id, universe)

        # Update cache
        self.device_cache[uid_str] = device
        self._emit("device_updated", device)

        return device

    def get_session_status(self, node_id: str) -> Optional[DiscoveryStatus]:
        """
        Get status of discovery session for a node.

        Args:
            node_id: Node identifier

        Returns:
            Session status or None if no session
        """
        session = self.sessions.get(node_id)
        return session.status if session else None

    def get_cached_devices(
        self,
        node_id: Optional[str] = None,
        universe: Optional[int] = None
    ) -> List[DiscoveredDevice]:
        """
        Get cached discovered devices.

        Args:
            node_id: Filter by node (optional)
            universe: Filter by universe (optional)

        Returns:
            List of cached devices matching filters
        """
        devices = list(self.device_cache.values())

        if node_id is not None:
            devices = [d for d in devices if d.node_id == node_id]

        if universe is not None:
            devices = [d for d in devices if d.universe == universe]

        return devices

    def get_cached_device(self, uid: str) -> Optional[DiscoveredDevice]:
        """
        Get a specific cached device by UID.

        Args:
            uid: Device UID string

        Returns:
            Cached device or None
        """
        return self.device_cache.get(uid)

    def clear_cache(self, node_id: Optional[str] = None) -> None:
        """
        Clear device cache.

        Args:
            node_id: Clear only devices from this node (optional)
        """
        if node_id is None:
            self.device_cache.clear()
        else:
            self.device_cache = {
                uid: dev for uid, dev in self.device_cache.items()
                if dev.node_id != node_id
            }

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Register event callback.

        Events:
            - discovery_started: Session started
            - uids_found: UIDs discovered
            - device_discovered: Device info retrieved
            - device_updated: Device info refreshed
            - discovery_complete: Session finished
            - discovery_error: Session failed

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Unregister event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event] = [
                cb for cb in self._callbacks[event] if cb != callback
            ]

    def _emit(self, event: str, data: Any) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")
