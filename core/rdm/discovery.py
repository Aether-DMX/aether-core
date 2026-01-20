"""
RDM Discovery - Device Discovery and Information Gathering

This module handles RDM device discovery on ESP32 nodes.
It manages discovery sessions, caches results, and provides
device information retrieval.

Classes:
    DiscoverySession: Tracks a single discovery operation
    RdmDiscovery: Main discovery coordinator

Usage:
    discovery = RdmDiscovery(transport)
    devices = await discovery.discover_node("192.168.1.100", universe=1)
    device = await discovery.refresh_device("192.168.1.100", uid)
"""

from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import asyncio
import logging

from .types import (
    RdmUid,
    DiscoveredDevice,
    RdmDeviceInfo,
    DiscoveryStatus,
    DiscoveryState,
)
from .transport import RdmTransport

logger = logging.getLogger(__name__)


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
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Mark session as started."""
        self.status.state = DiscoveryState.DISCOVERING
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
            DiscoveryState.DISCOVERING,
            DiscoveryState.QUERYING
        )

    def get_devices(self) -> List[DiscoveredDevice]:
        """Get list of discovered devices."""
        return list(self.devices.values())


class RdmDiscovery:
    """
    RDM Device Discovery Coordinator.

    Manages discovery sessions, caches results, and coordinates
    with transport layer for device communication.

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
        self._callbacks: Dict[str, List[Callable]] = {}

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
        node_id: str = None,
        universe: int = None
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
            universe = universe or cached.universe
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

    def on(self, event: str, callback: Callable) -> None:
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

    def off(self, event: str, callback: Callable) -> None:
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
