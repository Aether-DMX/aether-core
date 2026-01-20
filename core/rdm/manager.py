"""
RDM Manager - High-Level Facade for RDM Operations

This module provides the main entry point for all RDM operations.
It coordinates transport, discovery, and auto-patching, and emits
events for UI updates.

Classes:
    RdmManager: Main RDM operations facade

Usage:
    rdm = RdmManager(node_manager, fixture_library, db)

    # Discovery
    await rdm.start_discovery(node_id)
    status = rdm.get_discovery_status(node_id)
    devices = rdm.get_devices(node_id)

    # Device control
    await rdm.identify_device(uid, True)
    await rdm.set_device_address(uid, 25)

    # Auto-patch
    suggestion = rdm.get_patch_suggestion(uid, universe=1)
    fixture = rdm.apply_patch(suggestion)

    # Events
    rdm.on('device_discovered', handler)
"""

from typing import List, Dict, Optional, Callable, Any
import sqlite3
import logging
import asyncio

from .types import (
    RdmUid,
    DiscoveredDevice,
    PatchSuggestion,
    DiscoveryStatus,
)
from .transport import RdmTransport, UdpJsonRdmTransport
from .discovery import RdmDiscovery
from .auto_patch import AutoPatcher

logger = logging.getLogger(__name__)


# Type hints for external classes
NodeManager = Any
FixtureLibrary = Any
FixtureInstance = Any


class RdmManager:
    """
    High-level facade for all RDM operations.

    Coordinates transport, discovery, and auto-patching.
    Provides a clean API for the REST layer and emits events
    for real-time UI updates.

    Attributes:
        node_manager: NodeManager for getting node info
        fixture_library: FixtureLibrary for profiles/fixtures
        db: Database connection for persistence
        transport: RDM transport layer
        discovery: RDM discovery coordinator
        auto_patcher: Auto-patch generator
    """

    def __init__(
        self,
        node_manager: NodeManager,
        fixture_library: FixtureLibrary,
        db: sqlite3.Connection
    ):
        """
        Initialize RDM Manager.

        Args:
            node_manager: NodeManager instance for node lookup
            fixture_library: FixtureLibrary for profiles
            db: SQLite connection for persistence
        """
        self.node_manager = node_manager
        self.fixture_library = fixture_library
        self.db = db

        # Initialize components
        self.transport = UdpJsonRdmTransport()
        self.discovery = RdmDiscovery(self.transport)
        self.auto_patcher = AutoPatcher(fixture_library)

        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {}

        # Wire up discovery events
        self._setup_discovery_events()

    def _setup_discovery_events(self) -> None:
        """Wire up discovery events to manager events."""
        self.discovery.on("device_discovered", lambda d: self._emit("device_discovered", d))
        self.discovery.on("discovery_complete", lambda s: self._emit("discovery_complete", s))
        self.discovery.on("discovery_error", lambda e: self._emit("discovery_error", e))

    # ─────────────────────────────────────────────────────────
    # Discovery Methods
    # ─────────────────────────────────────────────────────────

    async def start_discovery(self, node_id: str) -> Dict[str, Any]:
        """
        Start RDM discovery on a node.

        Args:
            node_id: Node identifier

        Returns:
            Discovery session info dict

        Raises:
            ValueError: If node not found
            RuntimeError: If discovery already in progress
        """
        # Get node info
        node = self._get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        node_ip = node.get("ip") or node.get("address")
        universe = node.get("universe", 1)

        # Start discovery
        self._emit("discovery_started", {"node_id": node_id})

        try:
            devices = await self.discovery.discover_node(node_id, node_ip, universe)

            # Save to database
            for device in devices:
                self._save_device(device)

            return {
                "success": True,
                "node_id": node_id,
                "devices_found": len(devices),
                "devices": [d.to_dict() for d in devices]
            }

        except Exception as e:
            logger.error(f"Discovery failed for {node_id}: {e}")
            return {
                "success": False,
                "node_id": node_id,
                "error": str(e)
            }

    def get_discovery_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of discovery session for a node.

        Args:
            node_id: Node identifier

        Returns:
            Status dict or None if no session
        """
        status = self.discovery.get_session_status(node_id)
        return status.to_dict() if status else None

    def get_devices(
        self,
        node_id: Optional[str] = None,
        universe: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get discovered devices.

        Args:
            node_id: Filter by node (optional)
            universe: Filter by universe (optional)

        Returns:
            List of device dicts
        """
        # Get from cache first
        devices = self.discovery.get_cached_devices(node_id, universe)

        # If cache empty, load from database
        if not devices:
            devices = self._load_devices(node_id, universe)

        return [d.to_dict() for d in devices]

    def get_device(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific device by UID.

        Args:
            uid: Device UID string

        Returns:
            Device dict or None
        """
        device = self.discovery.get_cached_device(uid)
        if not device:
            device = self._load_device(uid)
        return device.to_dict() if device else None

    # ─────────────────────────────────────────────────────────
    # Device Control Methods
    # ─────────────────────────────────────────────────────────

    async def identify_device(self, uid: str, state: bool) -> bool:
        """
        Flash device identify LED.

        Args:
            uid: Device UID string
            state: True to enable, False to disable

        Returns:
            True if successful
        """
        device = self.discovery.get_cached_device(uid)
        if not device:
            raise ValueError(f"Device not found: {uid}")

        node = self._get_node(device.node_id)
        if not node:
            raise ValueError(f"Node not found: {device.node_id}")

        node_ip = node.get("ip") or node.get("address")
        rdm_uid = RdmUid.from_string(uid)

        return await self.transport.identify(node_ip, rdm_uid, state)

    async def set_device_address(self, uid: str, address: int) -> bool:
        """
        Change device DMX address.

        Args:
            uid: Device UID string
            address: New DMX address (1-512)

        Returns:
            True if successful
        """
        device = self.discovery.get_cached_device(uid)
        if not device:
            raise ValueError(f"Device not found: {uid}")

        node = self._get_node(device.node_id)
        if not node:
            raise ValueError(f"Node not found: {device.node_id}")

        node_ip = node.get("ip") or node.get("address")
        rdm_uid = RdmUid.from_string(uid)

        success = await self.transport.set_dmx_address(node_ip, rdm_uid, address)

        if success:
            # Update cache and database
            device.dmx_address = address
            self._save_device(device)
            self._emit("device_updated", device)

        return success

    async def set_device_label(self, uid: str, label: str) -> bool:
        """
        Set device user label.

        Args:
            uid: Device UID string
            label: New label

        Returns:
            True if successful
        """
        device = self.discovery.get_cached_device(uid)
        if not device:
            raise ValueError(f"Device not found: {uid}")

        node = self._get_node(device.node_id)
        if not node:
            raise ValueError(f"Node not found: {device.node_id}")

        node_ip = node.get("ip") or node.get("address")
        rdm_uid = RdmUid.from_string(uid)

        success = await self.transport.set_device_label(node_ip, rdm_uid, label)

        if success:
            device.device_label = label
            self._save_device(device)
            self._emit("device_updated", device)

        return success

    # ─────────────────────────────────────────────────────────
    # Auto-Patch Methods
    # ─────────────────────────────────────────────────────────

    def get_patch_suggestion(
        self,
        uid: str,
        universe: int
    ) -> Dict[str, Any]:
        """
        Get auto-patch suggestion for a device.

        Args:
            uid: Device UID string
            universe: Target universe

        Returns:
            PatchSuggestion dict
        """
        device = self.discovery.get_cached_device(uid)
        if not device:
            device = self._load_device(uid)
        if not device:
            raise ValueError(f"Device not found: {uid}")

        # Get existing fixtures for conflict check
        existing = self._get_fixtures(universe)

        suggestion = self.auto_patcher.suggest_patch(device, universe, existing)
        self._emit("patch_suggestion", suggestion)

        return suggestion.to_dict()

    def apply_patch(
        self,
        suggestion_data: Dict[str, Any],
        fixture_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply a patch suggestion, creating fixture instance.

        Args:
            suggestion_data: PatchSuggestion dict from get_patch_suggestion
            fixture_name: Optional custom name

        Returns:
            Created fixture dict

        Raises:
            ValueError: If suggestion has conflicts
        """
        # Reconstruct suggestion from dict
        # TODO: Implement proper deserialization
        # suggestion = PatchSuggestion.from_dict(suggestion_data)

        # fixture = self.auto_patcher.apply_patch(suggestion, fixture_name)

        # Mark device as patched
        # uid = suggestion_data.get("device", {}).get("uid")
        # if uid:
        #     device = self.discovery.get_cached_device(uid)
        #     if device:
        #         device.is_patched = True
        #         device.fixture_id = fixture.fixture_id
        #         self._save_device(device)

        # return fixture.to_dict()

        raise NotImplementedError("apply_patch not yet implemented")

    def suggest_next_address(
        self,
        footprint: int,
        universe: int
    ) -> int:
        """
        Suggest next available DMX address.

        Args:
            footprint: Required channel count
            universe: Target universe

        Returns:
            Suggested address or -1 if no space
        """
        existing = self._get_fixtures(universe)
        return self.auto_patcher.suggest_next_address(footprint, universe, existing)

    # ─────────────────────────────────────────────────────────
    # Event System
    # ─────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        """
        Register event callback.

        Events:
            - discovery_started: Discovery began
            - device_discovered: New device found
            - device_updated: Device info changed
            - discovery_complete: Discovery finished
            - discovery_error: Discovery failed
            - patch_suggestion: Patch suggestion generated

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

    # ─────────────────────────────────────────────────────────
    # Private Helper Methods
    # ─────────────────────────────────────────────────────────

    def _get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node info from node_manager."""
        # TODO: Implement - call node_manager.get_node()
        # return self.node_manager.get_node(node_id)
        return None

    def _get_fixtures(self, universe: int) -> List[FixtureInstance]:
        """Get fixtures in a universe."""
        # TODO: Implement - call fixture_library.get_fixtures()
        # return self.fixture_library.get_fixtures(universe=universe)
        return []

    def _save_device(self, device: DiscoveredDevice) -> None:
        """Save device to database."""
        # TODO: Implement - save to rdm_devices table
        pass

    def _load_device(self, uid: str) -> Optional[DiscoveredDevice]:
        """Load device from database."""
        # TODO: Implement - load from rdm_devices table
        return None

    def _load_devices(
        self,
        node_id: Optional[str] = None,
        universe: Optional[int] = None
    ) -> List[DiscoveredDevice]:
        """Load devices from database."""
        # TODO: Implement - load from rdm_devices table
        return []
