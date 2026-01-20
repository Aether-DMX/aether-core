"""
RDM Transport Layer - UDP JSON Communication with ESP32 Nodes

This module handles low-level RDM communication with ESP32 nodes
using the UDP JSON v2 protocol.

Classes:
    RdmTransport: Abstract base class for RDM communication
    UdpJsonRdmTransport: UDP JSON v2 implementation

Protocol:
    All messages use UDP JSON v2 format to ESP32 nodes on port 6455.
    ESP32 nodes act as RDM gateways, translating JSON to/from RDM.

Example:
    transport = UdpJsonRdmTransport(port=6455)
    uids = await transport.discover("192.168.1.100", universe=1)
    info = await transport.get_device_info("192.168.1.100", uids[0])
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import asyncio
import json
import socket
import logging

from .types import RdmUid, RdmDeviceInfo, RdmPersonality

logger = logging.getLogger(__name__)


class RdmTransport(ABC):
    """
    Abstract base class for RDM communication.

    Defines the interface for RDM transport implementations.
    All methods are async to support non-blocking I/O.
    """

    @abstractmethod
    async def discover(self, node_ip: str, universe: int) -> List[RdmUid]:
        """
        Start RDM discovery on a universe.

        Args:
            node_ip: IP address of ESP32 node
            universe: DMX universe to scan

        Returns:
            List of RDM UIDs found

        Raises:
            TimeoutError: If discovery times out
            ConnectionError: If node unreachable
        """
        pass

    @abstractmethod
    async def get_device_info(self, node_ip: str, uid: RdmUid) -> RdmDeviceInfo:
        """
        Get complete device information via RDM.

        Queries device for manufacturer, model, DMX info, personalities, etc.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID

        Returns:
            Complete device information

        Raises:
            TimeoutError: If device doesn't respond
            ValueError: If response is invalid
        """
        pass

    @abstractmethod
    async def identify(self, node_ip: str, uid: RdmUid, state: bool) -> bool:
        """
        Turn device identify mode on/off.

        Makes the device flash its LED or display to help locate it.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID
            state: True to enable, False to disable

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def set_dmx_address(self, node_ip: str, uid: RdmUid, address: int) -> bool:
        """
        Set device DMX start address.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID
            address: New DMX start address (1-512)

        Returns:
            True if successful

        Raises:
            ValueError: If address out of range
        """
        pass

    @abstractmethod
    async def get_personalities(self, node_ip: str, uid: RdmUid) -> List[RdmPersonality]:
        """
        Get available DMX personalities/modes.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID

        Returns:
            List of available personalities
        """
        pass

    @abstractmethod
    async def set_personality(self, node_ip: str, uid: RdmUid, personality: int) -> bool:
        """
        Set device DMX personality/mode.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID
            personality: Personality index (1-based)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def set_device_label(self, node_ip: str, uid: RdmUid, label: str) -> bool:
        """
        Set device user label.

        Args:
            node_ip: IP address of ESP32 node
            uid: Device RDM UID
            label: New label (max 32 chars)

        Returns:
            True if successful
        """
        pass


class UdpJsonRdmTransport(RdmTransport):
    """
    UDP JSON v2 implementation of RDM transport.

    Communicates with ESP32 nodes using UDP JSON protocol.
    Nodes translate JSON messages to/from RDM on the DMX bus.

    Attributes:
        port: UDP port for communication (default 6455)
        timeout: Response timeout in seconds
        retries: Number of retry attempts
    """

    DEFAULT_PORT = 6455
    DEFAULT_TIMEOUT = 5.0
    DEFAULT_RETRIES = 3

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES
    ):
        """
        Initialize UDP JSON RDM transport.

        Args:
            port: UDP port for node communication
            timeout: Response timeout in seconds
            retries: Number of retry attempts on failure
        """
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self._sequence = 0

    def _next_sequence(self) -> int:
        """Get next message sequence number."""
        self._sequence = (self._sequence + 1) % 65536
        return self._sequence

    async def _send_and_receive(
        self,
        node_ip: str,
        message: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Send UDP JSON message and wait for response.

        Args:
            node_ip: Target node IP
            message: Message dictionary
            timeout: Override default timeout

        Returns:
            Response dictionary

        Raises:
            TimeoutError: If no response received
            ConnectionError: If send fails
        """
        # TODO: Implement UDP send/receive
        raise NotImplementedError("UDP JSON transport not yet implemented")

    async def discover(self, node_ip: str, universe: int) -> List[RdmUid]:
        """
        Start RDM discovery on a universe.

        Sends discovery command to ESP32 node and waits for UID list.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"discover", "universe":1}
            Response: {"v":2, "type":"rdm_response", "uids":["02CA:12345678",...]}
        """
        message = {
            "v": 2,
            "type": "rdm",
            "action": "discover",
            "universe": universe,
            "seq": self._next_sequence()
        }

        # TODO: Implement discovery
        raise NotImplementedError("Discovery not yet implemented")

    async def get_device_info(self, node_ip: str, uid: RdmUid) -> RdmDeviceInfo:
        """
        Get complete device information via RDM.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"get_info", "uid":"02CA:12345678"}
            Response: {"v":2, "type":"rdm_response", "manufacturer_id":714, ...}
        """
        message = {
            "v": 2,
            "type": "rdm",
            "action": "get_info",
            "uid": str(uid),
            "seq": self._next_sequence()
        }

        # TODO: Implement get_device_info
        raise NotImplementedError("Get device info not yet implemented")

    async def identify(self, node_ip: str, uid: RdmUid, state: bool) -> bool:
        """
        Turn device identify mode on/off.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"identify", "uid":"...", "state":true}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        message = {
            "v": 2,
            "type": "rdm",
            "action": "identify",
            "uid": str(uid),
            "state": state,
            "seq": self._next_sequence()
        }

        # TODO: Implement identify
        raise NotImplementedError("Identify not yet implemented")

    async def set_dmx_address(self, node_ip: str, uid: RdmUid, address: int) -> bool:
        """
        Set device DMX start address.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_address", "uid":"...", "address":25}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        if not 1 <= address <= 512:
            raise ValueError(f"DMX address must be 1-512, got {address}")

        message = {
            "v": 2,
            "type": "rdm",
            "action": "set_address",
            "uid": str(uid),
            "address": address,
            "seq": self._next_sequence()
        }

        # TODO: Implement set_dmx_address
        raise NotImplementedError("Set DMX address not yet implemented")

    async def get_personalities(self, node_ip: str, uid: RdmUid) -> List[RdmPersonality]:
        """
        Get available DMX personalities/modes.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"get_personalities", "uid":"..."}
            Response: {"v":2, "type":"rdm_response", "personalities":[...]}
        """
        message = {
            "v": 2,
            "type": "rdm",
            "action": "get_personalities",
            "uid": str(uid),
            "seq": self._next_sequence()
        }

        # TODO: Implement get_personalities
        raise NotImplementedError("Get personalities not yet implemented")

    async def set_personality(self, node_ip: str, uid: RdmUid, personality: int) -> bool:
        """
        Set device DMX personality/mode.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_personality", "uid":"...", "personality":2}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        message = {
            "v": 2,
            "type": "rdm",
            "action": "set_personality",
            "uid": str(uid),
            "personality": personality,
            "seq": self._next_sequence()
        }

        # TODO: Implement set_personality
        raise NotImplementedError("Set personality not yet implemented")

    async def set_device_label(self, node_ip: str, uid: RdmUid, label: str) -> bool:
        """
        Set device user label.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_label", "uid":"...", "label":"My Light"}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        if len(label) > 32:
            label = label[:32]

        message = {
            "v": 2,
            "type": "rdm",
            "action": "set_label",
            "uid": str(uid),
            "label": label,
            "seq": self._next_sequence()
        }

        # TODO: Implement set_device_label
        raise NotImplementedError("Set device label not yet implemented")
