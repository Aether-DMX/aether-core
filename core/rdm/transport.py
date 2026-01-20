"""
RDM Transport Layer - UDP JSON Communication with ESP32 Nodes

This module handles low-level RDM communication with ESP32 nodes
using the UDP JSON v2 protocol. ESP32 nodes act as RDM gateways,
translating JSON commands to RDM on the DMX bus.

Classes:
    RdmTransport: Abstract base class for RDM communication
    UdpJsonRdmTransport: UDP JSON v2 implementation

Protocol:
    All messages use UDP JSON v2 format to ESP32 nodes on port 6455.
    Request:  {"v":2, "type":"rdm", "action":"discover", "universe":1, "seq":1}
    Response: {"v":2, "type":"rdm_response", "action":"discover", "uids":[...], "seq":1}

Example:
    transport = UdpJsonRdmTransport(port=6455)
    uids = await transport.discover("192.168.1.100", universe=1)
    info = await transport.get_device_info("192.168.1.100", uids[0])
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import json
import socket
import logging
import time

from .types import (
    RdmUid,
    RdmDeviceInfo,
    RdmPersonality,
    RdmCommand,
    RdmResponse,
    RdmCommandType,
    RdmResponseType,
)

logger = logging.getLogger(__name__)


class RdmTransportError(Exception):
    """Base exception for RDM transport errors."""
    pass


class RdmTimeoutError(RdmTransportError):
    """RDM command timed out."""
    pass


class RdmConnectionError(RdmTransportError):
    """Could not connect to node."""
    pass


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
            RdmTimeoutError: If discovery times out
            RdmConnectionError: If node unreachable
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
            RdmTimeoutError: If device doesn't respond
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
        discovery_timeout: Extended timeout for discovery (default 10s)
    """

    DEFAULT_PORT = 6455
    DEFAULT_TIMEOUT = 3.0
    DEFAULT_DISCOVERY_TIMEOUT = 10.0
    DEFAULT_RETRIES = 2
    RECV_BUFFER_SIZE = 4096

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        discovery_timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
        retries: int = DEFAULT_RETRIES
    ):
        """
        Initialize UDP JSON RDM transport.

        Args:
            port: UDP port for node communication
            timeout: Response timeout in seconds
            discovery_timeout: Extended timeout for discovery
            retries: Number of retry attempts on failure
        """
        self.port = port
        self.timeout = timeout
        self.discovery_timeout = discovery_timeout
        self.retries = retries
        self._sequence = 0
        self._pending_responses: Dict[int, asyncio.Future[Dict[str, Any]]] = {}

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
            RdmTimeoutError: If no response received
            RdmConnectionError: If send fails
        """
        timeout = timeout or self.timeout
        seq = message.get("seq", 0)

        # Create UDP socket
        loop = asyncio.get_event_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)

        try:
            # Send message
            data = json.dumps(message).encode('utf-8')
            logger.debug(f"RDM TX -> {node_ip}:{self.port}: {message}")

            await loop.sock_sendto(sock, data, (node_ip, self.port))

            # Wait for response with matching sequence
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Set socket timeout for this receive
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        break

                    # Use asyncio wait_for with sock_recv
                    response_data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, self.RECV_BUFFER_SIZE),
                        timeout=min(remaining, 1.0)
                    )

                    # Parse response
                    try:
                        response = json.loads(response_data.decode('utf-8'))
                        logger.debug(f"RDM RX <- {addr}: {response}")

                        # Check if this is our response (matching seq or rdm_response type)
                        if response.get("type") == "rdm_response":
                            resp_seq = response.get("seq", -1)
                            if resp_seq == seq or seq == 0:
                                return response

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON response: {e}")
                        continue

                except asyncio.TimeoutError:
                    continue

            # Timeout
            raise RdmTimeoutError(f"No response from {node_ip} within {timeout}s")

        except OSError as e:
            raise RdmConnectionError(f"Failed to send to {node_ip}: {e}")

        finally:
            sock.close()

    async def _send_command(
        self,
        node_ip: str,
        command: RdmCommand,
        timeout: Optional[float] = None
    ) -> RdmResponse:
        """
        Send RDM command and get response.

        Args:
            node_ip: Target node IP
            command: RDM command to send
            timeout: Optional timeout override

        Returns:
            RdmResponse object
        """
        message = command.to_udp_json()

        for attempt in range(self.retries + 1):
            try:
                response_dict = await self._send_and_receive(node_ip, message, timeout)
                return RdmResponse.from_udp_json(response_dict)

            except RdmTimeoutError:
                if attempt < self.retries:
                    logger.debug(f"Retry {attempt + 1}/{self.retries} for {command.action.value}")
                    continue
                raise

            except RdmConnectionError:
                raise

        # Should not reach here
        raise RdmTimeoutError(f"Command failed after {self.retries} retries")

    async def discover(self, node_ip: str, universe: int) -> List[RdmUid]:
        """
        Start RDM discovery on a universe.

        Sends discovery command to ESP32 node and waits for UID list.
        Uses extended timeout since discovery can take several seconds.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"discover", "universe":1}
            Response: {"v":2, "type":"rdm_response", "action":"discover", "uids":["02CA:12345678",...]}
        """
        command = RdmCommand(
            action=RdmCommandType.DISCOVER,
            universe=universe,
            seq=self._next_sequence()
        )

        try:
            response = await self._send_command(
                node_ip, command, timeout=self.discovery_timeout
            )

            if not response.success:
                logger.warning(f"Discovery failed: {response.error}")
                return []

            # Parse UIDs from response
            uids: List[RdmUid] = []
            uid_strings = response.data.get("uids", []) if response.data else []

            for uid_str in uid_strings:
                try:
                    uids.append(RdmUid.from_string(uid_str))
                except ValueError as e:
                    logger.warning(f"Invalid UID in discovery: {uid_str} - {e}")

            logger.info(f"Discovery found {len(uids)} devices on universe {universe}")
            return uids

        except RdmTimeoutError:
            logger.warning(f"Discovery timeout on {node_ip} universe {universe}")
            return []

        except RdmConnectionError as e:
            logger.error(f"Discovery connection error: {e}")
            return []

    async def get_device_info(self, node_ip: str, uid: RdmUid) -> RdmDeviceInfo:
        """
        Get complete device information via RDM.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"get_info", "uid":"02CA:12345678"}
            Response: {"v":2, "type":"rdm_response", "manufacturer_id":714, ...}
        """
        command = RdmCommand(
            action=RdmCommandType.GET_INFO,
            universe=0,  # Not needed for device-specific commands
            uid=str(uid),
            seq=self._next_sequence()
        )

        response = await self._send_command(node_ip, command)

        if not response.success:
            raise RdmTransportError(
                f"Failed to get device info for {uid}: {response.error}"
            )

        data = response.data or {}

        # Parse personalities
        personalities: List[RdmPersonality] = []
        for p in data.get("personalities", []):
            personalities.append(RdmPersonality(
                id=p.get("id", 1),
                name=p.get("name", "Unknown"),
                footprint=p.get("footprint", 1)
            ))

        return RdmDeviceInfo(
            uid=uid,
            manufacturer_id=data.get("manufacturer_id", 0),
            device_model_id=data.get("device_model_id", 0),
            manufacturer_label=data.get("manufacturer_label", ""),
            device_model=data.get("device_model", ""),
            device_label=data.get("device_label", ""),
            dmx_address=data.get("dmx_address", 1),
            dmx_footprint=data.get("dmx_footprint", 1),
            current_personality=data.get("current_personality", 1),
            personalities=personalities,
            software_version=data.get("software_version", ""),
            rdm_protocol_version=data.get("rdm_protocol_version", "1.0"),
        )

    async def identify(self, node_ip: str, uid: RdmUid, state: bool) -> bool:
        """
        Turn device identify mode on/off.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"identify", "uid":"...", "state":true}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        command = RdmCommand(
            action=RdmCommandType.IDENTIFY,
            universe=0,
            uid=str(uid),
            data={"state": state},
            seq=self._next_sequence()
        )

        try:
            response = await self._send_command(node_ip, command)
            return response.success

        except (RdmTimeoutError, RdmConnectionError) as e:
            logger.warning(f"Identify command failed for {uid}: {e}")
            return False

    async def set_dmx_address(self, node_ip: str, uid: RdmUid, address: int) -> bool:
        """
        Set device DMX start address.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_address", "uid":"...", "address":25}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        if not 1 <= address <= 512:
            raise ValueError(f"DMX address must be 1-512, got {address}")

        command = RdmCommand(
            action=RdmCommandType.SET_ADDRESS,
            universe=0,
            uid=str(uid),
            data={"address": address},
            seq=self._next_sequence()
        )

        try:
            response = await self._send_command(node_ip, command)

            if response.success:
                logger.info(f"Set DMX address for {uid} to {address}")

            return response.success

        except (RdmTimeoutError, RdmConnectionError) as e:
            logger.warning(f"Set address failed for {uid}: {e}")
            return False

    async def get_personalities(self, node_ip: str, uid: RdmUid) -> List[RdmPersonality]:
        """
        Get available DMX personalities/modes.

        Uses get_device_info since personalities are included in that response.
        """
        try:
            info = await self.get_device_info(node_ip, uid)
            return info.personalities

        except RdmTransportError:
            return []

    async def set_personality(self, node_ip: str, uid: RdmUid, personality: int) -> bool:
        """
        Set device DMX personality/mode.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_personality", "uid":"...", "personality":2}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        command = RdmCommand(
            action=RdmCommandType.SET_PERSONALITY,
            universe=0,
            uid=str(uid),
            data={"personality": personality},
            seq=self._next_sequence()
        )

        try:
            response = await self._send_command(node_ip, command)

            if response.success:
                logger.info(f"Set personality for {uid} to {personality}")

            return response.success

        except (RdmTimeoutError, RdmConnectionError) as e:
            logger.warning(f"Set personality failed for {uid}: {e}")
            return False

    async def set_device_label(self, node_ip: str, uid: RdmUid, label: str) -> bool:
        """
        Set device user label.

        Protocol:
            Request:  {"v":2, "type":"rdm", "action":"set_label", "uid":"...", "label":"My Light"}
            Response: {"v":2, "type":"rdm_response", "success":true}
        """
        # RDM labels are max 32 characters
        if len(label) > 32:
            label = label[:32]

        command = RdmCommand(
            action=RdmCommandType.SET_LABEL,
            universe=0,
            uid=str(uid),
            data={"label": label},
            seq=self._next_sequence()
        )

        try:
            response = await self._send_command(node_ip, command)

            if response.success:
                logger.info(f"Set label for {uid} to '{label}'")

            return response.success

        except (RdmTimeoutError, RdmConnectionError) as e:
            logger.warning(f"Set label failed for {uid}: {e}")
            return False


# ============================================================
# Factory Function
# ============================================================

def create_transport(
    transport_type: str = "udp_json",
    **kwargs: Any
) -> RdmTransport:
    """
    Create an RDM transport instance.

    Args:
        transport_type: Transport type ("udp_json")
        **kwargs: Transport-specific options

    Returns:
        RdmTransport instance

    Raises:
        ValueError: If transport type unknown
    """
    if transport_type == "udp_json":
        return UdpJsonRdmTransport(**kwargs)

    raise ValueError(f"Unknown transport type: {transport_type}")
