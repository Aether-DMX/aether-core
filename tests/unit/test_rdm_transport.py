"""
Unit Tests for RDM Transport Layer

Tests for:
- RdmUid parsing and formatting
- RdmCommand/RdmResponse serialization
- UdpJsonRdmTransport message handling
- Timeout and error handling
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.rdm.types import (
    RdmUid,
    RdmCommand,
    RdmResponse,
    RdmCommandType,
    RdmResponseType,
    RdmNackReason,
    RdmPid,
    RdmDeviceInfo,
    RdmPersonality,
)
from core.rdm.transport import (
    UdpJsonRdmTransport,
    RdmTransportError,
    RdmTimeoutError,
    RdmConnectionError,
    create_transport,
)


class TestRdmUid:
    """Tests for RDM UID parsing and formatting."""

    def test_uid_to_string(self):
        """Test UID string formatting."""
        uid = RdmUid(manufacturer_id=0x02CA, device_id=0x12345678)
        assert str(uid) == "02CA:12345678"

    def test_uid_to_string_zero_padded(self):
        """Test UID string zero-padding."""
        uid = RdmUid(manufacturer_id=0x0001, device_id=0x00000001)
        assert str(uid) == "0001:00000001"

    def test_uid_from_string(self):
        """Test UID parsing from string."""
        uid = RdmUid.from_string("02CA:12345678")
        assert uid.manufacturer_id == 0x02CA
        assert uid.device_id == 0x12345678

    def test_uid_from_string_lowercase(self):
        """Test UID parsing handles lowercase."""
        uid = RdmUid.from_string("02ca:12345678")
        assert uid.manufacturer_id == 0x02CA
        assert uid.device_id == 0x12345678

    def test_uid_from_string_invalid_format(self):
        """Test UID parsing rejects invalid format."""
        with pytest.raises(ValueError, match="Invalid RDM UID format"):
            RdmUid.from_string("invalid")

    def test_uid_from_string_missing_colon(self):
        """Test UID parsing rejects missing colon."""
        with pytest.raises(ValueError):
            RdmUid.from_string("02CA12345678")

    def test_uid_roundtrip(self):
        """Test UID string roundtrip."""
        original = RdmUid(manufacturer_id=0x1234, device_id=0xABCDEF01)
        parsed = RdmUid.from_string(str(original))
        assert parsed.manufacturer_id == original.manufacturer_id
        assert parsed.device_id == original.device_id


class TestRdmCommand:
    """Tests for RDM command serialization."""

    def test_discover_command(self):
        """Test discover command serialization."""
        cmd = RdmCommand(
            action=RdmCommandType.DISCOVER,
            universe=1,
            seq=42
        )
        msg = cmd.to_udp_json()

        assert msg["v"] == 2
        assert msg["type"] == "rdm"
        assert msg["action"] == "discover"
        assert msg["universe"] == 1
        assert msg["seq"] == 42
        assert "uid" not in msg

    def test_get_info_command(self):
        """Test get_info command serialization."""
        cmd = RdmCommand(
            action=RdmCommandType.GET_INFO,
            universe=0,
            uid="02CA:12345678",
            seq=1
        )
        msg = cmd.to_udp_json()

        assert msg["action"] == "get_info"
        assert msg["uid"] == "02CA:12345678"

    def test_set_address_command(self):
        """Test set_address command serialization."""
        cmd = RdmCommand(
            action=RdmCommandType.SET_ADDRESS,
            universe=0,
            uid="02CA:12345678",
            data={"address": 25},
            seq=3
        )
        msg = cmd.to_udp_json()

        assert msg["action"] == "set_address"
        assert msg["uid"] == "02CA:12345678"
        assert msg["address"] == 25

    def test_identify_command(self):
        """Test identify command serialization."""
        cmd = RdmCommand(
            action=RdmCommandType.IDENTIFY,
            universe=0,
            uid="02CA:12345678",
            data={"state": True},
            seq=5
        )
        msg = cmd.to_udp_json()

        assert msg["action"] == "identify"
        assert msg["state"] is True


class TestRdmResponse:
    """Tests for RDM response parsing."""

    def test_parse_ack_response(self):
        """Test parsing ACK response."""
        msg = {
            "v": 2,
            "type": "rdm_response",
            "action": "discover",
            "seq": 42,
            "uids": ["02CA:12345678", "02CA:87654321"]
        }
        resp = RdmResponse.from_udp_json(msg)

        assert resp.response_type == RdmResponseType.ACK
        assert resp.success is True
        assert resp.seq == 42
        assert resp.data["uids"] == ["02CA:12345678", "02CA:87654321"]

    def test_parse_error_response(self):
        """Test parsing error response."""
        msg = {
            "v": 2,
            "type": "rdm_response",
            "action": "get_info",
            "error": "Device not found",
            "seq": 1
        }
        resp = RdmResponse.from_udp_json(msg)

        assert resp.response_type == RdmResponseType.ERROR
        assert resp.success is False
        assert resp.error == "Device not found"

    def test_parse_timeout_response(self):
        """Test parsing timeout response."""
        msg = {
            "v": 2,
            "type": "rdm_response",
            "action": "get_info",
            "timeout": True,
            "seq": 2
        }
        resp = RdmResponse.from_udp_json(msg)

        assert resp.response_type == RdmResponseType.TIMEOUT
        assert resp.success is False

    def test_parse_nack_response(self):
        """Test parsing NACK response."""
        msg = {
            "v": 2,
            "type": "rdm_response",
            "action": "set_address",
            "nack": True,
            "nack_reason": 6,  # DATA_OUT_OF_RANGE
            "seq": 3
        }
        resp = RdmResponse.from_udp_json(msg)

        assert resp.response_type == RdmResponseType.NACK
        assert resp.success is False
        assert resp.nack_reason == RdmNackReason.DATA_OUT_OF_RANGE

    def test_response_to_dict(self):
        """Test response serialization."""
        resp = RdmResponse(
            action=RdmCommandType.DISCOVER,
            response_type=RdmResponseType.ACK,
            uid="02CA:12345678",
            data={"manufacturer_id": 714},
            seq=10
        )
        d = resp.to_dict()

        assert d["action"] == "discover"
        assert d["response_type"] == "ack"
        assert d["success"] is True
        assert d["uid"] == "02CA:12345678"


class TestRdmPid:
    """Tests for RDM Parameter IDs."""

    def test_standard_pids(self):
        """Test standard PID values."""
        assert RdmPid.DEVICE_INFO == 0x0060
        assert RdmPid.DMX_START_ADDRESS == 0x00F0
        assert RdmPid.IDENTIFY_DEVICE == 0x1000
        assert RdmPid.DMX_PERSONALITY == 0x00E0


class TestUdpJsonRdmTransport:
    """Tests for UDP JSON RDM transport."""

    def test_create_transport(self):
        """Test transport creation."""
        transport = create_transport("udp_json", port=6455, timeout=5.0)
        assert isinstance(transport, UdpJsonRdmTransport)
        assert transport.port == 6455
        assert transport.timeout == 5.0

    def test_create_transport_invalid_type(self):
        """Test invalid transport type."""
        with pytest.raises(ValueError, match="Unknown transport type"):
            create_transport("invalid")

    def test_sequence_increments(self):
        """Test sequence number incrementing."""
        transport = UdpJsonRdmTransport()
        seq1 = transport._next_sequence()
        seq2 = transport._next_sequence()
        seq3 = transport._next_sequence()

        assert seq2 == seq1 + 1
        assert seq3 == seq2 + 1

    def test_sequence_wraps(self):
        """Test sequence number wrapping at 65536."""
        transport = UdpJsonRdmTransport()
        transport._sequence = 65535
        seq = transport._next_sequence()
        assert seq == 0

    def test_set_dmx_address_validates_range(self):
        """Test DMX address validation."""
        transport = UdpJsonRdmTransport()
        uid = RdmUid(0x02CA, 0x12345678)

        with pytest.raises(ValueError, match="DMX address must be 1-512"):
            asyncio.get_event_loop().run_until_complete(
                transport.set_dmx_address("192.168.1.100", uid, 0)
            )

        with pytest.raises(ValueError, match="DMX address must be 1-512"):
            asyncio.get_event_loop().run_until_complete(
                transport.set_dmx_address("192.168.1.100", uid, 513)
            )


class TestUdpJsonRdmTransportAsync:
    """Async tests for UDP JSON RDM transport with mocked socket."""

    @pytest.fixture
    def transport(self):
        """Create transport instance."""
        return UdpJsonRdmTransport(timeout=0.5, retries=0)

    @pytest.mark.asyncio
    async def test_discover_parses_uids(self, transport):
        """Test discovery UID parsing."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "discover",
            "uids": ["02CA:12345678", "02CA:87654321"],
            "seq": 1
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uids = await transport.discover("192.168.1.100", 1)

            assert len(uids) == 2
            assert str(uids[0]) == "02CA:12345678"
            assert str(uids[1]) == "02CA:87654321"

    @pytest.mark.asyncio
    async def test_discover_returns_empty_on_timeout(self, transport):
        """Test discovery returns empty list on timeout."""
        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = RdmTimeoutError("Timeout")

            uids = await transport.discover("192.168.1.100", 1)

            assert uids == []

    @pytest.mark.asyncio
    async def test_get_device_info_parses_response(self, transport):
        """Test device info parsing."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "get_info",
            "manufacturer_id": 714,
            "device_model_id": 1234,
            "manufacturer_label": "Test Mfg",
            "device_model": "Test Model",
            "device_label": "My Light",
            "dmx_address": 25,
            "dmx_footprint": 8,
            "current_personality": 1,
            "personalities": [
                {"id": 1, "name": "8-Channel", "footprint": 8},
                {"id": 2, "name": "16-Channel", "footprint": 16}
            ],
            "software_version": "1.2.3",
            "seq": 2
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uid = RdmUid(0x02CA, 0x12345678)
            info = await transport.get_device_info("192.168.1.100", uid)

            assert info.manufacturer_id == 714
            assert info.device_model_id == 1234
            assert info.manufacturer_label == "Test Mfg"
            assert info.device_model == "Test Model"
            assert info.device_label == "My Light"
            assert info.dmx_address == 25
            assert info.dmx_footprint == 8
            assert len(info.personalities) == 2
            assert info.personalities[0].name == "8-Channel"

    @pytest.mark.asyncio
    async def test_identify_sends_correct_message(self, transport):
        """Test identify sends correct command."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "identify",
            "seq": 3
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uid = RdmUid(0x02CA, 0x12345678)
            result = await transport.identify("192.168.1.100", uid, True)

            assert result is True

            # Check the message sent
            call_args = mock_send.call_args
            msg = call_args[0][1]  # Second positional arg is the message
            assert msg["action"] == "identify"
            assert msg["uid"] == "02CA:12345678"
            assert msg["state"] is True

    @pytest.mark.asyncio
    async def test_set_address_sends_correct_message(self, transport):
        """Test set_address sends correct command."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "set_address",
            "seq": 4
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uid = RdmUid(0x02CA, 0x12345678)
            result = await transport.set_dmx_address("192.168.1.100", uid, 100)

            assert result is True

            call_args = mock_send.call_args
            msg = call_args[0][1]
            assert msg["action"] == "set_address"
            assert msg["address"] == 100

    @pytest.mark.asyncio
    async def test_set_label_truncates_long_labels(self, transport):
        """Test set_label truncates labels over 32 chars."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "set_label",
            "seq": 5
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uid = RdmUid(0x02CA, 0x12345678)
            long_label = "A" * 50  # 50 characters
            await transport.set_device_label("192.168.1.100", uid, long_label)

            call_args = mock_send.call_args
            msg = call_args[0][1]
            assert len(msg["label"]) == 32

    @pytest.mark.asyncio
    async def test_identify_returns_false_on_timeout(self, transport):
        """Test identify returns False on timeout."""
        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = RdmTimeoutError("Timeout")

            uid = RdmUid(0x02CA, 0x12345678)
            result = await transport.identify("192.168.1.100", uid, True)

            assert result is False

    @pytest.mark.asyncio
    async def test_get_device_info_raises_on_error(self, transport):
        """Test get_device_info raises on error response."""
        response = {
            "v": 2,
            "type": "rdm_response",
            "action": "get_info",
            "error": "Device not found",
            "seq": 6
        }

        with patch.object(transport, '_send_and_receive', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = response

            uid = RdmUid(0x02CA, 0x12345678)

            with pytest.raises(RdmTransportError, match="Failed to get device info"):
                await transport.get_device_info("192.168.1.100", uid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
