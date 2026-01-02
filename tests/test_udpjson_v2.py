#!/usr/bin/env python3
"""
AETHER UDPJSON Protocol v2 Test Suite

Tests the bulletproof UDPJSON protocol implementation for:
- Compact ch/fill/frame encodings
- Sequence number handling
- MTU compliance
- Error handling
- Backward compatibility

Usage:
    # Run all tests
    python test_udpjson_v2.py

    # Run specific tests
    python test_udpjson_v2.py TestV2Protocol.test_compact_encoding

    # Run with live node (requires node IP)
    python test_udpjson_v2.py --node-ip 192.168.50.100
"""

import socket
import json
import time
import argparse
import base64
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
DEFAULT_PORT = 6455
DEFAULT_UNIVERSE = 2
PROTOCOL_VERSION = 2


class UDPJSONClient:
    """Test client for sending UDPJSON packets."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self._seq = 0

    def next_seq(self):
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    def send(self, payload):
        """Send a JSON payload."""
        json_data = json.dumps(payload, separators=(',', ':'))
        self.sock.sendto(json_data.encode(), (self.host, self.port))
        return len(json_data)

    def receive(self, timeout=2.0):
        """Wait for a response."""
        self.sock.settimeout(timeout)
        try:
            data, addr = self.sock.recvfrom(2048)
            return json.loads(data.decode())
        except socket.timeout:
            return None

    def send_set(self, universe, channels, fade=0):
        """Send v2 set command with compact encoding."""
        ch_pairs = [[int(ch), int(val)] for ch, val in channels.items()]
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "set",
            "u": universe,
            "seq": self.next_seq(),
            "ch": ch_pairs
        }
        if fade > 0:
            payload["fade"] = fade
        return self.send(payload)

    def send_fill(self, universe, ranges, fade=0):
        """Send v2 fill command."""
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "fill",
            "u": universe,
            "seq": self.next_seq(),
            "ranges": ranges
        }
        if fade > 0:
            payload["fade"] = fade
        return self.send(payload)

    def send_frame(self, universe, frame_bytes, fade=0):
        """Send v2 frame command with base64 encoding."""
        b64 = base64.b64encode(frame_bytes).decode('ascii')
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "frame",
            "u": universe,
            "seq": self.next_seq(),
            "b64": b64
        }
        if fade > 0:
            payload["fade"] = fade
        return self.send(payload)

    def send_blackout(self, universe, fade=0):
        """Send v2 blackout command."""
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "blackout",
            "u": universe,
            "seq": self.next_seq()
        }
        if fade > 0:
            payload["fade"] = fade
        return self.send(payload)

    def send_panic(self, universe):
        """Send v2 panic command."""
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "panic",
            "u": universe,
            "seq": self.next_seq()
        }
        return self.send(payload)

    def send_ping(self):
        """Send v2 ping command."""
        payload = {
            "v": PROTOCOL_VERSION,
            "type": "ping",
            "seq": self.next_seq()
        }
        self.send(payload)
        return self.receive()

    def send_legacy_set(self, universe, channels):
        """Send legacy v1 set command for backward compatibility testing."""
        payload = {
            "type": "set",
            "universe": universe,
            "channels": channels,
            "ts": int(time.time())
        }
        return self.send(payload)

    def close(self):
        self.sock.close()


class TestV2Protocol(unittest.TestCase):
    """Unit tests for UDPJSON v2 protocol."""

    def test_compact_encoding(self):
        """Test that compact ch encoding produces valid JSON."""
        channels = {"1": 255, "2": 128, "170": 64}
        ch_pairs = [[int(ch), int(val)] for ch, val in channels.items()]

        payload = {
            "v": 2,
            "type": "set",
            "u": 2,
            "seq": 1000,
            "ch": ch_pairs
        }

        json_str = json.dumps(payload, separators=(',', ':'))
        self.assertLess(len(json_str), 100)  # Should be compact
        self.assertIn('"ch":[[', json_str)  # Array format

    def test_fill_encoding(self):
        """Test fill command encoding."""
        payload = {
            "v": 2,
            "type": "fill",
            "u": 2,
            "seq": 1001,
            "ranges": [[1, 512, 0]]
        }

        json_str = json.dumps(payload, separators=(',', ':'))
        self.assertLess(len(json_str), 60)  # Very compact
        parsed = json.loads(json_str)
        self.assertEqual(parsed["ranges"][0], [1, 512, 0])

    def test_frame_encoding(self):
        """Test base64 frame encoding."""
        frame = bytes([i % 256 for i in range(512)])
        b64 = base64.b64encode(frame).decode('ascii')

        payload = {
            "v": 2,
            "type": "frame",
            "u": 2,
            "seq": 1002,
            "b64": b64
        }

        json_str = json.dumps(payload, separators=(',', ':'))
        self.assertLess(len(json_str), 800)  # ~684 for b64 + overhead
        self.assertEqual(len(base64.b64decode(payload["b64"])), 512)

    def test_mtu_compliance(self):
        """Test that large payloads stay under MTU."""
        # 100 channels should fit easily
        channels = {str(i): i % 256 for i in range(1, 101)}
        ch_pairs = [[int(ch), int(val)] for ch, val in channels.items()]

        payload = {
            "v": 2,
            "type": "set",
            "u": 2,
            "seq": 1003,
            "ch": ch_pairs
        }

        json_str = json.dumps(payload, separators=(',', ':'))
        self.assertLess(len(json_str), 1200)  # Under MTU limit

    def test_sequence_number_increment(self):
        """Test sequence number increments correctly."""
        client = UDPJSONClient()
        seq1 = client.next_seq()
        seq2 = client.next_seq()
        seq3 = client.next_seq()
        self.assertEqual(seq2, seq1 + 1)
        self.assertEqual(seq3, seq2 + 1)

    def test_sequence_wraparound(self):
        """Test sequence number wraps at 2^32."""
        client = UDPJSONClient()
        client._seq = 0xFFFFFFFE
        seq1 = client.next_seq()  # 0xFFFFFFFF
        seq2 = client.next_seq()  # 0 (wrapped)
        self.assertEqual(seq1, 0xFFFFFFFF)
        self.assertEqual(seq2, 0)

    def test_payload_size_comparison(self):
        """Compare v1 vs v2 payload sizes."""
        # 50 channel update
        channels = {str(i): i % 256 for i in range(1, 51)}

        # V1 format
        v1_payload = {
            "type": "set",
            "universe": 2,
            "channels": channels,
            "ts": 1700000000
        }
        v1_size = len(json.dumps(v1_payload, separators=(',', ':')))

        # V2 format
        ch_pairs = [[int(ch), int(val)] for ch, val in channels.items()]
        v2_payload = {
            "v": 2,
            "type": "set",
            "u": 2,
            "seq": 1000,
            "ch": ch_pairs
        }
        v2_size = len(json.dumps(v2_payload, separators=(',', ':')))

        print(f"\n50 channel update: v1={v1_size} bytes, v2={v2_size} bytes ({100*v2_size/v1_size:.0f}%)")
        self.assertLess(v2_size, v1_size)  # v2 should be smaller


class TestLiveNode(unittest.TestCase):
    """Integration tests requiring a live node."""

    @classmethod
    def setUpClass(cls):
        cls.node_ip = getattr(cls, 'node_ip', None)
        if not cls.node_ip:
            raise unittest.SkipTest("No node IP provided. Use --node-ip to run live tests.")
        cls.client = UDPJSONClient(host=cls.node_ip)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'client'):
            cls.client.close()

    def test_ping_pong(self):
        """Test ping/pong health check."""
        response = self.client.send_ping()
        self.assertIsNotNone(response, "No pong response received")
        self.assertEqual(response.get("type"), "pong")
        self.assertIn("id", response)
        self.assertIn("rssi", response)
        self.assertIn("rx", response)
        print(f"\nPong from {response.get('id')}: RSSI={response.get('rssi')}")

    def test_sparse_update(self):
        """Test sparse channel update."""
        size = self.client.send_set(DEFAULT_UNIVERSE, {"1": 255, "2": 128, "3": 64})
        print(f"\nSparse update sent: {size} bytes")
        self.assertLess(size, 100)

    def test_fill_blackout(self):
        """Test fill range for blackout."""
        size = self.client.send_fill(DEFAULT_UNIVERSE, [[1, 512, 0]])
        print(f"\nFill blackout sent: {size} bytes")
        self.assertLess(size, 60)

    def test_fill_wipe(self):
        """Test fill for full white."""
        size = self.client.send_fill(DEFAULT_UNIVERSE, [[1, 512, 255]])
        print(f"\nFill full-on sent: {size} bytes")
        time.sleep(0.5)
        # Clean up
        self.client.send_fill(DEFAULT_UNIVERSE, [[1, 512, 0]])

    def test_frame_sync(self):
        """Test base64 frame sync."""
        # Create a gradient pattern
        frame = bytes([i % 256 for i in range(512)])
        size = self.client.send_frame(DEFAULT_UNIVERSE, frame)
        print(f"\nFrame sync sent: {size} bytes")
        self.assertLess(size, 800)
        time.sleep(0.5)
        # Clean up
        self.client.send_blackout(DEFAULT_UNIVERSE)

    def test_fade(self):
        """Test fade functionality."""
        # Set starting values
        self.client.send_set(DEFAULT_UNIVERSE, {"1": 0, "2": 0, "3": 0, "4": 0})
        time.sleep(0.1)

        # Fade up
        self.client.send_set(DEFAULT_UNIVERSE, {"1": 255, "2": 255, "3": 255, "4": 255}, fade=1000)
        print("\nFade up started (1s)")
        time.sleep(1.5)

        # Fade down
        self.client.send_set(DEFAULT_UNIVERSE, {"1": 0, "2": 0, "3": 0, "4": 0}, fade=1000)
        print("Fade down started (1s)")
        time.sleep(1.5)

    def test_panic(self):
        """Test panic immediate blackout."""
        # Set some values first
        self.client.send_set(DEFAULT_UNIVERSE, {"1": 255, "2": 255})
        time.sleep(0.2)

        # Panic should immediately blackout
        self.client.send_panic(DEFAULT_UNIVERSE)
        print("\nPanic command sent")

    def test_hold_on_silence(self):
        """Test that values hold when no packets are sent."""
        # Set values
        self.client.send_set(DEFAULT_UNIVERSE, {"1": 128, "2": 64, "3": 32, "4": 16})
        print("\nSet values, waiting 5s without packets...")
        time.sleep(5)

        # Ping to verify node still responding
        response = self.client.send_ping()
        self.assertIsNotNone(response)
        self.assertFalse(response.get("stale", True), "Node should not be stale yet")
        print(f"Node still alive after 5s silence: stale={response.get('stale')}")

        # Clean up
        self.client.send_blackout(DEFAULT_UNIVERSE)

    def test_sequence_drop(self):
        """Test that out-of-order packets are dropped."""
        # Send seq 100
        payload1 = {
            "v": 2,
            "type": "set",
            "u": DEFAULT_UNIVERSE,
            "seq": 100,
            "ch": [[1, 255]]
        }
        self.client.send(payload1)
        time.sleep(0.1)

        # Send seq 99 (should be dropped)
        payload2 = {
            "v": 2,
            "type": "set",
            "u": DEFAULT_UNIVERSE,
            "seq": 99,
            "ch": [[1, 0]]
        }
        self.client.send(payload2)
        time.sleep(0.1)

        # Ping to check seq_drop counter
        response = self.client.send_ping()
        self.assertIsNotNone(response)
        print(f"\nSequence drops: {response.get('rx_seq_drop', 0)}")

        # Clean up
        self.client.send_blackout(DEFAULT_UNIVERSE)

    def test_legacy_v1_compatibility(self):
        """Test that legacy v1 format still works."""
        size = self.client.send_legacy_set(DEFAULT_UNIVERSE, {"1": 255, "2": 128})
        print(f"\nLegacy v1 packet sent: {size} bytes")
        time.sleep(0.2)

        # Verify with ping
        response = self.client.send_ping()
        self.assertIsNotNone(response)

        # Clean up
        self.client.send_blackout(DEFAULT_UNIVERSE)


class TestMalformedPackets(unittest.TestCase):
    """Test node resilience to malformed packets."""

    @classmethod
    def setUpClass(cls):
        cls.node_ip = getattr(cls, 'node_ip', None)
        if not cls.node_ip:
            raise unittest.SkipTest("No node IP provided")
        cls.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cls.client = UDPJSONClient(host=cls.node_ip)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'sock'):
            cls.sock.close()
        if hasattr(cls, 'client'):
            cls.client.close()

    def send_raw(self, data):
        self.sock.sendto(data, (self.node_ip, DEFAULT_PORT))

    def test_invalid_json(self):
        """Test node handles invalid JSON."""
        self.send_raw(b'{broken json here}')
        time.sleep(0.1)

        # Node should still respond to ping
        response = self.client.send_ping()
        self.assertIsNotNone(response)
        print(f"\nParse failures after bad JSON: {response.get('rx_parse_fail', 0)}")

    def test_empty_packet(self):
        """Test node handles empty packet."""
        self.send_raw(b'')
        time.sleep(0.1)

        response = self.client.send_ping()
        self.assertIsNotNone(response)

    def test_oversized_packet(self):
        """Test node handles oversized packet."""
        # Send 2000 bytes (over MAX_UDP_PAYLOAD)
        big_data = b'x' * 2000
        self.send_raw(big_data)
        time.sleep(0.1)

        response = self.client.send_ping()
        self.assertIsNotNone(response)
        print(f"\nOversize packets dropped: {response.get('rx_oversize', 0)}")

    def test_wrong_universe(self):
        """Test packets to wrong universe are ignored."""
        wrong_universe = 999
        payload = {
            "v": 2,
            "type": "set",
            "u": wrong_universe,
            "seq": self.client.next_seq(),
            "ch": [[1, 255]]
        }
        self.client.send(payload)
        time.sleep(0.1)

        response = self.client.send_ping()
        self.assertIsNotNone(response)
        print(f"\nUniverse drops: {response.get('rx_universe_drop', 0)}")


def main():
    parser = argparse.ArgumentParser(description='UDPJSON v2 Protocol Test Suite')
    parser.add_argument('--node-ip', help='IP address of test node for live tests')
    parser.add_argument('tests', nargs='*', help='Specific tests to run')

    args, remaining = parser.parse_known_args()

    # Set node IP for test classes
    if args.node_ip:
        TestLiveNode.node_ip = args.node_ip
        TestMalformedPackets.node_ip = args.node_ip
        print(f"Testing against node: {args.node_ip}")

    # Build test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if args.tests:
        for test in args.tests:
            suite.addTests(loader.loadTestsFromName(test))
    else:
        # Add all test classes
        suite.addTests(loader.loadTestsFromTestCase(TestV2Protocol))
        if args.node_ip:
            suite.addTests(loader.loadTestsFromTestCase(TestLiveNode))
            suite.addTests(loader.loadTestsFromTestCase(TestMalformedPackets))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
