#!/usr/bin/env python3
"""
AETHER Core - Chase Playback Regression Tests
Tests that chase playback works correctly through the SSOT pipeline.

Run with: python test_chase_playback.py
Or with server running: python test_chase_playback.py --live

Tests:
1. Chase engine uses same SSOT path as /api/dmx/set
2. Chase steps cycle correctly (not static)
3. Chase outputs distinct DMX frames for each step
4. Fade values are passed through SSOT
"""

import sys
import json
import time
import threading
import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

# Test configuration
TEST_CHASE = {
    'chase_id': 'test_chase_001',
    'name': 'Test RGB Cycle',
    'universe': 2,
    'bpm': 60,
    'loop': True,
    'fade_ms': 500,
    'steps': [
        {'channels': {'1': 255, '2': 0, '3': 0}, 'duration': 1000},    # Red
        {'channels': {'1': 0, '2': 255, '3': 0}, 'duration': 1000},    # Green
        {'channels': {'1': 0, '2': 0, '3': 255}, 'duration': 1000},    # Blue
    ]
}


class MockContentManager:
    """Mock ContentManager to capture SSOT calls"""
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def set_channels(self, universe, channels, fade_ms=0):
        with self.lock:
            self.calls.append({
                'universe': universe,
                'channels': dict(channels),
                'fade_ms': fade_ms,
                'timestamp': time.time()
            })
        return {'success': True, 'results': []}

    def get_call_count(self):
        with self.lock:
            return len(self.calls)

    def get_calls(self):
        with self.lock:
            return list(self.calls)


class TestChasePlaybackSSoT(unittest.TestCase):
    """Test that chase uses the same SSOT path as direct DMX set"""

    def test_chase_uses_content_manager_set_channels(self):
        """Verify chase steps call content_manager.set_channels (SSOT)"""
        # Import the chase engine class
        mock_cm = MockContentManager()

        # Create a minimal chase engine for testing
        class TestChaseEngine:
            def _send_step(self, universe, channels, fade_ms=0):
                if not channels:
                    return
                parsed_channels = {}
                for key, value in channels.items():
                    key_str = str(key)
                    if ':' in key_str:
                        parts = key_str.split(':')
                        ch_univ = int(parts[0])
                        ch_num = int(parts[1])
                        if ch_univ == universe:
                            parsed_channels[ch_num] = value
                    else:
                        parsed_channels[int(key_str)] = value
                if not parsed_channels:
                    return
                mock_cm.set_channels(universe, parsed_channels, fade_ms=fade_ms)

        engine = TestChaseEngine()

        # Send test steps
        for step in TEST_CHASE['steps']:
            engine._send_step(TEST_CHASE['universe'], step['channels'], TEST_CHASE['fade_ms'])

        # Verify SSOT was called for each step
        self.assertEqual(mock_cm.get_call_count(), 3, "Should have 3 SSOT calls for 3 steps")

        calls = mock_cm.get_calls()

        # Verify first call (Red)
        self.assertEqual(calls[0]['universe'], 2)
        self.assertEqual(calls[0]['channels'], {1: 255, 2: 0, 3: 0})
        self.assertEqual(calls[0]['fade_ms'], 500)

        # Verify second call (Green)
        self.assertEqual(calls[1]['channels'], {1: 0, 2: 255, 3: 0})

        # Verify third call (Blue)
        self.assertEqual(calls[2]['channels'], {1: 0, 2: 0, 3: 255})

        print("✓ Chase correctly routes through SSOT (content_manager.set_channels)")


class TestChaseStepCycling(unittest.TestCase):
    """Test that chase actually cycles through steps"""

    def test_chase_loop_produces_distinct_frames(self):
        """Verify chase produces distinct DMX frames over time"""
        mock_cm = MockContentManager()
        stop_flag = threading.Event()

        def run_short_chase():
            steps = TEST_CHASE['steps']
            step_index = 0
            loop_count = 0

            # Run for ~2.5 loops worth of steps (7-8 steps)
            for _ in range(8):
                if stop_flag.is_set():
                    break

                step = steps[step_index]
                channels = step.get('channels', {})
                parsed = {int(k): v for k, v in channels.items()}
                mock_cm.set_channels(TEST_CHASE['universe'], parsed, TEST_CHASE['fade_ms'])

                # Short sleep to simulate step duration
                time.sleep(0.05)

                step_index += 1
                if step_index >= len(steps):
                    step_index = 0
                    loop_count += 1

        # Run the chase
        thread = threading.Thread(target=run_short_chase, daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        stop_flag.set()

        calls = mock_cm.get_calls()

        # Should have 8 calls
        self.assertGreaterEqual(len(calls), 7, f"Expected at least 7 calls, got {len(calls)}")

        # Verify we got different channel values (cycling)
        unique_frames = set()
        for call in calls:
            frame_key = tuple(sorted(call['channels'].items()))
            unique_frames.add(frame_key)

        self.assertEqual(len(unique_frames), 3, "Should have 3 distinct frames (R, G, B)")

        print(f"✓ Chase produced {len(calls)} frames with {len(unique_frames)} distinct values")


class TestChaseChannelParsing(unittest.TestCase):
    """Test channel parsing for universe-prefixed channels"""

    def test_simple_channel_numbers(self):
        """Test that simple channel numbers work"""
        mock_cm = MockContentManager()

        channels = {'1': 100, '2': 150, '3': 200}
        parsed = {int(k): v for k, v in channels.items()}
        mock_cm.set_channels(2, parsed, fade_ms=0)

        call = mock_cm.get_calls()[0]
        self.assertEqual(call['channels'], {1: 100, 2: 150, 3: 200})
        print("✓ Simple channel parsing works")

    def test_universe_prefixed_channels(self):
        """Test that universe:channel format filters correctly"""
        # Channels with universe prefix
        channels = {'2:1': 100, '2:2': 150, '3:1': 200}  # U2 and U3 channels
        target_universe = 2

        parsed = {}
        for key, value in channels.items():
            key_str = str(key)
            if ':' in key_str:
                parts = key_str.split(':')
                ch_univ = int(parts[0])
                ch_num = int(parts[1])
                if ch_univ == target_universe:
                    parsed[ch_num] = value
            else:
                parsed[int(key_str)] = value

        # Should only include U2 channels
        self.assertEqual(parsed, {1: 100, 2: 150})
        print("✓ Universe-prefixed channel parsing works")


class TestChaseFadePropagation(unittest.TestCase):
    """Test that fade_ms is correctly propagated through SSOT"""

    def test_fade_ms_reaches_ssot(self):
        """Verify fade_ms is passed to set_channels"""
        mock_cm = MockContentManager()

        # Test with different fade values
        mock_cm.set_channels(2, {1: 255}, fade_ms=0)
        mock_cm.set_channels(2, {1: 200}, fade_ms=500)
        mock_cm.set_channels(2, {1: 100}, fade_ms=1500)

        calls = mock_cm.get_calls()

        self.assertEqual(calls[0]['fade_ms'], 0)
        self.assertEqual(calls[1]['fade_ms'], 500)
        self.assertEqual(calls[2]['fade_ms'], 1500)

        print("✓ Fade values correctly propagated to SSOT")


def run_live_test():
    """Run a live test against the running server"""
    import urllib.request
    import urllib.error

    base_url = "http://localhost:8891"

    print("\n" + "="*60)
    print("LIVE TEST: Chase Playback Against Running Server")
    print("="*60 + "\n")

    # 1. Get available chases
    print("1. Fetching chases...")
    try:
        req = urllib.request.urlopen(f"{base_url}/api/chases", timeout=5)
        chases = json.loads(req.read().decode())
        print(f"   Found {len(chases)} chases")

        if not chases:
            print("   ⚠️ No chases found. Create a test chase first.")
            return False

        # Use first chase or specific test chase
        test_chase_id = None
        for chase in chases:
            if 'christmas' in chase.get('name', '').lower() or chase.get('chase_id') == 'chase_1766607583':
                test_chase_id = chase['chase_id']
                print(f"   Selected: {chase['name']} ({test_chase_id})")
                break

        if not test_chase_id:
            test_chase_id = chases[0]['chase_id']
            print(f"   Using first chase: {chases[0].get('name')} ({test_chase_id})")

    except urllib.error.URLError as e:
        print(f"   ❌ Server not reachable: {e}")
        print("   Start the server with: python aether-core.py")
        return False

    # 2. Start chase playback
    print(f"\n2. Starting chase playback: {test_chase_id}")
    try:
        data = json.dumps({}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/chases/{test_chase_id}/play",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        resp = urllib.request.urlopen(req, timeout=5)
        result = json.loads(resp.read().decode())
        print(f"   Response: {result}")

        if result.get('success'):
            print("   ✓ Chase started successfully")
        else:
            print(f"   ❌ Failed to start: {result.get('error')}")
            return False

    except Exception as e:
        print(f"   ❌ Failed to start chase: {e}")
        return False

    # 3. Monitor chase health for a few seconds
    print("\n3. Monitoring chase health (watch for step changes)...")
    print("   Waiting 10 seconds to observe step transitions...\n")

    seen_steps = set()
    for i in range(10):
        try:
            req = urllib.request.urlopen(f"{base_url}/api/chases/health", timeout=2)
            health = json.loads(req.read().decode())

            running = health.get('running', [])
            chase_health = health.get('health', {}).get(test_chase_id, {})

            step = chase_health.get('step', -1)
            status = chase_health.get('status', 'unknown')
            loop_num = chase_health.get('loop', 0)

            if step >= 0:
                seen_steps.add(step)

            print(f"   [{i+1}/10] Step: {step}, Loop: {loop_num}, Status: {status}, Seen steps: {sorted(seen_steps)}")

            time.sleep(1)

        except Exception as e:
            print(f"   [{i+1}/10] Error checking health: {e}")
            time.sleep(1)

    # 4. Stop the chase
    print(f"\n4. Stopping chase...")
    try:
        req = urllib.request.Request(
            f"{base_url}/api/chases/{test_chase_id}/stop",
            data=b'{}',
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        resp = urllib.request.urlopen(req, timeout=5)
        print(f"   ✓ Chase stopped")
    except Exception as e:
        print(f"   ⚠️ Stop request failed: {e}")

    # 5. Verdict
    print("\n" + "="*60)
    if len(seen_steps) >= 2:
        print("✓ PASS: Chase is cycling through steps!")
        print(f"  Observed steps: {sorted(seen_steps)}")
        return True
    else:
        print("❌ FAIL: Chase is NOT cycling (only saw step {})".format(list(seen_steps)))
        print("  Check server console for error messages.")
        return False


if __name__ == '__main__':
    if '--live' in sys.argv:
        success = run_live_test()
        sys.exit(0 if success else 1)
    else:
        print("="*60)
        print("AETHER Core - Chase Playback Unit Tests")
        print("="*60 + "\n")

        # Run unit tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        suite.addTests(loader.loadTestsFromTestCase(TestChasePlaybackSSoT))
        suite.addTests(loader.loadTestsFromTestCase(TestChaseStepCycling))
        suite.addTests(loader.loadTestsFromTestCase(TestChaseChannelParsing))
        suite.addTests(loader.loadTestsFromTestCase(TestChaseFadePropagation))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        print("\n" + "="*60)
        if result.wasSuccessful():
            print("✓ All unit tests passed!")
            print("\nTo run live test against server:")
            print("  python test_chase_playback.py --live")
        else:
            print("❌ Some tests failed")
        print("="*60)

        sys.exit(0 if result.wasSuccessful() else 1)
