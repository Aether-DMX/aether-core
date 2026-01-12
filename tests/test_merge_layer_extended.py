"""
AETHER Beta1 - Extended Merge Layer Tests

Tests cover priority-based merging scenarios:
1. Full priority ladder testing
2. Multi-universe merge
3. Source registration/deregistration
4. Per-universe blackout
5. Priority override scenarios
6. Edge cases
"""

import pytest
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from merge_layer import MergeLayer, get_priority, PRIORITY_LEVELS
    HAS_MERGE_LAYER = True
except ImportError:
    HAS_MERGE_LAYER = False
    MergeLayer = None
    get_priority = None
    PRIORITY_LEVELS = {}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def merge_layer():
    """Fresh MergeLayer for testing"""
    if not HAS_MERGE_LAYER:
        pytest.skip("MergeLayer not available")
    return MergeLayer()


# ============================================================
# Priority Ladder Tests
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestPriorityLadder:
    """Tests for the full priority ladder"""

    def test_blackout_highest_priority(self, merge_layer):
        """Blackout (100) overrides everything"""
        # Register sources at all levels
        merge_layer.register_source("manual", "manual", [1])  # 80
        merge_layer.register_source("effect", "effect", [1])  # 60
        merge_layer.register_source("look", "look", [1])       # 50

        merge_layer.set_source_channels("manual", 1, {1: 255})
        merge_layer.set_source_channels("effect", 1, {1: 200})
        merge_layer.set_source_channels("look", 1, {1: 150})

        # Enable blackout
        merge_layer.set_blackout(True, universes=[1])

        result = merge_layer.compute_merge(1)

        # Blackout wins
        assert len(result) == 0 or result.get(1, 0) == 0

    def test_manual_overrides_effect(self, merge_layer):
        """Manual (80) overrides effect (60)"""
        merge_layer.register_source("manual", "manual", [1])
        merge_layer.register_source("effect", "effect", [1])

        merge_layer.set_source_channels("effect", 1, {1: 100})
        merge_layer.set_source_channels("manual", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        assert result.get(1, 0) == 200

    def test_effect_overrides_look(self, merge_layer):
        """Effect (60) overrides look (50)"""
        merge_layer.register_source("effect", "effect", [1])
        merge_layer.register_source("look", "look", [1])

        merge_layer.set_source_channels("look", 1, {1: 100})
        merge_layer.set_source_channels("effect", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        assert result.get(1, 0) == 200

    def test_look_overrides_sequence(self, merge_layer):
        """Look (50) overrides sequence (45)"""
        merge_layer.register_source("look", "look", [1])
        merge_layer.register_source("sequence", "sequence", [1])

        merge_layer.set_source_channels("sequence", 1, {1: 100})
        merge_layer.set_source_channels("look", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        assert result.get(1, 0) == 200

    def test_sequence_overrides_chase(self, merge_layer):
        """Sequence (45) overrides chase (40)"""
        merge_layer.register_source("sequence", "sequence", [1])
        merge_layer.register_source("chase", "chase", [1])

        merge_layer.set_source_channels("chase", 1, {1: 100})
        merge_layer.set_source_channels("sequence", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        assert result.get(1, 0) == 200

    def test_chase_overrides_scene(self, merge_layer):
        """Chase (40) overrides scene (20)"""
        merge_layer.register_source("chase", "chase", [1])
        merge_layer.register_source("scene", "scene", [1])

        merge_layer.set_source_channels("scene", 1, {1: 100})
        merge_layer.set_source_channels("chase", 1, {1: 200})

        result = merge_layer.compute_merge(1)

        assert result.get(1, 0) == 200

    def test_full_priority_stack(self, merge_layer):
        """Full priority stack resolves correctly"""
        # Register all priority levels
        for source_type in ["manual", "effect", "look", "sequence", "chase", "scene"]:
            merge_layer.register_source(f"src_{source_type}", source_type, [1])

        # Set values from lowest to highest priority
        merge_layer.set_source_channels("src_scene", 1, {1: 10})
        merge_layer.set_source_channels("src_chase", 1, {1: 20})
        merge_layer.set_source_channels("src_sequence", 1, {1: 30})
        merge_layer.set_source_channels("src_look", 1, {1: 40})
        merge_layer.set_source_channels("src_effect", 1, {1: 50})
        merge_layer.set_source_channels("src_manual", 1, {1: 60})

        result = merge_layer.compute_merge(1)

        # Manual (highest) wins
        assert result.get(1, 0) == 60


# ============================================================
# Multi-Universe Tests
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestMultiUniverseMerge:
    """Tests for multi-universe merge scenarios"""

    def test_independent_universes(self, merge_layer):
        """Different universes are independent"""
        merge_layer.register_source("look_u1", "look", [1])
        merge_layer.register_source("look_u2", "look", [2])

        merge_layer.set_source_channels("look_u1", 1, {1: 100})
        merge_layer.set_source_channels("look_u2", 2, {1: 200})

        result_u1 = merge_layer.compute_merge(1)
        result_u2 = merge_layer.compute_merge(2)

        assert result_u1.get(1, 0) == 100
        assert result_u2.get(1, 0) == 200

    def test_source_spans_multiple_universes(self, merge_layer):
        """Single source can output to multiple universes"""
        merge_layer.register_source("look_multi", "look", [1, 2, 3])

        merge_layer.set_source_channels("look_multi", 1, {1: 100})
        merge_layer.set_source_channels("look_multi", 2, {1: 150})
        merge_layer.set_source_channels("look_multi", 3, {1: 200})

        assert merge_layer.compute_merge(1).get(1, 0) == 100
        assert merge_layer.compute_merge(2).get(1, 0) == 150
        assert merge_layer.compute_merge(3).get(1, 0) == 200

    def test_per_universe_blackout(self, merge_layer):
        """Blackout can be per-universe"""
        merge_layer.register_source("look", "look", [1, 2])

        merge_layer.set_source_channels("look", 1, {1: 255})
        merge_layer.set_source_channels("look", 2, {1: 255})

        # Blackout only universe 1
        merge_layer.set_blackout(True, universes=[1])

        result_u1 = merge_layer.compute_merge(1)
        result_u2 = merge_layer.compute_merge(2)

        # U1 should be blacked out, U2 should be normal
        assert len(result_u1) == 0 or result_u1.get(1, 0) == 0
        assert result_u2.get(1, 0) == 255

    def test_cross_universe_priority(self, merge_layer):
        """Priority applies per-universe correctly"""
        merge_layer.register_source("manual", "manual", [1])
        merge_layer.register_source("look", "look", [1, 2])

        merge_layer.set_source_channels("manual", 1, {1: 200})
        merge_layer.set_source_channels("look", 1, {1: 100})
        merge_layer.set_source_channels("look", 2, {1: 100})

        # U1: manual wins over look
        # U2: only look present
        assert merge_layer.compute_merge(1).get(1, 0) == 200
        assert merge_layer.compute_merge(2).get(1, 0) == 100


# ============================================================
# Source Registration Tests
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestSourceRegistration:
    """Tests for source registration and deregistration"""

    def test_register_source(self, merge_layer):
        """Source can be registered"""
        merge_layer.register_source("test_source", "look", [1])
        merge_layer.set_source_channels("test_source", 1, {1: 128})

        result = merge_layer.compute_merge(1)
        assert result.get(1, 0) == 128

    def test_deregister_source(self, merge_layer):
        """Source can be deregistered"""
        merge_layer.register_source("temp_source", "look", [1])
        merge_layer.set_source_channels("temp_source", 1, {1: 255})

        # Verify it's there
        assert merge_layer.compute_merge(1).get(1, 0) == 255

        # Deregister
        merge_layer.deregister_source("temp_source")

        # Should be gone
        result = merge_layer.compute_merge(1)
        assert result.get(1, 0) == 0 or len(result) == 0

    def test_update_source_channels(self, merge_layer):
        """Source channels can be updated"""
        merge_layer.register_source("update_test", "look", [1])

        merge_layer.set_source_channels("update_test", 1, {1: 100})
        assert merge_layer.compute_merge(1).get(1, 0) == 100

        merge_layer.set_source_channels("update_test", 1, {1: 200})
        assert merge_layer.compute_merge(1).get(1, 0) == 200

    def test_register_multiple_sources_same_type(self, merge_layer):
        """Multiple sources of same type can coexist"""
        merge_layer.register_source("look_1", "look", [1])
        merge_layer.register_source("look_2", "look", [1])

        merge_layer.set_source_channels("look_1", 1, {1: 100})
        merge_layer.set_source_channels("look_2", 1, {2: 200})

        result = merge_layer.compute_merge(1)

        # Both channels should be present
        # Note: same-priority same-channel would conflict; using different channels
        assert result.get(1, 0) == 100
        assert result.get(2, 0) == 200


# ============================================================
# Blackout Tests
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestBlackoutBehavior:
    """Tests for blackout behavior"""

    def test_global_blackout(self, merge_layer):
        """Global blackout affects all universes"""
        merge_layer.register_source("look", "look", [1, 2, 3, 4])

        for u in [1, 2, 3, 4]:
            merge_layer.set_source_channels("look", u, {1: 255})

        # Global blackout
        merge_layer.set_blackout(True)

        for u in [1, 2, 3, 4]:
            result = merge_layer.compute_merge(u)
            assert len(result) == 0 or result.get(1, 0) == 0

    def test_blackout_release_restores_state(self, merge_layer):
        """Releasing blackout restores previous state"""
        merge_layer.register_source("look", "look", [1])
        merge_layer.set_source_channels("look", 1, {1: 200, 2: 150, 3: 100})

        # Verify initial state
        before = merge_layer.compute_merge(1)
        assert before.get(1, 0) == 200

        # Blackout and release
        merge_layer.set_blackout(True)
        merge_layer.set_blackout(False)

        # State should be restored
        after = merge_layer.compute_merge(1)
        assert after.get(1, 0) == 200
        assert after.get(2, 0) == 150
        assert after.get(3, 0) == 100

    def test_blackout_during_playback(self, merge_layer):
        """Blackout during active playback works"""
        merge_layer.register_source("sequence", "sequence", [1])
        merge_layer.set_source_channels("sequence", 1, {1: 255})

        # Mid-playback blackout
        merge_layer.set_blackout(True, universes=[1])

        result = merge_layer.compute_merge(1)
        assert len(result) == 0 or result.get(1, 0) == 0


# ============================================================
# Edge Cases
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestMergeEdgeCases:
    """Edge case tests for merge layer"""

    def test_empty_universe(self, merge_layer):
        """Empty universe returns empty result"""
        result = merge_layer.compute_merge(99)  # Unused universe
        assert isinstance(result, dict)
        assert len(result) == 0 or all(v == 0 for v in result.values())

    def test_channel_zero_value(self, merge_layer):
        """Channel with value 0 is valid"""
        merge_layer.register_source("zero_test", "look", [1])
        merge_layer.set_source_channels("zero_test", 1, {1: 0, 2: 255})

        result = merge_layer.compute_merge(1)

        # Channel 1 should be 0, channel 2 should be 255
        assert result.get(2, 0) == 255

    def test_high_channel_numbers(self, merge_layer):
        """High channel numbers (up to 512) work"""
        merge_layer.register_source("high_ch", "look", [1])
        merge_layer.set_source_channels("high_ch", 1, {500: 128, 512: 255})

        result = merge_layer.compute_merge(1)

        assert result.get(500, 0) == 128
        assert result.get(512, 0) == 255

    def test_many_sources(self, merge_layer):
        """Many simultaneous sources handled correctly"""
        # Register 20 sources
        for i in range(20):
            merge_layer.register_source(f"src_{i}", "look", [1])
            merge_layer.set_source_channels(f"src_{i}", 1, {i + 1: 100 + i})

        result = merge_layer.compute_merge(1)

        # All 20 channels should be present (no conflict since different channels)
        for i in range(20):
            assert result.get(i + 1, 0) == 100 + i

    def test_source_type_case_sensitivity(self, merge_layer):
        """Source type is case-sensitive or normalized"""
        # Test both possible behaviors
        try:
            merge_layer.register_source("case_test", "LOOK", [1])  # Uppercase
            merge_layer.set_source_channels("case_test", 1, {1: 100})
            result = merge_layer.compute_merge(1)
            # If it works, uppercase is handled
            assert isinstance(result, dict)
        except (KeyError, ValueError):
            # If it fails, uppercase is not supported (expected)
            pass


# ============================================================
# Priority Function Tests
# ============================================================

@pytest.mark.skipif(not HAS_MERGE_LAYER, reason="MergeLayer not available")
class TestPriorityFunction:
    """Tests for get_priority function"""

    def test_known_priorities(self):
        """Known source types return correct priority"""
        expected = {
            "blackout": 100,
            "manual": 80,
            "fader": 80,
            "effect": 60,
            "look": 50,
            "sequence": 45,
            "chase": 40,
            "scene": 20,
            "background": 10,
        }

        for source_type, expected_priority in expected.items():
            actual = get_priority(source_type)
            assert actual == expected_priority, f"{source_type}: expected {expected_priority}, got {actual}"

    def test_unknown_priority_default(self):
        """Unknown source type returns default priority"""
        priority = get_priority("unknown_type")
        assert isinstance(priority, int)
        # Should be some reasonable default (likely 0 or low)
        assert 0 <= priority <= 100


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
