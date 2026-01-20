"""
Unit Tests for RDM Auto-Patch Engine

Tests for:
- AutoPatchSuggestion dataclass
- AutoPatchEngine suggestion generation
- Conflict detection
- Address assignment
- Personality reduction logic
- Validation
- RDM command application
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.rdm.auto_patch import (
    AutoPatchEngine,
    AutoPatcher,
    ProfileMatcher,
)
from core.rdm.types import (
    DiscoveredFixture,
    AutoPatchSuggestion,
    PatchConfidence,
)


class TestAutoPatchSuggestion:
    """Tests for AutoPatchSuggestion dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678",
            universe=1,
            start_address=1,
            channel_count=8,
            manufacturer="Test Mfg",
            model="Test Model"
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=1,
            suggested_start_address=25,
            personality_recommended=2,
            rationale="Moved to avoid conflict",
            confidence=0.85,
            requires_readdressing=True
        )

        d = suggestion.to_dict()

        assert d["suggested_universe"] == 1
        assert d["suggested_start_address"] == 25
        assert d["personality_recommended"] == 2
        assert d["rationale"] == "Moved to avoid conflict"
        assert d["confidence"] == 0.85
        assert d["requires_readdressing"] is True
        assert d["fixture"]["uid"] == "02CA:12345678"

    def test_needs_personality_change_true(self):
        """Test needs_personality_change when change needed."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=8
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=1,
            suggested_start_address=1,
            personality_recommended=2,
            confidence=0.7,
            requires_readdressing=True
        )

        assert suggestion.needs_personality_change() is True

    def test_needs_personality_change_false(self):
        """Test needs_personality_change when no change needed."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=8
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=1,
            suggested_start_address=1,
            personality_recommended=None,
            confidence=0.99,
            requires_readdressing=False
        )

        assert suggestion.needs_personality_change() is False

    def test_needs_address_change_true(self):
        """Test needs_address_change when address changed."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=8
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=1,
            suggested_start_address=25,
            confidence=0.85,
            requires_readdressing=True
        )

        assert suggestion.needs_address_change() is True

    def test_needs_address_change_universe(self):
        """Test needs_address_change when universe changed."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=8
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=2,
            suggested_start_address=1,
            confidence=0.5,
            requires_readdressing=True
        )

        assert suggestion.needs_address_change() is True

    def test_needs_address_change_false(self):
        """Test needs_address_change when no change."""
        fixture = DiscoveredFixture(
            uid="02CA:12345678", universe=1, start_address=1, channel_count=8
        )
        suggestion = AutoPatchSuggestion(
            fixture=fixture,
            suggested_universe=1,
            suggested_start_address=1,
            confidence=0.99,
            requires_readdressing=False
        )

        assert suggestion.needs_address_change() is False


class TestAutoPatchEngine:
    """Tests for AutoPatchEngine."""

    @pytest.fixture
    def mock_patch_manager(self):
        """Create mock patch manager with empty patch."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        return manager

    @pytest.fixture
    def mock_patch_manager_with_fixtures(self):
        """Create mock patch manager with existing fixtures."""
        manager = Mock()
        manager.get_patch.return_value = {
            'fixtures': [
                {'universe': 1, 'start_channel': 1, 'channel_count': 16},
                {'universe': 1, 'start_channel': 25, 'channel_count': 8},
            ]
        }
        return manager

    @pytest.fixture
    def engine(self, mock_patch_manager):
        """Create engine with empty patch."""
        return AutoPatchEngine(mock_patch_manager)

    @pytest.fixture
    def engine_with_fixtures(self, mock_patch_manager_with_fixtures):
        """Create engine with existing fixtures."""
        return AutoPatchEngine(mock_patch_manager_with_fixtures)

    def test_init(self, mock_patch_manager):
        """Test engine initialization."""
        engine = AutoPatchEngine(mock_patch_manager)
        assert engine.patch_manager == mock_patch_manager
        assert engine.fixture_db is None

    def test_init_with_fixture_db(self, mock_patch_manager):
        """Test engine initialization with fixture database."""
        fixture_db = Mock()
        engine = AutoPatchEngine(mock_patch_manager, fixture_db)
        assert engine.fixture_db == fixture_db

    def test_suggest_no_conflict(self, engine):
        """Test suggestion when no conflicts exist."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8
            )
        }

        suggestions = engine.suggest_patch(fixtures)

        assert len(suggestions) == 1
        assert suggestions[0].confidence == AutoPatchEngine.CONFIDENCE_NO_CHANGE
        assert suggestions[0].requires_readdressing is False
        assert suggestions[0].suggested_start_address == 1

    def test_suggest_with_conflict(self, engine_with_fixtures):
        """Test suggestion when conflict exists."""
        # Discovered fixture conflicts with existing (1-16)
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=5, channel_count=8
            )
        }

        suggestions = engine_with_fixtures.suggest_patch(fixtures)

        assert len(suggestions) == 1
        assert suggestions[0].requires_readdressing is True
        # Should be moved after existing fixtures
        assert suggestions[0].suggested_start_address >= 33  # After 25+8

    def test_suggest_multiple_fixtures(self, engine):
        """Test suggestions for multiple fixtures."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8
            ),
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=9, channel_count=8
            ),
            "uid3": DiscoveredFixture(
                uid="uid3", universe=1, start_address=17, channel_count=8
            ),
        }

        suggestions = engine.suggest_patch(fixtures)

        assert len(suggestions) == 3
        # All should be conflict-free (no existing fixtures)
        for s in suggestions:
            assert s.confidence == AutoPatchEngine.CONFIDENCE_NO_CHANGE

    def test_suggest_contiguous_preference(self, engine_with_fixtures):
        """Test contiguous addressing preference."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=100, channel_count=8
            )
        }

        # No conflict at 100, but prefer_contiguous might suggest moving
        suggestions = engine_with_fixtures.suggest_patch(
            fixtures, prefer_contiguous=True
        )

        assert len(suggestions) == 1
        # Since 100 doesn't conflict, should keep it
        assert suggestions[0].suggested_start_address == 100

    def test_suggest_sorts_by_confidence(self, engine_with_fixtures):
        """Test suggestions sorted by confidence."""
        # Create fixtures where some conflict and some don't
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=5, channel_count=8
            ),  # Conflicts
            "uid2": DiscoveredFixture(
                uid="uid2", universe=1, start_address=100, channel_count=8
            ),  # No conflict
        }

        suggestions = engine_with_fixtures.suggest_patch(fixtures)

        assert len(suggestions) == 2
        # First should be highest confidence (no change)
        assert suggestions[0].confidence > suggestions[1].confidence

    def test_validate_valid_patch(self, engine):
        """Test validation of valid patch."""
        fixtures = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=1,
                confidence=0.99,
                requires_readdressing=False
            ),
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid2", universe=1, start_address=9, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=9,
                confidence=0.99,
                requires_readdressing=False
            ),
        ]

        assert engine.validate_patch(fixtures) is True

    def test_validate_overlap_detected(self, engine):
        """Test validation catches overlapping fixtures."""
        fixtures = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=1,
                confidence=0.99,
                requires_readdressing=False
            ),
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid2", universe=1, start_address=5, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=5,  # Overlaps with uid1 (1-8)
                confidence=0.99,
                requires_readdressing=False
            ),
        ]

        assert engine.validate_patch(fixtures) is False

    def test_validate_invalid_address_low(self, engine):
        """Test validation catches address below 1."""
        fixtures = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=0,  # Invalid
                confidence=0.99,
                requires_readdressing=True
            ),
        ]

        assert engine.validate_patch(fixtures) is False

    def test_validate_invalid_address_high(self, engine):
        """Test validation catches address exceeding 512."""
        fixtures = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=510,  # Would end at 517
                confidence=0.99,
                requires_readdressing=True
            ),
        ]

        assert engine.validate_patch(fixtures) is False

    def test_validate_skips_no_space(self, engine):
        """Test validation skips fixtures with no space."""
        fixtures = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=1,
                confidence=AutoPatchEngine.CONFIDENCE_NO_SPACE,  # 0.0
                requires_readdressing=False
            ),
        ]

        # Should pass because no-space fixtures are skipped
        assert engine.validate_patch(fixtures) is True


class TestConflictDetection:
    """Tests for conflict detection."""

    @pytest.fixture
    def engine_full(self):
        """Create engine with nearly full universe."""
        manager = Mock()
        # Fill channels 1-500
        manager.get_patch.return_value = {
            'fixtures': [
                {'universe': 1, 'start_channel': 1, 'channel_count': 500},
            ]
        }
        return AutoPatchEngine(manager)

    def test_conflict_triggers_readdressing(self, engine_full):
        """Test conflict detection triggers readdressing."""
        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=100, channel_count=8
            )
        }

        suggestions = engine_full.suggest_patch(fixtures)

        assert len(suggestions) == 1
        assert suggestions[0].requires_readdressing is True
        # Should move to 501-508
        assert suggestions[0].suggested_start_address == 501

    def test_different_universe_no_conflict(self):
        """Test fixtures in different universes don't conflict."""
        manager = Mock()
        manager.get_patch.return_value = {
            'fixtures': [
                {'universe': 1, 'start_channel': 1, 'channel_count': 512},
            ]
        }
        engine = AutoPatchEngine(manager)

        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=2, start_address=1, channel_count=8
            )
        }

        suggestions = engine.suggest_patch(fixtures)

        assert len(suggestions) == 1
        assert suggestions[0].requires_readdressing is False
        assert suggestions[0].confidence == AutoPatchEngine.CONFIDENCE_NO_CHANGE


class TestAddressAssignment:
    """Tests for address assignment logic."""

    def test_find_free_address_empty(self):
        """Test finding address in empty universe."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        addr = engine._find_free_address(
            universe=1,
            channel_count=8,
            occupied={1: set()}
        )

        assert addr == 1

    def test_find_free_address_after_existing(self):
        """Test finding address after existing fixtures."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        occupied = {1: set(range(1, 17))}  # 1-16 occupied
        addr = engine._find_free_address(
            universe=1,
            channel_count=8,
            occupied=occupied,
            prefer_contiguous=True
        )

        assert addr == 17

    def test_find_free_address_in_gap(self):
        """Test finding address in gap between fixtures."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        # Occupied: 1-8 and 25-32, gap at 9-24
        occupied = {1: set(range(1, 9)) | set(range(25, 33))}
        addr = engine._find_free_address(
            universe=1,
            channel_count=8,
            occupied=occupied,
            prefer_contiguous=False
        )

        assert addr == 9

    def test_find_free_address_no_space(self):
        """Test no space available."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        # Fill entire universe
        occupied = {1: set(range(1, 513))}
        addr = engine._find_free_address(
            universe=1,
            channel_count=8,
            occupied=occupied
        )

        assert addr is None


class TestPersonalityReduction:
    """Tests for personality reduction logic."""

    def test_reduced_personality_suggested(self):
        """Test personality reduction when needed."""
        manager = Mock()
        # Leave only 4 channels free (505-508)
        manager.get_patch.return_value = {
            'fixtures': [
                {'universe': 1, 'start_channel': 1, 'channel_count': 504},
            ]
        }
        engine = AutoPatchEngine(manager)

        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=100,  # Conflicts
                channel_count=16  # Too big for remaining space
            )
        }

        suggestions = engine.suggest_patch(fixtures, max_universes=1)

        assert len(suggestions) == 1
        suggestion = suggestions[0]

        # Should suggest reduced personality
        if suggestion.personality_recommended is not None:
            assert suggestion.confidence == AutoPatchEngine.CONFIDENCE_PERSONALITY_REDUCTION
        else:
            # Or no space if reduction couldn't help
            assert suggestion.confidence <= AutoPatchEngine.CONFIDENCE_DIFFERENT_UNIVERSE

    def test_estimate_channels_for_personality(self):
        """Test channel count estimation for personality."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        fixture = DiscoveredFixture(
            uid="uid1", universe=1, start_address=1, channel_count=16
        )

        # Personality 1 should be ~1/4
        assert engine._estimate_channels_for_personality(fixture, 1) == 4
        # Personality 2 should be ~1/2
        assert engine._estimate_channels_for_personality(fixture, 2) == 8


class TestApplySuggestions:
    """Tests for applying suggestions via RDM."""

    @pytest.fixture
    def mock_rdm_manager(self):
        """Create mock RDM manager."""
        manager = Mock()
        manager.set_personality = AsyncMock(return_value=True)
        manager.set_start_address = AsyncMock(return_value=True)
        return manager

    @pytest.mark.asyncio
    async def test_apply_no_changes(self, mock_rdm_manager):
        """Test applying suggestions with no changes needed."""
        patch_manager = Mock()
        patch_manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(patch_manager)

        suggestions = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=1,
                confidence=0.99,
                requires_readdressing=False
            )
        ]

        results = await engine.apply_suggestions(
            suggestions, mock_rdm_manager, "192.168.1.100"
        )

        assert results == {"uid1": True}
        # No RDM calls should be made
        mock_rdm_manager.set_personality.assert_not_called()
        mock_rdm_manager.set_start_address.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_address_change(self, mock_rdm_manager):
        """Test applying address change."""
        patch_manager = Mock()
        patch_manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(patch_manager)

        suggestions = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=25,
                confidence=0.85,
                requires_readdressing=True
            )
        ]

        results = await engine.apply_suggestions(
            suggestions, mock_rdm_manager, "192.168.1.100"
        )

        assert results == {"uid1": True}
        mock_rdm_manager.set_start_address.assert_called_once_with(
            "192.168.1.100", "uid1", 25
        )

    @pytest.mark.asyncio
    async def test_apply_personality_and_address(self, mock_rdm_manager):
        """Test applying both personality and address change."""
        patch_manager = Mock()
        patch_manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(patch_manager)

        suggestions = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=16
                ),
                suggested_universe=1,
                suggested_start_address=50,
                personality_recommended=2,
                confidence=0.7,
                requires_readdressing=True
            )
        ]

        results = await engine.apply_suggestions(
            suggestions, mock_rdm_manager, "192.168.1.100"
        )

        assert results == {"uid1": True}
        # Personality change first, then address
        mock_rdm_manager.set_personality.assert_called_once_with(
            "192.168.1.100", "uid1", 2
        )
        mock_rdm_manager.set_start_address.assert_called_once_with(
            "192.168.1.100", "uid1", 50
        )

    @pytest.mark.asyncio
    async def test_apply_handles_failure(self, mock_rdm_manager):
        """Test handling of RDM command failure."""
        mock_rdm_manager.set_start_address = AsyncMock(return_value=False)

        patch_manager = Mock()
        patch_manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(patch_manager)

        suggestions = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=25,
                confidence=0.85,
                requires_readdressing=True
            )
        ]

        results = await engine.apply_suggestions(
            suggestions, mock_rdm_manager, "192.168.1.100"
        )

        assert results == {"uid1": False}

    @pytest.mark.asyncio
    async def test_apply_multiple_fixtures(self, mock_rdm_manager):
        """Test applying suggestions for multiple fixtures."""
        patch_manager = Mock()
        patch_manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(patch_manager)

        suggestions = [
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid1", universe=1, start_address=1, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=25,
                confidence=0.85,
                requires_readdressing=True
            ),
            AutoPatchSuggestion(
                fixture=DiscoveredFixture(
                    uid="uid2", universe=1, start_address=9, channel_count=8
                ),
                suggested_universe=1,
                suggested_start_address=33,
                confidence=0.85,
                requires_readdressing=True
            ),
        ]

        results = await engine.apply_suggestions(
            suggestions, mock_rdm_manager, "192.168.1.100"
        )

        assert results == {"uid1": True, "uid2": True}
        assert mock_rdm_manager.set_start_address.call_count == 2


class TestConfidenceCalculation:
    """Tests for confidence scoring."""

    def test_confidence_no_change(self):
        """Test confidence for no change needed."""
        manager = Mock()
        manager.get_patch.return_value = {'fixtures': []}
        engine = AutoPatchEngine(manager)

        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=1, channel_count=8
            )
        }

        suggestions = engine.suggest_patch(fixtures)

        assert suggestions[0].confidence == AutoPatchEngine.CONFIDENCE_NO_CHANGE

    def test_confidence_different_universe(self):
        """Test confidence for different universe."""
        manager = Mock()
        # Fill universe 1
        manager.get_patch.return_value = {
            'fixtures': [
                {'universe': 1, 'start_channel': 1, 'channel_count': 512},
            ]
        }
        engine = AutoPatchEngine(manager)

        fixtures = {
            "uid1": DiscoveredFixture(
                uid="uid1", universe=1, start_address=100, channel_count=8
            )
        }

        suggestions = engine.suggest_patch(fixtures, max_universes=2)

        # Should be moved to universe 2
        moved = [s for s in suggestions if s.suggested_universe == 2]
        if moved:
            assert moved[0].confidence == AutoPatchEngine.CONFIDENCE_DIFFERENT_UNIVERSE


class TestProfileMatcher:
    """Tests for legacy ProfileMatcher class."""

    def test_generic_profile_mapping(self):
        """Test generic profile selection by footprint."""
        library = Mock()
        matcher = ProfileMatcher(library)

        # Test known footprints
        assert matcher._match_generic(1) == ("generic-dimmer", "default")
        assert matcher._match_generic(3) == ("generic-rgb", "default")
        assert matcher._match_generic(4) == ("generic-rgbw", "default")

    def test_generic_fallback_to_closest(self):
        """Test fallback to closest generic profile."""
        library = Mock()
        matcher = ProfileMatcher(library)

        # Footprint 2 should fall back to next available (3 -> rgb)
        result = matcher._match_generic(2)
        assert result[0] == "generic-rgb"


class TestAutoPatcher:
    """Tests for legacy AutoPatcher class."""

    def test_find_conflicts_same_universe(self):
        """Test conflict detection in same universe."""
        library = Mock()
        patcher = AutoPatcher(library)

        existing = [
            Mock(universe=1, start_channel=1, channel_count=8),
            Mock(universe=1, start_channel=20, channel_count=8),
        ]

        # Overlaps with first fixture
        conflicts = patcher.find_conflicts(
            start_channel=5, footprint=8, universe=1, existing_fixtures=existing
        )

        assert len(conflicts) == 1

    def test_find_conflicts_different_universe(self):
        """Test no conflict in different universe."""
        library = Mock()
        patcher = AutoPatcher(library)

        existing = [
            Mock(universe=1, start_channel=1, channel_count=512),
        ]

        conflicts = patcher.find_conflicts(
            start_channel=1, footprint=8, universe=2, existing_fixtures=existing
        )

        assert len(conflicts) == 0

    def test_suggest_next_address_empty(self):
        """Test next address suggestion in empty universe."""
        library = Mock()
        patcher = AutoPatcher(library)

        addr = patcher.suggest_next_address(
            footprint=8, universe=1, existing_fixtures=[]
        )

        assert addr == 1

    def test_suggest_next_address_after_last(self):
        """Test next address after existing fixtures."""
        library = Mock()
        patcher = AutoPatcher(library)

        existing = [
            Mock(universe=1, start_channel=1, channel_count=16),
        ]

        addr = patcher.suggest_next_address(
            footprint=8, universe=1, existing_fixtures=existing
        )

        assert addr == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
