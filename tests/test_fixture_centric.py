"""
Unit Tests for Fixture-Centric Architecture

Tests for:
- Phase 0: RenderedFixtureFrame creation and conversion
- Phase 1: Distribution modes and render_with_distribution
- Phase 2: AI fixture advisor suggestions
- Phase 3: Final render pipeline

Run with: pytest tests/test_fixture_centric.py -v
"""

import pytest
import sys
import os
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Phase 0 Tests: Fixture Render
# ============================================================

class TestRenderedFixtureState:
    """Tests for RenderedFixtureState"""

    def test_create_basic_state(self):
        """Test creating a basic fixture state"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(
            fixture_id="par_1",
            attributes={"intensity": 255, "color": [255, 0, 0]}
        )

        assert state.fixture_id == "par_1"
        assert state.get_intensity() == 255
        assert state.get_color() == [255, 0, 0]

    def test_set_intensity(self):
        """Test setting intensity with clamping"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(fixture_id="test")

        state.set_intensity(200)
        assert state.get_intensity() == 200

        state.set_intensity(300)  # Should clamp to 255
        assert state.get_intensity() == 255

        state.set_intensity(-50)  # Should clamp to 0
        assert state.get_intensity() == 0

    def test_set_color(self):
        """Test setting RGB and RGBW colors"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(fixture_id="test")

        state.set_color(255, 128, 64)
        assert state.get_color() == [255, 128, 64]

        state.set_color(100, 100, 100, 50)  # RGBW
        assert state.attributes["color"] == [100, 100, 100, 50]

    def test_apply_intensity_multiplier(self):
        """Test applying brightness multiplier"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(
            fixture_id="test",
            attributes={"intensity": 200}
        )

        state.apply_intensity_multiplier(0.5)
        assert state.get_intensity() == 100

    def test_copy(self):
        """Test deep copying fixture state"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(
            fixture_id="test",
            attributes={"intensity": 255, "color": [255, 0, 0]}
        )

        copy = state.copy()

        # Modify original
        state.set_intensity(100)
        state.attributes["color"][0] = 0

        # Copy should be unchanged
        assert copy.get_intensity() == 255
        assert copy.attributes["color"][0] == 255

    def test_serialization(self):
        """Test to_dict and from_dict"""
        from fixture_render import RenderedFixtureState

        state = RenderedFixtureState(
            fixture_id="par_1",
            attributes={"intensity": 200, "color": [255, 128, 0]},
            source="test"
        )

        data = state.to_dict()
        restored = RenderedFixtureState.from_dict(data)

        assert restored.fixture_id == state.fixture_id
        assert restored.get_intensity() == state.get_intensity()
        assert restored.get_color() == state.get_color()


class TestRenderedFixtureFrame:
    """Tests for RenderedFixtureFrame"""

    def test_create_empty_frame(self):
        """Test creating empty frame"""
        from fixture_render import RenderedFixtureFrame

        frame = RenderedFixtureFrame()

        assert len(frame.fixtures) == 0
        assert frame.frame_number == 0

    def test_add_fixture(self):
        """Test adding fixtures to frame"""
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        frame = RenderedFixtureFrame()

        state = RenderedFixtureState(
            fixture_id="par_1",
            attributes={"intensity": 255}
        )
        frame.set_fixture(state)

        assert "par_1" in frame.fixtures
        assert frame.get_fixture("par_1").get_intensity() == 255

    def test_get_all_fixture_ids(self):
        """Test getting all fixture IDs"""
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        frame = RenderedFixtureFrame()
        frame.set_fixture(RenderedFixtureState("par_1", {}))
        frame.set_fixture(RenderedFixtureState("par_2", {}))
        frame.set_fixture(RenderedFixtureState("wash_1", {}))

        ids = frame.get_all_fixture_ids()
        assert len(ids) == 3
        assert "par_1" in ids
        assert "par_2" in ids
        assert "wash_1" in ids

    def test_apply_intensity_to_all(self):
        """Test applying intensity multiplier to all fixtures"""
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        frame = RenderedFixtureFrame()
        frame.set_fixture(RenderedFixtureState("par_1", {"intensity": 200}))
        frame.set_fixture(RenderedFixtureState("par_2", {"intensity": 100}))

        frame.apply_intensity_to_all(0.5)

        assert frame.get_fixture("par_1").get_intensity() == 100
        assert frame.get_fixture("par_2").get_intensity() == 50

    def test_copy_frame(self):
        """Test deep copying frame"""
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        frame = RenderedFixtureFrame(frame_number=5)
        frame.set_fixture(RenderedFixtureState("par_1", {"intensity": 200}))

        copy = frame.copy()

        # Modify original
        frame.get_fixture("par_1").set_intensity(50)

        # Copy should be unchanged
        assert copy.get_fixture("par_1").get_intensity() == 200

    def test_serialization(self):
        """Test frame serialization"""
        from fixture_render import RenderedFixtureFrame, RenderedFixtureState

        frame = RenderedFixtureFrame(frame_number=10, seed=12345)
        frame.set_fixture(RenderedFixtureState("par_1", {"intensity": 255}))

        data = frame.to_dict()
        restored = RenderedFixtureFrame.from_dict(data)

        assert restored.frame_number == 10
        assert restored.seed == 12345
        assert "par_1" in restored.fixtures


class TestFixtureFrameBuilder:
    """Tests for FixtureFrameBuilder"""

    def test_builder_pattern(self):
        """Test builder pattern usage"""
        from fixture_render import FixtureFrameBuilder

        frame = (FixtureFrameBuilder()
            .add_fixture("par_1", intensity=255, color=[255, 0, 0])
            .add_fixture("par_2", intensity=200, color=[0, 255, 0])
            .set_metadata(frame_number=1, seed=12345)
            .build())

        assert len(frame.fixtures) == 2
        assert frame.get_fixture("par_1").get_intensity() == 255
        assert frame.get_fixture("par_2").get_color() == [0, 255, 0]
        assert frame.frame_number == 1
        assert frame.seed == 12345

    def test_set_all_intensity(self):
        """Test setting intensity for all fixtures"""
        from fixture_render import FixtureFrameBuilder

        frame = (FixtureFrameBuilder()
            .add_fixture("par_1", intensity=255)
            .add_fixture("par_2", intensity=255)
            .set_all_intensity(100)
            .build())

        assert frame.get_fixture("par_1").get_intensity() == 100
        assert frame.get_fixture("par_2").get_intensity() == 100


class TestCreateFrameFromFixtureChannels:
    """Tests for create_frame_from_fixture_channels utility"""

    def test_create_from_dict(self):
        """Test creating frame from fixture_channels dict"""
        from fixture_render import create_frame_from_fixture_channels

        fixture_channels = {
            "par_1": {"intensity": 255, "color": [255, 0, 0]},
            "par_2": {"intensity": 200, "color": [0, 255, 0]}
        }

        frame = create_frame_from_fixture_channels(fixture_channels, frame_number=5)

        assert len(frame.fixtures) == 2
        assert frame.get_fixture("par_1").get_intensity() == 255
        assert frame.get_fixture("par_2").get_color() == [0, 255, 0]
        assert frame.frame_number == 5


# ============================================================
# Phase 1 Tests: Distribution Modes
# ============================================================

class TestDistributionMode:
    """Tests for DistributionMode enum"""

    def test_all_modes_exist(self):
        """Test all expected modes exist"""
        from distribution_modes import DistributionMode

        assert DistributionMode.SYNCED.value == "synced"
        assert DistributionMode.INDEXED.value == "indexed"
        assert DistributionMode.PHASED.value == "phased"
        assert DistributionMode.PIXELATED.value == "pixelated"
        assert DistributionMode.RANDOM.value == "random"
        assert DistributionMode.GROUPED.value == "grouped"


class TestDistributionConfig:
    """Tests for DistributionConfig"""

    def test_default_config(self):
        """Test default configuration"""
        from distribution_modes import DistributionConfig, DistributionMode

        config = DistributionConfig()

        assert config.mode == DistributionMode.SYNCED
        assert config.phase_offset == 0.0
        assert config.reverse == False

    def test_config_serialization(self):
        """Test config serialization"""
        from distribution_modes import DistributionConfig, DistributionMode

        config = DistributionConfig(
            mode=DistributionMode.PHASED,
            phase_offset=0.15,
            reverse=True
        )

        data = config.to_dict()
        restored = DistributionConfig.from_dict(data)

        assert restored.mode == DistributionMode.PHASED
        assert restored.phase_offset == 0.15
        assert restored.reverse == True


class TestDistributionCalculator:
    """Tests for DistributionCalculator"""

    def test_synced_mode_same_phase(self):
        """Test SYNCED mode returns same phase for all fixtures"""
        from distribution_modes import DistributionCalculator, DistributionConfig, DistributionMode

        calc = DistributionCalculator()
        config = DistributionConfig(mode=DistributionMode.SYNCED)

        phases = [
            calc.get_fixture_phase(config, i, 4, 0.5)
            for i in range(4)
        ]

        # All phases should be the same
        assert all(p == phases[0] for p in phases)

    def test_phased_mode_different_phases(self):
        """Test PHASED mode returns different phases per fixture"""
        from distribution_modes import DistributionCalculator, DistributionConfig, DistributionMode

        calc = DistributionCalculator()
        config = DistributionConfig(mode=DistributionMode.PHASED, phase_offset=0.1)

        phases = [
            calc.get_fixture_phase(config, i, 4, 0.0)
            for i in range(4)
        ]

        # Each fixture should have different phase
        assert len(set(phases)) == 4

    def test_indexed_mode_linear_scale(self):
        """Test INDEXED mode scales linearly"""
        from distribution_modes import DistributionCalculator, DistributionConfig, DistributionMode

        calc = DistributionCalculator()
        config = DistributionConfig(mode=DistributionMode.INDEXED, phase_offset=1.0)

        multipliers = [
            calc.get_fixture_multiplier(config, i, 4)
            for i in range(4)
        ]

        # Should be increasing
        assert multipliers == sorted(multipliers)

    def test_random_mode_deterministic(self):
        """Test RANDOM mode is deterministic with same seed"""
        from distribution_modes import DistributionCalculator, DistributionConfig, DistributionMode

        calc1 = DistributionCalculator(session_seed=12345)
        calc2 = DistributionCalculator(session_seed=12345)
        config = DistributionConfig(mode=DistributionMode.RANDOM)

        phases1 = [calc1.get_fixture_phase(config, i, 4, 0.0) for i in range(4)]
        phases2 = [calc2.get_fixture_phase(config, i, 4, 0.0) for i in range(4)]

        assert phases1 == phases2


class TestDistributionPresets:
    """Tests for distribution presets"""

    def test_get_preset(self):
        """Test getting a preset"""
        from distribution_modes import get_distribution_preset, DistributionMode

        preset = get_distribution_preset("chase_forward")

        assert preset is not None
        assert preset.mode == DistributionMode.PHASED
        assert preset.phase_offset > 0

    def test_list_presets(self):
        """Test listing all presets"""
        from distribution_modes import list_distribution_presets

        presets = list_distribution_presets()

        assert len(presets) > 0
        assert any(p['name'] == 'chase_forward' for p in presets)
        assert any(p['name'] == 'rainbow_spread' for p in presets)


class TestSuggestDistribution:
    """Tests for distribution suggestion function"""

    def test_suggest_for_wave(self):
        """Test suggestion for wave modifier"""
        from distribution_modes import suggest_distribution_for_effect, DistributionMode

        suggestion = suggest_distribution_for_effect("wave", fixture_count=5)

        assert suggestion.mode == DistributionMode.PHASED

    def test_suggest_for_rainbow(self):
        """Test suggestion for rainbow modifier"""
        from distribution_modes import suggest_distribution_for_effect, DistributionMode

        suggestion = suggest_distribution_for_effect("rainbow", fixture_count=4)

        assert suggestion.mode == DistributionMode.INDEXED

    def test_suggest_for_twinkle(self):
        """Test suggestion for twinkle modifier"""
        from distribution_modes import suggest_distribution_for_effect, DistributionMode

        suggestion = suggest_distribution_for_effect("twinkle", fixture_count=6)

        assert suggestion.mode == DistributionMode.RANDOM

    def test_single_fixture_returns_synced(self):
        """Test single fixture returns SYNCED"""
        from distribution_modes import suggest_distribution_for_effect, DistributionMode

        suggestion = suggest_distribution_for_effect("wave", fixture_count=1)

        assert suggestion.mode == DistributionMode.SYNCED


# ============================================================
# Phase 2 Tests: AI Fixture Advisor
# ============================================================

class TestAISuggestion:
    """Tests for AISuggestion"""

    def test_suggestion_creation(self):
        """Test creating a suggestion"""
        from ai_fixture_advisor import AISuggestion, SuggestionType, SuggestionPriority

        suggestion = AISuggestion(
            suggestion_id="test_1",
            suggestion_type=SuggestionType.DISTRIBUTION_MODE,
            suggestion={"mode": "phased", "phase_offset": 0.1},
            reason="Test reason",
            confidence=0.85
        )

        assert suggestion.suggestion_id == "test_1"
        assert suggestion.applied == False
        assert suggestion.dismissed == False
        assert suggestion.confidence == 0.85

    def test_suggestion_serialization(self):
        """Test suggestion serialization"""
        from ai_fixture_advisor import AISuggestion, SuggestionType

        suggestion = AISuggestion(
            suggestion_id="test_1",
            suggestion_type=SuggestionType.DISTRIBUTION_MODE,
            suggestion={"mode": "phased"},
            reason="Test",
            confidence=0.8
        )

        data = suggestion.to_dict()
        restored = AISuggestion.from_dict(data)

        assert restored.suggestion_id == suggestion.suggestion_id
        assert restored.confidence == suggestion.confidence


class TestAIFixtureAdvisor:
    """Tests for AIFixtureAdvisor"""

    def test_suggest_distribution_for_wave(self):
        """Test distribution suggestions for wave modifier"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        suggestions = advisor.suggest_distribution(
            modifier_type="wave",
            modifier_params={},
            fixture_count=5
        )

        assert len(suggestions) > 0
        # Wave should suggest PHASED
        assert any(s.suggestion.get("mode") == "phased" for s in suggestions)

    def test_suggest_distribution_for_rainbow(self):
        """Test distribution suggestions for rainbow modifier"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        suggestions = advisor.suggest_distribution(
            modifier_type="rainbow",
            modifier_params={},
            fixture_count=4
        )

        assert len(suggestions) > 0
        # Rainbow should suggest INDEXED
        assert any(s.suggestion.get("mode") == "indexed" for s in suggestions)

    def test_no_suggestions_for_single_fixture(self):
        """Test no suggestions for single fixture"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        suggestions = advisor.suggest_distribution(
            modifier_type="wave",
            modifier_params={},
            fixture_count=1
        )

        assert len(suggestions) == 0

    def test_apply_suggestion(self):
        """Test marking suggestion as applied"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        suggestions = advisor.suggest_distribution("wave", {}, 5)

        if suggestions:
            suggestion_id = suggestions[0].suggestion_id
            result = advisor.apply_suggestion(suggestion_id)
            assert result == True
            assert suggestions[0].applied == True

    def test_dismiss_suggestion(self):
        """Test dismissing suggestion"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        suggestions = advisor.suggest_distribution("wave", {}, 5)

        if suggestions:
            suggestion_id = suggestions[0].suggestion_id
            result = advisor.dismiss_suggestion(suggestion_id)
            assert result == True
            assert suggestions[0].dismissed == True

    def test_get_pending_suggestions(self):
        """Test getting pending suggestions"""
        from ai_fixture_advisor import AIFixtureAdvisor

        advisor = AIFixtureAdvisor()
        advisor.clear_suggestions()

        # Generate some suggestions
        advisor.suggest_distribution("wave", {}, 5)
        advisor.suggest_distribution("rainbow", {}, 4)

        pending = advisor.get_pending_suggestions()
        assert len(pending) > 0

        # Apply one
        advisor.apply_suggestion(pending[0].suggestion_id)

        # Should have one fewer pending
        new_pending = advisor.get_pending_suggestions()
        assert len(new_pending) == len(pending) - 1


# ============================================================
# Phase 3 Tests: Final Render Pipeline
# ============================================================

class TestRenderJob:
    """Tests for RenderJob"""

    def test_create_render_job(self):
        """Test creating a render job"""
        from final_render_pipeline import RenderJob

        job = RenderJob(
            job_id="test_job",
            fixture_ids=["par_1", "par_2"],
            fixture_channels={
                "par_1": {"intensity": 255, "color": [255, 0, 0]},
                "par_2": {"intensity": 200, "color": [0, 255, 0]}
            },
            universes=[2]
        )

        assert job.job_id == "test_job"
        assert len(job.fixture_ids) == 2
        assert job.has_fixture_data() == True

    def test_job_without_fixture_data(self):
        """Test job without fixture data falls back to legacy"""
        from final_render_pipeline import RenderJob

        job = RenderJob(
            job_id="legacy_job",
            channels={1: 255, 2: 128}
        )

        assert job.has_fixture_data() == False


class TestFinalRenderPipeline:
    """Tests for FinalRenderPipeline"""

    def test_pipeline_creation(self):
        """Test creating pipeline"""
        from final_render_pipeline import FinalRenderPipeline

        pipeline = FinalRenderPipeline()

        assert pipeline is not None
        assert pipeline.features.FIXTURE_CENTRIC_ENABLED == True

    def test_feature_flags(self):
        """Test feature flags"""
        from final_render_pipeline import FinalRenderPipeline

        pipeline = FinalRenderPipeline()

        # Disable fixture centric
        pipeline.features.FIXTURE_CENTRIC_ENABLED = False
        assert pipeline.features.FIXTURE_CENTRIC_ENABLED == False

        # Re-enable
        pipeline.features.FIXTURE_CENTRIC_ENABLED = True
        assert pipeline.features.FIXTURE_CENTRIC_ENABLED == True

    def test_get_status(self):
        """Test getting pipeline status"""
        from final_render_pipeline import FinalRenderPipeline

        pipeline = FinalRenderPipeline()
        status = pipeline.get_status()

        assert "render_count" in status
        assert "features" in status
        assert status["features"]["fixture_centric"] == True


class TestRenderTimeContext:
    """Tests for RenderTimeContext"""

    def test_create_context(self):
        """Test creating time context"""
        from final_render_pipeline import RenderTimeContext

        ctx = RenderTimeContext(
            absolute_time=time.time(),
            delta_time=1/30,
            elapsed_time=1.5,
            frame_number=45,
            seed=12345
        )

        assert ctx.frame_number == 45
        assert ctx.seed == 12345

    def test_convert_to_render_engine_ctx(self):
        """Test converting to render engine context"""
        from final_render_pipeline import RenderTimeContext

        ctx = RenderTimeContext(
            absolute_time=time.time(),
            delta_time=1/30,
            elapsed_time=1.5,
            frame_number=45,
            seed=12345
        )

        render_ctx = ctx.to_render_engine_ctx()

        assert render_ctx.frame_number == 45
        assert render_ctx.seed == 12345


# ============================================================
# Integration Tests
# ============================================================

class TestModifierWithDistribution:
    """Integration tests for Modifier with distribution"""

    def test_modifier_with_distribution_config(self):
        """Test Modifier with distribution configuration"""
        from unified_playback import Modifier
        from distribution_modes import DistributionConfig, DistributionMode

        dist_config = DistributionConfig(
            mode=DistributionMode.PHASED,
            phase_offset=0.15
        )

        modifier = Modifier(
            id="test_mod",
            type="wave",
            params={"speed": 1.0},
            distribution=dist_config.to_dict()
        )

        # Get distribution config back
        config = modifier.get_distribution_config()

        assert config.mode == DistributionMode.PHASED
        assert config.phase_offset == 0.15

    def test_modifier_default_distribution(self):
        """Test Modifier without distribution uses SYNCED default"""
        from unified_playback import Modifier
        from distribution_modes import DistributionMode

        modifier = Modifier(
            id="test_mod",
            type="pulse",
            params={}
        )

        config = modifier.get_distribution_config()

        assert config.mode == DistributionMode.SYNCED


class TestPlaybackSessionWithFixtures:
    """Integration tests for PlaybackSession with fixtures"""

    def test_session_with_fixture_channels(self):
        """Test PlaybackSession with fixture_channels"""
        from unified_playback import PlaybackSession, PlaybackType

        session = PlaybackSession(
            session_id="test_session",
            playback_type=PlaybackType.LOOK,
            name="Test Look",
            fixture_ids=["par_1", "par_2"],
            fixture_channels={
                "par_1": {"intensity": 255, "color": [255, 0, 0]},
                "par_2": {"intensity": 200, "color": [0, 255, 0]}
            }
        )

        assert len(session.fixture_ids) == 2
        assert "par_1" in session.fixture_channels
        assert session.fixture_channels["par_1"]["intensity"] == 255

    def test_session_backward_compat(self):
        """Test PlaybackSession backward compatibility with channels"""
        from unified_playback import PlaybackSession, PlaybackType

        session = PlaybackSession(
            session_id="legacy_session",
            playback_type=PlaybackType.LOOK,
            name="Legacy Look",
            channels={1: 255, 2: 128, 3: 64}
        )

        # Should work with empty fixture fields
        assert len(session.fixture_ids) == 0
        assert len(session.fixture_channels) == 0
        assert session.channels[1] == 255


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
