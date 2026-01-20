"""
Unit tests for RDM Discovery Panel UI components.

Tests the dual-mode UI for RDM discovery (touchscreen + desktop).
These tests focus on logic and state management, mocking PySide6.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from enum import Enum
import os
import sys

# Check if we can create Qt widgets (need display)
_CAN_CREATE_WIDGETS = False
_QAPP = None

try:
    from app.ui.patch.rdm_discovery_panel import HAS_PYSIDE6
    if HAS_PYSIDE6:
        from PySide6.QtWidgets import QApplication
        # Create QApplication if one doesn't exist
        if QApplication.instance() is None:
            # Check for display availability
            if sys.platform == 'win32' or os.environ.get('DISPLAY'):
                try:
                    _QAPP = QApplication([])
                    _CAN_CREATE_WIDGETS = True
                except Exception:
                    _CAN_CREATE_WIDGETS = False
            else:
                _CAN_CREATE_WIDGETS = False
        else:
            _CAN_CREATE_WIDGETS = True
except ImportError:
    HAS_PYSIDE6 = False


def requires_qt_widgets(func):
    """Decorator to skip tests that require Qt widget creation."""
    return pytest.mark.skipif(
        not _CAN_CREATE_WIDGETS,
        reason="Cannot create Qt widgets (no display or PySide6)"
    )(func)


class TestDiscoveryState:
    """Tests for DiscoveryState enum."""

    def test_discovery_state_values(self):
        """Test that DiscoveryState has correct values."""
        from app.ui.patch.rdm_discovery_panel import DiscoveryState

        assert DiscoveryState.IDLE.value == "idle"
        assert DiscoveryState.DISCOVERING.value == "discovering"
        assert DiscoveryState.REVIEWING.value == "reviewing"
        assert DiscoveryState.APPLYING.value == "applying"

    def test_discovery_state_members(self):
        """Test that all expected states exist."""
        from app.ui.patch.rdm_discovery_panel import DiscoveryState

        assert len(DiscoveryState) == 4
        states = [s.value for s in DiscoveryState]
        assert "idle" in states
        assert "discovering" in states
        assert "reviewing" in states
        assert "applying" in states


class TestDiscoveryWorker:
    """Tests for DiscoveryWorker thread."""

    def test_worker_init_without_pyside6(self):
        """Test worker initialization without PySide6."""
        with patch('app.ui.patch.rdm_discovery_panel.HAS_PYSIDE6', False):
            # Re-import to get stub class
            from importlib import reload
            import app.ui.patch.rdm_discovery_panel as panel_module

            # Worker should still be importable
            worker_class = panel_module.DiscoveryWorker

    def test_worker_attributes(self):
        """Test worker stores correct attributes."""
        from app.ui.patch.rdm_discovery_panel import DiscoveryWorker, HAS_PYSIDE6

        if not HAS_PYSIDE6:
            pytest.skip("PySide6 not available")

        rdm_manager = Mock()
        auto_patch_engine = Mock()
        node_ip = "192.168.1.50"
        universes = [1, 2]

        worker = DiscoveryWorker(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            node_ip=node_ip,
            universes=universes
        )

        assert worker.rdm_manager == rdm_manager
        assert worker.auto_patch_engine == auto_patch_engine
        assert worker.node_ip == node_ip
        assert worker.universes == universes
        assert worker._cancelled is False

    def test_worker_cancel(self):
        """Test worker cancellation."""
        from app.ui.patch.rdm_discovery_panel import DiscoveryWorker, HAS_PYSIDE6

        if not HAS_PYSIDE6:
            pytest.skip("PySide6 not available")

        worker = DiscoveryWorker(
            rdm_manager=Mock(),
            auto_patch_engine=Mock(),
            node_ip="192.168.1.50",
            universes=[1]
        )

        assert worker._cancelled is False
        worker.cancel()
        assert worker._cancelled is True


class TestConstants:
    """Tests for module constants."""

    def test_touchscreen_constants(self):
        """Test touchscreen-related constants."""
        from app.ui.patch.rdm_discovery_panel import (
            TOUCHSCREEN_WIDTH_THRESHOLD,
            TOUCHSCREEN_BUTTON_HEIGHT,
            TOUCHSCREEN_ROW_HEIGHT,
        )

        assert TOUCHSCREEN_WIDTH_THRESHOLD == 1000
        assert TOUCHSCREEN_BUTTON_HEIGHT == 60
        assert TOUCHSCREEN_ROW_HEIGHT == 70

    def test_desktop_constants(self):
        """Test desktop-related constants."""
        from app.ui.patch.rdm_discovery_panel import (
            DESKTOP_BUTTON_HEIGHT,
            DESKTOP_ROW_HEIGHT,
        )

        assert DESKTOP_BUTTON_HEIGHT == 40
        assert DESKTOP_ROW_HEIGHT == 50


class TestTouchscreenLayout:
    """Tests for TouchscreenLayout widget."""

    def test_touchscreen_layout_class_exists(self):
        """Test TouchscreenLayout class exists when PySide6 available."""
        from app.ui.patch.rdm_discovery_panel import HAS_PYSIDE6
        if HAS_PYSIDE6:
            from app.ui.patch.rdm_discovery_panel import TouchscreenLayout
            assert TouchscreenLayout is not None

    @requires_qt_widgets
    def test_touchscreen_layout_has_signals(self):
        """Test TouchscreenLayout has required signals."""
        from app.ui.patch.rdm_discovery_panel import TouchscreenLayout

        layout = TouchscreenLayout()

        assert hasattr(layout, 'discover_clicked')
        assert hasattr(layout, 'identify_clicked')
        assert hasattr(layout, 'apply_clicked')

    @requires_qt_widgets
    def test_touchscreen_layout_initial_state(self):
        """Test TouchscreenLayout initial state."""
        from app.ui.patch.rdm_discovery_panel import (
            TouchscreenLayout,
            DiscoveryState
        )

        layout = TouchscreenLayout()

        assert layout._state == DiscoveryState.IDLE
        assert layout._fixtures == {}
        assert layout._suggestions == {}

    @requires_qt_widgets
    def test_touchscreen_layout_set_state_idle(self):
        """Test set_state to IDLE."""
        from app.ui.patch.rdm_discovery_panel import (
            TouchscreenLayout,
            DiscoveryState
        )

        layout = TouchscreenLayout()
        layout.set_state(DiscoveryState.IDLE)

        assert layout._state == DiscoveryState.IDLE
        assert layout.discover_btn.isEnabled()
        assert layout.universe_combo.isEnabled()

    @requires_qt_widgets
    def test_touchscreen_layout_set_state_discovering(self):
        """Test set_state to DISCOVERING."""
        from app.ui.patch.rdm_discovery_panel import (
            TouchscreenLayout,
            DiscoveryState
        )

        layout = TouchscreenLayout()
        layout.set_state(DiscoveryState.DISCOVERING)

        assert layout._state == DiscoveryState.DISCOVERING
        assert not layout.discover_btn.isEnabled()
        assert not layout.universe_combo.isEnabled()

    @requires_qt_widgets
    def test_touchscreen_layout_set_state_reviewing(self):
        """Test set_state to REVIEWING."""
        from app.ui.patch.rdm_discovery_panel import (
            TouchscreenLayout,
            DiscoveryState
        )

        layout = TouchscreenLayout()
        layout.set_state(DiscoveryState.REVIEWING)

        assert layout._state == DiscoveryState.REVIEWING
        assert layout.discover_btn.isEnabled()
        assert layout.universe_combo.isEnabled()

    @requires_qt_widgets
    def test_touchscreen_layout_update_progress(self):
        """Test update_progress method."""
        from app.ui.patch.rdm_discovery_panel import TouchscreenLayout

        layout = TouchscreenLayout()
        layout.update_progress(50, "Scanning universe 2...")

        assert layout.progress_bar.value() == 50
        assert layout.status_label.text() == "Scanning universe 2..."

    @requires_qt_widgets
    def test_touchscreen_universe_selection_all(self):
        """Test universe selection - all universes."""
        from app.ui.patch.rdm_discovery_panel import TouchscreenLayout

        layout = TouchscreenLayout()

        emitted_universes = []
        layout.discover_clicked.connect(
            lambda u: emitted_universes.append(u)
        )

        layout.universe_combo.setCurrentIndex(0)
        layout._on_discover_clicked()

        assert emitted_universes == [[1, 2, 3, 4]]

    @requires_qt_widgets
    def test_touchscreen_universe_selection_single(self):
        """Test universe selection - single universe."""
        from app.ui.patch.rdm_discovery_panel import TouchscreenLayout

        layout = TouchscreenLayout()

        emitted_universes = []
        layout.discover_clicked.connect(
            lambda u: emitted_universes.append(u)
        )

        layout.universe_combo.setCurrentIndex(2)
        layout._on_discover_clicked()

        assert emitted_universes == [[2]]


class TestDesktopLayout:
    """Tests for DesktopLayout widget."""

    def test_desktop_layout_class_exists(self):
        """Test DesktopLayout class exists when PySide6 available."""
        from app.ui.patch.rdm_discovery_panel import HAS_PYSIDE6
        if HAS_PYSIDE6:
            from app.ui.patch.rdm_discovery_panel import DesktopLayout
            assert DesktopLayout is not None

    @requires_qt_widgets
    def test_desktop_layout_has_signals(self):
        """Test DesktopLayout has required signals."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        assert hasattr(layout, 'discover_clicked')
        assert hasattr(layout, 'identify_clicked')
        assert hasattr(layout, 'apply_clicked')
        assert hasattr(layout, 'mode_switch_clicked')

    @requires_qt_widgets
    def test_desktop_layout_initial_state(self):
        """Test DesktopLayout initial state."""
        from app.ui.patch.rdm_discovery_panel import (
            DesktopLayout,
            DiscoveryState
        )

        layout = DesktopLayout()

        assert layout._state == DiscoveryState.IDLE
        assert layout._fixtures == {}
        assert layout._suggestions == {}
        assert not layout.identify_btn.isEnabled()
        assert not layout.apply_btn.isEnabled()

    @requires_qt_widgets
    def test_desktop_layout_set_state_idle(self):
        """Test set_state to IDLE."""
        from app.ui.patch.rdm_discovery_panel import (
            DesktopLayout,
            DiscoveryState
        )

        layout = DesktopLayout()
        layout.set_state(DiscoveryState.IDLE)

        assert layout._state == DiscoveryState.IDLE
        assert layout.discover_btn.isEnabled()
        assert layout.universe_combo.isEnabled()
        assert not layout.identify_btn.isEnabled()
        assert not layout.apply_btn.isEnabled()

    @requires_qt_widgets
    def test_desktop_layout_set_state_discovering(self):
        """Test set_state to DISCOVERING."""
        from app.ui.patch.rdm_discovery_panel import (
            DesktopLayout,
            DiscoveryState
        )

        layout = DesktopLayout()
        layout.set_state(DiscoveryState.DISCOVERING)

        assert layout._state == DiscoveryState.DISCOVERING
        assert not layout.discover_btn.isEnabled()
        assert not layout.universe_combo.isEnabled()

    @requires_qt_widgets
    def test_desktop_layout_set_state_reviewing(self):
        """Test set_state to REVIEWING."""
        from app.ui.patch.rdm_discovery_panel import (
            DesktopLayout,
            DiscoveryState
        )

        layout = DesktopLayout()
        layout.set_state(DiscoveryState.REVIEWING)

        assert layout._state == DiscoveryState.REVIEWING
        assert layout.discover_btn.isEnabled()
        assert layout.identify_btn.isEnabled()
        assert layout.apply_btn.isEnabled()

    @requires_qt_widgets
    def test_desktop_layout_update_progress(self):
        """Test update_progress method."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()
        layout.update_progress(75, "Enriching fixture data...")

        assert layout.progress_bar.value() == 75
        assert layout.status_label.text() == "Enriching fixture data..."

    @requires_qt_widgets
    def test_desktop_has_7_columns(self):
        """Test results table has 7 columns."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        assert layout.results_table.columnCount() == 7

    @requires_qt_widgets
    def test_desktop_sortable_table(self):
        """Test results table is sortable."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        assert layout.results_table.isSortingEnabled()

    @requires_qt_widgets
    def test_desktop_universe_selection_all(self):
        """Test universe selection - all universes."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        emitted_universes = []
        layout.discover_clicked.connect(
            lambda u: emitted_universes.append(u)
        )

        layout.universe_combo.setCurrentIndex(0)
        layout._on_discover_clicked()

        assert emitted_universes == [[1, 2, 3, 4]]

    @requires_qt_widgets
    def test_desktop_universe_selection_single(self):
        """Test universe selection - single universe."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        emitted_universes = []
        layout.discover_clicked.connect(
            lambda u: emitted_universes.append(u)
        )

        layout.universe_combo.setCurrentIndex(3)
        layout._on_discover_clicked()

        assert emitted_universes == [[3]]


class TestRdmDiscoveryPanel:
    """Tests for RdmDiscoveryPanel main widget."""

    @pytest.fixture
    def mock_managers(self):
        """Create mock managers."""
        rdm_manager = Mock()
        rdm_manager.discovery_engine = Mock()
        rdm_manager.transport = Mock()

        auto_patch_engine = Mock()
        auto_patch_engine.suggest_patch = Mock(return_value=[])

        patch_manager = Mock()

        return rdm_manager, auto_patch_engine, patch_manager

    def test_panel_class_exists(self):
        """Test RdmDiscoveryPanel class exists when PySide6 available."""
        from app.ui.patch.rdm_discovery_panel import HAS_PYSIDE6
        if HAS_PYSIDE6:
            from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel
            assert RdmDiscoveryPanel is not None

    @requires_qt_widgets
    def test_panel_init(self, mock_managers):
        """Test panel initialization."""
        from app.ui.patch.rdm_discovery_panel import (
            RdmDiscoveryPanel,
            DiscoveryState
        )

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        assert panel.rdm_manager == rdm_manager
        assert panel.auto_patch_engine == auto_patch_engine
        assert panel.patch_manager == patch_manager
        assert panel._state == DiscoveryState.IDLE
        assert panel._worker is None
        assert panel._current_results == {}

    @requires_qt_widgets
    def test_panel_has_both_layouts(self, mock_managers):
        """Test panel has both touchscreen and desktop layouts."""
        from app.ui.patch.rdm_discovery_panel import (
            RdmDiscoveryPanel,
            TouchscreenLayout,
            DesktopLayout
        )

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        assert isinstance(panel.touchscreen_layout, TouchscreenLayout)
        assert isinstance(panel.desktop_layout, DesktopLayout)

    @requires_qt_widgets
    def test_panel_set_node_ip(self, mock_managers):
        """Test set_node_ip method."""
        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        assert panel._node_ip == "192.168.1.100"
        panel.set_node_ip("10.0.0.50")
        assert panel._node_ip == "10.0.0.50"

    @requires_qt_widgets
    def test_panel_get_current_mode_touchscreen(self, mock_managers):
        """Test get_current_mode returns touchscreen."""
        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._switch_to_touchscreen()

        assert panel.get_current_mode() == "touchscreen"

    @requires_qt_widgets
    def test_panel_get_current_mode_desktop(self, mock_managers):
        """Test get_current_mode returns desktop."""
        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._switch_to_desktop()

        assert panel.get_current_mode() == "desktop"

    @requires_qt_widgets
    def test_panel_set_state_updates_both_layouts(self, mock_managers):
        """Test _set_state updates both layouts."""
        from app.ui.patch.rdm_discovery_panel import (
            RdmDiscoveryPanel,
            DiscoveryState
        )

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._set_state(DiscoveryState.DISCOVERING)

        assert panel._state == DiscoveryState.DISCOVERING
        assert panel.touchscreen_layout._state == DiscoveryState.DISCOVERING
        assert panel.desktop_layout._state == DiscoveryState.DISCOVERING

    @requires_qt_widgets
    def test_panel_on_progress_updates_both_layouts(self, mock_managers):
        """Test _on_progress updates both layouts."""
        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._on_progress(42, "Testing progress...")

        assert panel.touchscreen_layout.progress_bar.value() == 42
        assert panel.touchscreen_layout.status_label.text() == "Testing progress..."
        assert panel.desktop_layout.progress_bar.value() == 42
        assert panel.desktop_layout.status_label.text() == "Testing progress..."

    @requires_qt_widgets
    def test_panel_stacked_layout(self, mock_managers):
        """Test panel uses stacked layout."""
        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        assert panel.stacked_layout is not None
        assert panel.stacked_layout.count() == 2


class TestResultsDisplay:
    """Tests for results display functionality."""

    @pytest.fixture
    def mock_fixture(self):
        """Create a mock discovered fixture."""
        fixture = Mock()
        fixture.uid = "02CA:12345678"
        fixture.universe = 1
        fixture.start_address = 1
        fixture.channel_count = 24
        fixture.manufacturer = "Generic"
        fixture.model = "LED Par"
        fixture.personality_label = "24ch Mode"
        fixture.conflicts = []
        return fixture

    @pytest.fixture
    def mock_suggestion(self, mock_fixture):
        """Create a mock patch suggestion."""
        suggestion = Mock()
        suggestion.fixture = mock_fixture
        suggestion.suggested_universe = 1
        suggestion.suggested_start_address = 1
        suggestion.personality_recommended = None
        suggestion.rationale = "No change needed"
        suggestion.confidence = 0.99
        suggestion.requires_readdressing = False
        return suggestion

    @requires_qt_widgets
    def test_touchscreen_set_results(self, mock_fixture, mock_suggestion):
        """Test TouchscreenLayout.set_results."""
        from app.ui.patch.rdm_discovery_panel import TouchscreenLayout

        layout = TouchscreenLayout()

        fixtures = {mock_fixture.uid: mock_fixture}
        suggestions = {mock_fixture.uid: mock_suggestion}

        layout.set_results(fixtures, suggestions)

        assert layout._fixtures == fixtures
        assert layout._suggestions == suggestions
        assert layout.results_table.rowCount() == 1

    @requires_qt_widgets
    def test_desktop_set_results(self, mock_fixture, mock_suggestion):
        """Test DesktopLayout.set_results."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        fixtures = {mock_fixture.uid: mock_fixture}
        suggestions = {mock_fixture.uid: mock_suggestion}

        layout.set_results(fixtures, suggestions)

        assert layout._fixtures == fixtures
        assert layout._suggestions == suggestions
        assert layout.results_table.rowCount() == 1

    @requires_qt_widgets
    def test_desktop_summary_label_updated(self, mock_fixture, mock_suggestion):
        """Test desktop summary label is updated."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        fixtures = {mock_fixture.uid: mock_fixture}
        suggestions = {mock_fixture.uid: mock_suggestion}

        layout.set_results(fixtures, suggestions)

        assert "1 fixtures" in layout.summary_label.text()

    @requires_qt_widgets
    def test_desktop_conflict_count(self, mock_fixture):
        """Test desktop conflict count in summary."""
        from app.ui.patch.rdm_discovery_panel import DesktopLayout

        layout = DesktopLayout()

        mock_fixture.conflicts = ["02CA:AAAABBBB"]

        fixtures = {mock_fixture.uid: mock_fixture}
        suggestions = {}

        layout.set_results(fixtures, suggestions)

        assert "1 conflicts" in layout.summary_label.text()


class TestModuleExports:
    """Tests for module exports."""

    def test_init_exports(self):
        """Test __init__.py exports correct symbols."""
        from app.ui.patch import (
            DiscoveryState,
            DiscoveryWorker,
            TOUCHSCREEN_WIDTH_THRESHOLD,
            HAS_PYSIDE6,
        )

        assert DiscoveryState is not None
        assert DiscoveryWorker is not None
        assert TOUCHSCREEN_WIDTH_THRESHOLD == 1000
        assert isinstance(HAS_PYSIDE6, bool)

    def test_init_exports_pyside6_dependent(self):
        """Test PySide6-dependent exports."""
        from app.ui.patch import HAS_PYSIDE6

        if HAS_PYSIDE6:
            from app.ui.patch import (
                TouchscreenLayout,
                DesktopLayout,
                RdmDiscoveryPanel,
            )
            assert TouchscreenLayout is not None
            assert DesktopLayout is not None
            assert RdmDiscoveryPanel is not None


class TestModeAutoSwitch:
    """Tests for automatic mode switching."""

    @pytest.fixture
    def mock_managers(self):
        """Create mock managers."""
        return Mock(), Mock(), Mock()

    @requires_qt_widgets
    def test_mode_switch_at_threshold(self, mock_managers):
        """Test mode switches at width threshold."""
        from app.ui.patch.rdm_discovery_panel import (
            RdmDiscoveryPanel,
            TOUCHSCREEN_WIDTH_THRESHOLD
        )
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QResizeEvent

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._switch_to_desktop()
        assert panel.get_current_mode() == "desktop"

        event = QResizeEvent(
            QSize(TOUCHSCREEN_WIDTH_THRESHOLD - 1, 480),
            QSize(1920, 1080)
        )
        panel.resizeEvent(event)

        assert panel.get_current_mode() == "touchscreen"

    @requires_qt_widgets
    def test_mode_stays_desktop_above_threshold(self, mock_managers):
        """Test mode stays desktop above threshold."""
        from app.ui.patch.rdm_discovery_panel import (
            RdmDiscoveryPanel,
            TOUCHSCREEN_WIDTH_THRESHOLD
        )
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QResizeEvent

        rdm_manager, auto_patch_engine, patch_manager = mock_managers

        panel = RdmDiscoveryPanel(
            rdm_manager=rdm_manager,
            auto_patch_engine=auto_patch_engine,
            patch_manager=patch_manager
        )

        panel._switch_to_desktop()

        event = QResizeEvent(
            QSize(TOUCHSCREEN_WIDTH_THRESHOLD + 100, 768),
            QSize(800, 600)
        )
        panel.resizeEvent(event)

        assert panel.get_current_mode() == "desktop"


class TestStubClasses:
    """Tests for stub classes when PySide6 is not available."""

    def test_stub_rdm_discovery_panel_raises(self):
        """Test stub RdmDiscoveryPanel raises ImportError."""
        # This test only makes sense when PySide6 is not available
        from app.ui.patch.rdm_discovery_panel import HAS_PYSIDE6

        if HAS_PYSIDE6:
            pytest.skip("PySide6 is available, skip stub test")

        from app.ui.patch.rdm_discovery_panel import RdmDiscoveryPanel

        with pytest.raises(ImportError) as exc_info:
            RdmDiscoveryPanel(Mock(), Mock(), Mock())

        assert "PySide6 is required" in str(exc_info.value)
