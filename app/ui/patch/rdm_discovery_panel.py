"""
RDM Discovery Panel - Dual Mode UI (Touchscreen + Desktop)

This module provides a responsive RDM discovery interface that works on:
- 7-inch touchscreen (800x480) - portrait, touch-first
- Desktop/laptop (1080p+) - landscape, mouse/keyboard

Classes:
    RdmDiscoveryPanel: Main widget with auto-switching layouts
    TouchscreenLayout: Touch-optimized layout for 7-inch displays
    DesktopLayout: Desktop-optimized layout with full features
    DiscoveryWorker: Background thread for RDM discovery

Usage:
    panel = RdmDiscoveryPanel(rdm_manager, auto_patch_engine, patch_manager)
    panel.show()
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

try:
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QStackedLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QProgressBar,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QScrollArea,
        QFrame,
        QSplitter,
        QMessageBox,
        QSizePolicy,
        QAbstractItemView,
    )
    from PySide6.QtCore import (
        Qt,
        Signal,
        Slot,
        QThread,
        QSize,
        QTimer,
    )
    from PySide6.QtGui import (
        QFont,
        QColor,
        QKeySequence,
        QShortcut,
    )
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    # Create stub classes for testing without PySide6
    class QWidget:  # type: ignore
        pass
    class Signal:  # type: ignore
        def __init__(self, *args: Any) -> None:
            pass
    class QThread:  # type: ignore
        pass

logger = logging.getLogger(__name__)

# Constants
TOUCHSCREEN_WIDTH_THRESHOLD = 1000
TOUCHSCREEN_BUTTON_HEIGHT = 60
DESKTOP_BUTTON_HEIGHT = 40
TOUCHSCREEN_ROW_HEIGHT = 70
DESKTOP_ROW_HEIGHT = 50


class DiscoveryState(Enum):
    """State machine states for discovery UI."""
    IDLE = "idle"
    DISCOVERING = "discovering"
    REVIEWING = "reviewing"
    APPLYING = "applying"


class DiscoveryWorker(QThread if HAS_PYSIDE6 else object):
    """
    Background worker thread for RDM discovery.

    Performs discovery operations without blocking the UI.

    Signals:
        progress_update: (int, str) - percent complete and status message
        discovery_complete: (dict) - discovered fixtures dictionary
        discovery_error: (str) - error message
    """

    if HAS_PYSIDE6:
        progress_update = Signal(int, str)
        discovery_complete = Signal(dict)
        discovery_error = Signal(str)

    def __init__(
        self,
        rdm_manager: Any,
        auto_patch_engine: Any,
        node_ip: str,
        universes: List[int],
        parent: Optional[Any] = None
    ):
        """
        Initialize discovery worker.

        Args:
            rdm_manager: RDM manager for discovery operations
            auto_patch_engine: Auto-patch engine for suggestions
            node_ip: IP address of ESP32 node
            universes: List of universes to scan
            parent: Parent QObject
        """
        if HAS_PYSIDE6:
            super().__init__(parent)
        self.rdm_manager = rdm_manager
        self.auto_patch_engine = auto_patch_engine
        self.node_ip = node_ip
        self.universes = universes
        self._cancelled = False

    def run(self) -> None:
        """Run discovery in background thread."""
        if not HAS_PYSIDE6:
            return

        try:
            self.progress_update.emit(0, "Starting discovery...")

            # Get discovery engine from rdm_manager
            discovery_engine = getattr(
                self.rdm_manager, 'discovery_engine', None
            )

            if not discovery_engine:
                self.discovery_error.emit("Discovery engine not available")
                return

            # Register progress callback
            def on_progress(percent: int, message: str) -> None:
                if not self._cancelled:
                    self.progress_update.emit(percent, message)

            discovery_engine.on_progress(on_progress)

            # Run discovery (this is async, need to handle in Qt thread)
            import asyncio

            async def do_discovery() -> Dict[str, Any]:
                return await discovery_engine.discover_universes(
                    self.node_ip, self.universes
                )

            # Run async in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                discovered = loop.run_until_complete(do_discovery())
            finally:
                loop.close()

            if self._cancelled:
                return

            # Generate patch suggestions
            self.progress_update.emit(90, "Generating patch suggestions...")
            suggestions = self.auto_patch_engine.suggest_patch(discovered)

            # Combine results
            result = {
                'fixtures': discovered,
                'suggestions': {s.fixture.uid: s for s in suggestions}
            }

            self.progress_update.emit(100, "Discovery complete!")
            self.discovery_complete.emit(result)

        except Exception as e:
            logger.error(f"Discovery error: {e}")
            self.discovery_error.emit(str(e))

    def cancel(self) -> None:
        """Cancel the discovery operation."""
        self._cancelled = True


if HAS_PYSIDE6:

    class TouchscreenLayout(QWidget):
        """
        Touch-optimized layout for 7-inch displays (800x480).

        Features:
        - Large buttons (60px+)
        - Portrait orientation
        - Single column table
        - Minimal text
        - Touch-friendly gestures
        """

        # Signals
        discover_clicked = Signal(list)  # universes
        identify_clicked = Signal(str)   # uid
        apply_clicked = Signal()

        def __init__(self, parent: Optional[QWidget] = None):
            """Initialize touchscreen layout."""
            super().__init__(parent)
            self._state = DiscoveryState.IDLE
            self._fixtures: Dict[str, Any] = {}
            self._suggestions: Dict[str, Any] = {}
            self._setup_ui()

        def _setup_ui(self) -> None:
            """Set up the touchscreen UI."""
            layout = QVBoxLayout(self)
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)

            # Title
            self.title_label = QLabel("AETHER RDM Discovery")
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_font = QFont()
            title_font.setBold(True)
            title_font.setPointSize(16)
            self.title_label.setFont(title_font)
            layout.addWidget(self.title_label)

            # Universe selector
            self.universe_combo = QComboBox()
            self.universe_combo.addItems([
                "All Universes",
                "Universe 1",
                "Universe 2",
                "Universe 3",
                "Universe 4"
            ])
            self.universe_combo.setMinimumHeight(44)
            layout.addWidget(self.universe_combo)

            # Discover button
            self.discover_btn = QPushButton("DISCOVER FIXTURES")
            self.discover_btn.setMinimumHeight(TOUCHSCREEN_BUTTON_HEIGHT)
            discover_font = QFont()
            discover_font.setBold(True)
            discover_font.setPointSize(16)
            self.discover_btn.setFont(discover_font)
            self.discover_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "border-radius: 8px; } "
                "QPushButton:pressed { background-color: #388E3C; }"
            )
            self.discover_btn.clicked.connect(self._on_discover_clicked)
            layout.addWidget(self.discover_btn)

            # Progress section (hidden initially)
            self.progress_frame = QFrame()
            progress_layout = QVBoxLayout(self.progress_frame)

            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimumHeight(40)
            self.progress_bar.setRange(0, 100)
            progress_layout.addWidget(self.progress_bar)

            self.status_label = QLabel("Ready")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_font = QFont()
            status_font.setPointSize(12)
            self.status_label.setFont(status_font)
            progress_layout.addWidget(self.status_label)

            self.progress_frame.hide()
            layout.addWidget(self.progress_frame)

            # Results table (hidden initially)
            self.results_frame = QFrame()
            results_layout = QVBoxLayout(self.results_frame)

            self.results_table = QTableWidget()
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels([
                "Fixture", "Current", "Suggested", "Action"
            ])
            self.results_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
            self.results_table.verticalHeader().setDefaultSectionSize(
                TOUCHSCREEN_ROW_HEIGHT
            )
            self.results_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            results_layout.addWidget(self.results_table)

            self.results_frame.hide()
            layout.addWidget(self.results_frame, 1)  # Stretch factor

            # Bottom action buttons (hidden initially)
            self.action_frame = QFrame()
            action_layout = QHBoxLayout(self.action_frame)
            action_layout.setSpacing(5)

            self.identify_btn = QPushButton("IDENTIFY")
            self.identify_btn.setMinimumHeight(50)
            self.identify_btn.clicked.connect(self._on_identify_clicked)
            action_layout.addWidget(self.identify_btn)

            self.apply_btn = QPushButton("APPLY PATCH")
            self.apply_btn.setMinimumHeight(50)
            self.apply_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "border-radius: 8px; } "
                "QPushButton:pressed { background-color: #388E3C; }"
            )
            self.apply_btn.clicked.connect(self._on_apply_clicked)
            action_layout.addWidget(self.apply_btn)

            self.action_frame.hide()
            layout.addWidget(self.action_frame)

        def _on_discover_clicked(self) -> None:
            """Handle discover button click."""
            index = self.universe_combo.currentIndex()
            if index == 0:
                universes = [1, 2, 3, 4]
            else:
                universes = [index]
            self.discover_clicked.emit(universes)

        def _on_identify_clicked(self) -> None:
            """Handle identify button click."""
            selected = self.results_table.selectedItems()
            if selected:
                row = selected[0].row()
                uid = self.results_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if uid:
                    self.identify_clicked.emit(uid)

        def _on_apply_clicked(self) -> None:
            """Handle apply button click."""
            self.apply_clicked.emit()

        def set_state(self, state: DiscoveryState) -> None:
            """Set UI state."""
            self._state = state

            if state == DiscoveryState.IDLE:
                self.discover_btn.setEnabled(True)
                self.universe_combo.setEnabled(True)
                self.progress_frame.hide()
                self.results_frame.hide()
                self.action_frame.hide()

            elif state == DiscoveryState.DISCOVERING:
                self.discover_btn.setEnabled(False)
                self.universe_combo.setEnabled(False)
                self.progress_frame.show()
                self.results_frame.hide()
                self.action_frame.hide()

            elif state == DiscoveryState.REVIEWING:
                self.discover_btn.setEnabled(True)
                self.universe_combo.setEnabled(True)
                self.progress_frame.hide()
                self.results_frame.show()
                self.action_frame.show()

            elif state == DiscoveryState.APPLYING:
                self.discover_btn.setEnabled(False)
                self.apply_btn.setEnabled(False)
                self.identify_btn.setEnabled(False)

        def update_progress(self, percent: int, message: str) -> None:
            """Update progress display."""
            self.progress_bar.setValue(percent)
            self.status_label.setText(message)

        def set_results(
            self,
            fixtures: Dict[str, Any],
            suggestions: Dict[str, Any]
        ) -> None:
            """Populate results table."""
            self._fixtures = fixtures
            self._suggestions = suggestions

            self.results_table.setRowCount(len(fixtures))

            for row, (uid, fixture) in enumerate(fixtures.items()):
                suggestion = suggestions.get(uid)

                # Fixture column
                fixture_item = QTableWidgetItem(
                    f"{fixture.manufacturer}\n{fixture.model}"
                )
                fixture_item.setData(Qt.ItemDataRole.UserRole, uid)
                self.results_table.setItem(row, 0, fixture_item)

                # Current address column
                current_item = QTableWidgetItem(
                    f"U{fixture.universe}:{fixture.start_address}"
                )
                self.results_table.setItem(row, 1, current_item)

                # Suggested column
                if suggestion:
                    if suggestion.requires_readdressing:
                        suggested_text = (
                            f"U{suggestion.suggested_universe}:"
                            f"{suggestion.suggested_start_address}"
                        )
                        if suggestion.personality_recommended:
                            suggested_text += " (resize)"
                    else:
                        suggested_text = "Keep"
                    suggested_item = QTableWidgetItem(suggested_text)

                    # Color based on confidence
                    if suggestion.confidence >= 0.95:
                        suggested_item.setBackground(QColor("#ccffcc"))
                    elif suggestion.confidence >= 0.70:
                        suggested_item.setBackground(QColor("#ffffcc"))
                    else:
                        suggested_item.setBackground(QColor("#ffcccc"))
                else:
                    suggested_item = QTableWidgetItem("N/A")

                self.results_table.setItem(row, 2, suggested_item)

                # Action column - identify button
                identify_btn = QPushButton("ID")
                identify_btn.setFixedSize(40, 40)
                identify_btn.clicked.connect(
                    lambda checked, u=uid: self.identify_clicked.emit(u)
                )
                self.results_table.setCellWidget(row, 3, identify_btn)


    class DesktopLayout(QWidget):
        """
        Desktop-optimized layout for 1080p+ displays.

        Features:
        - Sidebar + main area
        - Multi-column table
        - Keyboard shortcuts
        - Multi-select support
        - Sortable columns
        """

        # Signals
        discover_clicked = Signal(list)  # universes
        identify_clicked = Signal(str)   # uid
        apply_clicked = Signal()
        mode_switch_clicked = Signal()   # switch to touchscreen

        def __init__(self, parent: Optional[QWidget] = None):
            """Initialize desktop layout."""
            super().__init__(parent)
            self._state = DiscoveryState.IDLE
            self._fixtures: Dict[str, Any] = {}
            self._suggestions: Dict[str, Any] = {}
            self._setup_ui()
            self._setup_shortcuts()

        def _setup_ui(self) -> None:
            """Set up the desktop UI."""
            main_layout = QHBoxLayout(self)
            main_layout.setSpacing(10)
            main_layout.setContentsMargins(10, 10, 10, 10)

            # Left sidebar
            sidebar = QFrame()
            sidebar.setFixedWidth(200)
            sidebar.setStyleSheet(
                "QFrame { background-color: #f5f5f5; border-radius: 8px; }"
            )
            sidebar_layout = QVBoxLayout(sidebar)

            # Title
            title_label = QLabel("AETHER RDM\nDiscovery")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_font = QFont()
            title_font.setBold(True)
            title_font.setPointSize(14)
            title_label.setFont(title_font)
            sidebar_layout.addWidget(title_label)

            sidebar_layout.addSpacing(20)

            # Universe selector
            universe_label = QLabel("Universe:")
            sidebar_layout.addWidget(universe_label)

            self.universe_combo = QComboBox()
            self.universe_combo.addItems([
                "All Universes",
                "Universe 1",
                "Universe 2",
                "Universe 3",
                "Universe 4"
            ])
            self.universe_combo.setMinimumHeight(40)
            sidebar_layout.addWidget(self.universe_combo)

            sidebar_layout.addSpacing(10)

            # Discover button
            self.discover_btn = QPushButton("Discover")
            self.discover_btn.setMinimumHeight(DESKTOP_BUTTON_HEIGHT)
            self.discover_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "border-radius: 4px; } "
                "QPushButton:hover { background-color: #45a049; } "
                "QPushButton:pressed { background-color: #388E3C; }"
            )
            self.discover_btn.clicked.connect(self._on_discover_clicked)
            sidebar_layout.addWidget(self.discover_btn)

            sidebar_layout.addSpacing(20)

            # Progress section
            self.progress_frame = QFrame()
            progress_layout = QVBoxLayout(self.progress_frame)
            progress_layout.setContentsMargins(0, 0, 0, 0)

            progress_label = QLabel("Progress:")
            progress_layout.addWidget(progress_label)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            progress_layout.addWidget(self.progress_bar)

            self.status_label = QLabel("Ready")
            self.status_label.setWordWrap(True)
            status_font = QFont()
            status_font.setPointSize(10)
            self.status_label.setFont(status_font)
            progress_layout.addWidget(self.status_label)

            self.progress_frame.hide()
            sidebar_layout.addWidget(self.progress_frame)

            # Conflicts summary
            self.conflicts_frame = QFrame()
            conflicts_layout = QVBoxLayout(self.conflicts_frame)
            conflicts_layout.setContentsMargins(0, 0, 0, 0)

            conflicts_label = QLabel("Conflicts:")
            conflicts_label.setStyleSheet("font-weight: bold;")
            conflicts_layout.addWidget(conflicts_label)

            self.conflicts_list = QLabel("None")
            self.conflicts_list.setWordWrap(True)
            self.conflicts_list.setStyleSheet("color: #c00;")
            conflicts_layout.addWidget(self.conflicts_list)

            self.conflicts_frame.hide()
            sidebar_layout.addWidget(self.conflicts_frame)

            sidebar_layout.addStretch()

            # Mode switch button
            self.mode_btn = QPushButton("Switch to Touch Mode")
            self.mode_btn.setMinimumHeight(DESKTOP_BUTTON_HEIGHT)
            self.mode_btn.clicked.connect(self.mode_switch_clicked.emit)
            sidebar_layout.addWidget(self.mode_btn)

            main_layout.addWidget(sidebar)

            # Main area
            main_area = QFrame()
            main_area_layout = QVBoxLayout(main_area)

            # Results table
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(7)
            self.results_table.setHorizontalHeaderLabels([
                "Fixture", "Manufacturer", "Model",
                "Current", "Suggested", "Personality", "Conflicts"
            ])
            self.results_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Interactive
            )
            self.results_table.horizontalHeader().setStretchLastSection(True)
            self.results_table.verticalHeader().setDefaultSectionSize(
                DESKTOP_ROW_HEIGHT
            )
            self.results_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.results_table.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            self.results_table.setSortingEnabled(True)
            self.results_table.doubleClicked.connect(self._on_row_double_clicked)
            main_area_layout.addWidget(self.results_table)

            # Bottom toolbar
            toolbar = QFrame()
            toolbar_layout = QHBoxLayout(toolbar)
            toolbar_layout.setContentsMargins(0, 5, 0, 0)

            self.summary_label = QLabel("Ready to discover fixtures")
            toolbar_layout.addWidget(self.summary_label)

            toolbar_layout.addStretch()

            self.identify_btn = QPushButton("Identify Selected")
            self.identify_btn.setMinimumHeight(DESKTOP_BUTTON_HEIGHT)
            self.identify_btn.clicked.connect(self._on_identify_clicked)
            self.identify_btn.setEnabled(False)
            toolbar_layout.addWidget(self.identify_btn)

            self.apply_btn = QPushButton("Apply Patch")
            self.apply_btn.setMinimumHeight(DESKTOP_BUTTON_HEIGHT)
            self.apply_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "border-radius: 4px; } "
                "QPushButton:hover { background-color: #45a049; } "
                "QPushButton:pressed { background-color: #388E3C; }"
            )
            self.apply_btn.clicked.connect(self._on_apply_clicked)
            self.apply_btn.setEnabled(False)
            toolbar_layout.addWidget(self.apply_btn)

            main_area_layout.addWidget(toolbar)

            main_layout.addWidget(main_area, 1)  # Stretch factor

        def _setup_shortcuts(self) -> None:
            """Set up keyboard shortcuts."""
            # Ctrl+Enter to apply
            apply_shortcut = QShortcut(
                QKeySequence("Ctrl+Return"), self
            )
            apply_shortcut.activated.connect(self._on_apply_clicked)

            # Escape to cancel
            cancel_shortcut = QShortcut(QKeySequence("Escape"), self)
            cancel_shortcut.activated.connect(self._on_cancel)

            # Space to identify selected
            identify_shortcut = QShortcut(QKeySequence("Space"), self)
            identify_shortcut.activated.connect(self._on_identify_clicked)

        def _on_discover_clicked(self) -> None:
            """Handle discover button click."""
            index = self.universe_combo.currentIndex()
            if index == 0:
                universes = [1, 2, 3, 4]
            else:
                universes = [index]
            self.discover_clicked.emit(universes)

        def _on_identify_clicked(self) -> None:
            """Handle identify button click."""
            selected = self.results_table.selectedItems()
            if selected:
                row = selected[0].row()
                uid_item = self.results_table.item(row, 0)
                if uid_item:
                    uid = uid_item.data(Qt.ItemDataRole.UserRole)
                    if uid:
                        self.identify_clicked.emit(uid)

        def _on_apply_clicked(self) -> None:
            """Handle apply button click."""
            if self.apply_btn.isEnabled():
                self.apply_clicked.emit()

        def _on_cancel(self) -> None:
            """Handle cancel (Escape key)."""
            if self._state == DiscoveryState.DISCOVERING:
                # Cancel discovery - parent will handle
                pass

        def _on_row_double_clicked(self, index: Any) -> None:
            """Handle double-click on row."""
            row = index.row()
            uid_item = self.results_table.item(row, 0)
            if uid_item:
                uid = uid_item.data(Qt.ItemDataRole.UserRole)
                if uid:
                    self.identify_clicked.emit(uid)

        def set_state(self, state: DiscoveryState) -> None:
            """Set UI state."""
            self._state = state

            if state == DiscoveryState.IDLE:
                self.discover_btn.setEnabled(True)
                self.universe_combo.setEnabled(True)
                self.progress_frame.hide()
                self.conflicts_frame.hide()
                self.identify_btn.setEnabled(False)
                self.apply_btn.setEnabled(False)
                self.summary_label.setText("Ready to discover fixtures")

            elif state == DiscoveryState.DISCOVERING:
                self.discover_btn.setEnabled(False)
                self.universe_combo.setEnabled(False)
                self.progress_frame.show()
                self.identify_btn.setEnabled(False)
                self.apply_btn.setEnabled(False)

            elif state == DiscoveryState.REVIEWING:
                self.discover_btn.setEnabled(True)
                self.universe_combo.setEnabled(True)
                self.progress_frame.hide()
                self.conflicts_frame.show()
                self.identify_btn.setEnabled(True)
                self.apply_btn.setEnabled(True)

            elif state == DiscoveryState.APPLYING:
                self.discover_btn.setEnabled(False)
                self.apply_btn.setEnabled(False)
                self.identify_btn.setEnabled(False)

        def update_progress(self, percent: int, message: str) -> None:
            """Update progress display."""
            self.progress_bar.setValue(percent)
            self.status_label.setText(message)

        def set_results(
            self,
            fixtures: Dict[str, Any],
            suggestions: Dict[str, Any]
        ) -> None:
            """Populate results table."""
            self._fixtures = fixtures
            self._suggestions = suggestions

            self.results_table.setRowCount(len(fixtures))

            conflict_count = 0
            for row, (uid, fixture) in enumerate(fixtures.items()):
                suggestion = suggestions.get(uid)

                # Fixture name column
                name_item = QTableWidgetItem(
                    fixture.model if hasattr(fixture, 'model') else str(uid)
                )
                name_item.setData(Qt.ItemDataRole.UserRole, uid)
                self.results_table.setItem(row, 0, name_item)

                # Manufacturer column
                mfg_item = QTableWidgetItem(
                    fixture.manufacturer if hasattr(fixture, 'manufacturer') else ""
                )
                self.results_table.setItem(row, 1, mfg_item)

                # Model column
                model_item = QTableWidgetItem(
                    fixture.model if hasattr(fixture, 'model') else ""
                )
                self.results_table.setItem(row, 2, model_item)

                # Current address column
                current_text = (
                    f"U{fixture.universe}:{fixture.start_address}-"
                    f"{fixture.start_address + fixture.channel_count - 1}"
                )
                current_item = QTableWidgetItem(current_text)
                self.results_table.setItem(row, 3, current_item)

                # Suggested column
                if suggestion:
                    if suggestion.requires_readdressing:
                        suggested_text = (
                            f"U{suggestion.suggested_universe}:"
                            f"{suggestion.suggested_start_address}"
                        )
                    else:
                        suggested_text = "Keep"
                    suggested_item = QTableWidgetItem(suggested_text)

                    # Color based on confidence
                    if suggestion.confidence >= 0.95:
                        suggested_item.setBackground(QColor("#ccffcc"))
                    elif suggestion.confidence >= 0.70:
                        suggested_item.setBackground(QColor("#ffffcc"))
                    else:
                        suggested_item.setBackground(QColor("#ffcccc"))
                else:
                    suggested_item = QTableWidgetItem("N/A")

                self.results_table.setItem(row, 4, suggested_item)

                # Personality column
                pers_text = fixture.personality_label if hasattr(
                    fixture, 'personality_label'
                ) else ""
                if suggestion and suggestion.personality_recommended:
                    pers_text = f"-> Mode {suggestion.personality_recommended}"
                pers_item = QTableWidgetItem(pers_text)
                self.results_table.setItem(row, 5, pers_item)

                # Conflicts column
                conflicts_text = "None"
                if hasattr(fixture, 'conflicts') and fixture.conflicts:
                    conflicts_text = ", ".join(fixture.conflicts[:2])
                    conflict_count += 1
                conflicts_item = QTableWidgetItem(conflicts_text)
                if conflicts_text != "None":
                    conflicts_item.setForeground(QColor("#c00"))
                self.results_table.setItem(row, 6, conflicts_item)

            # Update summary
            self.summary_label.setText(
                f"Found {len(fixtures)} fixtures, {conflict_count} conflicts"
            )

            # Update conflicts list in sidebar
            if conflict_count > 0:
                conflict_texts = []
                for uid, fixture in fixtures.items():
                    if hasattr(fixture, 'conflicts') and fixture.conflicts:
                        conflict_texts.append(
                            f"- {fixture.model} @ ch{fixture.start_address}"
                        )
                self.conflicts_list.setText("\n".join(conflict_texts[:5]))
            else:
                self.conflicts_list.setText("None")


    class RdmDiscoveryPanel(QWidget):
        """
        Main RDM Discovery Panel with dual mode support.

        Automatically switches between touchscreen and desktop layouts
        based on screen size. Provides responsive UI for both 7-inch
        touchscreens and desktop/laptop displays.

        Attributes:
            rdm_manager: RDM manager for discovery operations
            auto_patch_engine: Auto-patch engine for suggestions
            patch_manager: Patch manager for current state
        """

        def __init__(
            self,
            rdm_manager: Any,
            auto_patch_engine: Any,
            patch_manager: Any,
            parent: Optional[QWidget] = None
        ):
            """
            Initialize RDM Discovery Panel.

            Args:
                rdm_manager: RDM manager instance
                auto_patch_engine: Auto-patch engine instance
                patch_manager: Patch manager instance
                parent: Parent widget
            """
            super().__init__(parent)
            self.rdm_manager = rdm_manager
            self.auto_patch_engine = auto_patch_engine
            self.patch_manager = patch_manager
            self._state = DiscoveryState.IDLE
            self._worker: Optional[DiscoveryWorker] = None
            self._current_results: Dict[str, Any] = {}
            self._node_ip = "192.168.1.100"  # Default, can be configured

            self._setup_ui()
            self._connect_signals()
            self._detect_initial_mode()

        def _setup_ui(self) -> None:
            """Set up the dual-mode UI."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            # Stacked layout for mode switching
            self.stacked_widget = QWidget()
            self.stacked_layout = QStackedLayout(self.stacked_widget)

            # Create both layouts
            self.touchscreen_layout = TouchscreenLayout()
            self.desktop_layout = DesktopLayout()

            self.stacked_layout.addWidget(self.touchscreen_layout)
            self.stacked_layout.addWidget(self.desktop_layout)

            layout.addWidget(self.stacked_widget)

        def _connect_signals(self) -> None:
            """Connect layout signals."""
            # Touchscreen signals
            self.touchscreen_layout.discover_clicked.connect(
                self._on_discover
            )
            self.touchscreen_layout.identify_clicked.connect(
                self._on_identify
            )
            self.touchscreen_layout.apply_clicked.connect(
                self._on_apply
            )

            # Desktop signals
            self.desktop_layout.discover_clicked.connect(
                self._on_discover
            )
            self.desktop_layout.identify_clicked.connect(
                self._on_identify
            )
            self.desktop_layout.apply_clicked.connect(
                self._on_apply
            )
            self.desktop_layout.mode_switch_clicked.connect(
                self._switch_to_touchscreen
            )

        def _detect_initial_mode(self) -> None:
            """Detect initial mode based on screen size."""
            screen = self.screen()
            if screen:
                size = screen.size()
                if size.width() < TOUCHSCREEN_WIDTH_THRESHOLD:
                    self.stacked_layout.setCurrentIndex(0)  # Touchscreen
                else:
                    self.stacked_layout.setCurrentIndex(1)  # Desktop

        def resizeEvent(self, event: Any) -> None:
            """Handle resize - switch modes if needed."""
            super().resizeEvent(event)
            width = event.size().width()

            if width < TOUCHSCREEN_WIDTH_THRESHOLD:
                if self.stacked_layout.currentIndex() != 0:
                    self.stacked_layout.setCurrentIndex(0)
            else:
                if self.stacked_layout.currentIndex() != 1:
                    self.stacked_layout.setCurrentIndex(1)

        def _switch_to_touchscreen(self) -> None:
            """Switch to touchscreen mode."""
            self.stacked_layout.setCurrentIndex(0)

        def _switch_to_desktop(self) -> None:
            """Switch to desktop mode."""
            self.stacked_layout.setCurrentIndex(1)

        def set_node_ip(self, ip: str) -> None:
            """Set the ESP32 node IP address."""
            self._node_ip = ip

        def _on_discover(self, universes: List[int]) -> None:
            """Start discovery operation."""
            if self._worker and self._worker.isRunning():
                return

            self._set_state(DiscoveryState.DISCOVERING)

            self._worker = DiscoveryWorker(
                rdm_manager=self.rdm_manager,
                auto_patch_engine=self.auto_patch_engine,
                node_ip=self._node_ip,
                universes=universes,
                parent=self
            )
            self._worker.progress_update.connect(self._on_progress)
            self._worker.discovery_complete.connect(self._on_discovery_complete)
            self._worker.discovery_error.connect(self._on_discovery_error)
            self._worker.start()

        def _on_progress(self, percent: int, message: str) -> None:
            """Handle progress update."""
            self.touchscreen_layout.update_progress(percent, message)
            self.desktop_layout.update_progress(percent, message)

        def _on_discovery_complete(self, results: Dict[str, Any]) -> None:
            """Handle discovery completion."""
            self._current_results = results
            fixtures = results.get('fixtures', {})
            suggestions = results.get('suggestions', {})

            self.touchscreen_layout.set_results(fixtures, suggestions)
            self.desktop_layout.set_results(fixtures, suggestions)

            self._set_state(DiscoveryState.REVIEWING)

        def _on_discovery_error(self, error: str) -> None:
            """Handle discovery error."""
            QMessageBox.critical(
                self,
                "Discovery Error",
                f"Failed to discover fixtures:\n{error}"
            )
            self._set_state(DiscoveryState.IDLE)

        def _on_identify(self, uid: str) -> None:
            """Identify a specific fixture via consolidated RDMManager."""
            import threading

            def run_identify() -> None:
                try:
                    # Use consolidated RDMManager.identify_by_uid()
                    self.rdm_manager.identify_by_uid(uid, True)
                except Exception as e:
                    print(f"⚠️ RDM identify failed for {uid}: {e}", flush=True)

            thread = threading.Thread(target=run_identify)
            thread.start()

        def _on_apply(self) -> None:
            """Apply patch suggestions."""
            suggestions = self._current_results.get('suggestions', {})
            if not suggestions:
                return

            # Confirm
            result = QMessageBox.question(
                self,
                "Apply Patch",
                f"Apply patch to {len(suggestions)} fixtures?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if result != QMessageBox.StandardButton.Yes:
                return

            self._set_state(DiscoveryState.APPLYING)

            # Apply in background
            import asyncio

            async def do_apply() -> Dict[str, bool]:
                suggestion_list = list(suggestions.values())
                return await self.auto_patch_engine.apply_suggestions(
                    suggestion_list,
                    self.rdm_manager,
                    self._node_ip
                )

            def run_apply() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(do_apply())
                    success_count = sum(1 for v in results.values() if v)
                    total = len(results)

                    # Show result on main thread
                    QTimer.singleShot(
                        0,
                        lambda: self._show_apply_result(success_count, total)
                    )
                except Exception as e:
                    QTimer.singleShot(
                        0,
                        lambda: self._on_apply_error(str(e))
                    )
                finally:
                    loop.close()

            import threading
            thread = threading.Thread(target=run_apply)
            thread.start()

        def _show_apply_result(self, success: int, total: int) -> None:
            """Show apply result."""
            QMessageBox.information(
                self,
                "Patch Applied",
                f"Successfully applied {success}/{total} patches."
            )
            self._set_state(DiscoveryState.REVIEWING)

        def _on_apply_error(self, error: str) -> None:
            """Handle apply error."""
            QMessageBox.critical(
                self,
                "Apply Error",
                f"Failed to apply patch:\n{error}"
            )
            self._set_state(DiscoveryState.REVIEWING)

        def _set_state(self, state: DiscoveryState) -> None:
            """Set UI state on both layouts."""
            self._state = state
            self.touchscreen_layout.set_state(state)
            self.desktop_layout.set_state(state)

        def get_current_mode(self) -> str:
            """Get current display mode."""
            if self.stacked_layout.currentIndex() == 0:
                return "touchscreen"
            return "desktop"

else:
    # Stub classes when PySide6 is not available
    class TouchscreenLayout:  # type: ignore
        """Stub for TouchscreenLayout when PySide6 is not available."""
        pass

    class DesktopLayout:  # type: ignore
        """Stub for DesktopLayout when PySide6 is not available."""
        pass

    class RdmDiscoveryPanel:  # type: ignore
        """Stub for RdmDiscoveryPanel when PySide6 is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PySide6 is required for RdmDiscoveryPanel. "
                "Install it with: pip install PySide6"
            )
