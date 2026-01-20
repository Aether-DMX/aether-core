"""
AETHER UI Patch Module

Contains UI components for RDM discovery and auto-patching.
"""

from .rdm_discovery_panel import (
    DiscoveryState,
    DiscoveryWorker,
    RdmDiscoveryPanel,
    TOUCHSCREEN_WIDTH_THRESHOLD,
    HAS_PYSIDE6,
)

# Only export PySide6-dependent classes if available
if HAS_PYSIDE6:
    from .rdm_discovery_panel import (
        TouchscreenLayout,
        DesktopLayout,
    )

__all__ = [
    "DiscoveryState",
    "DiscoveryWorker",
    "RdmDiscoveryPanel",
    "TouchscreenLayout",
    "DesktopLayout",
    "TOUCHSCREEN_WIDTH_THRESHOLD",
    "HAS_PYSIDE6",
]

__version__ = "0.1.0"
