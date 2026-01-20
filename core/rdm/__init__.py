"""
AETHER RDM Module - Remote Device Management for DMX Fixtures

This module provides RDM (Remote Device Management) support for AETHER,
enabling automatic discovery, configuration, and patching of DMX fixtures.

Key Components:
- RdmManager: High-level facade for all RDM operations
- RdmTransport: UDP JSON communication with ESP32 nodes
- RdmDiscovery: Device discovery and information gathering
- AutoPatcher: Automatic fixture patching from RDM devices

Usage:
    from core.rdm import RdmManager, DiscoveredDevice, PatchSuggestion

    rdm = RdmManager(node_manager, fixture_library, db)
    await rdm.start_discovery(node_id)
    devices = rdm.get_devices()
    suggestion = rdm.get_patch_suggestion(device.uid, universe=1)

Transport:
    All RDM communication uses UDP JSON v2 protocol to ESP32 nodes.
    No OLA or UART - ESP32 nodes act as RDM gateways.

Version: 0.1.0
"""

from .types import (
    RdmUid,
    DiscoveredDevice,
    RdmPersonality,
    RdmParameter,
    RdmDeviceInfo,
    PatchSuggestion,
    DiscoveryStatus,
)

from .transport import RdmTransport, UdpJsonRdmTransport
from .discovery import RdmDiscovery, DiscoverySession
from .auto_patch import AutoPatcher, ProfileMatcher
from .manager import RdmManager

__all__ = [
    # Types
    "RdmUid",
    "DiscoveredDevice",
    "RdmPersonality",
    "RdmParameter",
    "RdmDeviceInfo",
    "PatchSuggestion",
    "DiscoveryStatus",
    # Transport
    "RdmTransport",
    "UdpJsonRdmTransport",
    # Discovery
    "RdmDiscovery",
    "DiscoverySession",
    # Auto-Patch
    "AutoPatcher",
    "ProfileMatcher",
    # Manager
    "RdmManager",
]

__version__ = "0.1.0"
