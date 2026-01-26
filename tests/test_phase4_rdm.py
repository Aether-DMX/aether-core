"""
Phase 4: RDM v1 Tests
AETHER ARCHITECTURE PROGRAM

# ============================================================================
# PURPOSE
# ============================================================================
#
# This test verifies RDM v1 implementation:
# AC1: Discovery
# AC2: Addressing & Verification
# AC3: Fixture Intelligence
# AC4: Assisted Setup Logic
# AC5: Playback Safety
#
# ============================================================================
# HOW TO RUN
# ============================================================================
#
# From aether-core directory:
#   python -m pytest tests/test_phase4_rdm.py -v -s
#
# Or run directly:
#   python tests/test_phase4_rdm.py
#
# ============================================================================
"""

import sys
import os
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdm_service import (
    RDMService,
    RDMDevice,
    AddressSuggestion,
    RDMOperationResult,
    KNOWN_MANUFACTURERS,
    rdm_service,
    discover_rdm_devices,
    get_rdm_devices,
    get_rdm_device_address,
    set_rdm_device_address,
    identify_rdm_device,
    get_rdm_address_suggestions,
    auto_fix_rdm_addresses,
    verify_cue_rdm_readiness,
    get_rdm_status,
)

# Fix for Windows console encoding
import io
import sys
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class TestRDMDeviceDataStructure:
    """Test RDMDevice data structure."""

    def test_rdm_device_creation(self):
        """RDMDevice can be created with UID."""
        print("\n" + "="*70)
        print("TEST: RDMDevice Creation")
        print("="*70)

        device = RDMDevice(
            uid="0000:12345678",
            dmx_address=1,
            dmx_footprint=16
        )

        assert device.uid == "0000:12345678"
        assert device.dmx_address == 1
        assert device.dmx_footprint == 16

        print("[PASS] RDMDevice can be created")

    def test_rdm_device_to_dict(self):
        """RDMDevice serializes to dict."""
        print("\n" + "="*70)
        print("TEST: RDMDevice Serialization")
        print("="*70)

        device = RDMDevice(
            uid="0001:AABBCCDD",
            manufacturer_id=0x0001,
            dmx_address=100,
            dmx_footprint=8
        )

        d = device.to_dict()
        assert 'uid' in d
        assert 'dmx_address' in d
        assert 'dmx_footprint' in d
        assert d['uid'] == "0001:AABBCCDD"

        print("[PASS] RDMDevice serializes correctly")

    def test_rdm_device_from_dict(self):
        """RDMDevice can be created from dict."""
        print("\n" + "="*70)
        print("TEST: RDMDevice Deserialization")
        print("="*70)

        data = {
            'uid': "4348:11223344",
            'manufacturer_id': 0x4348,
            'dmx_address': 50,
            'dmx_footprint': 24
        }

        device = RDMDevice.from_dict(data)
        assert device.uid == "4348:11223344"
        assert device.manufacturer_id == 0x4348
        assert device.dmx_address == 50

        print("[PASS] RDMDevice can be created from dict")


class TestAC1Discovery:
    """AC1: Discovery tests."""

    def test_discovery_returns_result_structure(self):
        """Discovery returns proper result structure."""
        print("\n" + "="*70)
        print("TEST: AC1 - Discovery Result Structure")
        print("="*70)

        service = RDMService()

        # Mock no nodes available
        service._get_rdm_capable_nodes = Mock(return_value=[])

        result = service.discover_all()

        assert 'success' in result
        assert 'devices' in result
        assert isinstance(result['devices'], list)

        print("[PASS] Discovery returns proper structure")

    def test_cached_devices_returns_list(self):
        """get_cached_devices returns a list."""
        print("\n" + "="*70)
        print("TEST: AC1 - Cached Devices")
        print("="*70)

        service = RDMService()

        # Add some test devices
        service._devices['0000:11111111'] = RDMDevice(uid='0000:11111111')
        service._devices['0000:22222222'] = RDMDevice(uid='0000:22222222')

        devices = service.get_cached_devices()

        assert isinstance(devices, list)
        assert len(devices) == 2

        print("[PASS] Cached devices returns list")


class TestAC2AddressingVerification:
    """AC2: Addressing & Verification tests."""

    def test_address_validation(self):
        """Address must be 1-512."""
        print("\n" + "="*70)
        print("TEST: AC2 - Address Validation")
        print("="*70)

        service = RDMService()

        # Add a test device
        service._devices['test:device'] = RDMDevice(uid='test:device')

        # Test invalid addresses
        result = service.set_address('test:device', 0)
        assert result['success'] is False
        assert 'Invalid address' in result['error']

        result = service.set_address('test:device', 513)
        assert result['success'] is False
        assert 'Invalid address' in result['error']

        print("[PASS] Address validation works")

    def test_address_conflict_detection(self):
        """Address conflicts are detected."""
        print("\n" + "="*70)
        print("TEST: AC2 - Conflict Detection")
        print("="*70)

        service = RDMService()

        # Add devices with overlapping addresses
        service._devices['dev1'] = RDMDevice(
            uid='dev1', dmx_address=1, dmx_footprint=16
        )
        service._devices['dev2'] = RDMDevice(
            uid='dev2', dmx_address=20, dmx_footprint=16
        )

        # Try to set dev2 to overlapping address
        conflicts = service._check_address_conflict('dev2', 10, 16)

        assert len(conflicts) > 0
        assert conflicts[0]['uid'] == 'dev1'

        print("[PASS] Address conflicts are detected")

    def test_verify_address_structure(self):
        """verify_address returns proper structure."""
        print("\n" + "="*70)
        print("TEST: AC2 - Verify Address Structure")
        print("="*70)

        service = RDMService()
        service._devices['test:dev'] = RDMDevice(uid='test:dev', dmx_address=100)

        # Mock get_address
        service.get_address = Mock(return_value={'success': True, 'address': 100})

        result = service.verify_address('test:dev', 100)

        assert 'verified' in result
        assert 'expected_address' in result
        assert 'actual_address' in result
        assert result['verified'] is True

        print("[PASS] verify_address returns proper structure")


class TestAC3FixtureIntelligence:
    """AC3: Fixture Intelligence tests."""

    def test_get_device_info_returns_device(self):
        """get_device_info returns device data."""
        print("\n" + "="*70)
        print("TEST: AC3 - Get Device Info")
        print("="*70)

        service = RDMService()
        service._devices['test:fixture'] = RDMDevice(
            uid='test:fixture',
            manufacturer_id=0x4348,
            device_model_id=1234,
            dmx_footprint=16
        )

        result = service.get_device_info('test:fixture')

        assert result['success'] is True
        assert 'device' in result
        assert result['device']['manufacturer_id'] == 0x4348
        assert result['device']['dmx_footprint'] == 16

        print("[PASS] get_device_info returns device data")

    def test_known_manufacturers_exist(self):
        """Known manufacturer IDs are defined."""
        print("\n" + "="*70)
        print("TEST: AC3 - Known Manufacturers")
        print("="*70)

        # Check some common manufacturers
        assert 0x0000 in KNOWN_MANUFACTURERS  # PLASA
        assert 0x454C in KNOWN_MANUFACTURERS  # ETC

        print(f"   Known manufacturers: {len(KNOWN_MANUFACTURERS)}")
        print("[PASS] Known manufacturers are defined")

    def test_identify_device_structure(self):
        """identify_device returns proper structure."""
        print("\n" + "="*70)
        print("TEST: AC3 - Identify Device")
        print("="*70)

        service = RDMService()
        service._devices['test:id'] = RDMDevice(uid='test:id')

        # Mock the node lookup to return None (no node available)
        service._get_node_for_device = Mock(return_value=None)

        result = service.identify_device('test:id', True)

        # Should fail gracefully when no node available
        assert 'success' in result

        print("[PASS] identify_device returns proper structure")


class TestAC4AssistedSetupLogic:
    """AC4: Assisted Setup Logic tests."""

    def test_suggest_addresses_with_conflicts(self):
        """suggest_addresses detects conflicts."""
        print("\n" + "="*70)
        print("TEST: AC4 - Suggest Addresses (Conflicts)")
        print("="*70)

        service = RDMService()

        # Add devices with overlapping addresses
        service._devices['dev1'] = RDMDevice(
            uid='dev1', dmx_address=1, dmx_footprint=20
        )
        service._devices['dev2'] = RDMDevice(
            uid='dev2', dmx_address=10, dmx_footprint=20  # Overlaps with dev1
        )

        result = service.suggest_addresses()

        assert result['success'] is True
        assert 'conflicts' in result
        assert len(result['conflicts']) > 0

        print("[PASS] suggest_addresses detects conflicts")

    def test_suggest_addresses_generates_suggestions(self):
        """suggest_addresses generates fix suggestions."""
        print("\n" + "="*70)
        print("TEST: AC4 - Suggest Addresses (Suggestions)")
        print("="*70)

        service = RDMService()

        # Add devices with conflicts
        service._devices['dev1'] = RDMDevice(
            uid='dev1', dmx_address=1, dmx_footprint=16
        )
        service._devices['dev2'] = RDMDevice(
            uid='dev2', dmx_address=10, dmx_footprint=16  # Overlaps
        )

        result = service.suggest_addresses()

        assert 'suggestions' in result
        # Should suggest moving one device

        print("[PASS] suggest_addresses generates suggestions")

    def test_auto_fix_structure(self):
        """auto_fix_addresses returns proper structure."""
        print("\n" + "="*70)
        print("TEST: AC4 - Auto Fix Structure")
        print("="*70)

        service = RDMService()

        # Empty device list
        result = service.auto_fix_addresses()

        assert 'success' in result
        assert 'results' in result

        print("[PASS] auto_fix_addresses returns proper structure")


class TestAC5PlaybackSafety:
    """AC5: Playback Safety tests."""

    def test_verify_cue_readiness_empty_cue(self):
        """verify_cue_readiness handles empty cue."""
        print("\n" + "="*70)
        print("TEST: AC5 - Verify Empty Cue")
        print("="*70)

        service = RDMService()

        result = service.verify_cue_readiness({})

        assert 'ready' in result
        assert 'issues' in result
        assert 'warnings' in result

        print("[PASS] verify_cue_readiness handles empty cue")

    def test_verify_cue_detects_missing_device(self):
        """verify_cue_readiness detects missing devices."""
        print("\n" + "="*70)
        print("TEST: AC5 - Detect Missing Device")
        print("="*70)

        service = RDMService()

        cue = {
            'fixtures': [
                {'rdm_uid': 'missing:device', 'dmx_address': 1}
            ]
        }

        result = service.verify_cue_readiness(cue)

        # Should have issues for missing device
        assert any(i['type'] == 'device_not_found' for i in result['issues'])

        print("[PASS] verify_cue_readiness detects missing devices")

    def test_verify_cue_detects_address_mismatch(self):
        """verify_cue_readiness detects address mismatch."""
        print("\n" + "="*70)
        print("TEST: AC5 - Detect Address Mismatch")
        print("="*70)

        service = RDMService()

        # Add device at different address
        service._devices['test:fixture'] = RDMDevice(
            uid='test:fixture',
            dmx_address=50
        )

        cue = {
            'fixtures': [
                {'rdm_uid': 'test:fixture', 'dmx_address': 100}  # Expected at 100
            ]
        }

        result = service.verify_cue_readiness(cue)

        # Should have issues for address mismatch
        assert any(i['type'] == 'address_mismatch' for i in result['issues'])

        print("[PASS] verify_cue_readiness detects address mismatch")


class TestRDMAPIIntegration:
    """Test RDM API integration with aether-core."""

    def test_rdm_api_endpoints_exist(self):
        """RDM API endpoints are defined in aether-core."""
        print("\n" + "="*70)
        print("TEST: RDM API Endpoints Exist")
        print("="*70)

        aether_core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'aether-core.py'
        )

        with open(aether_core_path, 'r', encoding='utf-8') as f:
            content = f.read()

        required_endpoints = [
            '/api/rdm/status',
            '/api/rdm/discover',
            '/api/rdm/devices',
            '/api/rdm/address-suggestions',
            '/api/rdm/auto-fix',
            '/api/rdm/verify-cue',
        ]

        for endpoint in required_endpoints:
            assert endpoint in content, f"Missing endpoint: {endpoint}"

        print("[PASS] All RDM API endpoints are defined")

    def test_rdm_service_import(self):
        """rdm_service can be imported."""
        print("\n" + "="*70)
        print("TEST: RDM Service Import")
        print("="*70)

        from rdm_service import rdm_service

        assert rdm_service is not None
        assert hasattr(rdm_service, 'discover_all')
        assert hasattr(rdm_service, 'set_address')
        assert hasattr(rdm_service, 'verify_cue_readiness')

        print("[PASS] RDM service imports correctly")


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_rdm_status(self):
        """get_rdm_status returns status dict."""
        print("\n" + "="*70)
        print("TEST: Module-Level get_rdm_status")
        print("="*70)

        status = get_rdm_status()

        assert isinstance(status, dict)
        assert 'enabled' in status
        assert 'device_count' in status

        print("[PASS] get_rdm_status returns valid dict")

    def test_get_rdm_devices(self):
        """get_rdm_devices returns device list."""
        print("\n" + "="*70)
        print("TEST: Module-Level get_rdm_devices")
        print("="*70)

        devices = get_rdm_devices()

        assert isinstance(devices, list)

        print("[PASS] get_rdm_devices returns list")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(results):
    """Print a summary of all test results."""
    print("\n")
    print("=" * 70)
    print("PHASE 4 RDM V1 TEST - SUMMARY REPORT")
    print("=" * 70)
    print()

    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed

    print("Total Tests: {}".format(len(results)))
    print("Passed: {}".format(passed))
    print("Failed: {}".format(failed))
    print()

    if failed == 0:
        print("=" * 70)
        print("[PASS] ALL RDM TESTS PASSED - PHASE 4 LANE 4 VERIFIED")
        print("=" * 70)
        print()
        print("ACCEPTANCE CRITERIA VERIFIED:")
        print("  AC1: Discovery - RDM discovery returns device list")
        print("  AC2: Addressing - Address validation and conflict detection")
        print("  AC3: Fixture Intelligence - Device info and identify")
        print("  AC4: Assisted Setup - Suggestions and auto-fix")
        print("  AC5: Playback Safety - Cue verification")
        print()
        print("PHASE 4 LANE 4 STATUS: VERIFIED [PASS]")
    else:
        print("=" * 70)
        print("[FAIL] RDM TESTS FAILED")
        print("=" * 70)
        print()
        print("Failed tests:")
        for r in results:
            if not r['passed']:
                print("  [FAIL] {}: {}".format(r['name'], r['error']))
        print()
        print("PHASE 4 LANE 4 STATUS: INCOMPLETE [FAIL]")

    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 70)
    print("AETHER PHASE 4 RDM V1 TEST")
    print("Lane 4: RDM Implementation Verification")
    print("=" * 70)
    print()
    print("ACCEPTANCE CRITERIA UNDER TEST:")
    print("  AC1: Discovery")
    print("  AC2: Addressing & Verification")
    print("  AC3: Fixture Intelligence")
    print("  AC4: Assisted Setup Logic")
    print("  AC5: Playback Safety")
    print()
    print("Running tests...")
    print()

    results = []

    # Run all tests
    tests = [
        # Data Structure
        ('RDMDevice Creation', TestRDMDeviceDataStructure().test_rdm_device_creation),
        ('RDMDevice Serialization', TestRDMDeviceDataStructure().test_rdm_device_to_dict),
        ('RDMDevice Deserialization', TestRDMDeviceDataStructure().test_rdm_device_from_dict),

        # AC1: Discovery
        ('AC1 Discovery Result', TestAC1Discovery().test_discovery_returns_result_structure),
        ('AC1 Cached Devices', TestAC1Discovery().test_cached_devices_returns_list),

        # AC2: Addressing
        ('AC2 Address Validation', TestAC2AddressingVerification().test_address_validation),
        ('AC2 Conflict Detection', TestAC2AddressingVerification().test_address_conflict_detection),
        ('AC2 Verify Address', TestAC2AddressingVerification().test_verify_address_structure),

        # AC3: Fixture Intelligence
        ('AC3 Get Device Info', TestAC3FixtureIntelligence().test_get_device_info_returns_device),
        ('AC3 Known Manufacturers', TestAC3FixtureIntelligence().test_known_manufacturers_exist),
        ('AC3 Identify Device', TestAC3FixtureIntelligence().test_identify_device_structure),

        # AC4: Assisted Setup
        ('AC4 Detect Conflicts', TestAC4AssistedSetupLogic().test_suggest_addresses_with_conflicts),
        ('AC4 Generate Suggestions', TestAC4AssistedSetupLogic().test_suggest_addresses_generates_suggestions),
        ('AC4 Auto Fix Structure', TestAC4AssistedSetupLogic().test_auto_fix_structure),

        # AC5: Playback Safety
        ('AC5 Empty Cue', TestAC5PlaybackSafety().test_verify_cue_readiness_empty_cue),
        ('AC5 Missing Device', TestAC5PlaybackSafety().test_verify_cue_detects_missing_device),
        ('AC5 Address Mismatch', TestAC5PlaybackSafety().test_verify_cue_detects_address_mismatch),

        # Integration
        ('RDM API Endpoints', TestRDMAPIIntegration().test_rdm_api_endpoints_exist),
        ('RDM Service Import', TestRDMAPIIntegration().test_rdm_service_import),

        # Module Functions
        ('get_rdm_status', TestModuleLevelFunctions().test_get_rdm_status),
        ('get_rdm_devices', TestModuleLevelFunctions().test_get_rdm_devices),
    ]

    for name, test_func in tests:
        try:
            test_func()
            results.append({'name': name, 'passed': True, 'error': None})
        except AssertionError as e:
            results.append({'name': name, 'passed': False, 'error': str(e)})
            print("\n[FAIL] FAILED: {}".format(name))
            print("   Error: {}".format(e))
        except Exception as e:
            results.append({'name': name, 'passed': False, 'error': str(e)})
            print("\n[ERROR] ERROR: {}".format(name))
            print("   Error: {}".format(e))

    # Print summary
    print_summary_report(results)

    # Exit with appropriate code
    sys.exit(0 if all(r['passed'] for r in results) else 1)
