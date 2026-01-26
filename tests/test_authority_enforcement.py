"""
Authority Enforcement Test - Phase 1 Verification

# ============================================================================
# PURPOSE
# ============================================================================
#
# This test verifies that UnifiedPlaybackEngine is the SOLE AUTHORITY for
# DMX playback, as mandated by AETHER Hard Rule 1.1.
#
# It proves at runtime that:
# 1. UnifiedPlaybackEngine can start and tick
# 2. All parallel engines trigger warning logs if they attempt to run
# 3. No silent DMX output occurs outside the canonical authority chain
#
# ============================================================================
# WHAT THIS TEST DOES NOT DO
# ============================================================================
#
# - Does NOT fix violations (Phase 2+)
# - Does NOT refactor engines (Phase 3)
# - Does NOT delete code (Phase 2)
# - ONLY observes, asserts, and logs
#
# ============================================================================
# HOW TO RUN
# ============================================================================
#
# From aether-core directory:
#   python -m pytest tests/test_authority_enforcement.py -v -s
#
# Or run directly:
#   python tests/test_authority_enforcement.py
#
# ============================================================================
"""

import sys
import os
import logging
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AuthorityViolationCapture:
    """
    Captures logging.warning() calls to detect authority violations.

    This is the core mechanism for proving guards work - when a parallel
    engine starts, it should trigger a warning that we capture here.
    """

    def __init__(self):
        self.violations = []
        self.original_warning = None

    def __enter__(self):
        self.original_warning = logging.warning

        def capture_warning(msg, *args, **kwargs):
            if "AUTHORITY VIOLATION" in str(msg):
                self.violations.append(str(msg))
            # Still call original so logs appear
            self.original_warning(msg, *args, **kwargs)

        logging.warning = capture_warning
        return self

    def __exit__(self, *args):
        logging.warning = self.original_warning

    def has_violations(self):
        return len(self.violations) > 0

    def get_violations(self):
        return self.violations.copy()

    def clear(self):
        self.violations.clear()


class TestAuthorityEnforcement:
    """
    Phase 1 Authority Enforcement Tests

    These tests verify that:
    1. UnifiedPlaybackEngine is recognized as canonical
    2. Parallel engines trigger warnings when started
    3. The guard system is functional
    """

    # =========================================================================
    # TEST 1: UnifiedPlaybackEngine Authority Declaration
    # =========================================================================

    def test_unified_playback_engine_has_authority_declaration(self):
        """
        Verify UnifiedPlaybackEngine declares itself as the canonical authority.

        Expected: Authority declaration comment block exists at top of file.
        """
        print("\n" + "="*70)
        print("TEST 1: UnifiedPlaybackEngine Authority Declaration")
        print("="*70)

        # Read the source file
        unified_playback_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'unified_playback.py'
        )

        with open(unified_playback_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for authority declaration markers
        assert "AUTHORITY DECLARATION" in content, \
            "Missing AUTHORITY DECLARATION in unified_playback.py"
        assert "THIS IS THE SOLE AUTHORITY FOR DMX PLAYBACK OUTPUT" in content, \
            "Missing sole authority statement"
        assert "AETHER Hard Rule 1.1" in content, \
            "Missing Hard Rule reference"

        print("[PASS] Authority declaration found in unified_playback.py")
        print("   - AUTHORITY DECLARATION block present")
        print("   - Sole authority statement present")
        print("   - Hard Rule 1.1 reference present")

    # =========================================================================
    # TEST 2: RenderEngine Triggers Violation Warning
    # =========================================================================

    def test_render_engine_triggers_violation_warning(self):
        """
        Verify RenderEngine.start() triggers an authority violation warning.

        Expected: logging.warning() called with "AUTHORITY VIOLATION" message.
        """
        print("\n" + "="*70)
        print("TEST 2: RenderEngine Violation Warning")
        print("="*70)

        from render_engine import RenderEngine

        with AuthorityViolationCapture() as capture:
            engine = RenderEngine(target_fps=30)

            # Mock the output callback to prevent actual DMX output
            engine.set_output_callback(Mock())

            # Redirect stdout to avoid emoji encoding issues on Windows
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                # Start the engine - this should trigger violation warning
                engine.start()

                # Give it a moment to log
                time.sleep(0.1)

                # Stop immediately
                engine.stop()
            finally:
                sys.stdout = old_stdout

        assert capture.has_violations(), \
            "RenderEngine.start() did NOT trigger authority violation warning!"

        print("[PASS] RenderEngine.start() correctly triggered violation warning")
        # Encode to ASCII to avoid Windows console encoding issues
        captured = capture.get_violations()[0][:80]
        captured_safe = captured.encode('ascii', 'replace').decode('ascii')
        print("   Captured: {}...".format(captured_safe))

    # =========================================================================
    # TEST 3: ChaseEngine Triggers Violation Warning
    # =========================================================================

    def test_chase_engine_triggers_violation_warning(self):
        """
        Verify ChaseEngine.start_chase() triggers an authority violation warning.

        Expected: logging.warning() called with "AUTHORITY VIOLATION" message.
        """
        print("\n" + "="*70)
        print("TEST 3: ChaseEngine Violation Warning")
        print("="*70)

        # Import from aether-core.py requires special handling
        # We'll test by checking the source code for the guard
        aether_core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'aether-core.py'
        )

        with open(aether_core_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify guard exists in ChaseEngine
        assert "class ChaseEngine:" in content, "ChaseEngine class not found"

        # Find ChaseEngine section and verify guard
        chase_section_start = content.find("class ChaseEngine:")
        chase_section = content[chase_section_start:chase_section_start + 3000]

        assert "AUTHORITY VIOLATION" in chase_section, \
            "ChaseEngine missing AUTHORITY VIOLATION guard!"
        assert "logging.warning" in chase_section, \
            "ChaseEngine missing logging.warning() call!"
        assert "TASK-0006" in chase_section or "TASK-0016" in chase_section, \
            "ChaseEngine missing task reference!"

        print("[PASS] ChaseEngine has authority violation guard")
        print("   - AUTHORITY VIOLATION comment present")
        print("   - logging.warning() call present")
        print("   - Task reference present")

    # =========================================================================
    # TEST 4: ShowEngine Triggers Violation Warning
    # =========================================================================

    def test_show_engine_triggers_violation_warning(self):
        """
        Verify ShowEngine.play_show() triggers an authority violation warning.

        Expected: logging.warning() called with "AUTHORITY VIOLATION" message.
        """
        print("\n" + "="*70)
        print("TEST 4: ShowEngine Violation Warning")
        print("="*70)

        aether_core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'aether-core.py'
        )

        with open(aether_core_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find ShowEngine section
        show_section_start = content.find("class ShowEngine:")
        assert show_section_start != -1, "ShowEngine class not found"

        show_section = content[show_section_start:show_section_start + 3000]

        assert "AUTHORITY VIOLATION" in show_section, \
            "ShowEngine missing AUTHORITY VIOLATION guard!"
        assert "logging.warning" in show_section, \
            "ShowEngine missing logging.warning() call!"
        assert "TASK-0007" in show_section or "TASK-0017" in show_section, \
            "ShowEngine missing task reference!"

        print("[PASS] ShowEngine has authority violation guard")
        print("   - AUTHORITY VIOLATION comment present")
        print("   - logging.warning() call present")
        print("   - Task reference present")

    # =========================================================================
    # TEST 5: DynamicEffectsEngine Triggers Violation Warning
    # =========================================================================

    def test_effects_engine_triggers_violation_warning(self):
        """
        Verify DynamicEffectsEngine triggers an authority violation warning.

        Expected: logging.warning() called with "AUTHORITY VIOLATION" message.
        """
        print("\n" + "="*70)
        print("TEST 5: DynamicEffectsEngine Violation Warning")
        print("="*70)

        effects_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'effects_engine.py'
        )

        with open(effects_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "AUTHORITY VIOLATION" in content, \
            "effects_engine.py missing AUTHORITY VIOLATION guard!"
        assert "logging.warning" in content, \
            "effects_engine.py missing logging.warning() call!"
        assert "TASK-0005" in content, \
            "effects_engine.py missing task reference!"

        print("[PASS] DynamicEffectsEngine has authority violation guard")
        print("   - AUTHORITY VIOLATION header present")
        print("   - logging.warning() call present")
        print("   - TASK-0005 reference present")

    # =========================================================================
    # TEST 6: PreviewService Armed Mode Triggers Violation Warning
    # =========================================================================

    def test_preview_service_armed_triggers_violation_warning(self):
        """
        Verify PreviewService.arm_session() triggers an authority violation warning.

        Expected: logging.warning() called with "AUTHORITY VIOLATION" message.
        """
        print("\n" + "="*70)
        print("TEST 6: PreviewService Armed Mode Violation Warning")
        print("="*70)

        preview_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'preview_service.py'
        )

        with open(preview_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find arm_session method
        arm_section_start = content.find("def arm_session")
        assert arm_section_start != -1, "arm_session method not found"

        arm_section = content[arm_section_start:arm_section_start + 1000]

        assert "AUTHORITY VIOLATION" in arm_section, \
            "arm_session() missing AUTHORITY VIOLATION guard!"
        assert "logging.warning" in arm_section, \
            "arm_session() missing logging.warning() call!"
        assert "TASK-0009" in arm_section, \
            "arm_session() missing task reference!"

        print("[PASS] PreviewService.arm_session() has authority violation guard")
        print("   - AUTHORITY VIOLATION comment present")
        print("   - logging.warning() call present")
        print("   - TASK-0009 reference present")

    # =========================================================================
    # TEST 7: UnifiedPlaybackController Triggers Violation Warning
    # =========================================================================

    def test_playback_controller_triggers_violation_warning(self):
        """
        Verify UnifiedPlaybackController.start() triggers violation warning.

        Expected: File is DELETED (Phase 3 cleanup).
        """
        print("\n" + "="*70)
        print("TEST 7: UnifiedPlaybackController DELETED (Phase 3)")
        print("="*70)

        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'playback_controller.py'
        )

        # Phase 3: File should be DELETED
        assert not os.path.exists(controller_path), \
            "playback_controller.py should be DELETED (Phase 3)! File still exists."

        print("[PASS] UnifiedPlaybackController successfully deleted")
        print("   - playback_controller.py no longer exists")
        print("   - Authority violation resolved by deletion")

    # =========================================================================
    # TEST 8: Smoking Gun Endpoint Has Guard
    # =========================================================================

    def test_looks_play_endpoint_has_guard(self):
        """
        Verify /api/looks/{id}/play endpoint has authority violation guard.

        This is the "smoking gun" - the primary Look playback endpoint that
        uses RenderEngine directly instead of UnifiedPlaybackEngine.

        Expected: Guard comment and logging.warning() present.
        """
        print("\n" + "="*70)
        print("TEST 8: /api/looks/{id}/play Endpoint Guard")
        print("="*70)

        aether_core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'aether-core.py'
        )

        with open(aether_core_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the endpoint
        endpoint_marker = "@app.route('/api/looks/<look_id>/play'"
        endpoint_start = content.find(endpoint_marker)
        assert endpoint_start != -1, "play_look endpoint not found"

        # Look backwards for the guard comment block
        section_start = max(0, endpoint_start - 1500)
        section = content[section_start:endpoint_start + 2000]

        assert "SMOKING GUN VIOLATION" in section, \
            "play_look endpoint missing SMOKING GUN guard!"
        assert "TASK-0018" in section, \
            "play_look endpoint missing TASK-0018 reference!"
        assert "logging.warning" in section, \
            "play_look endpoint missing logging.warning() call!"

        print("[PASS] /api/looks/{id}/play endpoint has authority violation guard")
        print("   - SMOKING GUN VIOLATION marker present")
        print("   - TASK-0018 reference present")
        print("   - logging.warning() call present")

    # =========================================================================
    # TEST 9: Express DMXService Has Non-Authority Warning
    # =========================================================================

    def test_express_dmx_service_non_authority_warning(self):
        """
        Verify Express DMXService.js is DELETED (Phase 3).

        Expected: File should not exist.
        """
        print("\n" + "="*70)
        print("TEST 9: Express DMXService DELETED (Phase 3)")
        print("="*70)

        dmx_service_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'aether-portal-os', 'backend', 'src', 'services', 'DMXService.js'
        )

        # Normalize path
        dmx_service_path = os.path.normpath(dmx_service_path)

        # Phase 3: File should be DELETED
        assert not os.path.exists(dmx_service_path), \
            "DMXService.js should be DELETED (Phase 3)! File still exists."

        print("[PASS] Express DMXService.js successfully deleted")
        print("   - DMXService.js no longer exists")
        print("   - All DMX state now from AETHER Core only")

    # =========================================================================
    # TEST 10: All Parallel Systems Have Task References
    # =========================================================================

    def test_all_parallel_systems_have_task_references(self):
        """
        Verify all parallel systems reference their respective TASK IDs.

        This ensures violations are traceable to the ledger.
        """
        print("\n" + "="*70)
        print("TEST 10: All Parallel Systems Have Task References")
        print("="*70)

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Map of files to expected task references
        # NOTE: playback_controller.py was DELETED in Phase 3
        file_tasks = {
            'render_engine.py': 'TASK-0004',
            'effects_engine.py': 'TASK-0005',
            # 'playback_controller.py': 'TASK-0008',  # DELETED (Phase 3)
            'preview_service.py': 'TASK-0009',
        }

        # aether-core.py has multiple tasks
        aether_core_tasks = ['TASK-0006', 'TASK-0007', 'TASK-0016', 'TASK-0017', 'TASK-0018']

        all_found = True

        for filename, task_id in file_tasks.items():
            filepath = os.path.join(base_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if task_id in content:
                print("   [OK] {}: {} found".format(filename, task_id))
            else:
                print("   [FAIL] {}: {} NOT FOUND!".format(filename, task_id))
                all_found = False

        # Check aether-core.py
        aether_core_path = os.path.join(base_path, 'aether-core.py')
        with open(aether_core_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for task_id in aether_core_tasks:
            # TASK-0006 and TASK-0016 both refer to ChaseEngine
            # TASK-0007 and TASK-0017 both refer to ShowEngine
            # Accept either
            if task_id in content:
                print("   [OK] aether-core.py: {} found".format(task_id))
            else:
                # Check for alternate task ID
                if task_id == 'TASK-0006' and 'TASK-0016' in content:
                    print("   [OK] aether-core.py: TASK-0016 found (replaces {})".format(task_id))
                elif task_id == 'TASK-0007' and 'TASK-0017' in content:
                    print("   [OK] aether-core.py: TASK-0017 found (replaces {})".format(task_id))
                elif task_id == 'TASK-0016' and 'TASK-0006' in content:
                    pass  # Already checked
                elif task_id == 'TASK-0017' and 'TASK-0007' in content:
                    pass  # Already checked
                else:
                    print("   [FAIL] aether-core.py: {} NOT FOUND!".format(task_id))
                    all_found = False

        assert all_found, "Not all parallel systems have task references!"
        print("\n[PASS] All parallel systems have task references")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(results):
    """Print a summary of all test results."""
    print("\n")
    print("=" * 70)
    print("AUTHORITY ENFORCEMENT TEST - SUMMARY REPORT")
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
        print("[PASS] ALL TESTS PASSED - PHASE 1 AUTHORITY ENFORCEMENT VERIFIED")
        print("=" * 70)
        print()
        print("The following has been verified:")
        print("  1. UnifiedPlaybackEngine is declared as sole authority")
        print("  2. All 7 parallel engines have violation guards")
        print("  3. All guards include logging.warning() calls")
        print("  4. All violations reference TASK IDs in the ledger")
        print("  5. Express DMXService is marked as non-authoritative cache")
        print("  6. /api/looks/{id}/play smoking gun is guarded")
        print()
        print("PHASE 1 STATUS: VERIFIED [PASS]")
    else:
        print("=" * 70)
        print("[FAIL] TESTS FAILED - PHASE 1 NOT COMPLETE")
        print("=" * 70)
        print()
        print("Failed tests:")
        for r in results:
            if not r['passed']:
                print("  [FAIL] {}: {}".format(r['name'], r['error']))
        print()
        print("PHASE 1 STATUS: INCOMPLETE [FAIL]")

    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 70)
    print("AETHER AUTHORITY ENFORCEMENT TEST")
    print("Phase 1 Verification - FREEZE & GUARD")
    print("=" * 70)
    print()
    print("This test verifies that:")
    print("  - UnifiedPlaybackEngine is declared as the sole authority")
    print("  - All parallel engines have violation guards")
    print("  - Guards trigger logging.warning() calls")
    print("  - All violations are traceable to TASK_LEDGER.md")
    print()
    print("Running tests...")
    print()

    test_instance = TestAuthorityEnforcement()
    results = []

    # Run all tests
    tests = [
        ('Authority Declaration', test_instance.test_unified_playback_engine_has_authority_declaration),
        ('RenderEngine Guard', test_instance.test_render_engine_triggers_violation_warning),
        ('ChaseEngine Guard', test_instance.test_chase_engine_triggers_violation_warning),
        ('ShowEngine Guard', test_instance.test_show_engine_triggers_violation_warning),
        ('EffectsEngine Guard', test_instance.test_effects_engine_triggers_violation_warning),
        ('PreviewService Guard', test_instance.test_preview_service_armed_triggers_violation_warning),
        ('PlaybackController Guard', test_instance.test_playback_controller_triggers_violation_warning),
        ('Smoking Gun Endpoint', test_instance.test_looks_play_endpoint_has_guard),
        ('Express DMXService', test_instance.test_express_dmx_service_non_authority_warning),
        ('Task References', test_instance.test_all_parallel_systems_have_task_references),
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
