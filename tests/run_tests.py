#!/usr/bin/env python3
"""
Phase 8 Test Runner

Run all hardening tests:
  python run_tests.py

Run specific test file:
  python run_tests.py test_modifiers.py

Run with verbose output:
  python run_tests.py -v

Run specific test:
  python run_tests.py test_modifiers.py::TestModifierBounds::test_modifier_output_within_bounds
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Install with:")
        print("  pip install pytest pytest-timeout psutil")
        sys.exit(1)

    # Default args
    args = [
        "--tb=short",  # Short traceback format
        "-v",          # Verbose
        "--timeout=30",  # 30 second timeout per test
    ]

    # Add any command line args
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    else:
        # Run all tests in this directory
        args.append(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("AETHER Phase 8 - Hardening Tests")
    print("=" * 60)
    print()

    # Run pytest
    exit_code = pytest.main(args)

    print()
    print("=" * 60)
    if exit_code == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"TESTS FAILED (exit code: {exit_code})")
    print("=" * 60)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
