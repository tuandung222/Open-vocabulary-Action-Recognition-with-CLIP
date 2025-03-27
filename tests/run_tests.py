#!/usr/bin/env python
"""
Test runner for the CLIP HAR Project.

This script discovers and runs all tests in the project.
"""

import os
import sys
import unittest
import argparse
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_type=None, verbose=False, failfast=False):
    """
    Run tests of the specified type.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', or None for all)
        verbose: Whether to run tests in verbose mode
        failfast: Whether to stop at first failure
    
    Returns:
        unittest.TestResult: The test result object
    """
    # Configure test loader
    loader = unittest.TestLoader()
    
    # Determine test directory
    if test_type == 'unit':
        start_dir = os.path.join(project_root, 'tests', 'unit_tests')
    elif test_type == 'integration':
        start_dir = os.path.join(project_root, 'tests', 'integration_tests')
    else:
        start_dir = os.path.join(project_root, 'tests')
    
    # Discover tests
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    
    # Print test info
    test_count = suite.countTestCases()
    print(f"Running {test_count} tests from {start_dir}")
    
    # Time the test run
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print(f"\nRan {result.testsRun} tests in {end_time - start_time:.2f}s")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("All tests passed!")
        return 0
    else:
        print("Tests failed!")
        return 1


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run CLIP HAR Project tests")
    parser.add_argument(
        "--type", choices=["unit", "integration", "all"], default="all",
        help="Type of tests to run (unit, integration, or all)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", 
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "-f", "--failfast", action="store_true", 
        help="Stop on first test failure"
    )
    
    args = parser.parse_args()
    test_type = None if args.type == 'all' else args.type
    
    return run_tests(test_type=test_type, verbose=args.verbose, failfast=args.failfast)


if __name__ == "__main__":
    sys.exit(main()) 