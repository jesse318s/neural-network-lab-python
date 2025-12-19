"""
Test Runner for Neural Network Lab

Run this file to execute all tests in the test suite.
"""

import sys
import unittest


def main():
    """Main test runner entry point."""
    loader = unittest.TestLoader()
    suite = loader.discover(".", pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
