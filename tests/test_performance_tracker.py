"""Tests for performance tracking functionality."""

# pylint: disable=import-error
# pylint: disable=wrong-import-position

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from performance_tracker import PerformanceTracker


class TestPerformanceTrackerInitialization(unittest.TestCase):
    """Test PerformanceTracker initialization."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_with_output_dir(self):
        """Tracker should initialize with specified output directory."""
        tracker = PerformanceTracker(output_dir=self.temp_dir)
        self.assertEqual(tracker.output_dir, self.temp_dir)

    def test_initial_r2_value(self):
        """Tracker should initialize with R2 value of 0.0."""
        tracker = PerformanceTracker(output_dir=self.temp_dir)
        self.assertEqual(tracker.current_r2, 0.0)


class TestPerformanceTrackerSummary(unittest.TestCase):
    """Test performance summary generation."""

    def setUp(self):
        """Create temporary directory and tracker."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = PerformanceTracker(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_summary_returns_dict(self):
        """get_summary should return a dictionary."""
        summary = self.tracker.get_summary()
        self.assertIsInstance(summary, dict)

    def test_summary_contains_current_r2(self):
        """Summary should contain current_r2 key."""
        summary = self.tracker.get_summary()
        self.assertIn('current_r2', summary)


if __name__ == '__main__':
    unittest.main()
