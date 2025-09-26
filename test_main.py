"""
Test Suite for Neural Network Lab

This test suite verifies the functionality of various components in the neural network lab,
including weight constraints, adaptive loss functions, and performance tracking.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from weight_constraints import BinaryWeightConstraintMax, BinaryWeightConstraintChanges, OscillationDampener
    WEIGHT_CONSTRAINTS_AVAILABLE = True
except ImportError:
    WEIGHT_CONSTRAINTS_AVAILABLE = False

try:
    from ml_utils import create_adaptive_loss_fn, compute_loss_weights
    ADAPTIVE_LOSS_AVAILABLE = True
except ImportError:
    ADAPTIVE_LOSS_AVAILABLE = False

try:
    from performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False


class TestWeightConstraints(unittest.TestCase):
    """Test weight constraint functionality."""
    
    def setUp(self):
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
    
    def test_binary_weight_constraint_max(self):
        """Test binary weight constraint max functionality."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=8)
        weights = np.array([[0.125344, -0.875444], [1.5444, 0.75444]])
        result = constraint.apply_constraint(weights)
        
        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(result[0,0], weights[0,0])

    def test_binary_weight_constraint_changes(self):
        """Test binary weight constraint changes functionality."""
        constraint = BinaryWeightConstraintChanges(max_additional_digits=1)
        weights = np.array([[99.5123123112313999999999, -0.75], [1.25, 0.125]])
        constraint.previous_weights = np.array([[1, -0.7], [1.2, -1.4]])
        result = constraint.apply_constraint(weights)

        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(result[0,0], weights[0,0])
    
    def test_oscillation_dampener(self):
        """Test oscillation dampener functionality."""
        dampener = OscillationDampener(window_size=3)
        dampener_weight_values = [0.41, 0.51, 0.31]
        unstable_weights = np.array([[0.81]])
        
        for val in dampener_weight_values:
            dampener.add_weights(np.array([[val]]))

        result = dampener.detect_and_dampen_oscillations(unstable_weights)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, unstable_weights.shape)
        self.assertLess(result[0,0], unstable_weights[0,0])


class TestAdaptiveLoss(unittest.TestCase):
    """Test adaptive loss function functionality."""
    
    def setUp(self):
        if not ADAPTIVE_LOSS_AVAILABLE:
            self.skipTest("Adaptive loss module not available")
    
    def test_adaptive_loss_initialization(self):
        """Test adaptive loss function creation."""
        loss_fn = create_adaptive_loss_fn(strategy='epoch_based')

        self.assertTrue(callable(loss_fn))
        self.assertTrue(hasattr(loss_fn, 'update_state'))
        self.assertTrue(hasattr(loss_fn, 'get_current_info'))
        self.assertTrue(hasattr(loss_fn, 'get_history'))
    
    def test_adaptive_loss_get_weights(self):
        """Test getting weights from compute_loss_weights function."""
        mse_weight, mae_weight = compute_loss_weights('epoch_based', epoch=5)
        
        self.assertGreater(mse_weight, 0)
        self.assertGreater(mae_weight, 0)
        self.assertEqual(mse_weight + mae_weight, 1.0)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking functionality."""
    
    def setUp(self):
        if not PERFORMANCE_TRACKER_AVAILABLE:
            self.skipTest("Performance tracker module not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = PerformanceTracker(output_dir=self.temp_dir)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_performance_tracker_initialization(self):
        """Test initialization of PerformanceTracker."""
        self.assertEqual(self.tracker.output_dir, self.temp_dir)
        self.assertEqual(self.tracker.current_r2, 0.0)
    
    def test_get_summary(self):
        """Test getting performance summary."""
        summary = self.tracker.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('current_r2', summary)


class TestIntegration(unittest.TestCase):
    """Basic integration tests."""
    
    def test_component_availability(self):
        """Test that components can be imported."""
        components = {
            'weight_constraints': WEIGHT_CONSTRAINTS_AVAILABLE,
            'adaptive_loss': ADAPTIVE_LOSS_AVAILABLE,
            'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE
        }
        available_count = sum(components.values())

        self.assertEqual(available_count, len(components))
    
    def test_numpy_compatibility(self):
        """Test NumPy array handling."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        array = np.array([[0.5, 0.3], [0.7, 0.2]])
        result = constraint.apply_constraint(array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, array.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
