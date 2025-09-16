"""
Test Suite for Neural Network Lab
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with error handling
try:
    from weight_constraints import BinaryWeightConstraintMax, OscillationDampener
    WEIGHT_CONSTRAINTS_AVAILABLE = True
except ImportError:
    WEIGHT_CONSTRAINTS_AVAILABLE = False

try:
    from adaptive_loss import AdaptiveLossFunction
    ADAPTIVE_LOSS_AVAILABLE = True
except ImportError:
    ADAPTIVE_LOSS_AVAILABLE = False

try:
    from performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False

try:
    from advanced_neural_network import msaeRmseMaeR2_score
    MSAE_AVAILABLE = True
except ImportError:
    MSAE_AVAILABLE = False


class TestWeightConstraints(unittest.TestCase):
    """Test weight constraint functionality."""
    
    def setUp(self):
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
    
    def test_binary_weight_constraint_max(self):
        """Test binary weight constraint max functionality."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        
        weights = np.array([[0.125, 0.875], [1.5, 0.75]])
        result = constraint.apply_constraint(weights)
        
        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
    
    def test_oscillation_dampener(self):
        """Test oscillation dampener functionality."""
        dampener = OscillationDampener(window_size=3)
        
        weights = np.array([[0.5]])
        dampener.add_weights(weights)
        
        new_weights = np.array([[0.8]])
        result = dampener.detect_and_dampen_oscillations(new_weights)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, new_weights.shape)


class TestAdaptiveLoss(unittest.TestCase):
    """Test adaptive loss function functionality."""
    
    def setUp(self):
        if not ADAPTIVE_LOSS_AVAILABLE:
            self.skipTest("Adaptive loss module not available")
    
    def test_adaptive_loss_initialization(self):
        """Test AdaptiveLossFunction initialization."""
        loss_fn = AdaptiveLossFunction(weighting_strategy='epoch_based')
        self.assertEqual(loss_fn.weighting_strategy, 'epoch_based')
        self.assertEqual(loss_fn.current_epoch, 0)
    
    def test_adaptive_loss_get_weights(self):
        """Test getting weights from adaptive loss function."""
        loss_fn = AdaptiveLossFunction(weighting_strategy='epoch_based')
        
        mse_weight, mae_weight = loss_fn.get_weights()
        
        self.assertGreater(mse_weight, 0)
        self.assertGreater(mae_weight, 0)
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0, places=3)


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
        self.assertEqual(self.tracker.current_accuracy, 0.0)
    
    def test_get_summary(self):
        """Test getting performance summary."""
        summary = self.tracker.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('current_accuracy', summary)


class TestMSAE(unittest.TestCase):
    """Test MSAE metrics functionality."""
    
    def setUp(self):
        if not MSAE_AVAILABLE:
            self.skipTest("MSAE function not available")
        
        # Test data with known results
        self.y_test_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_1 = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
        
        # Perfect predictions
        self.y_test_2 = np.array([1.0, 2.0, 3.0])
        self.y_pred_2 = np.array([1.0, 2.0, 3.0])
    
    def test_function_returns_four_values(self):
        """Test that function returns exactly four values."""
        result = msaeRmseMaeR2_score(self.y_test_1, self.y_pred_1)
        self.assertEqual(len(result), 4)
    
    def test_return_types(self):
        """Test that all returned values are floats."""
        result = msaeRmseMaeR2_score(self.y_test_1, self.y_pred_1)
        for value in result:
            self.assertIsInstance(value, float)
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        mse, mae, rmse, r2 = msaeRmseMaeR2_score(self.y_test_2, self.y_pred_2)
        
        self.assertAlmostEqual(mse, 0.0, places=6)
        self.assertAlmostEqual(mae, 0.0, places=6)
        self.assertAlmostEqual(rmse, 0.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)
    
    def test_metrics_relationships(self):
        """Test mathematical relationships between metrics."""
        mse, mae, rmse, r2 = msaeRmseMaeR2_score(self.y_test_1, self.y_pred_1)
        
        # RMSE should be the square root of MSE
        self.assertAlmostEqual(rmse, np.sqrt(mse), places=6)
        
        # All metrics should be non-negative
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(mae, 0)
        self.assertGreaterEqual(rmse, 0)


class TestIntegration(unittest.TestCase):
    """Basic integration tests."""
    
    def test_component_availability(self):
        """Test that components can be imported."""
        components = {
            'weight_constraints': WEIGHT_CONSTRAINTS_AVAILABLE,
            'adaptive_loss': ADAPTIVE_LOSS_AVAILABLE,
            'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE,
            'msae_function': MSAE_AVAILABLE
        }
        
        # At least one component should be available
        available_count = sum(components.values())
        self.assertGreater(available_count, 0, "No components available")
    
    def test_numpy_compatibility(self):
        """Test NumPy array handling."""
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
        
        test_array = np.array([[0.5, 0.3], [0.7, 0.2]])
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        result = constraint.apply_constraint(test_array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_array.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)