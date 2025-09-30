"""
Test Suite for Neural Network Lab

This test suite verifies the functionality of various components in the neural network lab,
including weight constraints, adaptive loss functions, and performance tracking.
"""

import unittest
import numpy as np
import pandas as pd
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

try:
    from data_processing import generate_particle_data, load_and_validate_data, preprocess_for_training
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False


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
        dampener = OscillationDampener()
        dampener_weight_values = [0.41, 0.51, 0.31]
        unstable_weights = np.array([[0.81]])
        
        for val in dampener_weight_values:
            dampener.add_weights(np.array([[val]]))

        result = dampener.apply_constraint(unstable_weights)
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
        loss_fn_r2 = create_adaptive_loss_fn(strategy='r2_based')
        loss_fn_loss = create_adaptive_loss_fn(strategy='loss_based')
        loss_fn_combined = create_adaptive_loss_fn(strategy='combined')

        self.assertTrue(callable(loss_fn_r2))
        self.assertTrue(callable(loss_fn_loss))
        self.assertTrue(callable(loss_fn_combined))
    
    def test_adaptive_loss_get_weights(self):
        """Test getting weights from compute_loss_weights function."""
        mse_weight_r2, mae_weight_r2 = compute_loss_weights('r2_based')
        mse_weight_loss, mae_weight_loss = compute_loss_weights('loss_based')
        mse_weight_combined, mae_weight_combined = compute_loss_weights('combined')
        
        self.assertAlmostEqual(mse_weight_r2 + mae_weight_r2, 1.0)
        self.assertAlmostEqual(mse_weight_loss + mae_weight_loss, 1.0)
        self.assertAlmostEqual(mse_weight_combined + mae_weight_combined, 1.0)


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


class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        if not DATA_PROCESSING_AVAILABLE:
            self.skipTest("Data processing module not available")
    
    def test_generate_particle_data(self):
        """Test particle data generation."""
        data = generate_particle_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertEqual(data.shape, (10, 15)) 
    
    def test_load_and_validate_data(self):
        """Test loading and validating data."""
        data = load_and_validate_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertEqual(data.shape, (10, 15))
    
    def test_preprocess_for_training(self):
        """Test preprocessing data for training."""
        dataframe = pd.DataFrame({'mass': [1.0], 'kinetic_energy': [1.0]})
        processed_data = preprocess_for_training(dataframe)
        
        self.assertIsInstance(processed_data, tuple)
        self.assertEqual(len(processed_data), 6)


class TestIntegration(unittest.TestCase):
    """Basic integration tests."""
    
    def test_component_availability(self):
        """Test that components can be imported."""
        components = {
            'weight_constraints': WEIGHT_CONSTRAINTS_AVAILABLE,
            'adaptive_loss': ADAPTIVE_LOSS_AVAILABLE,
            'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE,
            'data_processing': DATA_PROCESSING_AVAILABLE
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
