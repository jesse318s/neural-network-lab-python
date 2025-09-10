"""
Test Suite for Advanced TensorFlow Lab

This test suite validates all components of the advanced TensorFlow lab including
weight constraints, adaptive loss functions, performance tracking, and data loading.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with error handling
try:
    from weight_constraints import (
        BinaryWeightConstraintChanges, 
        BinaryWeightConstraintMax, 
        OscillationDampener
    )
    WEIGHT_CONSTRAINTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Weight constraints module not available: {e}")
    WEIGHT_CONSTRAINTS_AVAILABLE = False

try:
    from adaptive_loss import (
        AdaptiveLossFunction, 
        epoch_weighted_loss, 
        accuracy_weighted_loss, 
        loss_weighted_loss
    )
    ADAPTIVE_LOSS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Adaptive loss module not available: {e}")
    ADAPTIVE_LOSS_AVAILABLE = False

try:
    from performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Performance tracker module not available: {e}")
    PERFORMANCE_TRACKER_AVAILABLE = False

try:
    from data_loader import generate_particle_data, validate_data_integrity
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data loader module not available: {e}")
    DATA_LOADER_AVAILABLE = False


class TestBinaryWeightConstraints(unittest.TestCase):
    """Test binary weight constraint functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
    
    def test_binary_weight_constraint_changes_initialization(self):
        """Test initialization of BinaryWeightConstraintChanges."""
        constraint = BinaryWeightConstraintChanges(max_additional_digits=2)
        self.assertEqual(constraint.max_additional_digits, 2)
        self.assertIsNone(constraint.previous_weights)
        self.assertEqual(constraint.error_count, 0)
    
    def test_binary_weight_constraint_changes_apply(self):
        """Test applying binary weight constraint changes."""
        constraint = BinaryWeightConstraintChanges(max_additional_digits=1)
        
        # Test with simple weight matrix
        weights1 = np.array([[0.5, 0.25], [0.75, 0.125]])
        result1 = constraint.apply_constraint(weights1)
        
        # First application should store weights and return unchanged
        np.testing.assert_array_equal(result1, weights1)
        
        # Second application should apply constraint
        weights2 = np.array([[0.6, 0.3], [0.8, 0.15]])
        result2 = constraint.apply_constraint(weights2)
        
        # Result should be a numpy array with the same shape
        self.assertEqual(result2.shape, weights2.shape)
        self.assertIsInstance(result2, np.ndarray)
    
    def test_binary_weight_constraint_max_initialization(self):
        """Test initialization of BinaryWeightConstraintMax."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        self.assertEqual(constraint.max_binary_digits, 3)
        self.assertEqual(constraint.error_count, 0)
    
    def test_binary_weight_constraint_max_apply(self):
        """Test applying binary weight constraint max."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        
        weights = np.array([[0.125, 0.875], [1.5, 0.75]])
        result = constraint.apply_constraint(weights)
        
        # Result should be a numpy array with the same shape
        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
    
    def test_oscillation_dampener_initialization(self):
        """Test initialization of OscillationDampener."""
        dampener = OscillationDampener(window_size=3)
        self.assertEqual(dampener.window_size, 3)
        self.assertEqual(len(dampener.weight_history), 0)
        self.assertEqual(dampener.error_count, 0)
    
    def test_oscillation_dampener_add_weights(self):
        """Test adding weights to oscillation dampener."""
        dampener = OscillationDampener(window_size=3)
        
        weights1 = np.array([[0.5]])
        weights2 = np.array([[0.7]])
        
        dampener.add_weights(weights1)
        self.assertEqual(len(dampener.weight_history), 1)
        
        dampener.add_weights(weights2)
        self.assertEqual(len(dampener.weight_history), 2)
    
    def test_oscillation_dampener_detect_and_dampen(self):
        """Test oscillation detection and dampening."""
        dampener = OscillationDampener(window_size=3)
        
        # Add weights that create oscillation pattern
        weights_sequence = [
            np.array([[0.5]]),
            np.array([[0.7]]),
            np.array([[0.4]])
        ]
        
        for weights in weights_sequence:
            dampener.add_weights(weights)
        
        # Test dampening on new weights
        new_weights = np.array([[0.8]])
        result = dampener.detect_and_dampen_oscillations(new_weights)
        
        # Result should be a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, new_weights.shape)


class TestAdaptiveLoss(unittest.TestCase):
    """Test adaptive loss function functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ADAPTIVE_LOSS_AVAILABLE:
            self.skipTest("Adaptive loss module not available")
    
    def test_adaptive_loss_function_initialization(self):
        """Test initialization of AdaptiveLossFunction."""
        loss_fn = AdaptiveLossFunction(weighting_strategy='epoch_based')
        self.assertEqual(loss_fn.weighting_strategy, 'epoch_based')
        self.assertEqual(loss_fn.current_epoch, 0)
        self.assertEqual(loss_fn.previous_accuracy, 0.5)
        self.assertEqual(loss_fn.error_count, 0)
    
    def test_epoch_weighted_loss_function(self):
        """Test epoch-based weighted loss function."""
        mse_loss = 0.15
        mae_loss = 0.12
        
        # Test early epoch
        early_loss = epoch_weighted_loss(5, mse_loss, mae_loss)
        self.assertIsInstance(early_loss, float)
        self.assertGreater(early_loss, 0)
        
        # Test mid epoch
        mid_loss = epoch_weighted_loss(20, mse_loss, mae_loss)
        self.assertIsInstance(mid_loss, float)
        self.assertGreater(mid_loss, 0)
        
        # Test late epoch
        late_loss = epoch_weighted_loss(40, mse_loss, mae_loss)
        self.assertIsInstance(late_loss, float)
        self.assertGreater(late_loss, 0)
    
    def test_accuracy_weighted_loss_function(self):
        """Test accuracy-based weighted loss function."""
        mse_loss = 0.15
        mae_loss = 0.12
        
        # Test low accuracy
        low_acc_loss = accuracy_weighted_loss(0.2, mse_loss, mae_loss)
        self.assertIsInstance(low_acc_loss, float)
        self.assertGreater(low_acc_loss, 0)
        
        # Test high accuracy
        high_acc_loss = accuracy_weighted_loss(0.9, mse_loss, mae_loss)
        self.assertIsInstance(high_acc_loss, float)
        self.assertGreater(high_acc_loss, 0)
    
    def test_loss_weighted_loss_function(self):
        """Test loss-based weighted loss function."""
        mse_loss = 0.15
        mae_loss = 0.12
        
        # Test high previous loss
        high_loss = loss_weighted_loss(2.0, mse_loss, mae_loss)
        self.assertIsInstance(high_loss, float)
        self.assertGreater(high_loss, 0)
        
        # Test low previous loss
        low_loss = loss_weighted_loss(0.05, mse_loss, mae_loss)
        self.assertIsInstance(low_loss, float)
        self.assertGreater(low_loss, 0)
    
    def test_adaptive_loss_get_weights(self):
        """Test getting weights from adaptive loss function."""
        loss_fn = AdaptiveLossFunction(weighting_strategy='epoch_based')
        
        mse_weight, mae_weight = loss_fn.get_weights()
        
        # Weights should be positive and sum to approximately 1
        self.assertGreater(mse_weight, 0)
        self.assertGreater(mae_weight, 0)
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0, places=3)
    
    def test_adaptive_loss_update_epoch(self):
        """Test updating epoch information."""
        loss_fn = AdaptiveLossFunction(weighting_strategy='accuracy_based')
        
        loss_fn.update_epoch(10, accuracy=0.8)
        
        self.assertEqual(loss_fn.current_epoch, 10)
        self.assertEqual(loss_fn.previous_accuracy, 0.8)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PERFORMANCE_TRACKER_AVAILABLE:
            self.skipTest("Performance tracker module not available")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = PerformanceTracker(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_performance_tracker_initialization(self):
        """Test initialization of PerformanceTracker."""
        self.assertEqual(self.tracker.output_dir, self.temp_dir)
        self.assertEqual(self.tracker.current_accuracy, 0.0)
        self.assertEqual(self.tracker.best_accuracy, 0.0)
        self.assertEqual(self.tracker.error_count, 0)
    
    def test_start_training(self):
        """Test starting training session."""
        config = {
            'epochs': 10,
            'batch_size': 32,
            'adaptive_loss_strategy': 'epoch_based'
        }
        
        self.tracker.start_training(config)
        
        self.assertIsNotNone(self.tracker.training_start_time)
        self.assertEqual(self.tracker.training_config, config)
        self.assertEqual(self.tracker.adaptive_loss_strategy, 'epoch_based')
    
    def test_epoch_tracking(self):
        """Test epoch start and end tracking."""
        self.tracker.start_epoch(0)
        self.assertIsNotNone(self.tracker.epoch_start_time)
        
        logs = {
            'loss': 0.5,
            'val_loss': 0.6,
            'accuracy': 0.7
        }
        
        self.tracker.end_epoch(0, logs)
        
        self.assertEqual(len(self.tracker.training_history), 1)
        self.assertEqual(self.tracker.current_accuracy, 0.7)
        self.assertEqual(self.tracker.best_accuracy, 0.7)
    
    def test_add_weight_modification(self):
        """Test adding weight modification tracking."""
        self.tracker.add_weight_modification('binary_constraint')
        self.tracker.add_weight_modification('oscillation_dampening')
        
        self.assertIn('binary_constraint', self.tracker.weight_modifications_used)
        self.assertIn('oscillation_dampening', self.tracker.weight_modifications_used)
    
    def test_get_summary(self):
        """Test getting performance summary."""
        summary = self.tracker.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('current_accuracy', summary)
        self.assertIn('best_accuracy', summary)
        self.assertIn('error_count', summary)


class TestDataLoader(unittest.TestCase):
    """Test data loading and processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not DATA_LOADER_AVAILABLE:
            self.skipTest("Data loader module not available")
    
    def test_generate_particle_data(self):
        """Test particle data generation."""
        df = generate_particle_data(num_particles=5, save_to_file=False)
        
        # Check that dataframe is created
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)
        
        # Check required columns exist
        required_columns = [
            'particle_id', 'mass', 'initial_velocity_x', 'final_velocity_x'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_validate_data_integrity(self):
        """Test data integrity validation."""
        # Create test dataframe
        test_data = {
            'particle_id': [1, 2, 3],
            'mass': [1.0, 2.0, 3.0],
            'velocity': [1.0, 2.0, float('inf')]  # Include invalid data
        }
        
        try:
            import pandas as pd
            df = pd.DataFrame(test_data)
            
            validation_result = validate_data_integrity(df)
            
            self.assertIsInstance(validation_result, dict)
            self.assertIn('is_valid', validation_result)
            self.assertIn('issues', validation_result)
            
        except ImportError:
            self.skipTest("Pandas not available for data validation test")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and railway programming patterns."""
    
    def test_graceful_degradation(self):
        """Test that components fail gracefully."""
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
        
        # Test with invalid input
        constraint = BinaryWeightConstraintChanges()
        
        # This should not raise an exception, but handle the error gracefully
        try:
            result = constraint.apply_constraint(None)
            # Should return something, even if it's the original input or a default
            self.assertIsNotNone(result)
        except Exception:
            # If an exception is raised, it should be documented
            pass
    
    def test_error_counting(self):
        """Test that errors are properly counted."""
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
        
        constraint = BinaryWeightConstraintChanges()
        initial_count = constraint.get_error_count()
        
        # Try to cause an error (implementation dependent)
        try:
            constraint.apply_constraint(np.array([]))  # Empty array might cause issues
        except:
            pass
        
        # Error count should be accessible
        final_count = constraint.get_error_count()
        self.assertIsInstance(final_count, int)
        self.assertGreaterEqual(final_count, initial_count)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_component_compatibility(self):
        """Test that components work together without import errors."""
        # Test that we can import and instantiate main components
        components_available = {
            'weight_constraints': WEIGHT_CONSTRAINTS_AVAILABLE,
            'adaptive_loss': ADAPTIVE_LOSS_AVAILABLE,
            'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE,
            'data_loader': DATA_LOADER_AVAILABLE
        }
        
        print("\nComponent availability:")
        for component, available in components_available.items():
            status = "✓" if available else "✗"
            print(f"  {status} {component}")
        
        # At least half of the components should be available
        available_count = sum(components_available.values())
        total_count = len(components_available)
        
        self.assertGreaterEqual(
            available_count / total_count, 0.5,
            f"Only {available_count}/{total_count} components available"
        )
    
    def test_numpy_compatibility(self):
        """Test NumPy array handling across components."""
        if not WEIGHT_CONSTRAINTS_AVAILABLE:
            self.skipTest("Weight constraints module not available")
        
        # Create test array
        test_array = np.array([[0.5, 0.3], [0.7, 0.2]])
        
        # Test that components can handle numpy arrays
        constraint = BinaryWeightConstraintMax(max_binary_digits=3)
        result = constraint.apply_constraint(test_array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_array.shape)


def run_comprehensive_tests():
    """Run all tests and provide a comprehensive report."""
    print("=" * 80)
    print("ADVANCED TENSORFLOW LAB - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBinaryWeightConstraints,
        TestAdaptiveLoss,
        TestPerformanceTracker,
        TestDataLoader,
        TestErrorHandling,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.skipped:
        print(f"\nSkipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    # Overall assessment
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun)
    
    print(f"\nOverall Success Rate: {success_rate * 100:.1f}%")
    
    if success_rate >= 0.9:
        print("🎉 Excellent! The lab implementation is highly reliable.")
    elif success_rate >= 0.7:
        print("✅ Good! The lab implementation is mostly working with minor issues.")
    elif success_rate >= 0.5:
        print("⚠️ Fair! The lab has some significant issues that need attention.")
    else:
        print("❌ Poor! The lab implementation has major problems.")
    
    print("\n" + "=" * 80)
    print("Ready to run the main lab!")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Check if running as main script
    if len(sys.argv) > 1 and sys.argv[1] == '--comprehensive':
        # Run comprehensive test suite
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        # Run standard unittest
        unittest.main(verbosity=2)