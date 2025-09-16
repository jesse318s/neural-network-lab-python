"""
Unit tests for the refactored msaeRmseMaeR2_score function.
This ensures the Keras-based implementation produces equivalent results
to the original implementation.
"""

import unittest
import numpy as np
from importsDo import msaeRmseMaeR2_score


class TestMsaeRmseMaeR2Score(unittest.TestCase):
    """Test the refactored msaeRmseMaeR2_score function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test data with known results
        self.y_test_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_1 = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
        
        # Perfect predictions
        self.y_test_2 = np.array([1.0, 2.0, 3.0])
        self.y_pred_2 = np.array([1.0, 2.0, 3.0])
        
        # More varied test data
        self.y_test_3 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        self.y_pred_3 = np.array([0.6, 1.4, 2.4, 3.6, 4.4])
    
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
        """Test with perfect predictions (should give R² = 1, MSE/MAE/RMSE = 0)."""
        mse, mae, rmse, r2 = msaeRmseMaeR2_score(self.y_test_2, self.y_pred_2)
        
        # Should be exactly 0 for perfect predictions
        self.assertAlmostEqual(mse, 0.0, places=6)
        self.assertAlmostEqual(mae, 0.0, places=6)
        self.assertAlmostEqual(rmse, 0.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)
    
    def test_known_values(self):
        """Test with known values and expected results."""
        mse, mae, rmse, r2 = msaeRmseMaeR2_score(self.y_test_1, self.y_pred_1)
        
        # Expected values calculated independently
        expected_mse = 0.016
        expected_mae = 0.12
        expected_rmse = np.sqrt(expected_mse)
        expected_r2 = 0.992
        
        self.assertAlmostEqual(mse, expected_mse, places=5)
        self.assertAlmostEqual(mae, expected_mae, places=5)
        self.assertAlmostEqual(rmse, expected_rmse, places=5)
        self.assertAlmostEqual(r2, expected_r2, places=3)
    
    def test_metrics_relationships(self):
        """Test mathematical relationships between metrics."""
        mse, mae, rmse, r2 = msaeRmseMaeR2_score(self.y_test_3, self.y_pred_3)
        
        # RMSE should be the square root of MSE
        self.assertAlmostEqual(rmse, np.sqrt(mse), places=6)
        
        # All metrics should be non-negative
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(mae, 0)
        self.assertGreaterEqual(rmse, 0)
        
        # R² should be <= 1 for reasonable data
        self.assertLessEqual(r2, 1.0)
    
    def test_edge_cases(self):
        """Test edge cases with small arrays."""
        y_test = np.array([1.0])
        y_pred = np.array([1.1])
        
        result = msaeRmseMaeR2_score(y_test, y_pred)
        
        # Should still return 4 values
        self.assertEqual(len(result), 4)
        
        # All should be valid numbers
        for value in result:
            self.assertFalse(np.isnan(value))
            self.assertFalse(np.isinf(value))
    
    def test_list_input_compatibility(self):
        """Test that function works with Python lists as well as numpy arrays."""
        y_test_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_list = [1.1, 2.2, 2.9, 4.1, 4.9]
        
        result_list = msaeRmseMaeR2_score(y_test_list, y_pred_list)
        result_array = msaeRmseMaeR2_score(self.y_test_1, self.y_pred_1)
        
        # Results should be the same regardless of input type
        for val_list, val_array in zip(result_list, result_array):
            self.assertAlmostEqual(val_list, val_array, places=6)


def original_msaeRmseMaeR2_score(y_test, y_pred):
    """
    Original implementation for comparison (with syntax errors fixed).
    This is used to verify our refactored version produces the same results.
    """
    diff_abs = np.abs(y_test - y_pred)
    mse, mae = float(np.mean((diff_abs) ** 2)), float(np.mean(diff_abs))
    rmse = float(np.sqrt(mse))
    # R² score
    ss_res, ss_tot = np.sum((diff_abs) ** 2), np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return mse, mae, rmse, r2_score


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the refactored function produces the same results as the original."""
    
    def test_equivalence_with_original(self):
        """Test that the Keras implementation matches the original."""
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.1, 2.2, 2.9, 4.1, 4.9])),
            (np.array([0.5, 1.5, 2.5]), np.array([0.6, 1.4, 2.4])),
            (np.array([10.0, 20.0, 30.0]), np.array([9.8, 20.1, 30.2])),
        ]
        
        for y_test, y_pred in test_cases:
            with self.subTest(y_test=y_test, y_pred=y_pred):
                original_result = original_msaeRmseMaeR2_score(y_test, y_pred)
                refactored_result = msaeRmseMaeR2_score(y_test, y_pred)
                
                # Compare each metric
                for i, (orig, refact) in enumerate(zip(original_result, refactored_result)):
                    self.assertAlmostEqual(
                        orig, refact, places=5,
                        msg=f"Metric {i} differs: original={orig}, refactored={refact}"
                    )


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)