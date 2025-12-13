"""Tests for adaptive loss function implementations."""

# pylint: disable=import-error
# pylint: disable=wrong-import-position

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_utils import create_adaptive_loss_fn, compute_loss_weights


class TestAdaptiveLossCreation(unittest.TestCase):
    """Test creation of adaptive loss functions."""

    def test_r2_based_strategy(self):
        """R2-based strategy should return callable loss function."""
        loss_fn = create_adaptive_loss_fn(strategy='r2_based')
        self.assertTrue(callable(loss_fn))

    def test_loss_based_strategy(self):
        """Loss-based strategy should return callable loss function."""
        loss_fn = create_adaptive_loss_fn(strategy='loss_based')
        self.assertTrue(callable(loss_fn))

    def test_combined_strategy(self):
        """Combined strategy should return callable loss function."""
        loss_fn = create_adaptive_loss_fn(strategy='combined')
        self.assertTrue(callable(loss_fn))

    def test_physics_aware_strategy(self):
        """Physics-aware strategy should return callable loss function."""
        loss_fn = create_adaptive_loss_fn(strategy='physics_aware')
        self.assertTrue(callable(loss_fn))


class TestLossWeightComputation(unittest.TestCase):
    """Test computation of loss weights for different strategies."""

    def test_r2_based_weights_sum_to_one(self):
        """R2-based weights should sum to 1.0."""
        mse_weight, mae_weight = compute_loss_weights('r2_based')
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0)

    def test_loss_based_weights_sum_to_one(self):
        """Loss-based weights should sum to 1.0."""
        mse_weight, mae_weight = compute_loss_weights('loss_based')
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0)

    def test_combined_weights_sum_to_one(self):
        """Combined weights should sum to 1.0."""
        mse_weight, mae_weight = compute_loss_weights('combined')
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0)

    def test_physics_aware_weights_sum_to_one(self):
        """Physics-aware weights should sum to 1.0."""
        mse_weight, mae_weight = compute_loss_weights('physics_aware')
        self.assertAlmostEqual(mse_weight + mae_weight, 1.0)

    def test_weights_are_non_negative(self):
        """All computed weights should be non-negative."""
        strategies = ['r2_based', 'loss_based', 'combined', 'physics_aware']
        for strategy in strategies:
            mse_weight, mae_weight = compute_loss_weights(strategy)
            self.assertGreaterEqual(mse_weight, 0.0)
            self.assertGreaterEqual(mae_weight, 0.0)


if __name__ == '__main__':
    unittest.main()
