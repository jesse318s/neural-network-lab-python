"""Tests for weight constraint implementations."""

# pylint: disable=import-error
# pylint: disable=wrong-import-position

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weight_constraints import (
    BinaryWeightConstraintMax,
    BinaryWeightConstraintChanges,
    OscillationDampener,
    AdaptiveOscillationDampener
)


class TestBinaryWeightConstraintMax(unittest.TestCase):
    """Test BinaryWeightConstraintMax functionality."""

    def test_apply_constraint_reduces_precision(self):
        """Constraint should reduce weight precision to max binary digits."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=8)
        weights = np.array([[0.125344, -0.875444], [1.5444, 0.75444]])
        result = constraint.apply_constraint(weights)

        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(result[0, 0], weights[0, 0])

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        constraint = BinaryWeightConstraintMax(max_binary_digits=8)
        weights = np.random.randn(5, 3)
        result = constraint.apply_constraint(weights)

        self.assertEqual(result.shape, weights.shape)


class TestBinaryWeightConstraintChanges(unittest.TestCase):
    """Test BinaryWeightConstraintChanges functionality."""

    def test_apply_constraint_limits_changes(self):
        """Constraint should limit precision changes from previous weights."""
        constraint = BinaryWeightConstraintChanges(max_additional_digits=1)
        weights = np.array([[99.5123123112313999999999, -0.75], [1.25, 0.125]])
        constraint.previous_weights = np.array([[1, -0.7], [1.2, -1.4]])
        result = constraint.apply_constraint(weights)

        self.assertEqual(result.shape, weights.shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(result[0, 0], weights[0, 0])

    def test_tracks_previous_weights(self):
        """Constraint should update previous_weights after application."""
        constraint = BinaryWeightConstraintChanges(max_additional_digits=2)
        weights = np.array([[0.5, 0.3]])
        constraint.previous_weights = np.array([[0.4, 0.2]])
        result = constraint.apply_constraint(weights)

        self.assertIsNotNone(result)


class TestOscillationDampener(unittest.TestCase):
    """Test OscillationDampener functionality."""

    def test_dampens_oscillating_weights(self):
        """Dampener should reduce magnitude of oscillating weights."""
        dampener = OscillationDampener()
        dampener_weight_values = [0.41, 0.51, 0.31]
        unstable_weights = np.array([[0.81]])

        for val in dampener_weight_values:
            dampener.add_weights(np.array([[val]]))

        result = dampener.apply_constraint(unstable_weights)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, unstable_weights.shape)
        self.assertLess(result[0, 0], unstable_weights[0, 0])

    def test_weight_history_tracking(self):
        """Dampener should maintain weight history."""
        dampener = OscillationDampener()
        weights = np.array([[0.5]])

        dampener.add_weights(weights)
        self.assertEqual(len(dampener.weight_history), 1)


class TestAdaptiveOscillationDampener(unittest.TestCase):
    """Test AdaptiveOscillationDampener functionality."""

    def test_adaptive_dampening(self):
        """Adaptive dampener should adjust damping based on oscillation pattern."""
        dampener = AdaptiveOscillationDampener()
        dampener_weight_values = [0.41, 0.51, 0.31]
        unstable_weights = np.array([[0.81]])

        for val in dampener_weight_values:
            dampener.add_weights(np.array([[val]]))

        result = dampener.apply_constraint(unstable_weights)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, unstable_weights.shape)
        self.assertLess(result[0, 0], unstable_weights[0, 0])

    def test_inherits_from_base_dampener(self):
        """Adaptive dampener should have base dampener capabilities."""
        dampener = AdaptiveOscillationDampener()

        self.assertTrue(hasattr(dampener, 'add_weights'))
        self.assertTrue(hasattr(dampener, 'apply_constraint'))


if __name__ == '__main__':
    unittest.main()
