"""Tests for data processing functionality."""

# pylint: disable=import-error
# pylint: disable=wrong-import-position

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import (
    generate_particle_data,
    load_and_validate_data,
    preprocess_for_training
)


class TestParticleDataGeneration(unittest.TestCase):
    """Test particle data generation."""

    def test_generate_returns_dataframe(self):
        """generate_particle_data should return a pandas DataFrame."""
        data = generate_particle_data()
        self.assertIsInstance(data, pd.DataFrame)

    def test_generate_creates_non_empty_data(self):
        """Generated data should not be empty."""
        data = generate_particle_data()
        self.assertGreater(len(data), 0)

    def test_generate_correct_shape(self):
        """Generated data should have expected shape (10, 15)."""
        data = generate_particle_data()
        self.assertEqual(data.shape, (10, 15))


class TestDataLoading(unittest.TestCase):
    """Test data loading and validation."""

    def tearDown(self):
        """Clean up generated data file after tests."""
        if os.path.exists('particle_data.csv'):
            os.remove('particle_data.csv')

    def test_load_returns_dataframe(self):
        """load_and_validate_data should return a pandas DataFrame."""
        data = load_and_validate_data()
        self.assertIsInstance(data, pd.DataFrame)

    def test_load_creates_non_empty_data(self):
        """Loaded data should not be empty."""
        data = load_and_validate_data()
        self.assertGreater(len(data), 0)

    def test_load_correct_shape(self):
        """Loaded data should have 15 columns."""
        data = load_and_validate_data()
        self.assertEqual(data.shape[1], 15)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing for training."""

    def test_preprocess_returns_tuple(self):
        """preprocess_for_training should return a tuple."""
        dataframe = pd.DataFrame({'mass': [1.0], 'kinetic_energy': [1.0]})
        processed_data = preprocess_for_training(dataframe)
        self.assertIsInstance(processed_data, tuple)

    def test_preprocess_returns_six_elements(self):
        """Preprocessed data should contain 6 elements (train/val/test splits)."""
        dataframe = pd.DataFrame({'mass': [1.0], 'kinetic_energy': [1.0]})
        processed_data = preprocess_for_training(dataframe)
        self.assertEqual(len(processed_data), 6)


if __name__ == '__main__':
    unittest.main()
