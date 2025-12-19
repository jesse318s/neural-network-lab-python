"""Integration tests for neural network components."""

# pylint: disable=import-error
# pylint: disable=wrong-import-position

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import AdvancedNeuralNetwork, complete_data_pipeline, train_with_tracking


class TestDataPipeline(unittest.TestCase):
    """Test complete data pipeline integration."""

    def tearDown(self):
        if os.path.exists('particle_data.csv'):
            os.remove('particle_data.csv')

        if os.path.exists('scaler_X.pkl') or os.path.exists('scaler_y.pkl'):
            os.remove('scaler_X.pkl')
            os.remove('scaler_y.pkl')

    def test_pipeline_returns_six_splits(self):
        """Data pipeline should return 6 data splits."""
        data_splits = complete_data_pipeline(num_particles=100)
        self.assertEqual(len(data_splits), 6)

    def test_pipeline_with_small_dataset(self):
        """Pipeline should work with small datasets."""
        data_splits = complete_data_pipeline(num_particles=50)
        x_train, x_val, x_test, y_train, y_val, y_test = data_splits

        self.assertIsNotNone(x_train)
        self.assertIsNotNone(x_val)
        self.assertIsNotNone(x_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_val)
        self.assertIsNotNone(y_test)


class TestModelCreation(unittest.TestCase):
    """Test neural network model creation."""

    def test_create_model_with_config(self):
        """Should create model with provided configuration."""
        model = AdvancedNeuralNetwork((10,), 2, config={})
        self.assertIsNotNone(model)

    def test_model_accepts_input_shape(self):
        """Model should accept specified input shape."""
        input_shape = (15,)
        output_dim = 3
        model = AdvancedNeuralNetwork(input_shape, output_dim, config={})

        self.assertIsNotNone(model)


class TestTrainingIntegration(unittest.TestCase):
    """Test training with performance tracking."""

    def tearDown(self):
        if os.path.exists('particle_data.csv'):
            os.remove('particle_data.csv')

        if os.path.exists('scaler_X.pkl') or os.path.exists('scaler_y.pkl'):
            os.remove('scaler_X.pkl')
            os.remove('scaler_y.pkl')

        if os.path.exists('saved_weights'):
            for weight_file in os.listdir('saved_weights'):
                os.remove(os.path.join('saved_weights', weight_file))

            os.rmdir('saved_weights')

    def test_training_completes(self):
        """Training should complete and return results."""
        data_splits = complete_data_pipeline(num_particles=100)
        x_train, x_val, x_test, y_train, y_val, y_test = data_splits
        training_config = {'epochs': 1, 'batch_size': 16}
        model = AdvancedNeuralNetwork(
            (x_train.shape[1],),
            y_train.shape[1],
            config={}
        )
        results = train_with_tracking(
            model, x_train, x_val, x_test,
            y_train, y_val, y_test, training_config
        )

        self.assertIn('test', results)

    def test_training_with_minimal_epochs(self):
        """Training should work with minimal configuration."""
        data_splits = complete_data_pipeline(num_particles=50)
        x_train, x_val, x_test, y_train, y_val, y_test = data_splits
        training_config = {'epochs': 1, 'batch_size': 8}
        model = AdvancedNeuralNetwork(
            (x_train.shape[1],),
            y_train.shape[1],
            config={}
        )
        results = train_with_tracking(
            model, x_train, x_val, x_test,
            y_train, y_val, y_test, training_config
        )

        self.assertIsInstance(results, dict)


if __name__ == '__main__':
    unittest.main()
