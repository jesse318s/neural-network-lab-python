import os
import time
from typing import Dict, List, Tuple, Any
import numpy as np

# Suppress TensorFlow warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Import TensorFlow with Keras
import tensorflow as tf

# Import custom modules
from weight_constraints import BinaryWeightConstraintChanges, BinaryWeightConstraintMax, OscillationDampener
from performance_tracker import PerformanceTracker
from ml_utils import create_adaptive_loss_fn


class AdvancedNeuralNetwork:
    """ 
    Neural network with custom weight constraints and adaptive loss functions.
    Implements railway-style error handling for robust execution. 
    """
    
    def __init__(self, input_shape: Tuple[int], output_shape: int, config: Dict[str, Any]):
        self.input_shape, self.output_shape  = input_shape, output_shape
        self.config, self.errors = config, []
        # Initialize components with error handling
        self.binary_constraint_changes = self._init_binary_changes()
        self.binary_constraint_max = self._init_binary_max()
        self.oscillation_dampener = self._init_oscillation_dampener()
        self.adaptive_loss = self._init_adaptive_loss()
        self.performance_tracker = self._init_performance_tracker()
        # Build model
        self.model, self.epoch_count = self._build_model(), 0
        # Data pipeline state
        self.data_pipeline_info = None
        
    def _init_binary_changes(self):
        """Initialize binary weight constraint for changes."""
        try:
            if self.config.get('enable_weight_constraints', True): return BinaryWeightConstraintChanges(max_additional_digits=self.config.get('max_additional_binary_digits', 1))
            return None
        except Exception as e:
            self.errors.append(f"Binary constraint changes failed: {e}")
            return None
    
    def _init_binary_max(self):
        """Initialize binary weight constraint for max precision."""
        try:
            if self.config.get('enable_weight_constraints', True): return BinaryWeightConstraintMax(max_binary_digits=self.config.get('max_binary_digits', 5))
            return None
        except Exception as e:
            self.errors.append(f"Binary constraint max failed: {e}")
            return None
    
    def _init_oscillation_dampener(self):
        """Initialize oscillation dampener."""
        try:
            if self.config.get('enable_weight_constraints', True): return OscillationDampener(window_size=self.config.get('oscillation_window', 3))
            return None
        except Exception as e:
            self.errors.append(f"Oscillation dampener failed: {e}")
            return None
    
    def _init_adaptive_loss(self):
        """Initialize adaptive loss function."""
        try:
            return create_adaptive_loss_fn(strategy=self.config.get('loss_weighting_strategy', 'epoch_based'))
        except Exception as e:
            self.errors.append(f"Adaptive loss failed: {e}")
            return None
            
    def _init_performance_tracker(self):
        """Initialize performance tracker."""
        try:
            return PerformanceTracker(output_dir=self.config.get('output_dir', 'training_output'))
        except Exception as e:
            self.errors.append(f"Performance tracker failed: {e}")
            return None
    
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model."""
        hidden_layers = self.config.get('hidden_layers', [64, 32, 16])
        activation = self.config.get('activation', 'relu')
        dropout_rate = self.config.get('dropout_rate', 0.2)  
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Dense(hidden_layers[0], activation=activation),
            tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else tf.keras.layers.Lambda(lambda x: x),
        ])
        
        for units in hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(units, activation=activation))

            if dropout_rate > 0: model.add(tf.keras.layers.Dropout(dropout_rate))
        
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear'))
        return model
    
    def compile_model(self) -> None:
        """Compile the model."""
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer_map = {'adam': tf.keras.optimizers.Adam,'sgd': tf.keras.optimizers.SGD,'rmsprop': tf.keras.optimizers.RMSprop}
        optimizer_class = optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)
        optimizer = optimizer_class(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    
    def apply_weight_constraints(self) -> List[str]:
        """Apply custom weight constraints to model weights."""
        applied_constraints = []
        try:
            for layer in self.model.layers:
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    weights = layer.get_weights()
                    modified_weights = []

                    for weight_matrix in weights:
                        current_weight = weight_matrix.copy()

                        # Apply constraints only to 2D weight matrices (not bias vectors)
                        if len(weight_matrix.shape) == 2:
                            if self.binary_constraint_changes:
                                try:
                                    current_weight = self.binary_constraint_changes.apply_constraint(current_weight)
                                    
                                    if 'binary_changes' not in applied_constraints: applied_constraints.append('binary_changes')
                                except Exception:
                                    pass
                            if self.binary_constraint_max:
                                try:
                                    current_weight = self.binary_constraint_max.apply_constraint(current_weight)
                                    
                                    if 'binary_max' not in applied_constraints:
                                        applied_constraints.append('binary_max')
                                except Exception:
                                    pass
                            
                            if self.oscillation_dampener:
                                try:
                                    self.oscillation_dampener.add_weights(current_weight)
                                    current_weight = self.oscillation_dampener.detect_and_dampen_oscillations(current_weight)
                                    
                                    if 'oscillation_dampening' not in applied_constraints: applied_constraints.append('oscillation_dampening')
                                except Exception:
                                    pass
                        modified_weights.append(current_weight)
                    layer.set_weights(modified_weights)
        except Exception as e:
            self.errors.append(f"Weight constraint application failed: {e}")
        
        return applied_constraints
    
    def custom_training_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, Any]:
        """Perform a custom training step with adaptive loss."""
        try:
            with tf.GradientTape() as tape:
                y_pred = self.model(X_batch, training=True)

                if self.adaptive_loss:
                    loss_value = self.adaptive_loss(y_batch, y_pred)
                    loss_strategy = self.adaptive_loss.get_current_info()
                else:
                    loss_value = tf.reduce_mean(tf.square(y_batch - y_pred))
                    loss_strategy = "mse_only"
            
            gradients = tape.gradient(loss_value, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            mse, mae = tf.reduce_mean(tf.square(y_batch - y_pred)), tf.reduce_mean(tf.abs(y_batch - y_pred))
            return {'loss': float(loss_value.numpy()),
                'mse': float(mse.numpy()),
                'mae': float(mae.numpy()),
                'loss_strategy': loss_strategy}
        except Exception as e:
            self.errors.append(f"Training step failed: {e}")
            return {'loss': 1.0, 'mse': 1.0, 'mae': 1.0, 'loss_strategy': 'error_fallback'}
    
    def train_with_custom_constraints(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                                    epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the model with custom weight constraints and adaptive loss."""
        # Start training tracking
        training_config = {
            'epochs': epochs, 'batch_size': batch_size, 'model_architecture': self.config,
            'adaptive_loss_strategy': self.config.get('loss_weighting_strategy', 'none') if self.adaptive_loss else 'none'}
        
        if self.performance_tracker: self.performance_tracker.start_training(training_config)
        
        # Training history
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 
                   'epoch_time': [], 'applied_constraints': [], 'loss_strategies': []}
        num_batches = max(1, len(X_train) // batch_size)
        
        for epoch in range(epochs):
            if self.performance_tracker: self.performance_tracker.start_epoch(epoch)
            
            epoch_start_time = time.time()
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled, y_train_shuffled  = X_train[indices], y_train[indices]
            epoch_losses, epoch_mse, epoch_mae, epoch_strategies = [], [], [], []
           
            # Training loop
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                X_batch, y_batch = X_train_shuffled[start_idx:end_idx], y_train_shuffled[start_idx:end_idx]
                metrics = self.custom_training_step(X_batch, y_batch)
                epoch_losses.append(metrics['loss']); epoch_mse.append(metrics['mse'])
                epoch_mae.append(metrics['mae']); epoch_strategies.append(metrics['loss_strategy'])
            
            # Apply weight constraints
            applied_constraints = self.apply_weight_constraints()

            # Update performance tracker
            for constraint in applied_constraints:
                if self.performance_tracker: self.performance_tracker.add_weight_modification(constraint)
            
            # Validation
            try:
                val_pred = self.model.predict(X_val, verbose=0)
                val_loss, val_mae, rmse, accuracy = self.calculate_regression_metrics(y_val, val_pred)     
            except Exception as e:
                val_loss, val_mae, accuracy = float(np.mean(epoch_losses)), float(np.mean(epoch_mae)), 0.0
            
            # Update adaptive loss function
            if self.adaptive_loss: self.adaptive_loss.update_state(epoch, accuracy)
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            epoch_loss, epoch_mae_val = np.mean(epoch_losses), np.mean(epoch_mae)
            history['loss'].append(epoch_loss); history['val_loss'].append(val_loss)
            history['mae'].append(epoch_mae_val); history['val_mae'].append(val_mae)
            history['epoch_time'].append(epoch_time); history['applied_constraints'].append(applied_constraints)
            history['loss_strategies'].append(epoch_strategies[0] if epoch_strategies else 'unknown')        
            # Update performance tracker
            logs = {'loss': epoch_loss, 'val_loss': val_loss, 'mae': epoch_mae_val, 'val_mae': val_mae, 'accuracy': accuracy}
            
            if self.performance_tracker: self.performance_tracker.end_epoch(epoch, logs)
            
            # Save weights periodically
            if epoch % 10 == 0 or epoch == epochs - 1:
                try:
                    weight_file = f"model_weights_epoch_{epoch}.weights.h5"
                    self.model.save_weights(weight_file)

                    if self.performance_tracker: self.performance_tracker.record_weight_file_size(weight_file)
                except Exception:
                    pass
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs} - Loss: {epoch_loss:.4f}, "f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                if applied_constraints: print(f"    Constraints: {', '.join(applied_constraints)}")
        # End training
        final_results = {}

        if self.performance_tracker:final_results = self.performance_tracker.end_training()
        
        # Get adaptive loss history
        adaptive_loss_history = {}

        if self.adaptive_loss: adaptive_loss_history = self.adaptive_loss.get_history()

        return {'history': history, 'final_results': final_results, 'adaptive_loss_history': adaptive_loss_history, 'errors': self.errors,
            'successful_constraints': list(set([c for sublist in history['applied_constraints'] for c in sublist]))}
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        try:
            # Measure inference time
            inference_time = 0.0

            if self.performance_tracker: inference_time = self.performance_tracker.measure_inference_time(self.model, X_test, num_runs=10)
            
            # Make predictions, calculate metrics
            y_pred = self.model.predict(X_test, verbose=0)
            mse, mae, rmse, r2_score = self.calculate_regression_metrics(y_test, y_pred)     
            return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2_score, 'inference_time': inference_time} 
        except Exception as e:
            self.errors.append(f"Model evaluation failed: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'r2_score': -1.0, 'inference_time': 0.0}
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data pipeline."""
        return self.data_pipeline_info or {'status': 'No data loaded yet'}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors encountered."""
        error_counts = {}

        for error in self.errors:
            error_type = error.split(':')[0] if ':' in error else 'general'
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {'total_errors': len(self.errors), 'error_breakdown': error_counts, 'recent_errors': self.errors[-5:] if self.errors else [],'all_errors': self.errors}
    
    @staticmethod
    def calculate_regression_metrics(y_test, y_pred):
        """Calculate MSE, MAE, RMSE, and R² score efficiently."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Convert to numpy arrays
        y_test, y_pred = np.asarray(y_test, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)
        # Calculate metrics
        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 0.0
        return mse, mae, rmse, r2
