"""Advanced TensorFlow Lab: Custom Weight Modification and Adaptive Loss Functions

This lab implements neural network training with binary weight constraints, 
oscillation dampening, adaptive loss functions, and performance tracking 
for particle physics simulations."""

import os, sys, time, warning
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def msaeRmseMaeR2_score(y_test, y_pred)          
    diff_abs=np.abs(y_test - y_pred)
    mse, mae = float(np.mean((diff_abs) ** 2)), float(np.mean(diff_abs))
    rmse = float(np.sqrt(mse))
    # R² score
    ss_res, ss_tot = np.sum((diff_abs) ** 2), np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))  if ss_tot==0 else r2_score = 1 - (ss_res / (ss_tot))
    return mse, mae, rmse, r2_score     

# Import custom modules and tensorflow
try:
    import tensorflow as tf
    from weight_constraints import (BinaryWeightConstraintChanges, BinaryWeightConstraintMax, OscillationDampener)
    from performance_tracker import PerformanceTracker
    from data_loader import load_and_prepare_data
    from adaptive_loss import AdaptiveLossFunction, epoch_weighted_loss, accuracy_weighted_loss, loss_weighted_loss
    print("✓ All modules loaded successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class AdvancedNeuralNetwork:
    """ Neural network with custom weight constraints and adaptive loss functions.
    Implements railway-style error handling for robust execution. """
    
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
            return AdaptiveLossFunction(weighting_strategy=self.config.get('loss_weighting_strategy', 'epoch_based'))
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
        
        model = tf.keras.Sequential([ tf.keras.layers.Dense(hidden_layers[0], activation=activation, input_shape=self.input_shape),
            tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else tf.keras.layers.Lambda(lambda x: x),])
        
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
                    loss_strategy = self.adaptive_loss.get_current_strategy_info()
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
        print(f"\n=== Starting Training === \n, Epochs: {epochs}, Batch size: {batch_size}")
        
        # Start training tracking
        training_config = {
            'epochs': epochs, 'batch_size': batch_size, 'model_architecture': self.config,
            'adaptive_loss_strategy': self.adaptive_loss.weighting_strategy if self.adaptive_loss else 'none'}
        if self.performance_tracker: self.performance_tracker.start_training(training_config)
        
        # Training history
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 'epoch_time': [], 'applied_constraints': [], 'loss_strategies': []}
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
                val_loss, val_mae, rmse, accuracy=msaeRmseMaeR2_score(y_test, y_pred)     
            except Exception as e:
                val_loss, val_mae, accruacy = float(np.mean(epoch_losses)), float(np.mean(epoch_mae)), accuracy = 0.0
            
            # Update adaptive loss function
            if self.adaptive_loss: self.adaptive_loss.update_epoch(epoch, accuracy)
            
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
        if self.adaptive_loss: adaptive_loss_history = self.adaptive_loss.get_loss_history()
        return {'history': history,
            'final_results': final_results, 'adaptive_loss_history': adaptive_loss_history, 'errors': self.errors,
            'successful_constraints': list(set([c for sublist in history['applied_constraints'] for c in sublist]))}
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        try:
            # Measure inference time
            inference_time = 0.0
            if self.performance_tracker: inference_time = self.performance_tracker.measure_inference_time(self.model, X_test, num_runs=10)
            # Make predictions, calculate metrics
            y_pred = self.model.predict(X_test, verbose=0)
            mse, mae, rmse, r2_score =msaeRmseMaeR2_score(y_test, y_pred)     
            return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2_score, 'inference_time': inference_time} 
        except Exception as e:
            self.errors.append(f"Model evaluation failed: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'r2_score': -1.0, 'inference_time': 0.0}
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors encountered."""
        error_counts = {}
        for error in self.errors:
            error_type = error.split(':')[0] if ':' in error else 'general'
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return {'total_errors': len(self.errors),
            'error_breakdown': error_counts, 'recent_errors': self.errors[-5:] if self.errors else [],
            'all_errors': self.errors}

def create_model(input_shape: Tuple[int], output_shape: int = 6, config: Optional[Dict[str, Any]] = None) -> AdvancedNeuralNetwork:
    """Create a neural network model with custom constraints."""
    if config is None:
        config = { 'hidden_layers': [64, 32, 16],
            'activation': 'relu', 'dropout_rate': 0.2, 'optimizer': 'adam', 'learning_rate': 0.001,
            'max_binary_digits': 5, 'max_additional_binary_digits': 1, 'oscillation_window': 3, 'loss_weighting_strategy': 'epoch_based',
            'output_dir': 'training_output'}
    return AdvancedNeuralNetwork(input_shape, output_shape, config)
def train_with_tracking(model: AdvancedNeuralNetwork, 
                       X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, 
                       y_val: np.ndarray, y_test: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Complete training pipeline with comprehensive tracking."""
    print("=== Starting Training Pipeline === /n")
    # Compile model
    try:
        model.compile_model()
    except Exception as e:
        return {'error': 'Model compilation failed', 'details': str(e)}   
    # Train with custom constraints
    training_results = model.train_with_custom_constraints(X_train, y_train, X_val, y_val,
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 32))
    # Evaluate on test set and save results
    test_results = model.evaluate_model(X_test, y_test)
    if model.performance_tracker:
        try:
            model.performance_tracker.save_results()
            print("✓ Performance results saved")
        except Exception as e:
            print(f"⚠ Failed to save performance results: {e}") 
    # Save loss history
    if 'adaptive_loss_history' in training_results and training_results['adaptive_loss_history']:
        try:
            loss_history = training_results['adaptive_loss_history'].get('losses', [])
            if loss_history and model.performance_tracker:
                model.performance_tracker.create_loss_history_csv(loss_history)
                print("✓ Loss history saved")
        except Exception as e:
            print(f"⚠ Failed to save loss history: {e}")
    # Save final model
    try:
        model.model.save_weights('model_weights.weights.h5')
        if model.performance_tracker: model.performance_tracker.record_weight_file_size('model_weights.weights.h5')
        print("✓ Final model weights saved")
    except Exception as e:
        print(f"⚠ Failed to save final model: {e}")
    # Get error summary
    error_summary = model.get_error_summary()
    return {'training': training_results, 'test': test_results,
        'performance_summary': model.performance_tracker.get_summary() if model.performance_tracker else {}, 'error_summary': error_summary}
def demonstrate_individual_components():
    """Demonstrate individual components with simplified error handling."""
    print("\n=== Demonstrating Individual Components ===, \n--- Binary Weight Constraint Changes ---")
     # Test binary weight constraints
    try:
        constraint_changes = BinaryWeightConstraintChanges(max_additional_digits=1)
        previous_weight = 0.625  # 1.010 in binary
        current_weights = np.array([[0.875], [0.6875], [1.125], [0.5625]])
        print(f"Previous weight: {previous_weight} /n Testing weight constraint changes:")
        # Initialize with previous weight
        constraint_changes.apply_constraint(np.array([[previous_weight]]))
        for i, weight in enumerate(current_weights.flatten()):
            test_array = np.array([[weight]])
            constrained = constraint_changes.apply_constraint(test_array)
            print(f"  Original: {weight:.4f} -> Constrained: {constrained[0,0]:.4f}")
        print(f"✓ Binary constraint changes test completed")
    except Exception as e:
        print(f"✗ Binary constraint changes test failed: {e}")
    # Test binary max constraints
    print("\n--- Binary Weight Constraint Max ---")
    try:
        constraint_max = BinaryWeightConstraintMax(max_binary_digits=3)
        test_weights = np.array([[0.125, 0.875], [1.5, 0.75]])
        print("Original weights: /n, test_weights")
        constrained_weights = constraint_max.apply_constraint(test_weights)
        print("Constrained weights: /n", constrained_weights), "✓ Binary constraint max test completed")
    except Exception as e:
        print(f"✗ Binary constraint max test failed: {e}")
    # Test oscillation dampening
    print("\n--- Oscillation Dampening ---")
    try:
        dampener = OscillationDampener(window_size=3)
        # Simulate oscillating weights
        weights_sequence = [np.array([[0.5]]), np.array([[0.7]]), np.array([[0.4]]), np.array([[0.8]])]
        print("Weight sequence (simulating oscillation):")
        for i, weights in enumerate(weights_sequence):
            dampener.add_weights(weights)
            dampened = dampener.detect_and_dampen_oscillations(weights)
            print(f"  Step {i+1}: Original: {weights[0,0]:.3f} -> Dampened: {dampened[0,0]:.3f}")
        print("✓ Oscillation dampening test completed")   
    except Exception as e:
        print(f"✗ Oscillation dampening test failed: {e}")
    # Test adaptive loss functions
    print("\n--- Adaptive Loss Functions ---")
    try:
        mse_loss, mae_loss = 0.15, 0.12
        print(f"Base losses - MSE: {mse_loss}, MAE: {mae_loss} /n Epoch-based weighting:")
        # Test epoch-based weighting
        for epoch in [5, 15, 25, 35]:
            combined_loss = epoch_weighted_loss(epoch, mse_loss, mae_loss)
            print(f"  Epoch {epoch:2d}: Combined loss = {combined_loss:.4f}")
        # Test accuracy-based weighting
        print("Accuracy-based weighting:")
        for accuracy in [0.3, 0.6, 0.8, 0.95]:
            combined_loss = accuracy_weighted_loss(accuracy, mse_loss, mae_loss)
            print(f"  Accuracy {accuracy:.2f}: Combined loss = {combined_loss:.4f}")
        # Test loss-based weighting
        print("Loss-based weighting:")
        for prev_loss in [2.0, 0.5, 0.1, 0.05]:
            combined_loss = loss_weighted_loss(prev_loss, mse_loss, mae_loss)
            print(f"  Prev loss {prev_loss:.2f}: Combined loss = {combined_loss:.4f}")
        print("✓ Adaptive loss functions test completed")
    except Exception as e:
        print(f"✗ Adaptive loss functions test failed: {e}")
def main():
    """Main function to run the complete TensorFlow lab."""
    print("=" * 60, "/n ADVANCED TENSORFLOW LAB: CUSTOM WEIGHT MODIFICATION, /n", "=" * 60)
    # Demonstrate individual components first
    demonstrate_individual_components()
    print("\n" + "=" * 40, " /n LOADING PARTICLE DATA /n ", "=" * 40)
    # Load and prepare data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, data_summary = load_and_prepare_data()
        print(f"✓ Data loaded successfully:")
        print(f"  Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]} -> {y_train.shape[1]}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    print("\n" + "=" * 40, "/n CREATING AND TRAINING MODEL", "=" * 40)
    
    # Create model with configuration
    model_config = {'hidden_layers': [64, 32, 16], 'activation': 'relu', 'dropout_rate': 0.2, 'optimizer': 'adam',
        'learning_rate': 0.001, 'max_binary_digits': 5, 'max_additional_binary_digits': 1,
        'oscillation_window': 3, 'loss_weighting_strategy': 'combined', 'output_dir': 'training_output','enable_weight_constraints': True} 
    try:
        model = create_model( input_shape=(X_train.shape[1],),  output_shape=y_train.shape[1], config=model_config)
        print("✓ Neural network created successfully") 
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    # Training configuration
    training_config = {'epochs': 30, 'batch_size': 16}
    print(f"\nTraining configuration: {training_config}, \n🚀 Starting training...")
    try:
        results = train_with_tracking( model, X_train, X_val, X_test, y_train, y_val, y_test, training_config)
        if 'error' in results:
            print(f"✗ Training failed: {results['error']}")
            return
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        return
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY /n", "=" * 40)
    # Display results
    try:
        performance_summary = results.get('performance_summary', {})
        test_results = results.get('test', {})
        error_summary = results.get('error_summary', {})
        print(f"📊 Performance Metrics:")
        print(f"  Final accuracy: {performance_summary.get('current_accuracy', 'N/A'):.4f}")
        print(f"  Best accuracy: {performance_summary.get('best_accuracy', 'N/A'):.4f} "
              f"at epoch {performance_summary.get('best_accuracy_epoch', 'N/A')}")
        print(f"  Average epoch time: {performance_summary.get('avg_epoch_time', 'N/A'):.2f}s")
        print(f"  Peak memory: {performance_summary.get('peak_memory_mb', 'N/A'):.1f} MB")
        
        print(f"\n🧪 Test Results:")
        print(f"  MSE: {test_results.get('mse', 'N/A'):.4f}")
        print(f"  MAE: {test_results.get('mae', 'N/A'):.4f}")
        print(f"  R² Score: {test_results.get('r2_score', 'N/A'):.4f}")
        print(f"  Inference time: {test_results.get('inference_time', 'N/A'):.4f}s")
        
        print(f"\n🔧 Weight Modifications:")
        weight_mods = performance_summary.get('weight_modifications_used', [])
        if weight_mods:
            for mod in weight_mods:
                print(f"  ✓ {mod}")
        else:
            print(f"  None applied")
        
        print(f"\n📈 Adaptive Loss Strategy:")
        adaptive_strategy = performance_summary.get('adaptive_loss_strategy', 'none')
        print(f"  Strategy: {adaptive_strategy}", \n⚠ Errors: {error_summary.get('total_errors', 0)}")
        
    except Exception as e:
        print(f"✗ Error displaying results: {e}")
    print("\n" + "=" * 40, "OUTPUT FILES \n", "=" * 40)
    # Check for output files
    output_files = [
        'training_output/training_results.csv', 'training_output/loss_history.csv',
        'training_output/training_log.txt','training_output/configuration_log.csv',
        'model_weights.weights.h5', 'particle_data.csv']
    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({file_size:,} bytes)")
        else: print(f"  ✗ {file_path} (not found)")
    
    print("\n" + "=" * 60)
    print("🎉 ADVANCED TENSORFLOW LAB COMPLETED!")
    print("=" * 60, "\n📁 Check 'training_output' directory for detailed results.")
    print("🔬 Lab demonstrated:")
    print("   • Binary weight precision constraints")
    print("   • Oscillation dampening for weight stability")
    print("   • Adaptive loss function combinations")
    print("   • Comprehensive performance tracking")
    print("   • Railway-style error handling")
    print("   • CSV data processing for particle physics simulations")
    
    # Final success/failure summary
    total_errors = results.get('error_summary', {}).get('total_errors', 0)
    if total_errors == 0: print("\n🏆 Lab completed with NO ERRORS!")
    elif total_errors < 5: print(f"\n⚠ Lab completed with {total_errors} minor errors")
    else: print(f"\n⚠ Lab completed with {total_errors} errors")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
    finally:
        print("\nThank you for using the Advanced TensorFlow Lab! 🧪🔬")
