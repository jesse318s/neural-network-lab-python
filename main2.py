"""Advanced TensorFlow Lab: Custom Weight Modification and Adaptive Loss Functions

This lab implements neural network training with binary weight constraints, 
oscillation dampening, adaptive l        print(f"📈 Adaptive Loss Strategy:")
        adaptive_strategy = performance_summary.get('adaptive_loss_strategy', 'none')
        print(f"  Strategy: {adaptive_strategy}")
        print(f"\n⚠ Errors: {error_summary.get('total_errors', 0)}") functions, and performance tracking 
for particle physics simulations."""

from import_manager import setup_environment_and_imports

# Set up environment and imports for main2.py
imports = setup_environment_and_imports()
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from advancedNeuralNetwork import AdvancedNeuralNetwork
# Use imported objects from import_manager
BinaryWeightConstraintChanges = imports['BinaryWeightConstraintChanges']
BinaryWeightConstraintMax = imports['BinaryWeightConstraintMax']
OscillationDampener = imports['OscillationDampener']
PerformanceTracker = imports['PerformanceTracker']
load_and_prepare_data = imports['load_and_prepare_data']
epoch_weighted_loss = imports['epoch_weighted_loss']
accuracy_weighted_loss = imports['accuracy_weighted_loss']
loss_weighted_loss = imports['loss_weighted_loss']
  
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
    print("=== Starting Training Pipeline ===\n")
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
        print(f"Previous weight: {previous_weight}\nTesting weight constraint changes:")
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
        print("Original weights:\n", test_weights)
        constrained_weights = constraint_max.apply_constraint(test_weights)
        print("Constrained weights:\n", constrained_weights)
        print("✓ Binary constraint max test completed")
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
        print(f"Base losses - MSE: {mse_loss}, MAE: {mae_loss}\nEpoch-based weighting:")
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
    print("=" * 60)
    print("ADVANCED TENSORFLOW LAB: CUSTOM WEIGHT MODIFICATION")
    print("=" * 60)
    # Demonstrate individual components first
    demonstrate_individual_components()
    print("\n" + "=" * 40)
    print("LOADING PARTICLE DATA")
    print("=" * 40)
    # Load and prepare data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, data_summary = load_and_prepare_data()
        print(f"✓ Data loaded successfully:")
        print(f"  Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]} -> {y_train.shape[1]}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    print("\n" + "=" * 40)
    print("CREATING AND TRAINING MODEL")
    print("=" * 40)
    
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
    print("RESULTS SUMMARY")
    print("=" * 40)
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
        else: print(f"  None applied")
        print(f"\n📈 Adaptive Loss Strategy:")
        adaptive_strategy = performance_summary.get('adaptive_loss_strategy', 'none')
        print(f"  Strategy: {adaptive_strategy}")
        print(f"\n⚠ Errors: {error_summary.get('total_errors', 0)}")
        
    except Exception as e:
        print(f"✗ Error displaying results: {e}")
    print("\n" + "=" * 40)
    print("OUTPUT FILES")
    print("=" * 40)
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
    print("\n🎉 ADVANCED TENSORFLOW LAB COMPLETED!")
    print("=" * 60)
    print("📁 Check 'training_output' directory for detailed results.")
    print("🔬 Lab demonstrated:")
    print("   • Binary weight precision constraints")
    print("   • Oscillation dampening for weight stability")
    print("   • Adaptive loss function combinations")
    print("   • Comprehensive performance tracking")
    print("   • Railway-style error handling")
    print("   • CSV data processing for particle physics simulations")
    
    # Final success/failure summary
    total_errors = results.get('error_summary', {}).get('total_errors', 0)
    if total_errors == 0:
        print("\n🏆 Lab completed with NO ERRORS!")
    else:
        print(f"\n⚠ Lab completed with {total_errors} minor errors")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
    finally:
        print("\nThank you for using the Advanced TensorFlow Lab! 🧪🔬")
