"""
Advanced TensorFlow Lab: Custom Weight Modification and Adaptive Loss Functions

This lab implements neural network training with binary weight constraints, 
oscillation dampening, adaptive loss functions, and performance tracking 
for particle physics simulations.
"""

import os
import numpy as np
from typing import Dict, Tuple, Any, Optional

# Import custom modules
from advanced_neural_network import AdvancedNeuralNetwork
from data_processing import complete_data_pipeline


def create_model(input_shape: Tuple[int], output_shape: int = 6, config: Optional[Dict[str, Any]] = None) -> AdvancedNeuralNetwork:
    """Create a neural network model with custom constraints."""
    if config is None:
        config = { 'hidden_layers': [64, 32, 16],
            'activation': 'relu', 'dropout_rate': 0.2, 'optimizer': 'adam', 'learning_rate': 0.001,
            'max_binary_digits': 5, 'max_additional_binary_digits': 1, 'oscillation_window': 3, 
            'loss_weighting_strategy': 'epoch_based', 'output_dir': 'training_output'}

    return AdvancedNeuralNetwork(input_shape, output_shape, config)


def train_with_tracking(model: AdvancedNeuralNetwork, 
                       X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, 
                       y_val: np.ndarray, y_test: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Complete training with comprehensive tracking."""
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
        'performance_summary': model.performance_tracker.get_summary() if model.performance_tracker else {}, 
        'error_summary': error_summary}


def main():
    """Main function to run the complete TensorFlow lab."""
    print("=" * 60)
    print("ADVANCED TENSORFLOW LAB")
    print("=" * 60)
    print("\n" + "=" * 40)
    print("LOADING PARTICLE DATA")
    print("=" * 40)

    # Load and prepare data
    try:
        data_splits, pipeline_info = complete_data_pipeline(num_particles=1000)
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
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
        'learning_rate': 0.001, 'max_binary_digits': 5, 'max_additional_binary_digits': 1, 'oscillation_window': 3,
        'loss_weighting_strategy': 'combined', 'output_dir': 'training_output', 'enable_weight_constraints': True}
    
    try:
        model = create_model(input_shape=(X_train.shape[1],), output_shape=y_train.shape[1], config=model_config)
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
    print("🎉 ADVANCED TENSORFLOW LAB COMPLETED!")
    print("=" * 60)
    print("📁 Check 'training_output' directory for detailed results.\n")
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
