"""
Advanced TensorFlow Lab: Custom Weight Modification and Adaptive Loss Functions

This lab implements neural network training with binary weight constraints, 
oscillation dampening, adaptive loss functions, and performance tracking 
for particle physics simulations.
"""

import json
import numpy as np
from typing import Dict, Any

# Import custom modules
from advanced_neural_network import AdvancedNeuralNetwork
from data_processing import complete_data_pipeline


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
        epochs=config.get('epochs', 30), batch_size=config.get('batch_size', 16))
    # Evaluate on test set
    test_results = model.evaluate_model(X_test, y_test)

    # Save training results
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

    # Get error summary
    error_summary = model.get_error_summary()
    return {'training': training_results, 'test': test_results,
        'performance_summary': model.performance_tracker.get_summary() if model.performance_tracker else {}, 
        'error_summary': error_summary}


def display_results(results: Dict[str, Any]) -> None:
    """Display a summary of training and evaluation results."""
    try:
        performance_summary = results.get('performance_summary', {})
        test_results = results.get('test', {})
        error_summary = results.get('error_summary', {})

        print(f"📊 Performance Metrics:")
        print(f"  Final R²: {performance_summary.get('current_r2', 'N/A'):.4f}")
        print(f"  Best R²: {performance_summary.get('best_r2', 'N/A'):.4f} "
              f"at epoch {performance_summary.get('best_r2_epoch', 'N/A')}")
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
        
        print(f"\n📈 Loss Weighting Strategy:")
        loss_strategy = performance_summary.get('loss_weighting_strategy', 'none')
        print(f"  Strategy: {loss_strategy}")
        print(f"\n⚠ Errors: {error_summary.get('total_errors', 0)}")
    except Exception as e:
        print(f"✗ Error displaying results: {e}")


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
        data_splits = complete_data_pipeline(num_particles=1000)
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

    model_config = {}
    training_config = {}

    # Load model config and training config
    try:
        with open('ml_config/model_config.json', 'r') as f:
            model_config = json.load(f)

        with open('ml_config/training_config.json', 'r') as f:
            training_config = json.load(f)
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")

    # Create model with configuration
    try:
        model = AdvancedNeuralNetwork((X_train.shape[1],), y_train.shape[1], model_config)
        print("✓ Neural network created successfully") 
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    # Train model with configuration
    print(f"\nTraining configuration: {training_config}, \n🚀 Starting training...")
    
    try:
        results = train_with_tracking(model, X_train, X_val, X_test, y_train, y_val, y_test, training_config)
        
        if 'error' in results:
            print(f"✗ Training failed: {results['error']}")
            return
        
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        return
    
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    # Display results
    display_results(results)  
    print("\n" + "=" * 60)
    print("🎉 ADVANCED TENSORFLOW LAB COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user\n\n")
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}\n\n")
    finally:
        print("Thank you for using the Advanced TensorFlow Lab! 🧪🔬")
