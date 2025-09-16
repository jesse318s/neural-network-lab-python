import os, sys, time, warnings
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
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

def msaeRmseMaeR2_score(y_test, y_pred):
    """
    Calculate MSE, MAE, RMSE, and R² score using Keras built-in functions.
    
    Args:
        y_test: True values
        y_pred: Predicted values
        
    Returns:
        tuple: (mse, mae, rmse, r2_score) as floats
    """
    # Import required modules
    from sklearn.metrics import r2_score
    
    # Convert to TensorFlow tensors for consistent data types
    y_true_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    # Use Keras built-in metrics
    mse_metric = tf.keras.metrics.MeanSquaredError()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()
    
    # Calculate metrics
    mse_metric.update_state(y_true_tf, y_pred_tf)
    mae_metric.update_state(y_true_tf, y_pred_tf)
    rmse_metric.update_state(y_true_tf, y_pred_tf)
    
    mse = float(mse_metric.result().numpy())
    mae = float(mae_metric.result().numpy())
    rmse = float(rmse_metric.result().numpy())
    
    # Use sklearn for R² score as Keras doesn't have a built-in R² metric
    # Handle edge case where R² is undefined (e.g., single sample)
    try:
        r2 = float(r2_score(y_test, y_pred))
        if np.isnan(r2):
            r2 = 0.0
    except ValueError:
        r2 = 0.0
    
    return mse, mae, rmse, r2 
