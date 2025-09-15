import os, sys, time, warning
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

def msaeRmseMaeR2_score(y_test, y_pred)          
    diff_abs=np.abs(y_test - y_pred)
    mse, mae = float(np.mean((diff_abs) ** 2)), float(np.mean(diff_abs))
    rmse = float(np.sqrt(mse))
    # R² score
    ss_res, ss_tot = np.sum((diff_abs) ** 2), np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))  if ss_tot==0 else r2_score = 1 - (ss_res / (ss_tot))
    return mse, mae, rmse, r2_score 
