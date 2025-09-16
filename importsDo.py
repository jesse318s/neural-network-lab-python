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
