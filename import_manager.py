import os, sys, warnings
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def setup_environment_and_imports():
    """
    Set up environment variables, suppress warnings, and import required modules for main2.py.
    Returns a dictionary of imported modules and classes for use in main2.py.
    """
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')

    # Try importing TensorFlow and custom modules
    try:
        import tensorflow as tf
        from weight_constraints import (BinaryWeightConstraintChanges, BinaryWeightConstraintMax, OscillationDampener)
        from performance_tracker import PerformanceTracker
        from data_loader import load_and_prepare_data
        from adaptive_loss import AdaptiveLossFunction, epoch_weighted_loss, accuracy_weighted_loss, loss_weighted_loss

        # Return all imported objects in a dictionary
        return {
            'tf': tf,
            'BinaryWeightConstraintChanges': BinaryWeightConstraintChanges,
            'BinaryWeightConstraintMax': BinaryWeightConstraintMax,
            'OscillationDampener': OscillationDampener,
            'PerformanceTracker': PerformanceTracker,
            'load_and_prepare_data': load_and_prepare_data,
            'AdaptiveLossFunction': AdaptiveLossFunction,
            'epoch_weighted_loss': epoch_weighted_loss,
            'accuracy_weighted_loss': accuracy_weighted_loss,
            'loss_weighted_loss': loss_weighted_loss
        }
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
