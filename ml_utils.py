"""
ML Utilities for Neural Network Training

This module combines adaptive loss functions and state management to enhance model training.
Includes physics-based loss functions and adaptive curve-fitting loss strategies.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Dict, Tuple, Optional, Any, List


# ============================================================================
# HELPER FUNCTIONS FOR ADAPTIVE LOSS
# ============================================================================

def unit_vec(x: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        x: Input vector or array
        
    Returns:
        Normalized unit vector
    """
    try:
        squared = np.square(x)
        scale = np.sum(squared)
        
        if scale == 0: return x
        
        return np.divide(x, np.sqrt(scale))
    except Exception:
        return x


def one_maker(x: np.ndarray) -> np.ndarray:
    """
    Makes all weights add up to one (or negative one) assuming all weights have the same sign.
    
    Args:
        x: Input array of weights
        
    Returns:
        Normalized weights that sum to 1
    """
    total = np.sum(np.abs(x))
    
    if total == 0: return x
    
    return x / total


def square_but_preserve_signs(x: float) -> float:
    """
    Square a value while preserving its sign.
    
    Args:
        x: Input value
        
    Returns:
        Squared value with original sign
    """
    if x < 0: return -1 * x**2
    
    return x**2


def squarer_diff_sign_preserver(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Calculate squared difference while preserving signs.
    
    Args:
        x1: First array
        x2: Second array
        
    Returns:
        Squared differences with signs preserved
    """
    diff = np.subtract(x1, x2)
    
    try:
        out = np.array([square_but_preserve_signs(d) for d in diff])
        return out
    except Exception:
        return square_but_preserve_signs(diff)


# ============================================================================
# EPOCH-BASED WEIGHT FUNCTIONS (SINE-BASED)
# ============================================================================

def epoch_weight_sine_for_one_weight(epoch: int, number_of_loss_funcs: int, number: int) -> float:
    """
    Fluctuates weights based on a sine curve, making sure they are always positive.
    
    Args:
        epoch: Current training epoch
        number_of_loss_funcs: Total number of loss functions
        number: Index of this particular loss function
        
    Returns:
        Weight value between 0 and 1
    """
    return (math.sin(epoch + 2 * math.pi * number / number_of_loss_funcs) + 1) / 2


def epoch_weight_sine_based(epoch: int, number_of_loss_funcs: int) -> np.ndarray:
    """
    Generate sine-based weights for all loss functions.
    
    Args:
        epoch: Current training epoch
        number_of_loss_funcs: Total number of loss functions
        
    Returns:
        Array of normalized weights
    """
    try:
        weights_array = []
        
        for number in range(0, number_of_loss_funcs):
            weights_array.append(epoch_weight_sine_for_one_weight(epoch, number_of_loss_funcs, number))
        
        return one_maker(np.array(weights_array))
    except Exception:
        return np.array([1.0])


# ============================================================================
# ADAPTIVE LOSS FUNCTIONS (CURVE FITTING)
# ============================================================================

def adaptive_loss_no_sin(loss_list: List[float], weight_list: List[np.ndarray]) -> np.ndarray:
    """
    Adaptive loss adjustment based on recent performance (without sine component).
    
    If weights are improving, increase bigger weights more and smaller weights less.
    Uses squared differences with sign preservation to explore different phases.
    
    Args:
        loss_list: History of loss values
        weight_list: History of weight arrays
        
    Returns:
        Unit vector for weight adjustment
    """
    if len(loss_list) < 2 or len(weight_list) < 2:
        return np.array([0.0])
    
    diff = loss_list[-1] - loss_list[-2]
    squared_diff = squarer_diff_sign_preserver(weight_list[-1], weight_list[-2])
    unit = unit_vec(squared_diff)
    
    if diff < 0: return -1 * unit
    
    return unit


def curve_fancy(loss_list: List[float], weight_list: List[np.ndarray], number_of_loss_funcs: int) -> np.ndarray:
    """
    Fancy curve fitting for adaptive loss weights.
    
    Combines adaptive gradient-based adjustment with sine-based exploration.
    Ensures no weight becomes zero and all weights remain positive.
    
    Args:
        loss_list: History of loss values
        weight_list: History of weight arrays
        number_of_loss_funcs: Number of loss functions
        
    Returns:
        Normalized weight array
    """
    min_to_do_fancy = number_of_loss_funcs + 2
    epoch = len(loss_list) if isinstance(loss_list, list) else 1
    
    if epoch > min_to_do_fancy:
        # Combine adaptive adjustment with sine exploration
        adaptive_component = adaptive_loss_no_sin(loss_list, weight_list)
        sine_component = epoch_weight_sine_based(epoch, number_of_loss_funcs)
        
        new_weights = np.add(adaptive_component, sine_component)
        
        # Protect only MSE (index 0) from zero - preserves physics
        if len(new_weights) > 0 and new_weights[0] == 0: new_weights[0] = 0.1
        
        # No negative weights
        min_val = np.min(new_weights)
        
        if min_val < 0: new_weights = new_weights - min_val
        
        return one_maker(new_weights)
    else:
        # Use sine-based exploration until we have enough data
        return one_maker(epoch_weight_sine_based(epoch, number_of_loss_funcs))


# ============================================================================
# STANDARD LOSS WEIGHT COMPUTATION
# ============================================================================

def compute_loss_weights(strategy: str, prev_r2: float = 0.5, prev_loss: float = 1.0, 
                        loss_history: Optional[List[float]] = None,
                        weight_history: Optional[List[np.ndarray]] = None,
                        epoch: int = 0) -> Tuple[float, float]:
    """
    Compute adaptive loss function weights based on training progress.
    
    Args:
        strategy: Weighting strategy ('r2_based', 'loss_based', 'combined', 'curve_fancy')
        prev_r2: Previous epoch's R² score
        prev_loss: Previous epoch's loss value
        loss_history: Complete loss history for curve_fancy strategy
        weight_history: Complete weight history for curve_fancy strategy
        epoch: Current epoch number
        
    Returns:
        Tuple of (mse_weight, mae_weight)
    """
    try:
        if strategy == 'r2_based':
            # More MSE as R² improves
            mse_weight = min(0.9, 0.2 + 0.7 * prev_r2)
            mae_weight = max(0.1, 0.8 - 0.7 * prev_r2)
            return mse_weight, mae_weight
        elif strategy == 'loss_based':
            # More MSE as loss decreases
            normalized_loss = min(1.0, max(0.0, prev_loss / 2.0))
            mse_weight = min(0.9, 0.2 + 0.7 * (1 - normalized_loss))
            mae_weight = max(0.1, 0.8 - 0.7 * (1 - normalized_loss))
            return mse_weight, mae_weight       
        elif strategy == 'combined':
            # Average the 2 strategies
            r2_mse, r2_mae = compute_loss_weights('r2_based', prev_r2, prev_loss)
            loss_mse, loss_mae = compute_loss_weights('loss_based', prev_r2, prev_loss)
            final_mse = (r2_mse + loss_mse) / 2
            final_mae = (r2_mae + loss_mae) / 2
            total = final_mse + final_mae
            return (final_mse / total, final_mae / total) if total > 0 else (0.5, 0.5)
        elif strategy == 'curve_fancy':
            # Adaptive curve fitting strategy
            if loss_history is None or weight_history is None or len(loss_history) < 2:
                # Not enough history, use sine-based exploration
                weights = epoch_weight_sine_based(epoch, 2)
                return float(weights[0]), float(weights[1])
            
            # Use curve fancy algorithm
            weights = curve_fancy(loss_history, weight_history, 2)
            
            if len(weights) >= 2: return float(weights[0]), float(weights[1])
            
            return 0.5, 0.5
        else: return 0.5, 0.5       
    except Exception as e:
        print(f"Warning: Error computing loss weights ({strategy}): {e}")
        return 0.5, 0.5


def create_adaptive_loss_fn(strategy: str = 'r2_based'):
    """
    Create an adaptive loss function with state management.
    
    Args:
        strategy: Weighting strategy to use ('r2_based', 'loss_based', 'combined', 'curve_fancy')
        
    Returns:
        Adaptive loss function with update capabilities
    """
    # State variables (using mutable default to maintain state)
    state = {
        'epoch': 0, 
        'prev_r2': 0.5, 
        'prev_loss': 1.0, 
        'history': [], 
        'error_count': 0,
        'loss_history': [],
        'weight_history': []
    }
    
    # Create loss functions
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    def adaptive_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute adaptive loss with current weights."""
        try:
            mse = mse_loss(y_true, y_pred)
            mae = mae_loss(y_true, y_pred)
            
            # Compute weights based on strategy
            mse_weight, mae_weight = compute_loss_weights(
                strategy, 
                state['prev_r2'], 
                state['prev_loss'],
                state['loss_history'],
                state['weight_history'],
                state['epoch']
            )
            
            combined_loss = mse_weight * mse + mae_weight * mae
            
            # Record history
            loss_info = {
                'epoch': state['epoch'], 
                'mse': float(mse.numpy()) if hasattr(mse, 'numpy') else float(mse),
                'mae': float(mae.numpy()) if hasattr(mae, 'numpy') else float(mae),
                'mse_weight': mse_weight, 
                'mae_weight': mae_weight,
                'combined_loss': float(combined_loss.numpy()) if hasattr(combined_loss, 'numpy') else float(combined_loss)
            }

            state['history'].append(loss_info)
            
            # Update weight history for curve_fancy strategy
            if strategy == 'curve_fancy':
                state['weight_history'].append(np.array([mse_weight, mae_weight]))
            
            return combined_loss
        except Exception as e:
            print(f"Warning: Adaptive loss computation failed: {e}")
            state['error_count'] += 1
            return mse_loss(y_true, y_pred)
    
    def update_state(epoch: int, prev_r2: Optional[float] = None) -> None:
        """Update loss function state."""
        state['epoch'] = epoch

        if prev_r2 is not None: state['prev_r2'] = prev_r2

        if state['history']:
            state['prev_loss'] = state['history'][-1]['combined_loss']
            state['loss_history'].append(state['history'][-1]['combined_loss'])
    
    def get_current_info() -> str:
        """Get current strategy information."""
        mse_weight, mae_weight = compute_loss_weights(
            strategy, 
            state['prev_r2'], 
            state['prev_loss'],
            state['loss_history'],
            state['weight_history'],
            state['epoch']
        )
        return f"{strategy} (MSE: {mse_weight:.3f}, MAE: {mae_weight:.3f})"
    
    def get_history() -> Dict[str, Any]:
        """Get complete loss history."""
        return {
            'losses': state['history'], 
            'strategy': strategy, 
            'error_count': state['error_count'],
            'loss_history': state['loss_history'],
            'weight_history': [w.tolist() for w in state['weight_history']]
        }
    
    # Attach methods to function
    adaptive_loss.update_state = update_state
    adaptive_loss.get_current_info = get_current_info
    adaptive_loss.get_history = get_history
    return adaptive_loss
