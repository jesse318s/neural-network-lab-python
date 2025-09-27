"""
ML Utilities for Neural Network Training

This module combines adaptive loss functions and state management to enhance model training.
"""

import tensorflow as tf
from typing import Dict, Tuple, Optional, Any


def compute_loss_weights(strategy: str, prev_r2: float = 0.5, prev_loss: float = 1.0) -> Tuple[float, float]:
    """
    Compute adaptive loss function weights based on training progress.
    
    Args:
        strategy: Weighting strategy ('r2_based', 'loss_based', 'combined')
        prev_r2: Previous epoch's R² score
        prev_loss: Previous epoch's loss value
        
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
        else: return 0.5, 0.5       
    except Exception as e:
        print(f"Warning: Error computing loss weights ({strategy}): {e}")
        return 0.5, 0.5


def create_adaptive_loss_fn(strategy: str = 'r2_based'):
    """
    Create an adaptive loss function with state management.
    
    Args:
        strategy: Weighting strategy to use
        
    Returns:
        Adaptive loss function with update capabilities
    """
    # State variables (using mutable default to maintain state)
    state = {'epoch': 0, 'prev_r2': 0.5, 'prev_loss': 1.0, 'history': [], 'error_count': 0}
    # Create loss functions
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    def adaptive_loss(y_true, y_pred):
        """Compute adaptive loss with current weights."""
        try:
            mse = mse_loss(y_true, y_pred)
            mae = mae_loss(y_true, y_pred)
            mse_weight, mae_weight = compute_loss_weights(strategy, state['prev_r2'], state['prev_loss'])
            combined_loss = mse_weight * mse + mae_weight * mae
            # Record history
            loss_info = {
                'epoch': state['epoch'], 'mse': float(mse.numpy()) if hasattr(mse, 'numpy') else float(mse),
                'mae': float(mae.numpy()) if hasattr(mae, 'numpy') else float(mae),
                'mse_weight': mse_weight, 'mae_weight': mae_weight,
                'combined_loss': float(combined_loss.numpy()) if hasattr(combined_loss, 'numpy') else float(combined_loss)
            }

            state['history'].append(loss_info)
            return combined_loss
        except Exception as e:
            print(f"Warning: Adaptive loss computation failed: {e}")
            state['error_count'] += 1
            return mse_loss(y_true, y_pred)
    
    def update_state(epoch: int, prev_r2: Optional[float] = None):
        """Update loss function state."""
        state['epoch'] = epoch

        if prev_r2 is not None: state['prev_r2'] = prev_r2

        if state['history']: state['prev_loss'] = state['history'][-1]['combined_loss']
    
    def get_current_info() -> str:
        """Get current strategy information."""
        mse_weight, mae_weight = compute_loss_weights(strategy, state['prev_r2'], state['prev_loss'])
        return f"{strategy} (MSE: {mse_weight:.3f}, MAE: {mae_weight:.3f})"
    
    def get_history() -> Dict[str, Any]:
        """Get complete loss history."""
        return {'losses': state['history'], 'strategy': strategy, 'error_count': state['error_count']}
    
    # Attach methods to function
    adaptive_loss.update_state = update_state
    adaptive_loss.get_current_info = get_current_info
    adaptive_loss.get_history = get_history
    return adaptive_loss
