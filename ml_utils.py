"""
ML Utilities for Neural Network Training

This module combines adaptive loss functions and data processing utilities into a 
comprehensive, compact ML utility suite for the advanced neural network project.
"""

import math
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any


def compute_loss_weights(strategy: str, epoch: int = 0, accuracy: float = 0.5, 
                        prev_loss: float = 1.0) -> Tuple[float, float]:
    """
    Compute adaptive loss function weights based on training progress.
    
    Args:
        strategy: Weighting strategy ('epoch_based', 'accuracy_based', 'loss_based', 'combined')
        epoch: Current training epoch
        accuracy: Previous epoch's accuracy
        prev_loss: Previous epoch's loss value
        
    Returns:
        Tuple of (mse_weight, mae_weight)
    """
    try:
        if strategy == 'epoch_based':
            if epoch < 10: return 0.3, 0.7
            elif epoch < 30:
                progress = (epoch - 10) / 20
                return 0.3 + 0.4 * progress, 0.7 - 0.4 * progress
            else: return 0.8, 0.2      
        elif strategy == 'accuracy_based':
            if accuracy < 0.3: return 0.2, 0.8
            elif accuracy < 0.6:
                progress = (accuracy - 0.3) / 0.3
                return 0.2 + 0.3 * progress, 0.8 - 0.3 * progress
            elif accuracy < 0.85:
                progress = (accuracy - 0.6) / 0.25
                return 0.5 + 0.3 * progress, 0.5 - 0.3 * progress
            else: return 0.9, 0.1
        elif strategy == 'loss_based':
            log_loss = math.log(max(prev_loss, 1e-8))

            if log_loss > 0: return 0.3, 0.7
            elif log_loss > -2:
                progress = (log_loss + 2) / 2
                return 0.3 + 0.4 * progress, 0.7 - 0.4 * progress
            else: return 0.8, 0.2       
        elif strategy == 'combined':
            # Get weights from each strategy
            epoch_mse, epoch_mae = compute_loss_weights('epoch_based', epoch, accuracy, prev_loss)
            acc_mse, acc_mae = compute_loss_weights('accuracy_based', epoch, accuracy, prev_loss)
            loss_mse, loss_mae = compute_loss_weights('loss_based', epoch, accuracy, prev_loss)
            
            # Weight strategies based on training progress
            if epoch < 5: weights = [0.6, 0.2, 0.2]
            elif epoch < 20: weights = [0.4, 0.4, 0.2]
            else: weights = [0.2, 0.5, 0.3]
            
            # Combine and normalize
            final_mse = weights[0] * epoch_mse + weights[1] * acc_mse + weights[2] * loss_mse
            final_mae = weights[0] * epoch_mae + weights[1] * acc_mae + weights[2] * loss_mae
            total = final_mse + final_mae
            return (final_mse / total, final_mae / total) if total > 0 else (0.5, 0.5)
        else: return 0.5, 0.5       
    except Exception as e:
        print(f"Warning: Error computing loss weights ({strategy}): {e}")
        return 0.5, 0.5


def create_adaptive_loss_fn(strategy: str = 'epoch_based'):
    """
    Create an adaptive loss function with state management.
    
    Args:
        strategy: Weighting strategy to use
        
    Returns:
        Adaptive loss function with update capabilities
    """
    # State variables (using mutable default to maintain state)
    state = {'epoch': 0, 'accuracy': 0.5, 'prev_loss': 1.0, 'history': [], 'error_count': 0}
    # Create loss functions
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    def adaptive_loss(y_true, y_pred):
        """Compute adaptive loss with current weights."""
        try:
            mse = mse_loss(y_true, y_pred)
            mae = mae_loss(y_true, y_pred)
            mse_weight, mae_weight = compute_loss_weights(strategy, state['epoch'], state['accuracy'], state['prev_loss'])
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
    
    def update_state(epoch: int, accuracy: Optional[float] = None):
        """Update loss function state."""
        state['epoch'] = epoch

        if accuracy is not None: state['accuracy'] = accuracy

        if state['history']: state['prev_loss'] = state['history'][-1]['combined_loss']
    
    def get_current_info() -> str:
        """Get current strategy information."""
        mse_weight, mae_weight = compute_loss_weights(strategy, state['epoch'], state['accuracy'], state['prev_loss'])
        return f"{strategy} (MSE: {mse_weight:.3f}, MAE: {mae_weight:.3f})"
    
    def get_history() -> Dict[str, Any]:
        """Get complete loss history."""
        return {'losses': state['history'], 'strategy': strategy, 'error_count': state['error_count']}
    
    # Attach methods to function
    adaptive_loss.update_state = update_state
    adaptive_loss.get_current_info = get_current_info
    adaptive_loss.get_history = get_history
    return adaptive_loss
