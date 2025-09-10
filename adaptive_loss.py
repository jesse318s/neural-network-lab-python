"""
Adaptive Loss Functions for Advanced TensorFlow Training

This module implements custom loss functions that adaptively combine multiple
standard loss functions using various weighting strategies based on training
progress, accuracy, and previous loss values.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Any, Callable
import math


class AdaptiveLossFunction:
    """
    Combines multiple loss functions adaptively using weighted averages.
    Supports epoch-based, accuracy-based, and loss-based weighting strategies.
    """
    
    def __init__(self, weighting_strategy: str = 'epoch_based'):
        """
        Initialize the adaptive loss function.
        
        Args:
            weighting_strategy: Strategy for weighting loss functions
                              ('epoch_based', 'accuracy_based', 'loss_based', 'combined')
        """
        self.weighting_strategy = weighting_strategy
        self.current_epoch = 0
        self.previous_accuracy = 0.5
        self.previous_loss = 1.0
        self.loss_history = []
        self.weight_history = []
        self.error_count = 0
        
        # Loss function components
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        
    def _calculate_epoch_weights(self, epoch: int) -> tuple:
        """
        Calculate loss function weights based on epoch number.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Tuple of (mse_weight, mae_weight)
        """
        try:
            # Early epochs: favor MAE for robust training
            # Later epochs: favor MSE for fine-tuning
            
            if epoch < 10:
                # Early training: emphasize MAE for outlier robustness
                mse_weight = 0.3
                mae_weight = 0.7
            elif epoch < 30:
                # Mid training: balanced approach
                progress = (epoch - 10) / 20  # 0 to 1
                mse_weight = 0.3 + 0.4 * progress  # 0.3 to 0.7
                mae_weight = 0.7 - 0.4 * progress  # 0.7 to 0.3
            else:
                # Late training: emphasize MSE for precision
                mse_weight = 0.8
                mae_weight = 0.2
            
            return mse_weight, mae_weight
            
        except Exception as e:
            print(f"Warning: Error calculating epoch weights: {e}")
            self.error_count += 1
            return 0.5, 0.5  # Fallback to equal weights
    
    def _calculate_accuracy_weights(self, accuracy: float) -> tuple:
        """
        Calculate loss function weights based on previous accuracy.
        
        Args:
            accuracy: Previous epoch's accuracy (0.0 to 1.0)
            
        Returns:
            Tuple of (mse_weight, mae_weight)
        """
        try:
            # Low accuracy: favor MAE for stability
            # High accuracy: favor MSE for precision
            
            if accuracy < 0.3:
                # Very low accuracy: heavy MAE emphasis
                mse_weight = 0.2
                mae_weight = 0.8
            elif accuracy < 0.6:
                # Low to medium accuracy: gradual transition
                progress = (accuracy - 0.3) / 0.3  # 0 to 1
                mse_weight = 0.2 + 0.3 * progress  # 0.2 to 0.5
                mae_weight = 0.8 - 0.3 * progress  # 0.8 to 0.5
            elif accuracy < 0.85:
                # Medium to high accuracy: favor MSE
                progress = (accuracy - 0.6) / 0.25  # 0 to 1
                mse_weight = 0.5 + 0.3 * progress  # 0.5 to 0.8
                mae_weight = 0.5 - 0.3 * progress  # 0.5 to 0.2
            else:
                # High accuracy: strong MSE emphasis
                mse_weight = 0.9
                mae_weight = 0.1
            
            return mse_weight, mae_weight
            
        except Exception as e:
            print(f"Warning: Error calculating accuracy weights: {e}")
            self.error_count += 1
            return 0.5, 0.5
    
    def _calculate_loss_weights(self, previous_loss: float) -> tuple:
        """
        Calculate loss function weights based on previous loss values.
        
        Args:
            previous_loss: Previous epoch's loss value
            
        Returns:
            Tuple of (mse_weight, mae_weight)
        """
        try:
            # High loss: favor MAE for stability
            # Low loss: favor MSE for precision
            
            # Use logarithmic scaling for loss
            log_loss = math.log(max(previous_loss, 1e-8))
            
            if log_loss > 0:  # Loss > 1
                # High loss: emphasize MAE
                mse_weight = 0.3
                mae_weight = 0.7
            elif log_loss > -2:  # Loss between 0.135 and 1
                # Medium loss: balanced approach
                progress = (log_loss + 2) / 2  # 0 to 1
                mse_weight = 0.3 + 0.4 * progress  # 0.3 to 0.7
                mae_weight = 0.7 - 0.4 * progress  # 0.7 to 0.3
            else:  # Loss < 0.135
                # Low loss: emphasize MSE
                mse_weight = 0.8
                mae_weight = 0.2
            
            return mse_weight, mae_weight
            
        except Exception as e:
            print(f"Warning: Error calculating loss weights: {e}")
            self.error_count += 1
            return 0.5, 0.5
    
    def _calculate_combined_weights(self, epoch: int, accuracy: float, loss: float) -> tuple:
        """
        Calculate weights using a combination of all strategies.
        
        Args:
            epoch: Current epoch
            accuracy: Previous accuracy
            loss: Previous loss
            
        Returns:
            Tuple of (mse_weight, mae_weight)
        """
        try:
            # Get weights from each strategy
            epoch_mse, epoch_mae = self._calculate_epoch_weights(epoch)
            acc_mse, acc_mae = self._calculate_accuracy_weights(accuracy)
            loss_mse, loss_mae = self._calculate_loss_weights(loss)
            
            # Weight the strategies themselves based on training progress
            if epoch < 5:
                # Early: rely more on epoch-based
                strategy_weights = [0.6, 0.2, 0.2]  # [epoch, accuracy, loss]
            elif epoch < 20:
                # Mid: balanced approach
                strategy_weights = [0.4, 0.4, 0.2]
            else:
                # Late: rely more on accuracy and loss
                strategy_weights = [0.2, 0.5, 0.3]
            
            # Combine strategies
            final_mse = (strategy_weights[0] * epoch_mse + 
                        strategy_weights[1] * acc_mse + 
                        strategy_weights[2] * loss_mse)
            
            final_mae = (strategy_weights[0] * epoch_mae + 
                        strategy_weights[1] * acc_mae + 
                        strategy_weights[2] * loss_mae)
            
            # Normalize to ensure they sum to 1
            total = final_mse + final_mae
            if total > 0:
                final_mse /= total
                final_mae /= total
            else:
                final_mse, final_mae = 0.5, 0.5
            
            return final_mse, final_mae
            
        except Exception as e:
            print(f"Warning: Error calculating combined weights: {e}")
            self.error_count += 1
            return 0.5, 0.5
    
    def get_weights(self) -> tuple:
        """
        Get current loss function weights based on the selected strategy.
        
        Returns:
            Tuple of (mse_weight, mae_weight)
        """
        try:
            if self.weighting_strategy == 'epoch_based':
                return self._calculate_epoch_weights(self.current_epoch)
            elif self.weighting_strategy == 'accuracy_based':
                return self._calculate_accuracy_weights(self.previous_accuracy)
            elif self.weighting_strategy == 'loss_based':
                return self._calculate_loss_weights(self.previous_loss)
            elif self.weighting_strategy == 'combined':
                return self._calculate_combined_weights(
                    self.current_epoch, self.previous_accuracy, self.previous_loss
                )
            else:
                print(f"Warning: Unknown weighting strategy '{self.weighting_strategy}', using equal weights")
                return 0.5, 0.5
                
        except Exception as e:
            print(f"Warning: Error getting weights: {e}")
            self.error_count += 1
            return 0.5, 0.5
    
    def __call__(self, y_true, y_pred):
        """
        Compute the adaptive loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Combined loss value
        """
        try:
            # Calculate individual losses
            mse = self.mse_loss(y_true, y_pred)
            mae = self.mae_loss(y_true, y_pred)
            
            # Get current weights
            mse_weight, mae_weight = self.get_weights()
            
            # Combine losses
            combined_loss = mse_weight * mse + mae_weight * mae
            
            # Store for history
            loss_info = {
                'epoch': self.current_epoch,
                'mse': float(mse.numpy()) if hasattr(mse, 'numpy') else float(mse),
                'mae': float(mae.numpy()) if hasattr(mae, 'numpy') else float(mae),
                'mse_weight': mse_weight,
                'mae_weight': mae_weight,
                'combined_loss': float(combined_loss.numpy()) if hasattr(combined_loss, 'numpy') else float(combined_loss)
            }
            self.loss_history.append(loss_info)
            self.weight_history.append((mse_weight, mae_weight))
            
            return combined_loss
            
        except Exception as e:
            print(f"Warning: Error computing adaptive loss: {e}")
            self.error_count += 1
            # Fallback to MSE
            return self.mse_loss(y_true, y_pred)
    
    def update_epoch(self, epoch: int, accuracy: Optional[float] = None):
        """
        Update the current epoch and accuracy for adaptive weighting.
        
        Args:
            epoch: Current training epoch
            accuracy: Current accuracy (optional)
        """
        try:
            self.current_epoch = epoch
            if accuracy is not None:
                self.previous_accuracy = accuracy
            
            # Update previous loss from history
            if self.loss_history:
                self.previous_loss = self.loss_history[-1]['combined_loss']
                
        except Exception as e:
            print(f"Warning: Error updating epoch information: {e}")
            self.error_count += 1
    
    def get_loss_history(self) -> Dict[str, List]:
        """
        Get the complete loss history.
        
        Returns:
            Dictionary containing loss history components
        """
        return {
            'losses': self.loss_history,
            'weights': self.weight_history,
            'strategy': self.weighting_strategy,
            'error_count': self.error_count
        }
    
    def get_current_strategy_info(self) -> str:
        """
        Get information about the current weighting strategy being used.
        
        Returns:
            String describing the current strategy
        """
        mse_weight, mae_weight = self.get_weights()
        return f"{self.weighting_strategy} (MSE: {mse_weight:.3f}, MAE: {mae_weight:.3f})"
    
    def get_error_count(self) -> int:
        """Get number of errors encountered."""
        return self.error_count


# Standalone functions for individual weighting strategies
def epoch_weighted_loss(epoch: int, mse_loss: float, mae_loss: float) -> float:
    """
    Weight loss functions based on epoch number.
    
    Args:
        epoch: Current epoch number
        mse_loss: Mean squared error loss
        mae_loss: Mean absolute error loss
        
    Returns:
        Combined weighted loss
    """
    try:
        if epoch < 10:
            mse_weight = 0.3
            mae_weight = 0.7
        elif epoch < 30:
            progress = (epoch - 10) / 20
            mse_weight = 0.3 + 0.4 * progress
            mae_weight = 0.7 - 0.4 * progress
        else:
            mse_weight = 0.8
            mae_weight = 0.2
        
        return mse_weight * mse_loss + mae_weight * mae_loss
        
    except Exception as e:
        print(f"Warning: Error in epoch weighted loss: {e}")
        return 0.5 * mse_loss + 0.5 * mae_loss


def accuracy_weighted_loss(previous_accuracy: float, mse_loss: float, mae_loss: float) -> float:
    """
    Weight loss functions based on previous accuracy.
    
    Args:
        previous_accuracy: Previous epoch's accuracy
        mse_loss: Mean squared error loss
        mae_loss: Mean absolute error loss
        
    Returns:
        Combined weighted loss
    """
    try:
        if previous_accuracy < 0.3:
            mse_weight = 0.2
            mae_weight = 0.8
        elif previous_accuracy < 0.6:
            progress = (previous_accuracy - 0.3) / 0.3
            mse_weight = 0.2 + 0.3 * progress
            mae_weight = 0.8 - 0.3 * progress
        elif previous_accuracy < 0.85:
            progress = (previous_accuracy - 0.6) / 0.25
            mse_weight = 0.5 + 0.3 * progress
            mae_weight = 0.5 - 0.3 * progress
        else:
            mse_weight = 0.9
            mae_weight = 0.1
        
        return mse_weight * mse_loss + mae_weight * mae_loss
        
    except Exception as e:
        print(f"Warning: Error in accuracy weighted loss: {e}")
        return 0.5 * mse_loss + 0.5 * mae_loss


def loss_weighted_loss(previous_loss: float, mse_loss: float, mae_loss: float) -> float:
    """
    Weight loss functions based on previous loss values.
    
    Args:
        previous_loss: Previous epoch's loss value
        mse_loss: Mean squared error loss
        mae_loss: Mean absolute error loss
        
    Returns:
        Combined weighted loss
    """
    try:
        log_loss = math.log(max(previous_loss, 1e-8))
        
        if log_loss > 0:
            mse_weight = 0.3
            mae_weight = 0.7
        elif log_loss > -2:
            progress = (log_loss + 2) / 2
            mse_weight = 0.3 + 0.4 * progress
            mae_weight = 0.7 - 0.4 * progress
        else:
            mse_weight = 0.8
            mae_weight = 0.2
        
        return mse_weight * mse_loss + mae_weight * mae_loss
        
    except Exception as e:
        print(f"Warning: Error in loss weighted loss: {e}")
        return 0.5 * mse_loss + 0.5 * mae_loss
