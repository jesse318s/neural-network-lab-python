"""
James-Stein Estimator Weight Constraint for TensorFlow/Keras

This module implements the James-Stein shrinkage estimator as a TensorFlow constraint.
The James-Stein estimator is a shrinkage estimator that dominates the maximum likelihood
estimator under squared error loss in dimensions three or higher. It shrinks estimates
toward a common target (typically zero), reducing variance at the cost of introducing bias.

Mathematical Foundation:
For a vector of parameters θ = (θ₁, θ₂, ..., θₚ) with p ≥ 3, the James-Stein estimator is:

    θ̂_JS = (1 - λ) × θ̂_MLE

where λ = (p - 2)σ²/||θ̂_MLE||² is the shrinkage factor, σ² is the noise variance estimate,
and θ̂_MLE is the maximum likelihood estimate (unconstrained weights).

For neural networks, we apply this shrinkage to weight matrices, treating each layer's
weights as a high-dimensional parameter vector.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class JamesSteinWeightConstraint(tf.keras.constraints.Constraint):
    """
    James-Stein shrinkage constraint for neural network weights.
    
    This constraint applies shrinkage toward a target value (default 0.0) to reduce
    variance in weight estimates. The shrinkage factor is computed based on the
    dimensionality of the weight tensor and the squared norm of the weights.
    
    Parameters
    ----------
    shrinkage_target : float, optional
        The target value toward which weights are shrunk (default: 0.0).
    min_norm : float, optional
        Minimum norm value to prevent division by zero (default: 1e-6).
    max_shrinkage : float, optional
        Maximum shrinkage factor to apply, preventing over-shrinkage (default: 0.99).
    """
    
    def __init__(
        self,
        shrinkage_target: float = 0.0,
        min_norm: float = 1e-6,
        max_shrinkage: float = 0.99
    ):
        self.shrinkage_target = shrinkage_target
        self.min_norm = min_norm
        self.max_shrinkage = max_shrinkage
    
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Apply James-Stein shrinkage to the weight tensor."""
        # Reshape weights to 1D vector for shrinkage computation
        original_shape = tf.shape(w)
        flat_weights = tf.reshape(w, [-1])
        
        # Compute dimension (number of parameters)
        p = tf.cast(tf.size(flat_weights), tf.float32)
        
        # Compute squared norm of weights
        norm_squared = tf.reduce_sum(tf.square(flat_weights))
        norm_squared = tf.maximum(norm_squared, self.min_norm)
        
        # Calculate James-Stein shrinkage factor
        # λ = (p - 2) / norm_squared
        # We cap this at max_shrinkage to prevent over-shrinkage
        shrinkage_factor_raw = (p - 2.0) / norm_squared
        shrinkage_factor = tf.minimum(shrinkage_factor_raw, self.max_shrinkage)
        
        # Ensure shrinkage factor is non-negative
        shrinkage_factor = tf.maximum(shrinkage_factor, 0.0)
        
        # Apply shrinkage: θ̂_JS = (1 - λ) × θ + λ × target
        # Equivalent to: θ̂_JS = θ - λ × (θ - target)
        shrinkage_amount = shrinkage_factor * (flat_weights - self.shrinkage_target)
        shrunk_weights = flat_weights - shrinkage_amount
        
        # Reshape back to original shape
        constrained_weights = tf.reshape(shrunk_weights, original_shape)
        
        return constrained_weights
    
    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        return {
            'shrinkage_target': self.shrinkage_target,
            'min_norm': self.min_norm,
            'max_shrinkage': self.max_shrinkage
        }


class AdaptiveJamesSteinConstraint(tf.keras.constraints.Constraint):
    """
    Adaptive James-Stein shrinkage constraint with dynamic target estimation.
    
    This variant estimates the shrinkage target from the weight distribution itself,
    using the mean of the weights as the target. This can be more robust when the
    optimal target is not known a priori.
    
    Parameters
    ----------
    min_norm : float, optional
        Minimum norm value to prevent division by zero (default: 1e-6).
    max_shrinkage : float, optional
        Maximum shrinkage factor to apply (default: 0.99).
    use_median : bool, optional
        If True, use median instead of mean as shrinkage target (default: False).
    """
    
    def __init__(
        self,
        min_norm: float = 1e-6,
        max_shrinkage: float = 0.99,
        use_median: bool = False
    ):
        self.min_norm = min_norm
        self.max_shrinkage = max_shrinkage
        self.use_median = use_median
    
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Apply adaptive James-Stein shrinkage to the weight tensor."""
        original_shape = tf.shape(w)
        flat_weights = tf.reshape(w, [-1])
        
        # Compute adaptive target (mean or median of weights)
        if self.use_median:
            # TensorFlow doesn't have a direct median function, approximate with percentile
            shrinkage_target = tfp.stats.percentile(flat_weights, 50.0)
        else:
            shrinkage_target = tf.reduce_mean(flat_weights)
        
        # Compute dimension
        p = tf.cast(tf.size(flat_weights), tf.float32)
        
        # Compute squared deviation from target
        deviations = flat_weights - shrinkage_target
        norm_squared = tf.reduce_sum(tf.square(deviations))
        norm_squared = tf.maximum(norm_squared, self.min_norm)
        
        # Calculate shrinkage factor
        shrinkage_factor_raw = (p - 2.0) / norm_squared
        shrinkage_factor = tf.minimum(shrinkage_factor_raw, self.max_shrinkage)
        shrinkage_factor = tf.maximum(shrinkage_factor, 0.0)
        
        # Apply shrinkage toward adaptive target
        shrunk_weights = flat_weights - shrinkage_factor * deviations
        
        # Reshape back
        constrained_weights = tf.reshape(shrunk_weights, original_shape)
        
        return constrained_weights
    
    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        return {
            'min_norm': self.min_norm,
            'max_shrinkage': self.max_shrinkage,
            'use_median': self.use_median
        }


def create_james_stein_model(
    input_shape: tuple,
    output_shape: int,
    hidden_layers: list = [64, 32, 16],
    activation: str = 'relu',
    dropout_rate: float = 0.0,
    shrinkage_target: float = 0.0,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Create a neural network model with James-Stein weight constraints.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input features.
    output_shape : int
        Number of output neurons.
    hidden_layers : list, optional
        List of hidden layer sizes (default: [64, 32, 16]).
    activation : str, optional
        Activation function for hidden layers (default: 'relu').
    dropout_rate : float, optional
        Dropout rate between layers (default: 0.0).
    shrinkage_target : float, optional
        Target value for James-Stein shrinkage (default: 0.0).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001).
    
    Returns
    -------
    tf.keras.Model
        Compiled Keras model with James-Stein constraints.
    """
    js_constraint = JamesSteinWeightConstraint(shrinkage_target=shrinkage_target)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_constraint=js_constraint
        ))

        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(
        output_shape,
        kernel_constraint=js_constraint
    ))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def compare_shrinkage_methods(
    weights: np.ndarray,
    show_diagnostics: bool = False
) -> dict:
    """
    Compare different shrinkage approaches on a weight array.
    
    Parameters
    ----------
    weights : np.ndarray
        Weight array to analyze.
    show_diagnostics : bool, optional
        If True, print diagnostic information (default: False).
    
    Returns
    -------
    dict
        Dictionary containing shrinkage statistics for different methods.
    """
    flat_weights = weights.flatten()
    
    # Original statistics
    original_norm = np.linalg.norm(flat_weights)
    original_mean = np.mean(flat_weights)
    original_std = np.std(flat_weights)
    
    # James-Stein shrinkage toward zero
    p = len(flat_weights)
    norm_squared = np.sum(flat_weights ** 2)
    js_factor = max(0, min(0.99, (p - 2) / norm_squared))
    js_weights = (1 - js_factor) * flat_weights
    
    # Adaptive James-Stein (shrink toward mean)
    deviations = flat_weights - original_mean
    dev_norm_squared = np.sum(deviations ** 2)
    adaptive_factor = max(0, min(0.99, (p - 2) / dev_norm_squared))
    adaptive_weights = flat_weights - adaptive_factor * deviations
    
    results = {
        'original': {
            'norm': original_norm,
            'mean': original_mean,
            'std': original_std,
            'sparsity': np.mean(np.abs(flat_weights) < 1e-3)
        },
        'james_stein': {
            'shrinkage_factor': js_factor,
            'norm': np.linalg.norm(js_weights),
            'mean': np.mean(js_weights),
            'std': np.std(js_weights),
            'sparsity': np.mean(np.abs(js_weights) < 1e-3),
            'norm_reduction': (original_norm - np.linalg.norm(js_weights)) / original_norm
        },
        'adaptive_james_stein': {
            'shrinkage_factor': adaptive_factor,
            'norm': np.linalg.norm(adaptive_weights),
            'mean': np.mean(adaptive_weights),
            'std': np.std(adaptive_weights),
            'sparsity': np.mean(np.abs(adaptive_weights) < 1e-3),
            'norm_reduction': (original_norm - np.linalg.norm(adaptive_weights)) / original_norm
        }
    }
    
    if show_diagnostics:
        print("Weight Shrinkage Comparison")
        print("=" * 50)
        for method, stats in results.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.6f}")
    
    return results
