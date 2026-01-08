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
            activation=None if activation.lower() == 'prelu' else activation,
            kernel_constraint=js_constraint
        ))
        
        if activation.lower() == 'prelu':
            model.add(tf.keras.layers.PReLU())

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
