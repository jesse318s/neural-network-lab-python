"""
Weight Constraint Classes for Advanced TensorFlow Training

This module implements custom weight constraint classes that manage binary precision
of neural network weights and prevent oscillations during training.
"""

import struct
import numpy as np
from typing import List
from abc import ABC, abstractmethod

class BinaryWeightConstraint(ABC):
    """Base class for binary weight constraints."""
    
    def __init__(self):
        self.error_count = 0
        
    def _float_to_binary_repr(self, value: float) -> str:
        """Convert float to binary representation string."""
        try:
            if value == 0.0: return "0"
            
            sign = "-" if value < 0 else ""
            abs_value = abs(value)
            integer_part = int(abs_value)
            fractional_part = abs_value - integer_part
            integer_binary = "0" if integer_part == 0 else bin(integer_part)[2:]
            fractional_binary = ""
            max_fractional_digits = 20

            while fractional_part > 0 and len(fractional_binary) < max_fractional_digits:
                fractional_part *= 2

                if fractional_part >= 1:
                    fractional_binary += "1"
                    fractional_part -= 1
                else:
                    fractional_binary += "0"
            
            result = f"{sign}{integer_binary}"

            if fractional_binary: result += f".{fractional_binary}"

            return result
        except Exception:
            return "0"
    
    def _count_significant_binary_digits(self, binary_str: str) -> int:
        """Count significant binary digits (excluding leading zeros and trailing zeros)."""
        try:
            if binary_str.startswith('-'): binary_str = binary_str[1:]
            
            if '.' in binary_str:
                integer_part, fractional_part = binary_str.split('.')
            else:
                integer_part, fractional_part = binary_str, ""
            
            integer_part = integer_part.lstrip('0')

            if not integer_part: integer_part = "0"

            fractional_part = fractional_part.rstrip('0')
            significant_digits = len(integer_part) + len(fractional_part)
            return max(significant_digits, 1)
        except Exception:
            return 1
    
    def _apply_to_weights(self, weights: np.ndarray, constraint_func) -> np.ndarray:
        """Apply constraint function to weights array of any shape."""
        try:
            flat_weights = weights.flatten()
            flat_constrained = np.array([constraint_func(w) for w in flat_weights])
            return flat_constrained.reshape(weights.shape)
        except Exception:
            self.error_count += 1
            return weights
    
    def get_error_count(self) -> int:
        """Get the number of errors encountered."""
        return self.error_count
    
    @abstractmethod
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Apply the binary constraint."""
        pass


class BinaryWeightConstraintChanges(BinaryWeightConstraint):
    """Manages binary precision of weight changes, allowing only one additional 
    significant figure in binary format compared to previous weights."""
    
    def __init__(self, max_additional_digits: int = 1):
        super().__init__()
        self.max_additional_digits = max_additional_digits
        self.previous_weights = None
        
    def _constrain_weight_change(self, current_weight: float, previous_weight: float) -> float:
        """Constrain a single weight to have at most one additional significant binary digit."""
        try:
            current_binary = self._float_to_binary_repr(current_weight)
            previous_binary = self._float_to_binary_repr(previous_weight)
            current_digits = self._count_significant_binary_digits(current_binary)
            previous_digits = self._count_significant_binary_digits(previous_binary)
            max_allowed_digits = previous_digits + self.max_additional_digits
            
            if current_digits <= max_allowed_digits: return current_weight
            
            # Reduce precision
            reduction_factor = current_digits - max_allowed_digits
            magnitude = 10 ** (-reduction_factor)
            return round(current_weight / magnitude) * magnitude
        except Exception:
            self.error_count += 1
            return current_weight
    
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Apply binary precision change constraint to weight matrix."""
        try:
            if self.previous_weights is None or self.previous_weights.shape != weights.shape:
                self.previous_weights = weights.copy()
                return weights
            
            flat_weights = weights.flatten()
            flat_previous = self.previous_weights.flatten()
            flat_constrained = np.array([
                self._constrain_weight_change(w, p) 
                for w, p in zip(flat_weights, flat_previous)
            ])
            constrained_weights = flat_constrained.reshape(weights.shape)
            self.previous_weights = constrained_weights.copy()
            return constrained_weights
        except Exception:
            self.error_count += 1

            if self.previous_weights is None or self.previous_weights.shape != weights.shape:
                self.previous_weights = weights.copy()
            
            return weights
    
    def reset(self):
        self.previous_weights = None
        self.error_count = 0


class BinaryWeightConstraintMax(BinaryWeightConstraint):
    """Manages binary precision of weights with maximum binary digits allowed."""
    
    def __init__(self, max_binary_digits: int = 5):
        super().__init__()
        self.max_binary_digits = max_binary_digits
    
    def _constrain_weight_max(self, weight: float) -> float:
        """Constrain a single weight to have at most max_binary_digits significant binary digits."""
        try:
            binary_repr = self._float_to_binary_repr(weight)
            current_digits = self._count_significant_binary_digits(binary_repr)
            
            if current_digits <= self.max_binary_digits: return weight
            
            # Reduce precision
            reduction_factor = current_digits - self.max_binary_digits
            magnitude = 10 ** (-reduction_factor)
            return round(weight / magnitude) * magnitude
        except Exception:
            self.error_count += 1
            return weight
    
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Apply maximum binary precision constraint to weight matrix."""
        return self._apply_to_weights(weights, self._constrain_weight_max)


class OscillationDampener:
    """Monitors weight changes and dampens oscillations by setting the smallest 
    non-zero binary digit to zero."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.weight_history: List[np.ndarray] = []
        self.error_count = 0
    
    def add_weights(self, weights: np.ndarray):
        """Add new weights to the history."""
        try:
            self.weight_history.append(weights.copy())

            if len(self.weight_history) > self.window_size:
                self.weight_history.pop(0)
        except Exception:
            self.error_count += 1
    
    def _detect_oscillation_pattern(self, values: List[float]) -> bool:
        """Detect if values show an oscillation pattern."""
        try:
            if len(values) < 3: return False

            return (values[0] < values[1] > values[2]) or (values[0] > values[1] < values[2])
        except Exception:
            return False
    
    def _set_smallest_binary_digit_to_zero(self, weight: float) -> float:
        """Set the smallest non-zero binary digit to zero."""
        try:
            if weight == 0.0: return weight
            
            sign = -1 if weight < 0 else 1
            abs_weight = abs(weight)    
            # Use bit manipulation approach
            packed = struct.pack('f', abs_weight)
            bits = struct.unpack('I', packed)[0]
            
            if bits == 0: return weight
            
            # Clear the least significant bit
            lsb_position = (bits & -bits).bit_length() - 1
            bits &= ~(1 << lsb_position)
            # Convert back to float
            modified_packed = struct.pack('I', bits)
            modified_weight = struct.unpack('f', modified_packed)[0]
            return sign * modified_weight
        except Exception:
            self.error_count += 1
            return weight * 0.99
    
    def detect_and_dampen_oscillations(self, current_weights: np.ndarray) -> np.ndarray:
        """Detect oscillations and apply dampening."""
        try:
            if len(self.weight_history) < self.window_size: return current_weights
            
            flat_current = current_weights.flatten()
            flat_dampened = flat_current.copy()
            
            for i in range(len(flat_current)):
                weight_sequence = []
                
                for hist_weights in self.weight_history:
                    if hist_weights.shape == current_weights.shape:
                        weight_sequence.append(hist_weights.flatten()[i])
                
                weight_sequence.append(flat_current[i])
                
                if len(weight_sequence) >= self.window_size:
                    recent_values = weight_sequence[-self.window_size:]

                    if self._detect_oscillation_pattern(recent_values):
                        flat_dampened[i] = self._set_smallest_binary_digit_to_zero(flat_current[i])
            
            return flat_dampened.reshape(current_weights.shape)
        except Exception:
            self.error_count += 1
            return current_weights
    
    def get_error_count(self) -> int:
        """Get the number of errors encountered."""
        return self.error_count
    
    def reset(self):
        """Reset the history and error count."""
        self.weight_history = []
        self.error_count = 0