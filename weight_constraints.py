"""
Weight Constraint Classes for Advanced TensorFlow Training

This module implements custom weight constraint classes that manage binary precision
of neural network weights and prevent oscillations during training.
The classes are:
- BinaryWeightConstraint - abstract base class: Provides common methods for binary representation and counting significant digits.
- BinaryWeightConstraintMax: Limits the number of significant binary digits in weights.
- BinaryWeightConstraintChanges: Restricts additional significant binary digits in weight changes compared to previous weights.
- OscillationDampener: Reduces oscillations by modifying the smallest non-zero binary digit in weights.
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
            
            # Convert float to its binary representation
            res = struct.unpack('!I', struct.pack('!f', value))[0]
            # Create and return a 32-bit binary string
            binary_str = f"{res:032b}"
            return binary_str
        except Exception:
            return "0"
    
    def _count_significant_binary_digits(self, binary_str: str) -> int:
        """Count significant binary digits."""
        try:
            if binary_str == "0": return 0

            # Find the first and last significant '1'
            first_one = binary_str.find('1')
            last_one = binary_str.rfind('1')
            # The significant part is the entire string between the first and last '1'
            significant_part = binary_str[first_one : last_one]
            # The count is the length of this part
            return len(significant_part)
        except (ValueError, TypeError):
            return 0
        
    def _binary_string_to_float(self, binary_str: str) -> float:
        """Convert binary string representation back to float."""
        try:
            if binary_str == "0": return 0.0

            # Pad to 32 bits
            padded_binary = binary_str.zfill(32)
            # Convert binary string to integer
            int_value = int(padded_binary, 2)
            # Convert binary integer to float
            return struct.unpack('!f', struct.pack('!I', int_value))[0]
        except Exception:
            return 0.0
    
    def get_error_count(self) -> int:
        """Get the number of errors encountered."""
        return self.error_count
    
    @abstractmethod
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Apply the binary constraint."""
        pass


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
            reduced_binary = binary_repr[:-reduction_factor] + '0' * reduction_factor
            return self._binary_string_to_float(reduced_binary)
        except Exception:
            self.error_count += 1
            return weight
    
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Apply maximum binary precision constraint to weight matrix."""
        try:
            flat_weights = weights.flatten()
            flat_constrained = np.array([self._constrain_weight_max(w) for w in flat_weights])
            constrained_weights = flat_constrained.reshape(weights.shape)
            return constrained_weights
        except Exception:
            self.error_count += 1
            return weights


class BinaryWeightConstraintChanges(BinaryWeightConstraint):
    """Manages binary precision of weight changes, limiting additional 
    significant digits in binary format compared to previous weights."""
    
    def __init__(self, max_additional_digits: int = 1):
        super().__init__()
        self.max_additional_digits = max_additional_digits
        self.previous_weights = None

    def _constrain_weight_change(self, current_weight: float, previous_weight: float) -> float:
        """Constrain a single weight to have at most max_additional_digits more"""
        try:
            current_binary = self._float_to_binary_repr(current_weight)
            previous_binary = self._float_to_binary_repr(previous_weight)
            current_digits = self._count_significant_binary_digits(current_binary)
            previous_digits = self._count_significant_binary_digits(previous_binary)
            max_allowed_digits = previous_digits + self.max_additional_digits

            if current_digits <= max_allowed_digits: return current_weight
            
            # Reduce precision
            reduction_factor = current_digits - max_allowed_digits
            reduced_binary = current_binary[:-reduction_factor] + '0' * reduction_factor
            return self._binary_string_to_float(reduced_binary)
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
            previous_flat_weights = self.previous_weights.flatten()
            flat_constrained = np.array([self._constrain_weight_change(w, pw) 
                    for w, pw in zip(flat_weights, previous_flat_weights)])
            constrained_weights = flat_constrained.reshape(weights.shape)
            self.previous_weights = constrained_weights.copy()
            return constrained_weights
        except Exception:
            self.error_count += 1
            return weights
    
    def reset(self):
        self.previous_weights = None
        self.error_count = 0


class OscillationDampener(BinaryWeightConstraint):
    """Monitors weight changes and dampens oscillations by setting the smallest 
    non-zero binary digits to zero when oscillation patterns are detected."""
    
    def __init__(self):
        super().__init__()
        self.weight_history: List[np.ndarray] = []
    
    def add_weights(self, weights: np.ndarray) -> None:
        """Add new weights to the history."""
        try:
            self.weight_history.append(weights.copy())
        except Exception:
            self.error_count += 1
    
    def _detect_oscillation_pattern(self, values: List[float]) -> bool:
        """Detect if values show an oscillation pattern."""
        try:
            if len(values) < 3: return False

            return (values[0] < values[1] > values[2]) or (values[0] > values[1] < values[2])
        except Exception:
            return False
    
    def _set_smallest_binary_digits_to_zero(self, weight: float) -> float:
        """Set a number of least significant binary digits to zero based on shrinkage from global mean.
        The number of digits zeroed increases with the difference from the mean.
        """
        try:
            if weight == 0.0: return weight

            # Get the significant bit count
            bit_count = self._count_significant_binary_digits(self._float_to_binary_repr(weight))
            # Calculate global mean of historical bias vector weights
            global_mean = np.mean([np.mean(hist_weights) 
                                   for hist_weights in self.weight_history]) if self.weight_history else 0.0
            # Calculate non-linear, decaying shrinkage factor based on distance from mean
            raw_factor = abs(weight - global_mean) / max(abs(global_mean), 1e-8)
            shrinkage_factor = 1 - np.exp(-2 * raw_factor)
            # Determine number of digits to zero from all significant bits (at least 1)
            digits_to_zero = max(1, int(1 + shrinkage_factor * (bit_count - 1)))
            # Zero out the specified number of least significant bits (reducing the precision)
            packed = struct.pack('f', weight)
            bits = struct.unpack('I', packed)[0]

            for _ in range(digits_to_zero):
                bits &= bits - 1

            modified_packed = struct.pack('I', bits)
            modified_weight = struct.unpack('f', modified_packed)[0]
            return modified_weight
        except Exception:
            self.error_count += 1
            return weight * 0.99
    
    def apply_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Detect oscillations and apply dampening."""
        try:
            if len(self.weight_history) < 2: return weights 

            flat_current = weights.flatten()
            flat_dampened = flat_current.copy()
            
            for i in range(len(flat_current)):
                weight_sequence = []
                
                for hist_weights in self.weight_history:
                    weight_sequence.append(hist_weights.flatten()[i])
                
                weight_sequence.append(flat_current[i])
                recent_values = weight_sequence[-3:]

                if self._detect_oscillation_pattern(recent_values):
                    flat_dampened[i] = self._set_smallest_binary_digits_to_zero(weight=flat_current[i])
            
            return flat_dampened.reshape(weights.shape)
        except Exception:
            self.error_count += 1
            return weights
    
    def reset(self) -> None:
        """Reset the history and error count."""
        self.weight_history = []
        self.error_count = 0
