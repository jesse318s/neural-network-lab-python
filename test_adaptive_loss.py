"""
Test script for adaptive loss functions and helper utilities.

This script demonstrates the curve_fancy adaptive loss strategy and 
physics-based loss functions implemented in ml_utils.py.
"""

import numpy as np
from ml_utils import (
    unit_vec, one_maker, square_but_preserve_signs, squarer_diff_sign_preserver,
    epoch_weight_sine_for_one_weight, epoch_weight_sine_based,
    adaptive_loss_no_sin, curve_fancy,
    kinetic_component, magnetic_potential, physics_loss
)


def test_helper_functions():
    """Test basic helper functions."""
    print("=" * 60)
    print("TESTING HELPER FUNCTIONS")
    print("=" * 60)
    
    # Test unit_vec
    print("\n1. Testing unit_vec:")
    test_vec = np.array([3, 4])
    result = unit_vec(test_vec)
    print(f"   Input: {test_vec}")
    print(f"   Unit vector: {result}")
    print(f"   Magnitude: {np.linalg.norm(result):.4f}")
    
    # Test one_maker
    print("\n2. Testing one_maker:")
    test_weights = np.array([1, 2, 3, 4])
    result = one_maker(test_weights)
    print(f"   Input: {test_weights}")
    print(f"   Normalized: {result}")
    print(f"   Sum: {np.sum(result):.4f}")
    
    # Test square_but_preserve_signs
    print("\n3. Testing square_but_preserve_signs:")
    
    for x in [-3, -1, 0, 1, 3]:
        result = square_but_preserve_signs(x)
        print(f"   {x} -> {result}")


def test_sine_based_weights():
    """Test sine-based weight generation."""
    print("\n" + "=" * 60)
    print("TESTING SINE-BASED WEIGHT GENERATION")
    print("=" * 60)
    
    num_loss_funcs = 2
    
    print(f"\nWeight evolution over 10 epochs ({num_loss_funcs} loss functions):")
    print(f"{'Epoch':<8} {'Weight 0':<12} {'Weight 1':<12} {'Sum':<12}")
    print("-" * 50)
    
    for epoch in range(10):
        weights = epoch_weight_sine_based(epoch, num_loss_funcs)
        print(f"{epoch:<8} {weights[0]:<12.4f} {weights[1]:<12.4f} {np.sum(weights):<12.4f}")


def test_adaptive_loss():
    """Test adaptive loss adjustment."""
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE LOSS ADJUSTMENT")
    print("=" * 60)
    
    # Test case 1: Improving performance
    print("\n1. Improving performance (loss decreasing):")
    loss_list = [1.0, 0.8, 0.6]
    weight_list = [np.array([0.5, 0.5]), np.array([0.6, 0.4]), np.array([0.7, 0.3])]
    result = adaptive_loss_no_sin(loss_list, weight_list)
    print(f"   Loss: {loss_list}")
    print(f"   Adjustment: {result}")
    
    # Test case 2: Worsening performance
    print("\n2. Worsening performance (loss increasing):")
    loss_list = [0.6, 0.8, 1.0]
    weight_list = [np.array([0.7, 0.3]), np.array([0.6, 0.4]), np.array([0.5, 0.5])]
    result = adaptive_loss_no_sin(loss_list, weight_list)
    print(f"   Loss: {loss_list}")
    print(f"   Adjustment: {result}")


def test_curve_fancy():
    """Test the curve_fancy adaptive strategy."""
    print("\n" + "=" * 60)
    print("TESTING CURVE_FANCY ADAPTIVE STRATEGY")
    print("=" * 60)
    
    num_loss_funcs = 2
    
    # Early training (not enough data)
    print("\n1. Early training (epoch 2):")
    loss_list = [1.0, 0.9]
    weight_list = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    result = curve_fancy(loss_list, weight_list, num_loss_funcs)
    print(f"   Loss history: {loss_list}")
    print(f"   Weights: {result}")
    
    # Mid training (enough data for curve fitting)
    print("\n2. Mid training (epoch 10):")
    loss_list = [1.0, 0.9, 0.85, 0.82, 0.80, 0.78, 0.75, 0.73, 0.71, 0.70]
    weight_list = [np.array([0.5, 0.5]) for _ in range(10)]
    result = curve_fancy(loss_list, weight_list, num_loss_funcs)
    print(f"   Loss history (last 5): {loss_list[-5:]}")
    print(f"   Weights: {result}")
    print(f"   Sum: {np.sum(result):.4f}")


def test_physics_loss():
    """Test physics-based loss functions."""
    print("\n" + "=" * 60)
    print("TESTING PHYSICS-BASED LOSS FUNCTIONS")
    print("=" * 60)
    
    # Test kinetic component
    print("\n1. Kinetic component:")
    
    for x_vel, y_vel in [(3, 4), (0, 5), (-2, -2)]:
        kc = kinetic_component(x_vel, y_vel)
        print(f"   v=({x_vel}, {y_vel}) -> KE={kc:.4f}")
    
    # Test magnetic potential
    print("\n2. Magnetic potential:")
    mag_field, charge = 1.0, 1.0
    
    for y_pos, x_pos, x_vel, y_vel in [(1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 1, 1)]:
        mp = magnetic_potential(mag_field, y_pos, x_pos, x_vel, y_vel, charge)
        print(f"   pos=({x_pos}, {y_pos}), v=({x_vel}, {y_vel}) -> MP={mp:.4f}")
    
    # Test complete physics loss
    print("\n3. Complete physics loss (energy conservation):")
    mag_field, charge = 1.0, 1.0
    
    # Perfect conservation (should be ~0)
    loss = physics_loss(mag_field, charge, 0, 1, 1, 0, 1, 0, 0, 1)
    print(f"   Perfect conservation: loss={loss:.6f}")
    
    # Imperfect conservation
    loss = physics_loss(mag_field, charge, 0, 1, 1, 0, 1, 0, 0.5, 0.5)
    print(f"   Imperfect conservation: loss={loss:.6f}")


def test_integration_example():
    """Demonstrate how the adaptive strategy evolves over training."""
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE: SIMULATED TRAINING")
    print("=" * 60)
    
    num_loss_funcs = 2
    num_epochs = 20
    
    # Simulate decreasing loss over training
    np.random.seed(42)
    loss_list = []
    weight_list = []
    
    print(f"\n{'Epoch':<8} {'Loss':<12} {'MSE Weight':<12} {'MAE Weight':<12}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Simulate loss decreasing with some noise
        loss = 1.0 * np.exp(-epoch / 10) + 0.1 * np.random.randn()
        loss = max(0.1, loss)
        loss_list.append(loss)
        
        # Calculate weights using curve_fancy
        weights = curve_fancy(loss_list, weight_list, num_loss_funcs)
        weight_list.append(weights)
        
        print(f"{epoch:<8} {loss:<12.4f} {weights[0]:<12.4f} {weights[1]:<12.4f}")
    
    print("\nObservations:")
    print("- Early epochs use sine-based exploration")
    print("- Later epochs adapt based on loss trajectory")
    print("- Weights remain normalized and positive")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("ADAPTIVE LOSS FUNCTION TEST SUITE")
    print("*" * 60)
    
    test_helper_functions()
    test_sine_based_weights()
    test_adaptive_loss()
    test_curve_fancy()
    test_physics_loss()
    test_integration_example()
    
    print("\n" + "*" * 60)
    print("ALL TESTS COMPLETED")
    print("*" * 60)
    print()


if __name__ == "__main__":
    main()
