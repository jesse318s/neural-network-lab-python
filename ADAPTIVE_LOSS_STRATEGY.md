# Adaptive Loss Strategy Documentation

## Overview

The adaptive loss strategy implements dynamic loss function weighting during neural network training. Instead of using fixed weights for combining multiple loss metrics (MSE and MAE), this strategy adjusts weights based on training progress to optimize model performance.

## Strategies Available

### 1. `r2_based`
Adjusts weights based on R² score improvement:
- **Higher R²** → More emphasis on MSE (fine-tuning)
- **Lower R²** → More emphasis on MAE (robustness)

### 2. `loss_based`
Adjusts weights based on loss value trends:
- **Decreasing loss** → More emphasis on MSE
- **Higher loss** → More emphasis on MAE

### 3. `combined`
Averages the `r2_based` and `loss_based` strategies for balanced adaptation.

### 4. `curve_fancy` ⭐ **NEW**
Advanced adaptive strategy using curve-fitting and sine-based exploration:
- **Early epochs**: Uses sine-based weight oscillation to explore different weight combinations
- **Later epochs**: Adapts based on loss trajectory using gradient-like adjustments
- **Features**:
  - Preserves sign information in weight adjustments
  - Ensures all weights remain positive and normalized
  - Prevents any weight from becoming zero (minimum 0.1)
  - Combines exploration (sine) with exploitation (gradient)

## Mathematical Foundation

### Sine-Based Exploration

For weight `i` at epoch `e` with `N` total loss functions:

```
weight_i(e) = (sin(e + 2π * i / N) + 1) / 2
```

This creates oscillating weights that:
- Always stay between 0 and 1
- Are evenly phase-shifted across loss functions
- Explore different weight combinations systematically

### Adaptive Adjustment

When sufficient training history exists:

1. **Calculate weight change**: 
   ```
   Δw = w_t - w_{t-1}
   ```

2. **Square while preserving sign**:
   ```
   Δw' = sign(Δw) * Δw²
   ```

3. **Normalize to unit vector**:
   ```
   u = Δw' / ||Δw'||
   ```

4. **Direction based on loss change**:
   ```
   adjustment = -u  if loss_t < loss_{t-1}  (improving)
                 u  if loss_t ≥ loss_{t-1}  (not improving)
   ```

5. **Combine with exploration**:
   ```
   new_weights = adjustment + sine_weights(epoch)
   ```

6. **Ensure positivity and normalization**:
   ```
   new_weights = max(0.1, new_weights - min(new_weights))
   new_weights = new_weights / sum(new_weights)
   ```

## Physics-Based Loss Functions

The module also includes physics-informed loss functions for particle dynamics:

### Kinetic Energy Component
```python
KE = (v_x² + v_y²) / 2
```
(Mass factored out as it's constant)

### Magnetic Potential Energy
```python
U_mag = |q * B * (y * v_x - x * v_y)|
```
Where:
- `q` = particle charge
- `B` = magnetic field strength
- `(x, y)` = particle position
- `(v_x, v_y)` = particle velocity

### Physics Loss (Energy Conservation)
```python
loss = |KE_new - KE_orig + U_mag_new - U_mag_orig|
```

This ensures total energy (kinetic + potential) remains constant.

## Usage

### Configuration

Edit `ml_config/model_config.json`:

```json
{
  "loss_weighting_strategy": "curve_fancy"
}
```

Available options:
- `"none"` - Fixed 50/50 weights
- `"r2_based"` - R² score adaptive
- `"loss_based"` - Loss value adaptive
- `"combined"` - Combined strategy
- `"curve_fancy"` - Advanced curve-fitting strategy

### In Code

```python
from ml_utils import create_adaptive_loss_fn

# Create adaptive loss function
adaptive_loss = create_adaptive_loss_fn(strategy='curve_fancy')

# During training, update state each epoch
adaptive_loss.update_state(epoch=epoch_num, prev_r2=r2_score)

# Get current weights info
info = adaptive_loss.get_current_info()
print(info)  # "curve_fancy (MSE: 0.653, MAE: 0.347)"

# Get complete history
history = adaptive_loss.get_history()
```

### Testing

Run the test suite:

```bash
python test_adaptive_loss.py
```

This demonstrates:
- Helper function behavior
- Sine-based weight generation
- Adaptive loss adjustment
- Curve fancy strategy evolution
- Physics-based loss calculations
- Simulated training progression

## How It Works During Training

### Phase 1: Exploration (Epochs 1-4)
- Insufficient data for curve fitting
- Uses sine-based oscillation
- Systematically explores weight combinations
- Weights oscillate smoothly between configurations

### Phase 2: Adaptive Learning (Epochs 5+)
- Sufficient history for gradient calculation
- Combines sine exploration with adaptive adjustment
- If loss improving: moves in current direction (squared for emphasis)
- If loss worsening: reverses direction
- Maintains exploration component to avoid local optima

### Example Evolution

```
Epoch    Loss      MSE Weight    MAE Weight
-----    ----      ----------    ----------
0        1.0000    0.5000        0.5000     (sine exploration)
1        0.9123    0.8535        0.1465     (sine exploration)
2        0.8456    0.5000        0.5000     (sine exploration)
3        0.7891    0.1465        0.8535     (sine exploration)
4        0.7345    0.5000        0.5000     (transition)
5        0.6823    0.6234        0.3766     (adaptive + sine)
10       0.5123    0.7145        0.2855     (adaptive + sine)
20       0.3456    0.8023        0.1977     (adaptive + sine)
```

## Key Advantages

1. **Automatic Tuning**: No manual hyperparameter tuning for loss weights
2. **Exploration**: Sine component prevents premature convergence
3. **Exploitation**: Gradient component accelerates in promising directions
4. **Robustness**: All weights bounded and normalized
5. **Interpretability**: Weight history shows training dynamics
6. **Stability**: Minimum weight threshold prevents division by zero

## Implementation Details

### State Management
The adaptive loss function maintains:
- Current epoch number
- Previous R² score
- Previous loss value
- Complete loss history
- Complete weight history
- Error count

### Error Handling
- Falls back to MSE on computation errors
- Tracks error count in history
- Continues training even if adaptation fails

### Performance
- Minimal computational overhead
- History stored for analysis
- Efficient NumPy operations

## Customization

To modify the strategy:

1. **Adjust exploration period**: Change `min_to_do_fancy` in `curve_fancy()`
2. **Change weight bounds**: Modify minimum weight (default 0.1)
3. **Alter oscillation frequency**: Adjust sine wave parameters
4. **Add new strategies**: Extend `compute_loss_weights()` with new cases

## References

The curve fancy strategy is inspired by:
- Adaptive optimization techniques
- Multi-objective optimization
- Curriculum learning
- Exploration-exploitation trade-offs in reinforcement learning

## Future Enhancements

Potential improvements:
- [ ] Add physics loss integration for particle simulation datasets
- [ ] Implement multi-loss (>2) support
- [ ] Add momentum term to weight adjustments
- [ ] Create visualization tools for weight evolution
- [ ] Benchmark against fixed weight strategies
