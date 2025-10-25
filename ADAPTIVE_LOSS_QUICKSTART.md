# Quick Start Guide: Adaptive Loss Strategy

## What is it?

An advanced loss function weighting strategy that automatically adjusts how much emphasis to place on different error metrics (MSE vs MAE) during training. Think of it as the model learning which type of error correction is most helpful at each stage of training.

## How to Use

### 1. Choose Your Strategy

Edit `ml_config/model_config.json`:

```json
{
  "loss_weighting_strategy": "curve_fancy"
}
```

**Available strategies:**
- `"curve_fancy"` - **Recommended** - Smart adaptive with exploration
- `"combined"` - Balanced R² and loss-based adaptation
- `"r2_based"` - Adapts based on R² score
- `"loss_based"` - Adapts based on loss values
- `"none"` - Fixed 50/50 weights (baseline)

### 2. Run Training

```bash
python main.py
```

The adaptive loss will automatically:
- Explore different weight combinations early on
- Adapt based on what's working as training progresses
- Keep weights balanced and stable

### 3. View Results

Check the training output for loss weight information:
```
Epoch   5/2000 - Train Loss: 0.4523, Val Loss: 0.4891, R²: 0.6234
    Loss weights: curve_fancy (MSE: 0.653, MAE: 0.347)
```

After training, examine `training_output/adaptive_loss_history.csv` to see how weights evolved.

## Understanding `curve_fancy`

### The Strategy

**Epochs 1-4:** Exploration Phase
- Uses sine waves to systematically try different weight combinations
- Weights oscillate between emphasizing MSE and MAE
- Gathers data about what works

**Epochs 5+:** Adaptive Phase
- Analyzes loss trajectory
- If loss improving → continues in current direction (amplified)
- If loss worsening → reverses direction
- Still includes exploration component to avoid getting stuck

### Why It Works

1. **Early exploration** finds promising weight ranges
2. **Adaptive adjustment** capitalizes on what's working
3. **Continuous exploration** prevents premature convergence
4. **Guaranteed stability** - weights always positive and normalized

### Example Evolution

```
Early:     MSE=0.50, MAE=0.50  (exploring)
           MSE=0.85, MAE=0.15  (exploring)
           MSE=0.50, MAE=0.50  (exploring)
           MSE=0.15, MAE=0.85  (exploring)

Mid:       MSE=0.62, MAE=0.38  (adapting + exploring)
           MSE=0.71, MAE=0.29  (adapting + exploring)

Late:      MSE=0.80, MAE=0.20  (fine-tuning emphasis)
```

## Testing

Run the test suite to see the strategy in action:

```bash
python test_adaptive_loss.py
```

This shows:
- How weights evolve over epochs
- Adaptive adjustments based on loss changes
- Physics-based loss calculations (for particle data)
- Complete simulated training example

## When to Use Each Strategy

### Use `curve_fancy` when:
- ✅ You want the best performance with minimal tuning
- ✅ Training for many epochs (50+)
- ✅ You're not sure which metrics matter most
- ✅ You want interpretable weight evolution

### Use `combined` when:
- ✅ You want something simpler but still adaptive
- ✅ Training for fewer epochs
- ✅ You want smoother, more predictable behavior

### Use `r2_based` when:
- ✅ R² score is your primary metric
- ✅ You want weights tied directly to model performance

### Use `loss_based` when:
- ✅ You care more about absolute loss values
- ✅ You want simple loss-driven adaptation

### Use `none` when:
- ✅ You want a baseline for comparison
- ✅ You already know optimal weights
- ✅ You're debugging

## Troubleshooting

### Weights seem random
**Issue:** Seeing lots of variation in weights epoch-to-epoch  
**Solution:** This is normal for `curve_fancy` - it includes exploration. The overall trend matters more than individual values.

### All weight going to one metric
**Issue:** MSE weight = 1.0, MAE weight = 0.0  
**Solution:** The strategy found this works best. Check if your data has outliers (favors MAE) or needs precision (favors MSE).

### Want to understand what's happening
**Issue:** Not sure why weights are changing  
**Solution:** 
1. Run `test_adaptive_loss.py` to see the algorithm behavior
2. Check `training_output/adaptive_loss_history.csv`
3. Read `ADAPTIVE_LOSS_STRATEGY.md` for full mathematical details

## Advanced: Physics-Based Loss

For particle physics simulations, you can also use energy conservation loss:

```python
from ml_utils import physics_loss

# Calculate physics-informed loss
energy_loss = physics_loss(
    mag_field, charge,
    x_pos_orig, y_pos_orig, x_vel_orig, y_vel_orig,
    x_pos_pred, y_pos_pred, x_vel_pred, y_vel_pred
)
```

This ensures predictions conserve energy (kinetic + magnetic potential).

## More Information

- **Full documentation**: `ADAPTIVE_LOSS_STRATEGY.md`
- **Test examples**: `test_adaptive_loss.py`
- **Implementation**: `ml_utils.py`

## Quick Tips

💡 **Tip 1:** Start with `curve_fancy` and only change if you have a specific reason  
💡 **Tip 2:** Run for at least 100 epochs to see adaptation benefits  
💡 **Tip 3:** Compare against `none` strategy to measure improvement  
💡 **Tip 4:** Check weight evolution in CSV to understand your training dynamics  
💡 **Tip 5:** Physics loss works best when combined with standard losses
