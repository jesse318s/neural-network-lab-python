# neural-network-lab-python

A TensorFlow implementation featuring custom weight constraints, adaptive loss functions, performance tracking, and experiment analysis for neural network training with particle physics simulation data.

## Features

- **Binary Weight Constraints**: Control binary precision of neural network weights
- **Oscillation Dampening**: Prevent weight oscillations during training
- **Adaptive Loss Functions**: Dynamically adjusts MSE/MAE based on training
- **Performance Tracking**: Comprehensive metrics collection with CSV export
- **Experiment Analysis Framework**: Tools for analyzing training experiments
- **Error Resilience**: Graceful degradation on component failures

## Core Components

### Weight Constraints

- **Binary Precision Control**: Limits binary digits in weight representations
- **Oscillation Dampening**: Detects and prevents weight oscillation patterns

### Adaptive Loss Functions

- **R²-Based**: Modifies weights based on validation R² score
- **Loss-Based**: Adapts based on previous loss values
- **Combined Strategy**: Intelligently combines both strategies

### Performance Tracking

- Training metrics and result tracking
- Memory usage and timing measurements
- CSV export with comprehensive logging

### Experiment Analysis Framework

- **Comprehensive Visualization Dashboards**: Training dynamics, residual analysis, hyperparameter impact
- **Statistical Hyperparameter Recommendations**: Data-driven suggestions with confidence intervals
- **Baseline Benchmarking**: Comparison against mean, linear regression, and standard neural networks
- **James-Stein Estimator Comparison**: Rigorous evaluation of binary weight constraints vs statistical shrinkage
- **Automated Report Generation**: Publication-quality figures and JSON summary exports
- **Modular Analysis Sections**: Independent execution of specific analysis components

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: TensorFlow, NumPy, Pandas, scikit-learn, psutil

## Usage

### Quick Start

```bash
# Run the complete training pipeline
python main.py

# Run tests
python test_main.py
```

### Basic Example

```python
from data_processing import complete_data_pipeline
from advanced_neural_network import AdvancedNeuralNetwork
from main import train_with_tracking

# Load data and create model
data_splits = complete_data_pipeline(num_particles=1000)
X_train, X_val, X_test, y_train, y_val, y_test = data_splits
config = {'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001, 'dropout_rate': 0.02}
model = AdvancedNeuralNetwork((X_train.shape[1],), y_train.shape[1], config)

# Train with tracking
results = train_with_tracking(model, X_train, X_val, X_test, y_train, y_val, y_test, config)
```

## Project Structure

```
├── ml_config/                        # ML configuration files
|   ├── model_config.json
|   └── training_config.json
├── advanced_neural_network.py        # Core neural network implementation
├── data_processing.py                # Data processing functionality
├── experiment_analysis_framework_refactored.ipynb  # Refactored experiment analysis notebook
├── experiment_analysis_utils.py      # Analysis utility functions
├── james_stein_constraint.py         # James-Stein weight constraint implementation
├── main.py                           # Main training script
├── ml_utils.py                       # ML utilities (adaptive loss)
├── performance_tracker.py            # Metrics tracking and CSV output
├── requirements.txt                  # Dependencies
├── test_main.py                      # Test suite
├── weight_constraints.py             # Binary weight management
├── saved_weights/                    # Model weights generated during training
└── training_output/                  # Generated results (name may vary based on config)
    ├── analysis/
    |   ├── figures/                  # Publication-quality visualizations (300 dpi)
    |   └── analysis_summary_*.json   # Timestamped analysis exports
    ├── training_results.csv
    ├── loss_history.csv
    ├── training_log.txt
    └── configuration_log.csv
```

## Output Files

Training generates comprehensive logs and metrics:

- Training and validation metrics per epoch
- Weight evolution and constraint applications
- Loss function component tracking
- Model weights and performance statistics
- Error logs and configuration records

Analysis tools provide comprehensive experiment diagnostics:

- **Training Dynamics Dashboard**: Loss curves, R² progression, memory usage, generalization gap
- **Residual Analysis Suite**: Distribution histograms, Q-Q plots, scatter plots, per-target breakdowns
- **Hyperparameter Impact Heatmap**: Correlation matrix showing parameter-performance relationships
- **James-Stein Comparison**: Statistical comparison of weight constraint methods
- **Baseline Benchmarking**: Performance vs mean predictor, linear regression, unconstrained models

## Key Features

### Binary Weight Constraints

Controls weight precision at the binary level, preventing explosive growth while maintaining numerical stability.

### Oscillation Dampening

Detects weight oscillation patterns across epochs and applies dampening to stabilize training.

### Adaptive Loss Functions

Dynamically adjusts MSE/MAE weighting based on R² and loss history.

### Error Resilience

Implements graceful degradation - training continues even when individual components encounter errors.

## Testing

```bash
python test_main.py # Run standard tests
```

Tests cover weight constraints, adaptive loss functions, performance tracking, and error handling.

## Experiment Analysis Framework

### Overview

The refactored experiment analysis framework provides a comprehensive toolkit for analyzing neural network training experiments with statistical rigor and publication-quality visualizations.

### Usage

#### Running the Analysis

Open and execute `experiment_analysis_framework_refactored.ipynb` in Jupyter:

```bash
jupyter notebook experiment_analysis_framework_refactored.ipynb
```

Or in VS Code with the Jupyter extension:
1. Open the notebook file
2. Select Python kernel
3. Run all cells sequentially

#### What It Does

The notebook automatically:
1. **Validates artifacts**: Checks for required training outputs, configs, and checkpoints
2. **Loads data**: Ingests training logs, particle data, scalers, and model weights
3. **Generates visualizations**: Creates 8+ publication-quality figures (300 dpi)
4. **Performs benchmarking**: Compares model against baselines and James-Stein shrinkage
5. **Provides recommendations**: Statistical hyperparameter suggestions with confidence intervals
6. **Exports results**: Saves figures and JSON summary with key metrics

### Key Features

#### 1. Training Dynamics Dashboard

Comprehensive 2×2 visualization showing:
- Training/validation loss with confidence bands
- R² score progression with best epoch marker
- Computational resource usage (time and memory)
- Generalization gap evolution

#### 2. Residual Analysis Suite

Four-panel diagnostic including:
- Residual distribution histogram with normal fit
- Q-Q plot for normality testing
- Residuals vs predicted values scatter
- Per-target residual boxplots

#### 3. James-Stein Estimator Comparison

Rigorous evaluation of weight constraint effectiveness:
- Trains three model variants with identical hyperparameters
- Compares binary constraints vs James-Stein shrinkage vs no constraints
- Provides convergence speed, performance metrics, and weight distributions
- Includes statistical significance testing (Wilcoxon signed-rank, Cohen's d)

**Example Results:**
```
Model                              Best Val Loss  Convergence Epoch
Binary Constraints                      0.0234                42
James-Stein                             0.0241                38
No Constraints                          0.0298                45
```

#### 4. Advanced Hyperparameter Recommendations

Data-driven suggestions with statistical backing:
- **Learning rate sensitivity**: Detects plateaus and suggests adjustments
- **Overfitting detection**: One-sample t-test for train/val gap significance
- **Batch size optimization**: Analyzes memory headroom and epoch time
- **Convergence analysis**: Extrapolates improvement potential
- **Historical pattern recognition**: Identifies optimal settings from past runs

Each recommendation includes:
- Current value and suggested alternatives
- Confidence level (High/Medium/Low)
- Statistical evidence (p-values, test statistics, correlations)
- Expected impact quantification
- Priority ranking

#### 5. Hyperparameter Impact Heatmap

Color-coded correlation matrix showing:
- Relationship between hyperparameters (learning rate, dropout, batch size)
- Impact on performance metrics (R², MAE, training time)
- Strongest positive/negative correlations
- Interaction effects

#### 6. Baseline Model Comparison

Benchmarking against:
- **Mean Baseline**: Predicts training set mean for all samples
- **Linear Regression**: Standard scikit-learn LinearRegression
- **Advanced NN (Binary Constraints)**: Current model

Metrics compared: R², MAE, RMSE, MAPE

### Outputs

All analysis artifacts are saved to `training_output/analysis/`:

#### Figures (PNG, 300 dpi)
- `training_dynamics_dashboard.png` - Training curves and resource usage
- `learning_rate_analysis.png` - LR impact on performance
- `residual_analysis_suite.png` - Comprehensive residual diagnostics
- `prediction_scatter_plots.png` - Actual vs predicted for each target
- `baseline_comparison.png` - Benchmarking bar charts
- `james_stein_comparison.png` - Weight constraint evaluation
- `hyperparameter_impact_heatmap.png` - Parameter correlation matrix

#### Data Exports
- `analysis_summary_YYYYMMDD_HHMMSS.json` - Timestamped metrics summary

Example JSON structure:
```json
{
  "analysis_timestamp": "2025-10-11T14:30:00",
  "best_validation_loss": 0.0234,
  "best_r2_score": 0.9567,
  "total_training_epochs": 100,
  "prediction_mae": 0.0421,
  "james_stein_comparison": {
    "binary_constraints_best_loss": 0.0234,
    "james_stein_best_loss": 0.0241,
    "winner": "Binary Constraints"
  }
}
```

### Customization

#### Adjusting Sample Size

Control prediction sample size for faster analysis:

```python
SAMPLE_SIZE = 256  # Reduce for speed, increase for accuracy
```

#### Skipping James-Stein Comparison

Comment out or skip the James-Stein comparison cells to save time (trains 3 models).

#### Custom Visualizations

The notebook uses modular utility functions from `experiment_analysis_utils.py`. Import and use them in custom cells:

```python
from experiment_analysis_utils import compute_predictions, summarize_run_performance

# Generate custom analysis
custom_residuals, custom_metrics = compute_predictions(
    model, scaler_X, scaler_y, particle_df, sample_size=1000
)
```

### Interpreting Results

#### Status Indicators
- ✓ **Success**: Metric within healthy range
- ⚠ **Warning**: Attention recommended but not critical
- ✗ **Critical**: Immediate action required

#### Recommendation Priorities
- **High Priority**: Significant impact expected, implement immediately
- **Medium Priority**: Moderate improvements, consider for next experiment
- **Low Priority**: Minor optimizations, optional refinements

#### Statistical Evidence
- **p-value < 0.05**: Statistically significant finding (α=0.05)
- **Cohen's d**: Effect size (0.2=small, 0.5=medium, 0.8=large)
- **95% CI**: Confidence interval for expected performance

### Troubleshooting

**Issue**: Missing artifacts error
- **Solution**: Run `main.py` to generate training outputs first

**Issue**: Kernel crashes during James-Stein comparison
- **Solution**: Reduce `common_config["epochs"]` in that cell or increase available memory

**Issue**: Figures not displaying
- **Solution**: Check `paths["figures_dir"]` exists and has write permissions

**Issue**: Historical data empty
- **Solution**: Run multiple training experiments to populate config history

## Configuration

Key configuration options:

```json
{
  "enable_weight_oscillation_dampener": true,
  "use_adaptive_oscillation_dampener": true,
  "enable_binary_change_max": true,
  "max_additional_binary_digits": 16,
  "enable_binary_precision_max": true,
  "max_binary_digits": 24,
  "loss_weighting_strategy": "combined"
}
```
