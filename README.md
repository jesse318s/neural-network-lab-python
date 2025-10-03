# neural-network-lab-python

A TensorFlow implementation featuring custom weight constraints, adaptive loss functions, and performance tracking for neural network training with particle physics simulation data.

## Features

- **Binary Weight Constraints**: Control binary precision of neural network weights
- **Oscillation Dampening**: Prevent weight oscillations during training
- **Adaptive Loss Functions**: Dynamically adjusts MSE/MAE based on training
- **Performance Tracking**: Comprehensive metrics collection with CSV export
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
├── ml_config/                   # ML configuration files
|   ├── model_config.json
|   └── training_config.json
├── advanced_neural_network.py   # Core neural network implementation
├── data_processing.py           # Data processing functionality
├── main.py                      # Main training script
├── ml_utils.py                  # ML utilities (adaptive loss)
├── performance_tracker.py       # Metrics tracking and CSV output
├── requirements.txt             # Dependencies
├── test_main.py                 # Test suite
├── weight_constraints.py        # Binary weight management
├── saved_weights/               # Model weights generated during training
└── training_output/             # Generated results (name may vary based on config)
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

## Configuration

Key configuration options:

```json
{
  "enable_weight_oscillation_dampener": true,
  "enable_binary_change_max": true,
  "max_additional_binary_digits": 16,
  "enable_binary_precision_max": true,
  "max_binary_digits": 24,
  "loss_weighting_strategy": "combined"
}
```
