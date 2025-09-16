# Neural Network Lab - Advanced Training Techniques

A TensorFlow implementation featuring custom weight constraints, adaptive loss functions, and performance tracking for neural network training with particle physics simulation data.

## Features

- **Binary Weight Constraints**: Control binary precision of neural network weights
- **Oscillation Dampening**: Prevent weight oscillations during training  
- **Adaptive Loss Functions**: Dynamically combine MSE and MAE based on training progress
- **Performance Tracking**: Comprehensive metrics collection with CSV export

## Core Components

### Weight Constraints
- **Binary Precision Control**: Limits binary digits in weight representations
- **Oscillation Dampening**: Detects and prevents weight oscillation patterns

### Adaptive Loss Functions
- **Epoch-Based**: Adjusts MSE/MAE ratio based on training progress
- **Accuracy-Based**: Modifies weights based on validation accuracy
- **Loss-Based**: Adapts based on previous loss values
- **Combined Strategy**: Intelligently combines all strategies

### Performance Tracking
- Training metrics and accuracy tracking
- Memory usage and timing measurements
- CSV export with comprehensive logging

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: TensorFlow, NumPy, Pandas, scikit-learn, matplotlib, psutil

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
from main import create_model, train_with_tracking
from data_loader import load_and_prepare_data

# Load data and create model
X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_prepare_data()
model = create_model(input_shape=(X_train.shape[1],), output_shape=y_train.shape[1])

# Train with tracking
config = {'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001}
results = train_with_tracking(model, X_train, X_val, X_test, y_train, y_val, y_test, config)
```

## Project Structure

```
├── main.py                      # Main training script
├── advanced_neural_network.py   # Core neural network implementation
├── weight_constraints.py        # Binary weight management
├── adaptive_loss.py             # Custom loss functions
├── performance_tracker.py       # Metrics tracking and CSV output
├── data_loader.py               # Data loading and preprocessing
├── test_main.py                 # Test suite
├── requirements.txt             # Dependencies
└── training_output/             # Generated results
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
Dynamically adjusts MSE/MAE weighting based on training progress, accuracy, and loss history.

### Error Resilience
Implements graceful degradation - training continues even when individual components encounter errors.

## Testing

```bash
python test_main.py              # Run standard tests
python test_main.py --comprehensive  # Run detailed test suite
```

Tests cover weight constraints, adaptive loss functions, performance tracking, and error handling.

## Configuration

Key configuration options:

```python
config = {
    'hidden_layers': [64, 32, 16],
    'max_binary_digits': 5,
    'oscillation_window': 3,
    'loss_weighting_strategy': 'combined'
}
```

## License

This project is part of the Code Lab Assist educational framework.
