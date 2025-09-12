# Systematic Neural Network Experimentation Framework - Usage Guide

This document provides comprehensive usage examples for the new systematic experimentation capabilities integrated into the neural network lab.

## Overview

The systematic experimentation framework enables comprehensive testing of:
- **Weight Constraints**: Binary precision, oscillation dampening, and combinations
- **Adaptive Loss Functions**: Epoch-based, accuracy-based, loss-based strategies
- **Architectural Variations**: Shallow wide vs deep narrow networks
- **Physics Simulation Complexity**: Variable particle counts with enhanced features

## Quick Start

### 1. Running Systematic Experiments

```bash
# Run comprehensive systematic experiments
python main.py --systematic

# Run systematic experiments with custom configuration
python main.py --systematic --epochs 20 --output_dir custom_experiments

# Run original single experiment mode (backward compatible)
python main.py --epochs 10
```

### 2. Custom Experiment Configuration

```python
from experiment_config import ExperimentConfigManager
from experiment_runner import ExperimentRunner
from experiment_analysis import ExperimentAnalyzer

# Create configuration manager
config_manager = ExperimentConfigManager("my_experiments")

# Generate targeted configurations
configs = config_manager.generate_subset_configurations(
    particle_counts=[100, 500, 1000],
    architectures=['shallow_wide', 'deep_narrow'],
    constraints=['none', 'binary_changes_only', 'all_constraints'],
    loss_strategies=['mse_only', 'combined_adaptive'],
    random_seeds=[42, 123, 456],
    epochs=30
)

# Run experiments
runner = ExperimentRunner("my_experiments")
success = runner.run_all_experiments(configs)

# Analyze results
analyzer = ExperimentAnalyzer(runner.results_file)
report = analyzer.generate_comprehensive_report()
```

### 3. Enhanced Data Generation

```python
from data_loader import generate_enhanced_particle_data, get_data_complexity_info

# Generate complex physics simulation data
df = generate_enhanced_particle_data(
    num_particles=1000,
    complexity_level='complex',  # 'simple', 'medium', or 'complex'
    random_seed=42,
    save_to_file=True
)

# Get complexity information for different particle counts
for particles in [100, 500, 1000, 2000]:
    info = get_data_complexity_info(particles)
    print(f"{particles} particles: {info}")
```

## Configuration Options

### Architecture Types
- **shallow_wide**: [64, 32] hidden layers with relu activation
- **deep_narrow**: [32, 16, 8] hidden layers with relu activation  
- **mixed_activation**: [64, 32] with mixed relu/tanh activations

### Weight Constraints
- **none**: No constraints applied
- **binary_changes_only**: Only binary precision constraint
- **binary_max_only**: Only binary maximum constraint
- **oscillation_only**: Only oscillation dampening
- **binary_combined**: Binary precision + maximum
- **all_constraints**: All constraints combined

### Loss Strategies
- **mse_only**: Standard MSE loss
- **epoch_based**: Adaptive loss based on training epoch
- **accuracy_based**: Adaptive loss based on model accuracy
- **loss_based**: Adaptive loss based on loss reduction
- **combined_adaptive**: Combination of all adaptive strategies

### Complexity Levels
- **simple**: Basic particle physics with minimal features
- **medium**: Enhanced physics with kinetic energy and trajectories
- **complex**: Full physics simulation with all advanced features

## Analysis Features

### Statistical Analysis
```python
# Perform ANOVA analysis (requires scipy)
anova_results = analyzer.perform_anova_analysis(
    metric='final_r2_score',
    factors=['architecture_name', 'constraints_name']
)

# Identify optimal configurations
optimal = analyzer.identify_optimal_configurations(
    metrics=['final_r2_score', 'training_time_seconds'],
    top_n=5
)

# Analyze generalizability across particle counts
generalization = analyzer.analyze_generalizability()
```

### Results Export
All results are automatically saved to CSV files with comprehensive metrics:
- **Experiment Results**: `experiment_results.csv`
- **Analysis Reports**: JSON format with detailed statistics
- **Configuration Logs**: Complete parameter tracking

## Example Workflows

### 1. Performance Optimization
```python
# Focus on finding fastest configurations
configs = config_manager.generate_subset_configurations(
    particle_counts=[100],  # Small dataset for speed
    architectures=['shallow_wide'],  # Simpler architecture
    constraints=['none'],  # No constraint overhead
    loss_strategies=['mse_only'],  # Simple loss
    epochs=10
)
```

### 2. Accuracy Maximization
```python
# Focus on highest accuracy configurations
configs = config_manager.generate_subset_configurations(
    particle_counts=[2000],  # Large dataset
    architectures=['deep_narrow', 'mixed_activation'],
    constraints=['all_constraints'],  # All regularization
    loss_strategies=['combined_adaptive'],  # Advanced loss
    epochs=50
)
```

### 3. Comprehensive Study
```python
# Full systematic study
configs = config_manager.generate_all_configurations(epochs=30)
runner.run_all_experiments(configs, max_parallel=4)
```

## Integration with Existing Code

The framework is fully backward compatible:

```python
# Existing single experiment still works
from main import create_model, AdvancedNeuralNetwork
from data_loader import load_and_prepare_data

# Original workflow unchanged
X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_prepare_data()
model = create_model(input_shape=X_train.shape[1:], output_shape=y_train.shape[1])

# New systematic capabilities available alongside
config_manager = ExperimentConfigManager("experiments")
# ... systematic experiment code ...
```

## Best Practices

### 1. Start Small
Begin with subset configurations to verify setup:
```python
configs = config_manager.generate_subset_configurations(
    particle_counts=[100],
    architectures=['shallow_wide'],
    constraints=['none'],
    loss_strategies=['mse_only'],
    epochs=5
)
```

### 2. Monitor Progress
Use the built-in progress tracking:
```python
# Experiments save progress every 5 runs by default
runner.run_all_experiments(configs, save_frequency=5)
```

### 3. Analyze Incrementally
Analyze results as experiments complete:
```python
# Check results periodically
analyzer = ExperimentAnalyzer(runner.results_file)
if len(analyzer.successful_df) > 10:
    preliminary_report = analyzer.generate_comprehensive_report()
```

### 4. Resource Management
For large studies, use experiment subsets:
```python
# Run experiments in batches
total_configs = config_manager.generate_all_configurations()
batch_size = 20

for i in range(0, len(total_configs), batch_size):
    batch = total_configs[i:i+batch_size]
    runner.run_all_experiments(batch)
    
    # Analyze batch results
    analyzer = ExperimentAnalyzer(runner.results_file)
    batch_report = analyzer.generate_comprehensive_report(f"batch_{i//batch_size}_report.json")
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce particle counts or batch sizes
2. **Long training times**: Use fewer epochs or simpler architectures
3. **Import errors**: Ensure all dependencies are installed
4. **Analysis limitations**: Install scipy for full statistical analysis

### Performance Tips
- Use smaller particle counts (100-500) for initial exploration
- Limit epochs (5-20) for configuration screening
- Use `save_frequency` parameter for progress tracking
- Monitor disk space for large experiment sets

This framework provides a comprehensive foundation for systematic neural network experimentation while maintaining full compatibility with existing workflows.