"""
Experiment Configuration Manager for Systematic Neural Network Experimentation

This module generates systematic combinations of experimental parameters including
particle counts, architectures, weight constraints, and adaptive loss strategies
to enable comprehensive testing for physics simulation neural networks.
"""

import itertools
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Iterator, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: int
    particle_count: int
    train_batch_size: int
    val_batch_size: int
    architecture: Dict[str, Any]
    constraints: Dict[str, Any]
    loss_config: Dict[str, Any]
    epochs: int
    random_seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_config_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = f"{self.particle_count}_{self.architecture['name']}_{self.constraints['name']}_{self.loss_config['name']}_{self.random_seed}"
        return config_str


class ExperimentConfigManager:
    """
    Manages systematic generation of experiment configurations for comprehensive
    neural network testing with physics simulations.
    """
    
    def __init__(self, output_dir: str = "physics_simulation_experiments"):
        """
        Initialize the experiment configuration manager.
        
        Args:
            output_dir: Directory to save experiment configurations and results
        """
        self.output_dir = output_dir
        self.experiment_configs = []
        self.config_summary = {}
        
        # Create output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create experiment directory {self.output_dir}: {e}")
            self.output_dir = "."
    
    def get_particle_counts(self) -> List[int]:
        """Get particle counts for generalizability testing."""
        return [100, 500, 1000, 2000]
    
    def get_batch_size_pairs(self) -> List[Tuple[int, int]]:
        """Get (train_batch_size, val_batch_size) pairs."""
        return [(32, 16), (64, 32), (128, 64)]
    
    def get_architectures(self) -> List[Dict[str, Any]]:
        """Get neural network architectures for comparison."""
        architectures = [
            {
                'name': 'shallow_wide',
                'hidden_layers': [128, 64],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'description': 'Shallow network with wide layers'
            },
            {
                'name': 'deep_narrow', 
                'hidden_layers': [64, 64, 32, 32],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'description': 'Deep network with narrow layers'
            },
            {
                'name': 'mixed_activation',
                'hidden_layers': [96, 48, 24],
                'activation': 'relu',  # Primary activation
                'mixed_activations': ['relu', 'tanh', 'sigmoid'],
                'dropout_rate': 0.25,
                'description': 'Mixed activation functions per layer'
            }
        ]
        return architectures
    
    def get_weight_constraints(self) -> List[Dict[str, Any]]:
        """Get weight constraint configurations."""
        constraints = [
            {
                'name': 'none',
                'binary_changes': False,
                'binary_max': False,
                'oscillation': False,
                'description': 'No weight constraints applied'
            },
            {
                'name': 'binary_changes_only',
                'binary_changes': True,
                'binary_max': False,
                'oscillation': False,
                'max_additional_binary_digits': 1,
                'description': 'Binary precision constraints on weight changes only'
            },
            {
                'name': 'binary_max_only',
                'binary_changes': False,
                'binary_max': True,
                'oscillation': False,
                'max_binary_digits': 5,
                'description': 'Binary precision constraints on maximum weight precision'
            },
            {
                'name': 'oscillation_only',
                'binary_changes': False,
                'binary_max': False,
                'oscillation': True,
                'oscillation_window': 3,
                'description': 'Oscillation dampening only'
            },
            {
                'name': 'binary_combined',
                'binary_changes': True,
                'binary_max': True,
                'oscillation': False,
                'max_additional_binary_digits': 1,
                'max_binary_digits': 5,
                'description': 'Both binary precision constraints combined'
            },
            {
                'name': 'all_constraints',
                'binary_changes': True,
                'binary_max': True,
                'oscillation': True,
                'max_additional_binary_digits': 1,
                'max_binary_digits': 5,
                'oscillation_window': 3,
                'description': 'All weight constraints applied together'
            }
        ]
        return constraints
    
    def get_loss_strategies(self) -> List[Dict[str, Any]]:
        """Get adaptive loss function strategies."""
        strategies = [
            {
                'name': 'mse_only',
                'strategy': 'mse_only',
                'adaptive': False,
                'description': 'Mean Squared Error loss function only'
            },
            {
                'name': 'epoch_based',
                'strategy': 'epoch_based',
                'adaptive': True,
                'description': 'Adaptive weighting based on epoch number'
            },
            {
                'name': 'accuracy_based',
                'strategy': 'accuracy_based', 
                'adaptive': True,
                'description': 'Adaptive weighting based on previous accuracy'
            },
            {
                'name': 'loss_based',
                'strategy': 'loss_based',
                'adaptive': True,
                'description': 'Adaptive weighting based on previous loss values'
            },
            {
                'name': 'combined_adaptive',
                'strategy': 'combined',
                'adaptive': True,
                'description': 'Combined adaptive weighting using all strategies'
            }
        ]
        return strategies
    
    def get_random_seeds(self) -> List[int]:
        """Get random seeds for reproducible results."""
        return [42, 123, 456]
    
    def generate_all_configurations(self, epochs: int = 50) -> List[ExperimentConfig]:
        """
        Generate all possible combinations of experimental parameters.
        
        Args:
            epochs: Number of training epochs for all experiments
            
        Returns:
            List of ExperimentConfig objects
        """
        print("Generating systematic experiment configurations...")
        
        # Get all parameter options
        particle_counts = self.get_particle_counts()
        batch_pairs = self.get_batch_size_pairs()
        architectures = self.get_architectures()
        constraints = self.get_weight_constraints()
        loss_strategies = self.get_loss_strategies()
        random_seeds = self.get_random_seeds()
        
        # Generate all combinations
        experiment_id = 0
        configs = []
        
        for particle_count in particle_counts:
            for train_batch, val_batch in batch_pairs:
                for architecture in architectures:
                    for constraint in constraints:
                        for loss_config in loss_strategies:
                            for seed in random_seeds:
                                
                                config = ExperimentConfig(
                                    experiment_id=experiment_id,
                                    particle_count=particle_count,
                                    train_batch_size=train_batch,
                                    val_batch_size=val_batch,
                                    architecture=architecture,
                                    constraints=constraint,
                                    loss_config=loss_config,
                                    epochs=epochs,
                                    random_seed=seed
                                )
                                
                                configs.append(config)
                                experiment_id += 1
        
        self.experiment_configs = configs
        
        # Generate summary statistics
        self._generate_config_summary()
        
        print(f"Generated {len(configs)} experiment configurations")
        print(f"Configuration breakdown:")
        print(f"  Particle counts: {len(particle_counts)} options")
        print(f"  Batch size pairs: {len(batch_pairs)} options")
        print(f"  Architectures: {len(architectures)} options")
        print(f"  Weight constraints: {len(constraints)} options")
        print(f"  Loss strategies: {len(loss_strategies)} options")
        print(f"  Random seeds: {len(random_seeds)} options")
        
        return configs
    
    def generate_subset_configurations(self, 
                                     particle_counts: Optional[List[int]] = None,
                                     architectures: Optional[List[str]] = None,
                                     constraints: Optional[List[str]] = None,
                                     loss_strategies: Optional[List[str]] = None,
                                     random_seeds: Optional[List[int]] = None,
                                     epochs: int = 50) -> List[ExperimentConfig]:
        """
        Generate a subset of configurations for testing or focused experiments.
        
        Args:
            particle_counts: Subset of particle counts to test
            architectures: Subset of architecture names to test
            constraints: Subset of constraint names to test
            loss_strategies: Subset of loss strategy names to test
            random_seeds: Subset of random seeds to test
            epochs: Number of training epochs
            
        Returns:
            List of ExperimentConfig objects for the subset
        """
        print("Generating subset experiment configurations...")
        
        # Use defaults if not specified
        if particle_counts is None:
            particle_counts = [500, 1000]  # Smaller subset
        if architectures is None:
            architectures = ['shallow_wide', 'deep_narrow']
        if constraints is None:
            constraints = ['none', 'all_constraints']
        if loss_strategies is None:
            loss_strategies = ['mse_only', 'combined_adaptive']
        if random_seeds is None:
            random_seeds = [42]
        
        # Get full parameter sets
        all_architectures = {arch['name']: arch for arch in self.get_architectures()}
        all_constraints = {const['name']: const for const in self.get_weight_constraints()}
        all_loss_strategies = {loss['name']: loss for loss in self.get_loss_strategies()}
        batch_pairs = self.get_batch_size_pairs()
        
        # Generate subset combinations
        experiment_id = 0
        configs = []
        
        for particle_count in particle_counts:
            if particle_count not in self.get_particle_counts():
                print(f"Warning: Particle count {particle_count} not in standard set")
                continue
                
            for train_batch, val_batch in batch_pairs:
                for arch_name in architectures:
                    if arch_name not in all_architectures:
                        print(f"Warning: Architecture '{arch_name}' not found")
                        continue
                        
                    for const_name in constraints:
                        if const_name not in all_constraints:
                            print(f"Warning: Constraint '{const_name}' not found")
                            continue
                            
                        for loss_name in loss_strategies:
                            if loss_name not in all_loss_strategies:
                                print(f"Warning: Loss strategy '{loss_name}' not found")
                                continue
                                
                            for seed in random_seeds:
                                
                                config = ExperimentConfig(
                                    experiment_id=experiment_id,
                                    particle_count=particle_count,
                                    train_batch_size=train_batch,
                                    val_batch_size=val_batch,
                                    architecture=all_architectures[arch_name],
                                    constraints=all_constraints[const_name],
                                    loss_config=all_loss_strategies[loss_name],
                                    epochs=epochs,
                                    random_seed=seed
                                )
                                
                                configs.append(config)
                                experiment_id += 1
        
        self.experiment_configs = configs
        self._generate_config_summary()
        
        print(f"Generated {len(configs)} subset experiment configurations")
        return configs
    
    def save_configurations(self, filename: str = "experiment_configurations.json"):
        """Save all configurations to JSON file."""
        try:
            file_path = os.path.join(self.output_dir, filename)
            
            config_data = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_experiments': len(self.experiment_configs),
                'configuration_summary': self.config_summary,
                'experiment_configs': [config.to_dict() for config in self.experiment_configs]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Experiment configurations saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving configurations: {e}")
    
    def load_configurations(self, filename: str = "experiment_configurations.json") -> bool:
        """Load configurations from JSON file."""
        try:
            file_path = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"Configuration file {file_path} not found")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Convert dictionaries back to ExperimentConfig objects
            self.experiment_configs = []
            for config_dict in config_data.get('experiment_configs', []):
                config = ExperimentConfig(**config_dict)
                self.experiment_configs.append(config)
            
            self.config_summary = config_data.get('configuration_summary', {})
            
            print(f"Loaded {len(self.experiment_configs)} configurations from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error loading configurations: {e}")
            return False
    
    def get_config_by_id(self, experiment_id: int) -> Optional[ExperimentConfig]:
        """Get configuration by experiment ID."""
        for config in self.experiment_configs:
            if config.experiment_id == experiment_id:
                return config
        return None
    
    def get_configs_by_criteria(self, **criteria) -> List[ExperimentConfig]:
        """
        Get configurations matching specific criteria.
        
        Args:
            **criteria: Key-value pairs to match in configurations
            
        Returns:
            List of matching ExperimentConfig objects
        """
        matching_configs = []
        
        for config in self.experiment_configs:
            match = True
            
            for key, value in criteria.items():
                if key == 'architecture_name':
                    if config.architecture.get('name') != value:
                        match = False
                        break
                elif key == 'constraints_name':
                    if config.constraints.get('name') != value:
                        match = False
                        break
                elif key == 'loss_strategy_name':
                    if config.loss_config.get('name') != value:
                        match = False
                        break
                elif hasattr(config, key):
                    if getattr(config, key) != value:
                        match = False
                        break
            
            if match:
                matching_configs.append(config)
        
        return matching_configs
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        return self.config_summary.copy()
    
    def _generate_config_summary(self):
        """Generate summary statistics for configurations."""
        if not self.experiment_configs:
            self.config_summary = {}
            return
        
        # Count unique values for each parameter
        particle_counts = set()
        architectures = set()
        constraints = set()
        loss_strategies = set()
        batch_pairs = set()
        seeds = set()
        
        for config in self.experiment_configs:
            particle_counts.add(config.particle_count)
            architectures.add(config.architecture['name'])
            constraints.add(config.constraints['name'])
            loss_strategies.add(config.loss_config['name'])
            batch_pairs.add((config.train_batch_size, config.val_batch_size))
            seeds.add(config.random_seed)
        
        self.config_summary = {
            'total_experiments': len(self.experiment_configs),
            'unique_particle_counts': sorted(list(particle_counts)),
            'unique_architectures': sorted(list(architectures)),
            'unique_constraints': sorted(list(constraints)),
            'unique_loss_strategies': sorted(list(loss_strategies)),
            'unique_batch_pairs': sorted(list(batch_pairs)),
            'unique_seeds': sorted(list(seeds)),
            'estimated_total_runtime_hours': len(self.experiment_configs) * 0.5,  # Rough estimate
            'parameter_counts': {
                'particle_counts': len(particle_counts),
                'architectures': len(architectures),
                'constraints': len(constraints),
                'loss_strategies': len(loss_strategies),
                'batch_pairs': len(batch_pairs),
                'seeds': len(seeds)
            }
        }


def create_quick_test_config() -> ExperimentConfig:
    """Create a single configuration for quick testing."""
    config = ExperimentConfig(
        experiment_id=0,
        particle_count=100,
        train_batch_size=32,
        val_batch_size=16,
        architecture={
            'name': 'shallow_wide',
            'hidden_layers': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'description': 'Quick test architecture'
        },
        constraints={
            'name': 'none',
            'binary_changes': False,
            'binary_max': False,
            'oscillation': False,
            'description': 'No constraints for quick testing'
        },
        loss_config={
            'name': 'mse_only',
            'strategy': 'mse_only',
            'adaptive': False,
            'description': 'MSE only for quick testing'
        },
        epochs=10,
        random_seed=42
    )
    
    return config


def main():
    """Demonstrate experiment configuration generation."""
    print("=== Physics Simulation Experiment Configuration Manager ===")
    
    # Create manager
    manager = ExperimentConfigManager()
    
    # Generate full set of configurations
    print("\n--- Generating Full Configuration Set ---")
    full_configs = manager.generate_all_configurations(epochs=30)
    
    # Save configurations
    manager.save_configurations()
    
    # Generate subset for testing
    print("\n--- Generating Test Subset ---")
    test_configs = manager.generate_subset_configurations(
        particle_counts=[500, 1000],
        architectures=['shallow_wide', 'deep_narrow'],
        constraints=['none', 'all_constraints'],
        loss_strategies=['mse_only', 'combined_adaptive'],
        epochs=15
    )
    
    # Display summary
    print("\n--- Configuration Summary ---")
    summary = manager.get_configuration_summary()
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Estimated runtime: {summary['estimated_total_runtime_hours']:.1f} hours")
    
    print("\nParameter variations:")
    for param, count in summary['parameter_counts'].items():
        print(f"  {param}: {count} options")
    
    # Show example configurations
    print("\n--- Example Configurations ---")
    for i, config in enumerate(test_configs[:3]):
        print(f"\nConfig {i+1} (ID: {config.experiment_id}):")
        print(f"  Particles: {config.particle_count}")
        print(f"  Architecture: {config.architecture['name']}")
        print(f"  Constraints: {config.constraints['name']}")
        print(f"  Loss: {config.loss_config['name']}")
        print(f"  Batches: {config.train_batch_size}/{config.val_batch_size}")
        print(f"  Seed: {config.random_seed}")


if __name__ == "__main__":
    main()