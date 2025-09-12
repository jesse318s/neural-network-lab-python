"""
Comprehensive Experiment Runner for Systematic Neural Network Testing

This module executes systematic experiments with comprehensive metrics collection,
CSV saving, error handling, and progress tracking for physics simulation
neural network optimization.
"""

import os
import time
import csv
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import tensorflow as tf
    import psutil
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except ImportError as e:
    print(f"Warning: Required packages not available: {e}")

# Import existing modules
try:
    from experiment_config import ExperimentConfig, ExperimentConfigManager
    from main import AdvancedNeuralNetwork
    from data_loader import generate_particle_data, preprocess_data
    from performance_tracker import PerformanceTracker
    from weight_constraints import (
        BinaryWeightConstraintChanges, 
        BinaryWeightConstraintMax, 
        OscillationDampener
    )
    from adaptive_loss import AdaptiveLossFunction
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: int
    config_hash: str
    start_time: str
    end_time: str
    duration_seconds: float
    success: bool
    error_message: Optional[str]
    
    # Model performance metrics
    final_train_loss: float
    final_val_loss: float
    final_test_loss: float
    final_train_mse: float
    final_val_mse: float
    final_test_mse: float
    final_train_mae: float
    final_val_mae: float
    final_test_mae: float
    final_r2_score: float
    
    # Training progression metrics
    best_val_loss: float
    best_val_loss_epoch: int
    convergence_epoch: Optional[int]
    stability_score: float
    generalization_gap: float
    
    # Performance metrics
    training_time_seconds: float
    inference_time_ms: float
    model_size_mb: float
    peak_memory_mb: float
    
    # Constraint application metrics
    binary_changes_applied: int
    binary_max_applied: int
    oscillation_dampening_applied: int
    total_constraint_applications: int
    
    # Configuration details (for analysis)
    particle_count: int
    architecture_name: str
    constraints_name: str
    loss_strategy_name: str
    train_batch_size: int
    val_batch_size: int
    epochs: int
    random_seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class ExperimentRunner:
    """
    Executes systematic experiments with comprehensive metrics collection
    and robust error handling for physics simulation neural networks.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the experiment runner.
        
        Args:
            experiment_dir: Directory to save experiment results
        """
        self.experiment_dir = experiment_dir
        self.results_file = os.path.join(experiment_dir, "experiment_results.csv")
        self.progress_file = os.path.join(experiment_dir, "experiment_progress.json")
        self.error_log_file = os.path.join(experiment_dir, "error_log.txt")
        
        # Runtime state
        self.experiment_results: List[ExperimentResult] = []
        self.failed_experiments: List[int] = []
        self.current_experiment_id: Optional[int] = None
        self.start_time: Optional[float] = None
        
        # Performance tracking
        self.total_experiments = 0
        self.completed_experiments = 0
        self.successful_experiments = 0
        
        # Create output directory
        try:
            os.makedirs(self.experiment_dir, exist_ok=True)
            print(f"Experiment runner initialized - Output: {self.experiment_dir}")
        except Exception as e:
            print(f"Warning: Could not create experiment directory {self.experiment_dir}: {e}")
            self.experiment_dir = "."
    
    def run_all_experiments(self, 
                          configs: List[ExperimentConfig], 
                          experiment_subset: Optional[List[int]] = None,
                          save_frequency: int = 5,
                          continue_from_checkpoint: bool = True) -> bool:
        """
        Run all experiments with progress tracking and checkpointing.
        
        Args:
            configs: List of experiment configurations
            experiment_subset: Optional list of experiment IDs to run (for testing)
            save_frequency: Save results every N experiments
            continue_from_checkpoint: Whether to continue from previous checkpoint
            
        Returns:
            True if all experiments completed successfully
        """
        print("=== Starting Systematic Experiment Execution ===")
        
        # Load previous progress if requested
        if continue_from_checkpoint:
            self._load_progress()
        
        # Filter to subset if requested
        if experiment_subset is not None:
            configs = [config for config in configs if config.experiment_id in experiment_subset]
            print(f"Running subset of {len(configs)} experiments")
        
        self.total_experiments = len(configs)
        self.start_time = time.time()
        
        print(f"Total experiments to run: {self.total_experiments}")
        print(f"Results will be saved to: {self.results_file}")
        
        # Initialize results CSV file
        self._initialize_results_file()
        
        # Run each experiment
        for i, config in enumerate(configs):
            
            # Skip if already completed
            if config.experiment_id in [r.experiment_id for r in self.experiment_results]:
                print(f"Skipping experiment {config.experiment_id} (already completed)")
                continue
            
            self.current_experiment_id = config.experiment_id
            
            print(f"\n--- Experiment {i+1}/{len(configs)} (ID: {config.experiment_id}) ---")
            print(f"Config: {config.particle_count} particles, {config.architecture['name']}, "
                  f"{config.constraints['name']}, {config.loss_config['name']}")
            
            # Run single experiment
            try:
                result = self._run_single_experiment(config)
                self.experiment_results.append(result)
                
                if result.success:
                    self.successful_experiments += 1
                    print(f"✓ Experiment completed successfully")
                else:
                    self.failed_experiments.append(config.experiment_id)
                    print(f"✗ Experiment failed: {result.error_message}")
                
                self.completed_experiments += 1
                
                # Save progress periodically
                if (i + 1) % save_frequency == 0:
                    self._save_progress()
                    self._append_result_to_csv(result)
                    print(f"Progress saved ({self.completed_experiments}/{self.total_experiments})")
                
            except Exception as e:
                # Handle catastrophic failures
                error_msg = f"Catastrophic failure in experiment {config.experiment_id}: {str(e)}"
                print(f"💥 {error_msg}")
                self._log_error(config.experiment_id, error_msg, traceback.format_exc())
                self.failed_experiments.append(config.experiment_id)
                self.completed_experiments += 1
            
            # Progress update
            elapsed = time.time() - self.start_time
            if self.completed_experiments > 0:
                avg_time_per_exp = elapsed / self.completed_experiments
                remaining_time = avg_time_per_exp * (self.total_experiments - self.completed_experiments)
                print(f"Progress: {self.completed_experiments}/{self.total_experiments} "
                      f"({100 * self.completed_experiments / self.total_experiments:.1f}%) - "
                      f"ETA: {remaining_time/3600:.1f}h")
        
        # Final save
        self._save_final_results()
        
        # Summary
        print(f"\n=== Experiment Execution Complete ===")
        print(f"Total experiments: {self.total_experiments}")
        print(f"Successful: {self.successful_experiments}")
        print(f"Failed: {len(self.failed_experiments)}")
        print(f"Success rate: {100 * self.successful_experiments / self.total_experiments:.1f}%")
        print(f"Total time: {(time.time() - self.start_time) / 3600:.2f} hours")
        
        return len(self.failed_experiments) == 0
    
    def _run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment and collect comprehensive metrics.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult object with all metrics
        """
        start_time = time.time()
        start_time_str = datetime.now().isoformat()
        
        # Initialize result with default values
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config_hash=config.get_config_hash(),
            start_time=start_time_str,
            end_time="",
            duration_seconds=0.0,
            success=False,
            error_message=None,
            
            # Default metrics (will be updated if successful)
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            final_test_loss=float('inf'),
            final_train_mse=float('inf'),
            final_val_mse=float('inf'),
            final_test_mse=float('inf'),
            final_train_mae=float('inf'),
            final_val_mae=float('inf'),
            final_test_mae=float('inf'),
            final_r2_score=-1.0,
            
            best_val_loss=float('inf'),
            best_val_loss_epoch=-1,
            convergence_epoch=None,
            stability_score=0.0,
            generalization_gap=float('inf'),
            
            training_time_seconds=0.0,
            inference_time_ms=0.0,
            model_size_mb=0.0,
            peak_memory_mb=0.0,
            
            binary_changes_applied=0,
            binary_max_applied=0,
            oscillation_dampening_applied=0,
            total_constraint_applications=0,
            
            # Configuration details
            particle_count=config.particle_count,
            architecture_name=config.architecture['name'],
            constraints_name=config.constraints['name'],
            loss_strategy_name=config.loss_config['name'],
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
            epochs=config.epochs,
            random_seed=config.random_seed
        )
        
        try:
            # Set random seed for reproducibility
            np.random.seed(config.random_seed)
            tf.random.set_seed(config.random_seed)
            
            # Generate data with specified particle count
            data_df = generate_particle_data(
                num_particles=config.particle_count, 
                save_to_file=False
            )
            
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
                data_df, 
                test_size=0.2, 
                val_size=0.2,
                random_state=config.random_seed
            )
            
            # Create model configuration
            model_config = self._create_model_config(config)
            
            # Create and compile model
            model = AdvancedNeuralNetwork(
                input_shape=(X_train.shape[1],),
                output_shape=y_train.shape[1],
                config=model_config
            )
            model.compile_model()
            
            # Train model and collect metrics
            training_start = time.time()
            
            training_results = model.train_with_custom_constraints(
                X_train, y_train, X_val, y_val,
                epochs=config.epochs,
                batch_size=config.train_batch_size
            )
            
            training_time = time.time() - training_start
            
            # Evaluate model
            test_results = model.evaluate_model(X_test, y_test)
            
            # Calculate comprehensive metrics
            result = self._calculate_comprehensive_metrics(
                result, config, training_results, test_results, 
                X_train, X_val, X_test, y_train, y_val, y_test,
                model, training_time
            )
            
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            self._log_error(config.experiment_id, str(e), traceback.format_exc())
        
        # Finalize result timing
        result.end_time = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time
        
        return result
    
    def _create_model_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Create model configuration from experiment config."""
        model_config = {
            'hidden_layers': config.architecture['hidden_layers'],
            'activation': config.architecture['activation'],
            'dropout_rate': config.architecture['dropout_rate'],
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'output_dir': self.experiment_dir,
            'enable_weight_constraints': True
        }
        
        # Add constraint parameters
        if config.constraints['binary_changes']:
            model_config['max_additional_binary_digits'] = config.constraints.get('max_additional_binary_digits', 1)
        
        if config.constraints['binary_max']:
            model_config['max_binary_digits'] = config.constraints.get('max_binary_digits', 5)
        
        if config.constraints['oscillation']:
            model_config['oscillation_window'] = config.constraints.get('oscillation_window', 3)
        
        # Add loss strategy
        if config.loss_config['adaptive']:
            model_config['loss_weighting_strategy'] = config.loss_config['strategy']
        else:
            model_config['loss_weighting_strategy'] = 'none'
        
        return model_config
    
    def _calculate_comprehensive_metrics(self, 
                                       result: ExperimentResult,
                                       config: ExperimentConfig,
                                       training_results: Dict[str, Any],
                                       test_results: Dict[str, Any],
                                       X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                                       y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                                       model: Any,
                                       training_time: float) -> ExperimentResult:
        """Calculate comprehensive metrics from training and test results."""
        
        try:
            # Basic performance metrics
            result.training_time_seconds = training_time
            
            # Get training history
            history = training_results.get('history', {})
            
            if history:
                # Final metrics
                if 'loss' in history and history['loss']:
                    result.final_train_loss = float(history['loss'][-1])
                if 'val_loss' in history and history['val_loss']:
                    result.final_val_loss = float(history['val_loss'][-1])
                if 'mae' in history and history['mae']:
                    result.final_train_mae = float(history['mae'][-1])
                if 'val_mae' in history and history['val_mae']:
                    result.final_val_mae = float(history['val_mae'][-1])
                
                # Best validation loss
                if 'val_loss' in history and history['val_loss']:
                    val_losses = [float(x) for x in history['val_loss'] if not np.isinf(x)]
                    if val_losses:
                        result.best_val_loss = min(val_losses)
                        result.best_val_loss_epoch = val_losses.index(result.best_val_loss)
                
                # Convergence detection (when val loss stops improving significantly)
                if 'val_loss' in history and history['val_loss']:
                    result.convergence_epoch = self._detect_convergence(history['val_loss'])
                
                # Stability score (inverse of validation loss variance in last 10 epochs)
                if 'val_loss' in history and len(history['val_loss']) >= 10:
                    recent_losses = history['val_loss'][-10:]
                    recent_losses = [float(x) for x in recent_losses if not np.isinf(x)]
                    if recent_losses:
                        result.stability_score = 1.0 / (1.0 + np.var(recent_losses))
            
            # Test set metrics
            result.final_test_mse = test_results.get('mse', float('inf'))
            result.final_test_mae = test_results.get('mae', float('inf'))
            result.final_r2_score = test_results.get('r2_score', -1.0)
            result.inference_time_ms = test_results.get('inference_time', 0.0) * 1000
            
            # Calculate MSE for train/val sets
            try:
                train_pred = model.model.predict(X_train, verbose=0)
                val_pred = model.model.predict(X_val, verbose=0)
                
                result.final_train_mse = float(mean_squared_error(y_train, train_pred))
                result.final_val_mse = float(mean_squared_error(y_val, val_pred))
                
                # Generalization gap
                result.generalization_gap = abs(result.final_val_mse - result.final_train_mse)
                
            except Exception as e:
                print(f"Warning: Could not calculate MSE metrics: {e}")
            
            # Model size estimation
            try:
                total_params = model.model.count_params()
                result.model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            except Exception:
                result.model_size_mb = 0.0
            
            # Memory usage
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                result.peak_memory_mb = memory_info.rss / 1024 / 1024
            except Exception:
                result.peak_memory_mb = 0.0
            
            # Constraint application counts
            if 'successful_constraints' in training_results:
                constraints = training_results['successful_constraints']
                result.binary_changes_applied = constraints.count('binary_changes')
                result.binary_max_applied = constraints.count('binary_max')
                result.oscillation_dampening_applied = constraints.count('oscillation_dampening')
                result.total_constraint_applications = len(constraints)
            
        except Exception as e:
            print(f"Warning: Error calculating comprehensive metrics: {e}")
        
        return result
    
    def _detect_convergence(self, val_losses: List[float], 
                          patience: int = 5, 
                          min_improvement: float = 0.001) -> Optional[int]:
        """Detect convergence epoch based on validation loss improvement."""
        try:
            val_losses = [float(x) for x in val_losses if not np.isinf(x)]
            if len(val_losses) < patience + 1:
                return None
            
            best_loss = float('inf')
            no_improvement_count = 0
            
            for epoch, loss in enumerate(val_losses):
                if loss < best_loss - min_improvement:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        return epoch - patience
            
            return None
        except Exception:
            return None
    
    def _initialize_results_file(self):
        """Initialize CSV results file with headers."""
        try:
            # Check if file already exists
            if os.path.exists(self.results_file):
                return
            
            # Create dummy result to get field names
            dummy_result = ExperimentResult(
                experiment_id=0, config_hash="", start_time="", end_time="",
                duration_seconds=0.0, success=False, error_message=None,
                final_train_loss=0.0, final_val_loss=0.0, final_test_loss=0.0,
                final_train_mse=0.0, final_val_mse=0.0, final_test_mse=0.0,
                final_train_mae=0.0, final_val_mae=0.0, final_test_mae=0.0,
                final_r2_score=0.0, best_val_loss=0.0, best_val_loss_epoch=0,
                convergence_epoch=None, stability_score=0.0, generalization_gap=0.0,
                training_time_seconds=0.0, inference_time_ms=0.0, model_size_mb=0.0,
                peak_memory_mb=0.0, binary_changes_applied=0, binary_max_applied=0,
                oscillation_dampening_applied=0, total_constraint_applications=0,
                particle_count=0, architecture_name="", constraints_name="",
                loss_strategy_name="", train_batch_size=0, val_batch_size=0,
                epochs=0, random_seed=0
            )
            
            # Write header
            with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=dummy_result.to_dict().keys())
                writer.writeheader()
            
        except Exception as e:
            print(f"Warning: Could not initialize results file: {e}")
    
    def _append_result_to_csv(self, result: ExperimentResult):
        """Append single result to CSV file."""
        try:
            with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=result.to_dict().keys())
                writer.writerow(result.to_dict())
        except Exception as e:
            print(f"Warning: Could not save result to CSV: {e}")
    
    def _save_progress(self):
        """Save current progress to checkpoint file."""
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'total_experiments': self.total_experiments,
                'completed_experiments': self.completed_experiments,
                'successful_experiments': self.successful_experiments,
                'failed_experiments': self.failed_experiments,
                'current_experiment_id': self.current_experiment_id,
                'experiment_results_count': len(self.experiment_results)
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def _load_progress(self):
        """Load progress from checkpoint file."""
        try:
            if not os.path.exists(self.progress_file):
                return
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            self.completed_experiments = progress_data.get('completed_experiments', 0)
            self.successful_experiments = progress_data.get('successful_experiments', 0)
            self.failed_experiments = progress_data.get('failed_experiments', [])
            
            print(f"Loaded progress: {self.completed_experiments} completed, "
                  f"{self.successful_experiments} successful")
            
            # Load existing results
            if os.path.exists(self.results_file):
                df = pd.read_csv(self.results_file)
                print(f"Found {len(df)} existing results in CSV file")
                
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
    
    def _save_final_results(self):
        """Save final results and generate summary."""
        try:
            # Save all results to CSV
            for result in self.experiment_results:
                self._append_result_to_csv(result)
            
            # Save final progress
            self._save_progress()
            
            # Generate summary report
            summary_file = os.path.join(self.experiment_dir, "experiment_summary.json")
            
            summary_data = {
                'execution_timestamp': datetime.now().isoformat(),
                'total_experiments': self.total_experiments,
                'successful_experiments': self.successful_experiments,
                'failed_experiments': len(self.failed_experiments),
                'success_rate': self.successful_experiments / self.total_experiments if self.total_experiments > 0 else 0,
                'total_execution_time_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0,
                'failed_experiment_ids': self.failed_experiments,
                'results_file': self.results_file,
                'total_results_saved': len(self.experiment_results)
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"Final results saved to {self.results_file}")
            print(f"Summary saved to {summary_file}")
            
        except Exception as e:
            print(f"Error saving final results: {e}")
    
    def _log_error(self, experiment_id: int, error_message: str, traceback_str: str):
        """Log error to error log file."""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Experiment ID: {experiment_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_message}\n")
                f.write(f"Traceback:\n{traceback_str}\n")
        except Exception as e:
            print(f"Warning: Could not log error: {e}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of experiment results."""
        if not self.experiment_results:
            return {'message': 'No results available'}
        
        successful_results = [r for r in self.experiment_results if r.success]
        
        if not successful_results:
            return {'message': 'No successful experiments'}
        
        # Calculate summary statistics
        r2_scores = [r.final_r2_score for r in successful_results if r.final_r2_score > -1]
        training_times = [r.training_time_seconds for r in successful_results]
        
        summary = {
            'total_experiments': len(self.experiment_results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(self.experiment_results),
            'best_r2_score': max(r2_scores) if r2_scores else 0,
            'mean_r2_score': np.mean(r2_scores) if r2_scores else 0,
            'mean_training_time': np.mean(training_times) if training_times else 0,
            'total_runtime_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
        return summary


def main():
    """Demonstrate experiment runner functionality."""
    print("=== Physics Simulation Experiment Runner ===")
    
    # Create configuration manager and generate test configs
    config_manager = ExperimentConfigManager("test_experiments")
    
    # Generate small subset for testing
    test_configs = config_manager.generate_subset_configurations(
        particle_counts=[100, 500],
        architectures=['shallow_wide'],
        constraints=['none', 'binary_changes_only'],
        loss_strategies=['mse_only'],
        random_seeds=[42],
        epochs=10
    )
    
    # Create experiment runner
    runner = ExperimentRunner("test_experiments")
    
    # Run experiments
    success = runner.run_all_experiments(
        test_configs, 
        experiment_subset=list(range(2))  # Run first 2 experiments
    )
    
    # Display results
    summary = runner.get_results_summary()
    print(f"\nResults Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()