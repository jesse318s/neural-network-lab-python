"""
Performance Tracking and Metrics Collection for Advanced TensorFlow Lab
Implements railway-style error handling for robust execution.
"""

import os
import time
import csv
import json
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


class PerformanceTracker:
    """
    Tracks and logs training metrics including R², loss, memory usage,
    and comprehensive performance data with CSV output capabilities.
    """
    
    def __init__(self, output_dir: str = "training_output"):
        self.output_dir = output_dir
        self.training_start_time = None
        self.training_end_time = None
        self.epoch_start_time = None
        # Performance metrics
        self.current_r2 = 0.0
        self.best_r2 = 0.0
        self.best_r2_epoch = 0
        self.greatest_improvement = 0.0
        self.greatest_improvement_epoch = 0
        self.previous_r2 = 0.0
        # Timing metrics
        self.epoch_times = []
        self.avg_epoch_time = 0.0
        self.total_training_time = 0.0
        # Memory metrics
        self.current_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.weight_file_sizes = {}
        # Training history
        self.training_history = []
        # Configuration tracking
        self.training_config = {}
        self.weight_modifications_used = []
        # Error tracking
        self.error_count = 0
        self.errors_log = []
        
        # Create output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create output directory {self.output_dir}: {e}")
            self.output_dir = "."
    
    def start_training(self, config: Dict[str, Any]):
        """Start training session and initialize tracking."""
        try:
            self.training_start_time = time.time()
            self.training_config = config.copy()
            print(f"Performance tracking started - Output: {self.output_dir}")  
        except Exception as e:
            self._handle_error(f"Error starting training tracking: {e}")
    
    def start_epoch(self, epoch: int):
        """Start epoch tracking."""
        try:
            self.epoch_start_time = time.time()
            self._update_memory_usage()
        except Exception as e:
            self._handle_error(f"Error starting epoch {epoch}: {e}")
    
    def end_epoch(self, epoch: int, logs: Dict[str, float]):
        """End epoch tracking and record metrics."""
        try:
            if self.epoch_start_time is not None:
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                self.avg_epoch_time = np.mean(self.epoch_times)
            else:
                epoch_time = 0.0
            
            current_r2 = logs.get('r2_score', logs.get('val_r2', 0.0))
            self.current_r2 = current_r2
            
            if current_r2 > self.best_r2:
                self.best_r2 = current_r2
                self.best_r2_epoch = epoch
            
            if epoch > 0:
                improvement = current_r2 - self.previous_r2

                if improvement > self.greatest_improvement:
                    self.greatest_improvement = improvement
                    self.greatest_improvement_epoch = epoch
            
            self.previous_r2 = current_r2
            self._update_memory_usage()    
            history_entry = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'epoch_time': epoch_time,
                'memory_mb': self.current_memory_mb,
                **logs
            }
            self.training_history.append(history_entry) 
        except Exception as e:
            self._handle_error(f"Error ending epoch {epoch}: {e}")
    
    def end_training(self) -> Dict[str, Any]:
        """End training session and finalize tracking."""
        try:
            self.training_end_time = time.time()
            
            if self.training_start_time is not None:
                self.total_training_time = self.training_end_time - self.training_start_time
            
            self._update_memory_usage()
            return {
                'total_training_time': self.total_training_time,
                'current_r2': self.current_r2,
                'best_r2': self.best_r2,
                'best_r2_epoch': self.best_r2_epoch,
                'greatest_improvement': self.greatest_improvement,
                'greatest_improvement_epoch': self.greatest_improvement_epoch,
                'avg_epoch_time': self.avg_epoch_time,
                'current_memory_mb': self.current_memory_mb,
                'peak_memory_mb': self.peak_memory_mb,
                'loss_weighting_strategy': self.training_config.get('loss_weighting_strategy', 'none'),
                'weight_modifications_used': self.weight_modifications_used,
                'error_count': self.error_count
            }   
        except Exception as e:
            self._handle_error(f"Error ending training: {e}")
            return {}
    
    def record_weight_file_size(self, file_path: str) -> None:
        """Record the size of a weight file."""
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.weight_file_sizes[file_path] = file_size
        except Exception as e:
            self._handle_error(f"Error recording weight file size for {file_path}: {e}")
    
    def measure_inference_time(self, model: Optional[Any], test_data: np.ndarray, num_runs: int = 10) -> float:
        """Measure model inference time."""
        try:
            if model is None:
                raise ValueError("Model is None, cannot measure inference time.")

            inference_times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = model.predict(test_data, verbose=0)
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            return float(np.mean(inference_times))
        except ValueError as ve:
            self._handle_error(f"Validation error in inference timing: {ve}")
            return 0.0
        except Exception as e:
            self._handle_error(f"Error measuring inference time: {e}")
            return 0.0
    
    def add_weight_modification(self, modification_name: str) -> None:
        """Record that a weight modification technique was used."""
        try:
            if modification_name not in self.weight_modifications_used:
                self.weight_modifications_used.append(modification_name)
        except Exception as e:
            self._handle_error(f"Error adding weight modification {modification_name}: {e}")
    
    def save_results(self) -> None:
        """Save all tracked results to CSV and JSON files."""
        try:
            self._save_training_results_csv()
            self._save_configuration_log()
            self._save_training_log()
            print(f"Results saved to {self.output_dir}")
        except Exception as e:
            self._handle_error(f"Error saving results: {e}")
    
    def create_loss_history_csv(self, loss_history: List[Dict[str, Any]]) -> None:
        """Create CSV file with loss history components."""
        try:
            file_path = os.path.join(self.output_dir, "loss_history.csv")
            
            if not loss_history:
                return
            
            all_keys = set()

            for entry in loss_history:
                all_keys.update(entry.keys())
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(loss_history)
        except Exception as e:
            self._handle_error(f"Error creating loss history CSV: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance metrics."""
        try:
            return {
                'current_r2': self.current_r2,
                'best_r2': self.best_r2,
                'best_r2_epoch': self.best_r2_epoch,
                'greatest_improvement': self.greatest_improvement,
                'greatest_improvement_epoch': self.greatest_improvement_epoch,
                'avg_epoch_time': self.avg_epoch_time,
                'total_training_time': self.total_training_time,
                'current_memory_mb': self.current_memory_mb,
                'peak_memory_mb': self.peak_memory_mb,
                'weight_modifications_used': self.weight_modifications_used,
                'weight_file_sizes': self.weight_file_sizes,
                'error_count': self.error_count,
                'total_epochs': len(self.training_history)
            }
        except Exception as e:
            self._handle_error(f"Error getting summary: {e}")
            return {}
    
    def _update_memory_usage(self) -> None:
        """Update current memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.current_memory_mb = memory_info.rss / 1024 / 1024
            
            if self.current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = self.current_memory_mb
        except Exception as e:
            self._handle_error(f"Error updating memory usage: {e}")
    
    def _save_training_results_csv(self) -> None:
        """Save training results to CSV file."""
        try:
            file_path = os.path.join(self.output_dir, "training_results.csv")
            
            if not self.training_history:
                return
            
            all_keys = set()

            for entry in self.training_history:
                all_keys.update(entry.keys())
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(self.training_history) 
        except Exception as e:
            self._handle_error(f"Error saving training results CSV: {e}")
    
    def _save_configuration_log(self) -> None:
        """Save configuration and settings to JSON and CSV files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_id = f"training_config_{timestamp}"
            config_data = {
                'config_id': config_id,
                'timestamp': datetime.now().isoformat(),
                'training_config': self.training_config,
                'performance_summary': self.get_summary()
            }
            json_file_path = os.path.join(self.output_dir, f"{config_id}.json")

            with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(config_data, jsonfile, indent=2)
            
            csv_file_path = os.path.join(self.output_dir, "configuration_log.csv")
            csv_row = {
                'config_id': config_id,
                'timestamp': config_data['timestamp'],
                'loss_weighting_strategy': self.training_config.get('loss_weighting_strategy', 'none'),
                'weight_modifications_used': ', '.join(self.weight_modifications_used) if self.weight_modifications_used else 'none',
                'final_r2': self.current_r2,
                'best_r2': self.best_r2,
                'total_training_time': self.total_training_time,
                'avg_epoch_time': self.avg_epoch_time,
                'peak_memory_mb': self.peak_memory_mb,
                'error_count': self.error_count
            }
            file_exists = os.path.exists(csv_file_path)
            
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_row.keys())

                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_row)
        except Exception as e:
            self._handle_error(f"Error saving configuration log: {e}")
    
    def _save_training_log(self) -> None:
        """Save detailed training log to text file."""
        try:
            file_path = os.path.join(self.output_dir, "training_log.txt")
            
            with open(file_path, 'w', encoding='utf-8') as logfile:
                logfile.write("=== Advanced TensorFlow Lab Training Log ===\n\n")
                logfile.write(f"Training started: {datetime.fromtimestamp(self.training_start_time).isoformat() if self.training_start_time else 'Unknown'}\n")
                logfile.write(f"Training ended: {datetime.fromtimestamp(self.training_end_time).isoformat() if self.training_end_time else 'Unknown'}\n")
                logfile.write(f"Total training time: {self.total_training_time:.2f} seconds\n\n")
                logfile.write("=== Configuration ===\n")

                for key, value in self.training_config.items():
                    logfile.write(f"{key}: {value}\n")

                logfile.write(f"Weight modifications used: {', '.join(self.weight_modifications_used) if self.weight_modifications_used else 'none'}\n\n")
                logfile.write("=== Performance Summary ===\n")
                logfile.write(f"Final training R²: {self.current_r2:.4f}\n")
                logfile.write(f"Best R²: {self.best_r2:.4f} at epoch {self.best_r2_epoch}\n")
                logfile.write(f"Greatest R² improvement: {self.greatest_improvement:.4f} at epoch {self.greatest_improvement_epoch}\n")
                logfile.write(f"Average epoch time: {self.avg_epoch_time:.2f} seconds\n")
                logfile.write(f"Peak memory usage: {self.peak_memory_mb:.1f} MB\n\n")
                logfile.write("=== Weight File Sizes ===\n")

                for file_path, size in self.weight_file_sizes.items():
                    logfile.write(f"{file_path}: {size} bytes\n")
                
                if self.error_count > 0:
                    logfile.write(f"\n=== Errors Encountered ===\n")
                    logfile.write(f"Total errors: {self.error_count}\n")

                    for error in self.errors_log[-10:]:
                        logfile.write(f"- {error}\n")
        except Exception as e:
            self._handle_error(f"Error saving training log: {e}")
    
    def _handle_error(self, error_message: str) -> None:
        """Handle and log errors."""
        self.error_count += 1
        self.errors_log.append(f"{datetime.now().isoformat()}: {error_message}")
        print(f"Performance Tracker Error: {error_message}")
