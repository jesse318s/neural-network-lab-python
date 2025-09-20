"""Performance Tracking and Metrics Collection for Advanced TensorFlow Training 242
This module implements comprehensive performance tracking for the TensorFlow lab,
including training metrics, CSV output, memory monitoring, and configuration logging."""
import os, time, csv, json, psutil; import numpy as np; from datetime import datetime
from typing import Dict, List, Any, Optional
class PerformanceTracker:
    def __init__(self, output_dir: str = "training_output"):
        self.output_dir = output_dir
        self.training_start_time, self.training_end_time = None, None
        self.epoch_start_time = None
        # Performance metrics, after 3 lines timing and then memory metrics
        self.current_accuracy, self.best_accuracy = 0.0, 0.0
        self.best_accuracy_epoch, self.previous_accuracy  = 0, 0.0
        self.greatest_improvement, self.greatest_improvement_epoch = 0.0, 0
        self.avg_epoch_time, self.total_training_time, self.epoch_times   = 0.0, 0.0, []
        self.current_memory_mb,self.peak_memory_mb = 0.0,0.0
        self.weight_file_sizes = {}
        # Configuration tracking, below error tracking, and below below output directory
        self.training_history = []
        self.training_config, self.weight_modifications_used = {}, []
        self.adaptive_loss_strategy = "none"
        self.error_count, self.errors_log = 0, []
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create output directory {self.output_dir}: {e}")
            self.output_dir = "."
    def start_training(self, config: Dict[str, Any]): """Start training session and initialize tracking."""
        try:
            self.training_start_time = time.time()
            self.training_config = config.copy()
            self.adaptive_loss_strategy = config.get('adaptive_loss_strategy', 'none')
            print(f"Performance tracking started - Output: {self.output_dir}")  
        except Exception as e:
            self._handle_error(f"Error starting training tracking: {e}")
    def start_epoch(self, epoch: int): """Start epoch tracking."""
        try:
            self.epoch_start_time = time.time()
            self._update_memory_usage()
        except Exception as e:
            self._handle_error(f"Error starting epoch {epoch}: {e}")
    def end_epoch(self, epoch: int, logs: Dict[str, float]): """End epoch tracking and record metrics."""
        try: # Calculate epoch time
            if self.epoch_start_time is not None:
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time) 
                self.avg_epoch_time = np.mean(self.epoch_times)
            else: epoch_time = 0.0 
            # Update accuracy metrics including best accuracy and then greatest improvement
            current_accuracy = logs.get('accuracy', logs.get('val_accuracy', 0.0))
            self.current_accuracy = current_accuracy
            if current_accuracy > self.best_accuracy: self.best_accuracy, self.best_accuracy_epoc= current_accuracy, epoch
            if epoch > 0:
                improvement = current_accuracy - self.previous_accuracy
                if improvement > self.greatest_improvement: self.greatest_improvement, self.greatest_improvement_epoch  = improvement, epoch
            #update accuracy, memory usage, and record training history
            self.previous_accuracy = current_accuracy
            self._update_memory_usage()    
            history_entry = { 'epoch': epoch, 'timestamp': datetime.now().isoformat(), 'epoch_time': epoch_time, 
                             'memory_mb': self.current_memory_mb, **logs}
            self.training_history.append(history_entry) 
        except Exception as e:
            self._handle_error(f"Error ending epoch {epoch}: {e}")
    def end_training(self) -> Dict[str, Any]: """End training session and finalize tracking."""
        try:
            self.training_end_time = time.time()
            if self.training_start_time is not None: self.total_training_time = self.training_end_time - self.training_start_time
            self._update_memory_usage()
            return {'total_training_time': self.total_training_time, 'current_accuracy': self.current_accuracy, 
                'best_accuracy': self.best_accuracy, 'best_accuracy_epoch': self.best_accuracy_epoch,
                'greatest_improvement': self.greatest_improvement, 'greatest_improvement_epoch': self.greatest_improvement_epoch,
                'avg_epoch_time': self.avg_epoch_time, 'current_memory_mb': self.current_memory_mb, 'peak_memory_mb': self.peak_memory_mb,
                'adaptive_loss_strategy': self.adaptive_loss_strategy,
                'weight_modifications_used': self.weight_modifications_used, 'error_count': self.error_count}   
        except Exception as e:
            self._handle_error(f"Error ending training: {e}")
            return {}
    def record_weight_file_size(self, file_path: str): """Record the size of a weight file."""
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.weight_file_sizes[file_path] = file_size
        except Exception as e:
            self._handle_error(f"Error recording weight file size for {file_path}: {e}")
    def measure_inference_time(self, model: Optional[Any], test_data: np.ndarray, num_runs: int = 10) -> float: """Measure model inference time."""
        try:
            inference_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = model.predict(test_data, verbose=0)
                end_time = time.time()
                inference_times.append(end_time - start_time)
            return float(np.mean(inference_times))  
        except Exception as e:
            self._handle_error(f"Error measuring inference time: {e}")
            return 0.0
    def add_weight_modification(self, modification_name: str): """Record that a weight modification technique was used."""
        try:
            if modification_name not in self.weight_modifications_used: self.weight_modifications_used.append(modification_name)
        except Exception as e:
            self._handle_error(f"Error adding weight modification {modification_name}: {e}")
    def save_results(self): """Save all tracked results to CSV and JSON files."""
        try:
            self._save_training_results_csv(); self._save_configuration_log(); self._save_training_log() 
            print(f"Results saved to {self.output_dir}")
        except Exception as e:
            self._handle_error(f"Error saving results: {e}")
    def create_loss_history_csv(self, loss_history: List[Dict[str, Any]]):
        try:
            file_path = os.path.join(self.output_dir, "loss_history.csv")
            if not loss_history:
                return
            all_keys = set()
            for entry in loss_history: all_keys.update(entry.keys())
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader(); writer.writerows(loss_history)
            print(f"Loss history saved to {file_path}")
        except Exception as e:
            self._handle_error(f"Error creating loss history CSV: {e}")
    def get_summary(self) -> Dict[str, Any]: """Get a summary of all performance metrics."""
        try:
            return {'current_accuracy': self.current_accuracy,'best_accuracy': self.best_accuracy,
                'best_accuracy_epoch': self.best_accuracy_epoch, 'greatest_improvement': self.greatest_improvement,
                'greatest_improvement_epoch': self.greatest_improvement_epoch, 'avg_epoch_time': self.avg_epoch_time,
                 'total_training_time': self.total_training_time, 'current_memory_mb': self.current_memory_mb,
                 'peak_memory_mb': self.peak_memory_mb, 'adaptive_loss_strategy': self.adaptive_loss_strategy, 
                'weight_modifications_used': self.weight_modifications_used, 'weight_file_sizes': self.weight_file_sizes, 
                'error_count': self.error_count,'total_epochs': len(self.training_history)}
        except Exception as e:
            self._handle_error(f"Error getting summary: {e}")
            return {} 
    def _update_memory_usage(self):  """Update current memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.current_memory_mb = memory_info.rss / 1024 / 1024
            if self.current_memory_mb > self.peak_memory_mb: self.peak_memory_mb = self.current_memory_mb
        except Exception:
            pass
    def _save_training_results_csv(self):
        try:
            file_path = os.path.join(self.output_dir, "training_results.csv")
            if not self.training_history:
                return
            all_keys = set()
            for entry in self.training_history: all_keys.update(entry.keys())
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader(); writer.writerows(self.training_history) 
        except Exception as e:
            self._handle_error(f"Error saving training results CSV: {e}")
    def _save_configuration_log(self)#Save configuration and settings to JSON and CSV files
        try:
            # Create unique ID based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_id = f"training_config_{timestamp}"
            config_data = { 'config_id': config_id, 'timestamp': datetime.now().isoformat(), 'training_config': self.training_config,
                 'adaptive_loss_strategy': self.adaptive_loss_strategy,  'weight_modifications_used': self.weight_modifications_used,
                'performance_summary': self.get_summary(), 'weight_file_sizes': self.weight_file_sizes}
            json_file_path = os.path.join(self.output_dir, f"{config_id}.json")
            with open(json_file_path, 'w', encoding='utf-8') as jsonfile: wijson.dump(config_data, jsonfile, indent=2)  # Save/append to CSV for easy comparison
            csv_file_path = os.path.join(self.output_dir, "configuration_log.csv")
            csv_row = { 'config_id': config_id, 'timestamp': config_data['timestamp'],
                'adaptive_loss_strategy': self.adaptive_loss_strategy, 'weight_modifications_used': ',
                 '.join(self.weight_modifications_used) if self.weight_modifications_used else 'none',
                'final_accuracy': self.current_accuracy, 'best_accuracy': self.best_accuracy, 'total_training_time': self.total_training_time, 
                'avg_epoch_time': self.avg_epoch_time, 'peak_memory_mb': self.peak_memory_mb, 'error_count': self.error_count}
            file_exists = os.path.exists(csv_file_path)
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_row.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(csv_row)
        except Exception as e:
            self._handle_error(f"Error saving configuration log: {e}")





