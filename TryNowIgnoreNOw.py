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

def _save_training_log(self): """Save detailed training log to text file."""
        try:
            file_path = os.path.join(self.output_dir, "training_log.txt")
            with open(file_path, 'w', encoding='utf-8') as logfile:
                logfile.write("=== Advanced TensorFlow Lab Training Log ===\n\n")
                logfile.write(f"Training started: {datetime.fromtimestamp(self.training_start_time).isoformat() if self.training_start_time else 'Unknown'}\n")
                logfile.write(f"Training ended: {datetime.fromtimestamp(self.training_end_time).isoformat() if self.training_end_time else 'Unknown'}\n")
                logfile.write(f"Total training time: {self.total_training_time:.2f} seconds\n\n"); logfile.write("=== Configuration ===\n")
                for key, value in self.training_config.items(): logfile.write(f"{key}: {value}\n")
                logfile.write(f"Adaptive loss strategy: {self.adaptive_loss_strategy}\n")
                logfile.write(f"Weight modifications used: {', '.join(self.weight_modifications_used) if self.weight_modifications_used else 'none'}\n\n")
                logfile.write("=== Performance Summary ===\n"); logfile.write(f"Final training accuracy: {self.current_accuracy:.4f}\n")
                logfile.write(f"Best accuracy: {self.best_accuracy:.4f} at epoch {self.best_accuracy_epoch}\n")
                logfile.write(f"Greatest accuracy improvement: {self.greatest_improvement:.4f} at epoch {self.greatest_improvement_epoch}\n")
                logfile.write(f"Average epoch time: {self.avg_epoch_time:.2f} seconds\n"); logfile.write(f"Peak memory usage: {self.peak_memory_mb:.1f} MB\n\n")
                logfile.write("=== Weight File Sizes ===\n")
                for file_path, size in self.weight_file_sizes.items(): logfile.write(f"{file_path}: {size} bytes\n")
                if self.error_count > 0:
                    logfile.write(f"\n=== Errors Encountered ===\n"); logfile.write(f"Total errors: {self.error_count}\n")
                    for error in self.errors_log[-10:]: logfile.write(f"- {error}\n")
        except Exception as e:
            self._handle_error(f"Error saving training log: {e}")
    def _handle_error(self, error_message: str): """Handle and log errors."""
        self.error_count += 1; self.errors_log.append(f"{datetime.now().isoformat()}: {error_message}")
        print(f"Performance Tracker Error: {error_message}")
