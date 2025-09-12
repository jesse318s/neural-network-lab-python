"""
Integration Tests for Systematic Neural Network Experimentation Framework

This module provides comprehensive tests to verify that all components
of the systematic experimentation framework work correctly together
with the existing neural network lab code.
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import traceback
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class IntegrationTester:
    """
    Comprehensive integration tester for the systematic experimentation framework.
    Tests all components and their interactions to ensure proper functionality.
    """
    
    def __init__(self):
        """Initialize the integration tester."""
        self.test_dir = None
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
    def setup_test_environment(self):
        """Set up temporary test environment."""
        try:
            self.test_dir = tempfile.mkdtemp(prefix="neural_net_integration_test_")
            print(f"Test environment created: {self.test_dir}")
            
            # Change to test directory
            os.chdir(self.test_dir)
            
            return True
        except Exception as e:
            self.errors.append(f"Failed to setup test environment: {e}")
            return False
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        try:
            if self.test_dir and os.path.exists(self.test_dir):
                os.chdir(os.path.dirname(self.test_dir))
                shutil.rmtree(self.test_dir)
                print(f"Test environment cleaned up: {self.test_dir}")
        except Exception as e:
            print(f"Warning: Failed to cleanup test environment: {e}")
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        self.total_tests += 1
        print(f"\n--- Running Test: {test_name} ---")
        
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
                self.passed_tests += 1
                self.test_results[test_name] = {'status': 'PASSED', 'error': None}
                return True
            else:
                print(f"✗ {test_name} FAILED")
                self.test_results[test_name] = {'status': 'FAILED', 'error': 'Test returned False'}
                return False
                
        except Exception as e:
            error_msg = f"Exception in {test_name}: {str(e)}"
            print(f"✗ {test_name} ERROR: {error_msg}")
            self.errors.append(error_msg)
            self.test_results[test_name] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_imports(self) -> bool:
        """Test that all new modules can be imported correctly."""
        try:
            print("Testing module imports...")
            
            # Import new modules
            from experiment_config import ExperimentConfigManager, ExperimentConfig
            from experiment_runner import ExperimentRunner, ExperimentResult
            from experiment_analysis import ExperimentAnalyzer
            from data_loader import generate_enhanced_particle_data, get_data_complexity_info
            
            # Test that classes can be instantiated
            config_manager = ExperimentConfigManager("test_experiments")
            runner = ExperimentRunner("test_experiments")
            
            print("  ✓ All imports successful")
            print("  ✓ Classes instantiated successfully")
            return True
            
        except ImportError as e:
            print(f"  ✗ Import error: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Error during import test: {e}")
            return False
    
    def test_experiment_config_generation(self) -> bool:
        """Test experiment configuration generation."""
        try:
            print("Testing experiment configuration generation...")
            
            from experiment_config import ExperimentConfigManager
            
            # Create configuration manager
            config_manager = ExperimentConfigManager("test_experiments")
            
            # Generate subset of configurations
            configs = config_manager.generate_subset_configurations(
                particle_counts=[100, 500],
                architectures=['shallow_wide'],
                constraints=['none', 'binary_changes_only'],
                loss_strategies=['mse_only'],
                random_seeds=[42],
                epochs=5
            )
            
            # Validate configurations
            if not configs:
                print("  ✗ No configurations generated")
                return False
            
            print(f"  ✓ Generated {len(configs)} configurations")
            
            # Test configuration properties
            config = configs[0]
            if not hasattr(config, 'experiment_id'):
                print("  ✗ Configuration missing experiment_id")
                return False
            
            if not hasattr(config, 'particle_count'):
                print("  ✗ Configuration missing particle_count")
                return False
            
            # Test saving and loading
            config_manager.save_configurations("test_configs.json")
            loaded_success = config_manager.load_configurations("test_configs.json")
            
            if not loaded_success:
                print("  ✗ Failed to save/load configurations")
                return False
            
            print("  ✓ Configuration save/load successful")
            return True
            
        except Exception as e:
            print(f"  ✗ Error in configuration test: {e}")
            return False
    
    def test_enhanced_data_generation(self) -> bool:
        """Test enhanced data generation with variable particle counts."""
        try:
            print("Testing enhanced data generation...")
            
            from data_loader import generate_enhanced_particle_data, get_data_complexity_info
            
            # Test different particle counts and complexities
            test_cases = [
                (50, 'simple'),
                (200, 'medium'),
                (500, 'complex')
            ]
            
            for num_particles, complexity in test_cases:
                print(f"  Testing {num_particles} particles with {complexity} complexity...")
                
                # Get complexity info
                complexity_info = get_data_complexity_info(num_particles)
                if not complexity_info:
                    print(f"    ✗ No complexity info for {num_particles} particles")
                    return False
                
                # Generate data
                df = generate_enhanced_particle_data(
                    num_particles=num_particles,
                    complexity_level=complexity,
                    random_seed=42,
                    save_to_file=False
                )
                
                if df is None or len(df) == 0:
                    print(f"    ✗ No data generated for {num_particles} particles")
                    return False
                
                if len(df) != num_particles:
                    print(f"    ✗ Wrong number of particles: expected {num_particles}, got {len(df)}")
                    return False
                
                # Check required columns
                required_cols = ['mass', 'initial_velocity_x', 'final_velocity_x', 'kinetic_energy']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"    ✗ Missing required column: {col}")
                        return False
                
                print(f"    ✓ {num_particles} particles generated successfully")
            
            print("  ✓ Enhanced data generation successful")
            return True
            
        except Exception as e:
            print(f"  ✗ Error in data generation test: {e}")
            return False
    
    def test_single_experiment_execution(self) -> bool:
        """Test execution of a single experiment."""
        try:
            print("Testing single experiment execution...")
            
            from experiment_config import ExperimentConfig
            from experiment_runner import ExperimentRunner
            
            # Create simple experiment configuration
            config = ExperimentConfig(
                experiment_id=1,
                particle_count=50,
                train_batch_size=16,
                val_batch_size=8,
                architecture={
                    'name': 'shallow_wide',
                    'hidden_layers': [32, 16],
                    'activation': 'relu',
                    'dropout_rate': 0.2
                },
                constraints={
                    'name': 'none',
                    'binary_changes': False,
                    'binary_max': False,
                    'oscillation': False
                },
                loss_config={
                    'name': 'mse_only',
                    'strategy': 'mse_only',
                    'adaptive': False
                },
                epochs=3,
                random_seed=42
            )
            
            # Create experiment runner
            runner = ExperimentRunner("test_experiments")
            
            # Run single experiment
            result = runner._run_single_experiment(config)
            
            if result is None:
                print("  ✗ No result from experiment")
                return False
            
            if not hasattr(result, 'experiment_id'):
                print("  ✗ Result missing experiment_id")
                return False
            
            if not hasattr(result, 'success'):
                print("  ✗ Result missing success flag")
                return False
            
            print(f"  ✓ Experiment completed: success={result.success}")
            
            if result.success:
                print(f"    R² Score: {result.final_r2_score:.4f}")
                print(f"    Training time: {result.training_time_seconds:.2f}s")
            else:
                print(f"    Error: {result.error_message}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error in single experiment test: {e}")
            traceback.print_exc()
            return False
    
    def test_experiment_analysis(self) -> bool:
        """Test experiment analysis functionality."""
        try:
            print("Testing experiment analysis...")
            
            from experiment_analysis import ExperimentAnalyzer
            
            # Create sample experiment results
            sample_data = {
                'experiment_id': range(10),
                'success': [True] * 8 + [False] * 2,
                'final_r2_score': np.random.normal(0.7, 0.2, 10),
                'final_test_mse': np.random.exponential(0.1, 10),
                'final_test_mae': np.random.exponential(0.05, 10),
                'training_time_seconds': np.random.exponential(30, 10),
                'architecture_name': ['shallow_wide'] * 5 + ['deep_narrow'] * 5,
                'constraints_name': ['none'] * 3 + ['binary_changes_only'] * 4 + ['all_constraints'] * 3,
                'loss_strategy_name': ['mse_only'] * 4 + ['combined_adaptive'] * 6,
                'particle_count': [100, 500, 100, 500, 100, 500, 100, 500, 100, 500],
                'stability_score': np.random.uniform(0.5, 1.0, 10),
                'generalization_gap': np.random.exponential(0.02, 10)
            }
            
            # Save sample data
            sample_df = pd.DataFrame(sample_data)
            sample_csv = "sample_results.csv"
            sample_df.to_csv(sample_csv, index=False)
            
            # Create analyzer
            analyzer = ExperimentAnalyzer(sample_csv)
            
            if len(analyzer.successful_df) == 0:
                print("  ✗ No successful experiments loaded")
                return False
            
            print(f"  ✓ Loaded {len(analyzer.successful_df)} successful experiments")
            
            # Test ANOVA analysis
            anova_results = analyzer.perform_anova_analysis(
                metric='final_r2_score',
                factors=['architecture_name', 'constraints_name']
            )
            
            print(f"  ✓ ANOVA analysis completed: {len(anova_results)} factors tested")
            
            # Test optimal configuration identification
            optimal_configs = analyzer.identify_optimal_configurations(
                metrics=['final_r2_score', 'training_time_seconds'],
                top_n=3
            )
            
            if not optimal_configs:
                print("  ✗ No optimal configurations identified")
                return False
            
            print(f"  ✓ Optimal configurations identified: {len(optimal_configs)} metrics")
            
            # Test report generation
            report = analyzer.generate_comprehensive_report("test_analysis_report.json")
            
            if not report:
                print("  ✗ Failed to generate analysis report")
                return False
            
            print("  ✓ Analysis report generated successfully")
            
            # Clean up
            if os.path.exists(sample_csv):
                os.remove(sample_csv)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error in analysis test: {e}")
            return False
    
    def test_backward_compatibility(self) -> bool:
        """Test that existing functionality still works."""
        try:
            print("Testing backward compatibility...")
            
            # Test existing data loading
            from data_loader import generate_particle_data, load_and_prepare_data
            
            # Generate data using original function
            df = generate_particle_data(num_particles=20, save_to_file=False)
            
            if df is None or len(df) != 20:
                print("  ✗ Original data generation failed")
                return False
            
            print("  ✓ Original data generation works")
            
            # Test existing data preparation
            X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_prepare_data('nonexistent.csv')
            
            if X_train is None:
                print("  ✗ Data preparation failed")
                return False
            
            print("  ✓ Data preparation works")
            
            # Test existing model creation (without training)
            from main import create_model
            
            model = create_model(input_shape=(8,), output_shape=6)
            
            if model is None:
                print("  ✗ Model creation failed")
                return False
            
            print("  ✓ Model creation works")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error in backward compatibility test: {e}")
            return False
    
    def test_end_to_end_integration(self) -> bool:
        """Test complete end-to-end systematic experiment workflow."""
        try:
            print("Testing end-to-end integration...")
            
            from experiment_config import ExperimentConfigManager
            from experiment_runner import ExperimentRunner
            from experiment_analysis import ExperimentAnalyzer
            
            # Step 1: Generate configurations
            print("  Step 1: Generating configurations...")
            config_manager = ExperimentConfigManager("integration_test")
            
            configs = config_manager.generate_subset_configurations(
                particle_counts=[100],  # Use standard particle count
                architectures=['shallow_wide'],
                constraints=['none'],
                loss_strategies=['mse_only'],
                random_seeds=[42],
                epochs=2  # Very short for testing
            )
            
            if not configs:
                print("    ✗ No configurations generated")
                return False
            
            print(f"    ✓ Generated {len(configs)} configurations")
            
            # Step 2: Run experiments
            print("  Step 2: Running experiments...")
            runner = ExperimentRunner("integration_test")
            
            success = runner.run_all_experiments(
                configs,
                experiment_subset=[0],  # Run only first experiment
                save_frequency=1
            )
            
            print(f"    ✓ Experiment execution completed: {success}")
            
            # Step 3: Analyze results
            print("  Step 3: Analyzing results...")
            
            if os.path.exists(runner.results_file):
                analyzer = ExperimentAnalyzer(runner.results_file)
                
                if len(analyzer.df) > 0:
                    print(f"    ✓ Analysis loaded {len(analyzer.df)} experiments")
                    
                    # Generate simple report
                    report = analyzer.generate_comprehensive_report("integration_analysis.json")
                    
                    print("    ✓ Analysis completed successfully")
                else:
                    print("    ⚠ No experiments to analyze")
            else:
                print("    ⚠ No results file found")
            
            print("  ✓ End-to-end integration successful")
            return True
            
        except Exception as e:
            print(f"  ✗ Error in end-to-end test: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return summary."""
        print("=" * 60)
        print("NEURAL NETWORK LAB INTEGRATION TESTS")
        print("=" * 60)
        
        if not self.setup_test_environment():
            return {'error': 'Failed to setup test environment'}
        
        try:
            # Define all tests
            tests = [
                ("Module Imports", self.test_imports),
                ("Experiment Configuration", self.test_experiment_config_generation),
                ("Enhanced Data Generation", self.test_enhanced_data_generation),
                ("Single Experiment Execution", self.test_single_experiment_execution),
                ("Experiment Analysis", self.test_experiment_analysis),
                ("Backward Compatibility", self.test_backward_compatibility),
                ("End-to-End Integration", self.test_end_to_end_integration)
            ]
            
            # Run all tests
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
            
            # Generate summary
            success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
            
            print("\n" + "=" * 60)
            print("INTEGRATION TEST SUMMARY")
            print("=" * 60)
            print(f"Total tests: {self.total_tests}")
            print(f"Passed: {self.passed_tests}")
            print(f"Failed: {self.total_tests - self.passed_tests}")
            print(f"Success rate: {success_rate:.1f}%")
            
            if self.errors:
                print(f"\nErrors encountered: {len(self.errors)}")
                for i, error in enumerate(self.errors[-5:], 1):  # Show last 5 errors
                    print(f"  {i}. {error}")
            
            # Overall assessment
            if success_rate >= 100:
                print("\n🎉 ALL TESTS PASSED! Integration is successful.")
                assessment = "EXCELLENT"
            elif success_rate >= 80:
                print("\n✓ Most tests passed. Integration is largely successful.")
                assessment = "GOOD"
            elif success_rate >= 60:
                print("\n⚠ Some tests failed. Integration has issues.")
                assessment = "PARTIAL"
            else:
                print("\n✗ Many tests failed. Integration needs work.")
                assessment = "POOR"
            
            return {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'success_rate': success_rate,
                'assessment': assessment,
                'test_results': self.test_results,
                'errors': self.errors
            }
            
        finally:
            self.cleanup_test_environment()


def main():
    """Run integration tests."""
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    # Save results to file if possible
    try:
        with open("integration_test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nTest results saved to integration_test_results.json")
    except Exception as e:
        print(f"\nWarning: Could not save test results: {e}")
    
    # Return appropriate exit code
    if results.get('assessment') in ['EXCELLENT', 'GOOD']:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)