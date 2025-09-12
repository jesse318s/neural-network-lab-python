"""
Statistical Analysis Module for Systematic Neural Network Experiments

This module performs comprehensive statistical analysis of experiment results
including ANOVA testing, pairwise comparisons, and optimal configuration
identification for physics simulation neural networks.
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

try:
    import scipy.stats as stats
    from scipy.stats import f_oneway, tukey_hsd
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy and/or matplotlib not available. Some analysis features will be limited.")
    HAS_SCIPY = False

try:
    from experiment_runner import ExperimentResult
    from experiment_config import ExperimentConfig
except ImportError as e:
    print(f"Warning: Could not import experiment modules: {e}")


@dataclass
class StatisticalResult:
    """Results from statistical analysis."""
    test_name: str
    p_value: float
    statistic: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""
    groups_compared: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'p_value': self.p_value,
            'statistic': self.statistic,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'interpretation': self.interpretation,
            'groups_compared': self.groups_compared or []
        }


@dataclass
class OptimalConfiguration:
    """Optimal configuration for a specific metric."""
    metric_name: str
    optimal_value: float
    experiment_id: int
    config_details: Dict[str, Any]
    rank: int
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'optimal_value': self.optimal_value,
            'experiment_id': self.experiment_id,
            'config_details': self.config_details,
            'rank': self.rank,
            'confidence_score': self.confidence_score
        }


class ExperimentAnalyzer:
    """
    Comprehensive statistical analysis of systematic neural network experiments
    with ANOVA testing, pairwise comparisons, and optimal configuration identification.
    """
    
    def __init__(self, results_csv_path: str, output_dir: Optional[str] = None):
        """
        Initialize the experiment analyzer.
        
        Args:
            results_csv_path: Path to experiment results CSV file
            output_dir: Directory to save analysis results (defaults to same dir as CSV)
        """
        self.results_csv_path = results_csv_path
        self.output_dir = output_dir or os.path.dirname(results_csv_path)
        
        # Load and validate data
        self.df = self._load_and_validate_data()
        self.successful_df = self.df[self.df['success'] == True].copy()
        
        # Analysis results storage
        self.statistical_tests: List[StatisticalResult] = []
        self.optimal_configs: Dict[str, OptimalConfiguration] = {}
        self.analysis_summary = {}
        
        print(f"Loaded {len(self.df)} total experiments, {len(self.successful_df)} successful")
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate experiment results data."""
        try:
            if not os.path.exists(self.results_csv_path):
                raise FileNotFoundError(f"Results file not found: {self.results_csv_path}")
            
            df = pd.read_csv(self.results_csv_path)
            
            # Validate required columns
            required_columns = [
                'experiment_id', 'success', 'final_r2_score', 'final_test_mse',
                'architecture_name', 'constraints_name', 'loss_strategy_name',
                'particle_count', 'training_time_seconds'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in results: {missing_columns}")
            
            # Convert data types
            numeric_columns = [
                'final_r2_score', 'final_test_mse', 'final_test_mae',
                'training_time_seconds', 'inference_time_ms', 'model_size_mb',
                'stability_score', 'generalization_gap'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error loading results data: {e}")
            # Return empty dataframe with basic structure
            return pd.DataFrame({
                'experiment_id': [], 'success': [], 'final_r2_score': [],
                'architecture_name': [], 'constraints_name': [], 'loss_strategy_name': []
            })
    
    def perform_anova_analysis(self, 
                             metric: str = 'final_r2_score',
                             factors: List[str] = None,
                             alpha: float = 0.05) -> Dict[str, StatisticalResult]:
        """
        Perform ANOVA analysis to test for significant differences between groups.
        
        Args:
            metric: Performance metric to analyze
            factors: List of factors to test (e.g., ['architecture_name', 'constraints_name'])
            alpha: Significance level
            
        Returns:
            Dictionary of statistical results by factor
        """
        if not HAS_SCIPY:
            print("Warning: scipy not available. ANOVA analysis skipped.")
            return {}
        
        if factors is None:
            factors = ['architecture_name', 'constraints_name', 'loss_strategy_name']
        
        print(f"\n=== ANOVA Analysis for {metric} ===")
        
        anova_results = {}
        
        if len(self.successful_df) < 3:
            print("Insufficient data for ANOVA analysis")
            return anova_results
        
        for factor in factors:
            if factor not in self.successful_df.columns:
                print(f"Factor '{factor}' not found in data")
                continue
            
            try:
                # Group data by factor
                groups = []
                group_names = []
                
                for group_name in self.successful_df[factor].unique():
                    group_data = self.successful_df[self.successful_df[factor] == group_name][metric]
                    group_data = group_data.dropna()
                    
                    if len(group_data) >= 2:  # Need at least 2 samples per group
                        groups.append(group_data.values)
                        group_names.append(str(group_name))
                
                if len(groups) < 2:
                    print(f"Insufficient groups for factor '{factor}'")
                    continue
                
                # Perform one-way ANOVA
                statistic, p_value = f_oneway(*groups)
                
                # Calculate effect size (eta-squared)
                total_ss = np.sum([(np.array(group) - np.mean(np.concatenate(groups)))**2 
                                  for group in groups])
                within_ss = np.sum([np.sum((np.array(group) - np.mean(group))**2) 
                                   for group in groups])
                eta_squared = 1 - (within_ss / total_ss) if total_ss > 0 else 0
                
                # Interpretation
                significant = p_value < alpha
                if significant:
                    interpretation = f"Significant difference between {factor} groups (p < {alpha})"
                else:
                    interpretation = f"No significant difference between {factor} groups (p >= {alpha})"
                
                result = StatisticalResult(
                    test_name=f"ANOVA_{factor}_{metric}",
                    p_value=p_value,
                    statistic=statistic,
                    significant=significant,
                    effect_size=eta_squared,
                    interpretation=interpretation,
                    groups_compared=group_names
                )
                
                anova_results[factor] = result
                self.statistical_tests.append(result)
                
                print(f"{factor}: F = {statistic:.4f}, p = {p_value:.6f}, η² = {eta_squared:.4f}")
                print(f"  {interpretation}")
                
            except Exception as e:
                print(f"Error performing ANOVA for factor '{factor}': {e}")
        
        return anova_results
    
    def perform_pairwise_comparisons(self, 
                                   metric: str = 'final_r2_score',
                                   factor: str = 'architecture_name',
                                   method: str = 'tukey') -> List[StatisticalResult]:
        """
        Perform pairwise comparisons between groups.
        
        Args:
            metric: Performance metric to analyze
            factor: Factor to perform comparisons on
            method: Method for multiple comparison correction
            
        Returns:
            List of pairwise comparison results
        """
        if not HAS_SCIPY:
            print("Warning: scipy not available. Pairwise comparisons skipped.")
            return []
        
        print(f"\n=== Pairwise Comparisons: {factor} on {metric} ===")
        
        pairwise_results = []
        
        if len(self.successful_df) < 6:  # Need sufficient data
            print("Insufficient data for pairwise comparisons")
            return pairwise_results
        
        try:
            # Prepare data for pairwise comparison
            groups = []
            group_names = []
            
            for group_name in self.successful_df[factor].unique():
                group_data = self.successful_df[self.successful_df[factor] == group_name][metric]
                group_data = group_data.dropna()
                
                if len(group_data) >= 2:
                    groups.extend(group_data.values)
                    group_names.extend([str(group_name)] * len(group_data))
            
            if len(set(group_names)) < 2:
                print("Insufficient groups for pairwise comparison")
                return pairwise_results
            
            # Perform Tukey HSD test
            if method.lower() == 'tukey' and len(groups) > 0:
                try:
                    tukey_result = tukey_hsd(*[self.successful_df[self.successful_df[factor] == group][metric].dropna().values 
                                             for group in self.successful_df[factor].unique() 
                                             if len(self.successful_df[self.successful_df[factor] == group][metric].dropna()) >= 2])
                    
                    # Extract pairwise comparisons
                    unique_groups = [group for group in self.successful_df[factor].unique() 
                                   if len(self.successful_df[self.successful_df[factor] == group][metric].dropna()) >= 2]
                    
                    for i in range(len(unique_groups)):
                        for j in range(i+1, len(unique_groups)):
                            p_val = tukey_result.pvalue[i, j] if hasattr(tukey_result, 'pvalue') else 0.5
                            significant = p_val < 0.05
                            
                            result = StatisticalResult(
                                test_name=f"TukeyHSD_{factor}_{metric}",
                                p_value=p_val,
                                statistic=0.0,  # Tukey doesn't have a single statistic
                                significant=significant,
                                interpretation=f"{'Significant' if significant else 'No significant'} difference between {unique_groups[i]} and {unique_groups[j]}",
                                groups_compared=[str(unique_groups[i]), str(unique_groups[j])]
                            )
                            
                            pairwise_results.append(result)
                            self.statistical_tests.append(result)
                            
                            print(f"{unique_groups[i]} vs {unique_groups[j]}: p = {p_val:.6f} {'*' if significant else ''}")
                
                except Exception as e:
                    print(f"Error in Tukey HSD: {e}")
                    # Fallback to simple t-tests
                    self._perform_simple_pairwise_tests(metric, factor, pairwise_results)
            
        except Exception as e:
            print(f"Error performing pairwise comparisons: {e}")
        
        return pairwise_results
    
    def _perform_simple_pairwise_tests(self, metric: str, factor: str, pairwise_results: List):
        """Fallback method for pairwise comparisons using t-tests."""
        try:
            unique_groups = list(self.successful_df[factor].unique())
            
            for i in range(len(unique_groups)):
                for j in range(i+1, len(unique_groups)):
                    group1_data = self.successful_df[self.successful_df[factor] == unique_groups[i]][metric].dropna()
                    group2_data = self.successful_df[self.successful_df[factor] == unique_groups[j]][metric].dropna()
                    
                    if len(group1_data) >= 2 and len(group2_data) >= 2:
                        statistic, p_value = stats.ttest_ind(group1_data, group2_data)
                        significant = p_value < 0.05
                        
                        result = StatisticalResult(
                            test_name=f"t_test_{factor}_{metric}",
                            p_value=p_value,
                            statistic=statistic,
                            significant=significant,
                            interpretation=f"{'Significant' if significant else 'No significant'} difference between {unique_groups[i]} and {unique_groups[j]}",
                            groups_compared=[str(unique_groups[i]), str(unique_groups[j])]
                        )
                        
                        pairwise_results.append(result)
                        self.statistical_tests.append(result)
                        
        except Exception as e:
            print(f"Error in simple pairwise tests: {e}")
    
    def identify_optimal_configurations(self, 
                                      metrics: List[str] = None,
                                      top_n: int = 5) -> Dict[str, List[OptimalConfiguration]]:
        """
        Identify optimal configurations for different metrics.
        
        Args:
            metrics: List of metrics to optimize
            top_n: Number of top configurations to return per metric
            
        Returns:
            Dictionary of optimal configurations by metric
        """
        if metrics is None:
            metrics = ['final_r2_score', 'final_test_mse', 'training_time_seconds', 'stability_score']
        
        print(f"\n=== Identifying Optimal Configurations ===")
        
        optimal_configs = {}
        
        if len(self.successful_df) == 0:
            print("No successful experiments to analyze")
            return optimal_configs
        
        for metric in metrics:
            if metric not in self.successful_df.columns:
                print(f"Metric '{metric}' not found in data")
                continue
            
            print(f"\n--- Top {top_n} configurations for {metric} ---")
            
            # Determine if higher or lower is better
            ascending = metric in ['final_test_mse', 'final_test_mae', 'training_time_seconds', 
                                 'inference_time_ms', 'generalization_gap']
            
            # Sort and get top configurations
            metric_data = self.successful_df[metric].dropna()
            if len(metric_data) == 0:
                continue
            
            sorted_df = self.successful_df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            top_configs = sorted_df.head(top_n)
            
            metric_optimal_configs = []
            
            for rank, (_, row) in enumerate(top_configs.iterrows(), 1):
                # Calculate confidence score based on consistency across seeds
                confidence_score = self._calculate_confidence_score(row, metric)
                
                config_details = {
                    'particle_count': row.get('particle_count', 0),
                    'architecture_name': row.get('architecture_name', ''),
                    'constraints_name': row.get('constraints_name', ''),
                    'loss_strategy_name': row.get('loss_strategy_name', ''),
                    'train_batch_size': row.get('train_batch_size', 0),
                    'val_batch_size': row.get('val_batch_size', 0),
                    'random_seed': row.get('random_seed', 0)
                }
                
                optimal_config = OptimalConfiguration(
                    metric_name=metric,
                    optimal_value=row[metric],
                    experiment_id=row['experiment_id'],
                    config_details=config_details,
                    rank=rank,
                    confidence_score=confidence_score
                )
                
                metric_optimal_configs.append(optimal_config)
                
                print(f"  {rank}. ID {row['experiment_id']}: {metric} = {row[metric]:.4f}")
                print(f"     Config: {row.get('particle_count', 0)} particles, "
                      f"{row.get('architecture_name', '')}, {row.get('constraints_name', '')}, "
                      f"{row.get('loss_strategy_name', '')}")
                print(f"     Confidence: {confidence_score:.3f}")
            
            optimal_configs[metric] = metric_optimal_configs
        
        self.optimal_configs.update(optimal_configs)
        return optimal_configs
    
    def _calculate_confidence_score(self, row: pd.Series, metric: str) -> float:
        """Calculate confidence score for a configuration based on consistency."""
        try:
            # Find similar configurations (same architecture, constraints, loss strategy)
            similar_mask = (
                (self.successful_df['architecture_name'] == row.get('architecture_name', '')) &
                (self.successful_df['constraints_name'] == row.get('constraints_name', '')) &
                (self.successful_df['loss_strategy_name'] == row.get('loss_strategy_name', '')) &
                (self.successful_df['particle_count'] == row.get('particle_count', 0))
            )
            
            similar_experiments = self.successful_df[similar_mask]
            
            if len(similar_experiments) <= 1:
                return 0.5  # Low confidence if no replicates
            
            # Calculate coefficient of variation (lower is better)
            values = similar_experiments[metric].dropna()
            if len(values) <= 1 or values.mean() == 0:
                return 0.5
            
            cv = values.std() / abs(values.mean())
            confidence = max(0.0, min(1.0, 1.0 - cv))  # Convert to 0-1 scale
            
            return confidence
            
        except Exception:
            return 0.5
    
    def analyze_generalizability(self) -> Dict[str, Any]:
        """Analyze how well configurations generalize across different particle counts."""
        print(f"\n=== Generalizability Analysis ===")
        
        generalizability_results = {}
        
        if 'particle_count' not in self.successful_df.columns:
            print("Particle count data not available for generalizability analysis")
            return generalizability_results
        
        try:
            # Group by configuration (excluding particle count and seed)
            config_cols = ['architecture_name', 'constraints_name', 'loss_strategy_name']
            available_cols = [col for col in config_cols if col in self.successful_df.columns]
            
            if not available_cols:
                print("Configuration columns not available")
                return generalizability_results
            
            grouped = self.successful_df.groupby(available_cols)
            
            generalization_scores = []
            
            for config, group in grouped:
                if len(group) < 2:  # Need multiple particle counts
                    continue
                
                # Calculate performance consistency across particle counts
                r2_scores = group['final_r2_score'].dropna()
                
                if len(r2_scores) >= 2:
                    # Generalization score: 1 - coefficient_of_variation
                    cv = r2_scores.std() / r2_scores.mean() if r2_scores.mean() != 0 else float('inf')
                    generalization_score = max(0.0, 1.0 - cv)
                    
                    config_dict = dict(zip(available_cols, config))
                    config_dict['generalization_score'] = generalization_score
                    config_dict['mean_r2_score'] = r2_scores.mean()
                    config_dict['std_r2_score'] = r2_scores.std()
                    config_dict['particle_counts_tested'] = sorted(group['particle_count'].unique())
                    
                    generalization_scores.append(config_dict)
            
            # Sort by generalization score
            generalization_scores.sort(key=lambda x: x['generalization_score'], reverse=True)
            
            print("Top 5 most generalizable configurations:")
            for i, config in enumerate(generalization_scores[:5]):
                print(f"  {i+1}. Score: {config['generalization_score']:.3f}, "
                      f"Mean R²: {config['mean_r2_score']:.3f}")
                print(f"     Config: {config.get('architecture_name', '')}, "
                      f"{config.get('constraints_name', '')}, {config.get('loss_strategy_name', '')}")
                print(f"     Tested on: {config['particle_counts_tested']}")
            
            generalizability_results = {
                'top_generalizable_configs': generalization_scores[:10],
                'mean_generalization_score': np.mean([c['generalization_score'] for c in generalization_scores]),
                'total_configs_analyzed': len(generalization_scores)
            }
            
        except Exception as e:
            print(f"Error in generalizability analysis: {e}")
        
        return generalizability_results
    
    def generate_comprehensive_report(self, output_file: str = "experiment_analysis_report.json"):
        """Generate comprehensive analysis report."""
        print(f"\n=== Generating Comprehensive Analysis Report ===")
        
        try:
            # Perform all analyses
            anova_results = self.perform_anova_analysis()
            pairwise_results = self.perform_pairwise_comparisons()
            optimal_configs = self.identify_optimal_configurations()
            generalizability = self.analyze_generalizability()
            
            # Generate summary statistics
            summary_stats = self._generate_summary_statistics()
            
            # Compile comprehensive report
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_experiments': len(self.df),
                    'successful_experiments': len(self.successful_df),
                    'success_rate': len(self.successful_df) / len(self.df) if len(self.df) > 0 else 0,
                    'unique_configurations': len(self.successful_df.drop_duplicates(['architecture_name', 'constraints_name', 'loss_strategy_name'])),
                },
                'summary_statistics': summary_stats,
                'anova_results': {factor: result.to_dict() for factor, result in anova_results.items()},
                'pairwise_comparisons': [result.to_dict() for result in pairwise_results],
                'optimal_configurations': {
                    metric: [config.to_dict() for config in configs] 
                    for metric, configs in optimal_configs.items()
                },
                'generalizability_analysis': generalizability,
                'recommendations': self._generate_recommendations(anova_results, optimal_configs, generalizability)
            }
            
            # Save report
            report_path = os.path.join(self.output_dir, output_file)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
            
            print(f"Comprehensive analysis report saved to: {report_path}")
            
            # Print key findings
            self._print_key_findings(report)
            
            return report
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            return {}
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        if len(self.successful_df) == 0:
            return {}
        
        numeric_columns = ['final_r2_score', 'final_test_mse', 'final_test_mae', 
                          'training_time_seconds', 'stability_score']
        
        stats = {}
        
        for col in numeric_columns:
            if col in self.successful_df.columns:
                data = self.successful_df[col].dropna()
                if len(data) > 0:
                    stats[col] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'median': float(data.median()),
                        'q25': float(data.quantile(0.25)),
                        'q75': float(data.quantile(0.75))
                    }
        
        return stats
    
    def _generate_recommendations(self, anova_results: Dict, optimal_configs: Dict, 
                                generalizability: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        try:
            # ANOVA-based recommendations
            for factor, result in anova_results.items():
                if result.significant:
                    recommendations.append(
                        f"The choice of {factor} significantly affects performance (p = {result.p_value:.4f}). "
                        f"Consider focusing optimization efforts on this factor."
                    )
            
            # Optimal configuration recommendations
            if 'final_r2_score' in optimal_configs:
                best_config = optimal_configs['final_r2_score'][0]
                recommendations.append(
                    f"For best R² score, use: {best_config.config_details['architecture_name']} architecture "
                    f"with {best_config.config_details['constraints_name']} constraints and "
                    f"{best_config.config_details['loss_strategy_name']} loss strategy."
                )
            
            # Generalizability recommendations
            if generalizability and 'top_generalizable_configs' in generalizability:
                if generalizability['top_generalizable_configs']:
                    top_general = generalizability['top_generalizable_configs'][0]
                    recommendations.append(
                        f"For best generalizability across particle counts, use: "
                        f"{top_general.get('architecture_name', '')} architecture with "
                        f"{top_general.get('constraints_name', '')} constraints."
                    )
            
            # Performance vs efficiency trade-offs
            if 'training_time_seconds' in optimal_configs and 'final_r2_score' in optimal_configs:
                fast_config = optimal_configs['training_time_seconds'][0]
                accurate_config = optimal_configs['final_r2_score'][0]
                
                if fast_config.experiment_id != accurate_config.experiment_id:
                    recommendations.append(
                        f"Trade-off identified: fastest training uses {fast_config.config_details['architecture_name']} "
                        f"while best accuracy uses {accurate_config.config_details['architecture_name']}."
                    )
            
        except Exception as e:
            print(f"Warning: Error generating recommendations: {e}")
        
        if not recommendations:
            recommendations.append("Insufficient data for specific recommendations. Collect more experimental data.")
        
        return recommendations
    
    def _print_key_findings(self, report: Dict[str, Any]):
        """Print key findings from the analysis."""
        print(f"\n=== Key Findings ===")
        
        try:
            # Data overview
            data_summary = report.get('data_summary', {})
            print(f"Analyzed {data_summary.get('successful_experiments', 0)} successful experiments")
            print(f"Success rate: {data_summary.get('success_rate', 0):.1%}")
            
            # Best performing configuration
            optimal_configs = report.get('optimal_configurations', {})
            if 'final_r2_score' in optimal_configs and optimal_configs['final_r2_score']:
                best = optimal_configs['final_r2_score'][0]
                print(f"Best R² score: {best['optimal_value']:.4f} (Experiment ID: {best['experiment_id']})")
            
            # Significant factors
            anova_results = report.get('anova_results', {})
            significant_factors = [factor for factor, result in anova_results.items() 
                                 if result.get('significant', False)]
            if significant_factors:
                print(f"Significant factors: {', '.join(significant_factors)}")
            
            # Recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                print(f"\nTop recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
        except Exception as e:
            print(f"Error printing key findings: {e}")


def main():
    """Demonstrate experiment analysis functionality."""
    print("=== Physics Simulation Experiment Analysis ===")
    
    # Create sample data for testing
    sample_data = {
        'experiment_id': range(20),
        'success': [True] * 18 + [False] * 2,
        'final_r2_score': np.random.normal(0.85, 0.1, 20),
        'final_test_mse': np.random.exponential(0.1, 20),
        'architecture_name': ['shallow_wide'] * 10 + ['deep_narrow'] * 10,
        'constraints_name': ['none'] * 5 + ['binary_changes_only'] * 5 + ['all_constraints'] * 10,
        'loss_strategy_name': ['mse_only'] * 7 + ['combined_adaptive'] * 13,
        'particle_count': [100, 500, 1000, 2000] * 5,
        'training_time_seconds': np.random.exponential(100, 20)
    }
    
    # Save sample data
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = "sample_experiment_results.csv"
    sample_df.to_csv(sample_csv_path, index=False)
    
    # Analyze results
    analyzer = ExperimentAnalyzer(sample_csv_path)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report("sample_analysis_report.json")
    
    # Cleanup
    if os.path.exists(sample_csv_path):
        os.remove(sample_csv_path)
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()