"""
ML Utilities for Neural Network Training

This module combines adaptive loss functions and data processing utilities into a 
comprehensive, compact ML utility suite for the advanced neural network project.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# ADAPTIVE LOSS UTILITIES
# ============================================================================

def compute_loss_weights(strategy: str, epoch: int = 0, accuracy: float = 0.5, 
                        prev_loss: float = 1.0) -> Tuple[float, float]:
    """
    Compute adaptive loss function weights based on training progress.
    
    Args:
        strategy: Weighting strategy ('epoch_based', 'accuracy_based', 'loss_based', 'combined')
        epoch: Current training epoch
        accuracy: Previous epoch's accuracy
        prev_loss: Previous epoch's loss value
        
    Returns:
        Tuple of (mse_weight, mae_weight)
    """
    try:
        if strategy == 'epoch_based':
            if epoch < 10: return 0.3, 0.7
            elif epoch < 30:
                progress = (epoch - 10) / 20
                return 0.3 + 0.4 * progress, 0.7 - 0.4 * progress
            else: return 0.8, 0.2      
        elif strategy == 'accuracy_based':
            if accuracy < 0.3: return 0.2, 0.8
            elif accuracy < 0.6:
                progress = (accuracy - 0.3) / 0.3
                return 0.2 + 0.3 * progress, 0.8 - 0.3 * progress
            elif accuracy < 0.85:
                progress = (accuracy - 0.6) / 0.25
                return 0.5 + 0.3 * progress, 0.5 - 0.3 * progress
            else: return 0.9, 0.1
        elif strategy == 'loss_based':
            log_loss = math.log(max(prev_loss, 1e-8))

            if log_loss > 0: return 0.3, 0.7
            elif log_loss > -2:
                progress = (log_loss + 2) / 2
                return 0.3 + 0.4 * progress, 0.7 - 0.4 * progress
            else: return 0.8, 0.2       
        elif strategy == 'combined':
            # Get weights from each strategy
            epoch_mse, epoch_mae = compute_loss_weights('epoch_based', epoch, accuracy, prev_loss)
            acc_mse, acc_mae = compute_loss_weights('accuracy_based', epoch, accuracy, prev_loss)
            loss_mse, loss_mae = compute_loss_weights('loss_based', epoch, accuracy, prev_loss)
            
            # Weight strategies based on training progress
            if epoch < 5: weights = [0.6, 0.2, 0.2]
            elif epoch < 20: weights = [0.4, 0.4, 0.2]
            else: weights = [0.2, 0.5, 0.3]
            
            # Combine and normalize
            final_mse = weights[0] * epoch_mse + weights[1] * acc_mse + weights[2] * loss_mse
            final_mae = weights[0] * epoch_mae + weights[1] * acc_mae + weights[2] * loss_mae
            total = final_mse + final_mae
            return (final_mse / total, final_mae / total) if total > 0 else (0.5, 0.5)
        else: return 0.5, 0.5       
    except Exception as e:
        print(f"Warning: Error computing loss weights ({strategy}): {e}")
        return 0.5, 0.5


def create_adaptive_loss_fn(strategy: str = 'epoch_based'):
    """
    Create an adaptive loss function with state management.
    
    Args:
        strategy: Weighting strategy to use
        
    Returns:
        Adaptive loss function with update capabilities
    """
    # State variables (using mutable default to maintain state)
    state = {'epoch': 0, 'accuracy': 0.5, 'prev_loss': 1.0, 'history': [], 'error_count': 0}
    # Create loss functions
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    def adaptive_loss(y_true, y_pred):
        """Compute adaptive loss with current weights."""
        try:
            mse = mse_loss(y_true, y_pred)
            mae = mae_loss(y_true, y_pred)
            mse_weight, mae_weight = compute_loss_weights(strategy, state['epoch'], state['accuracy'], state['prev_loss'])
            combined_loss = mse_weight * mse + mae_weight * mae
            # Record history
            loss_info = {
                'epoch': state['epoch'], 'mse': float(mse.numpy()) if hasattr(mse, 'numpy') else float(mse),
                'mae': float(mae.numpy()) if hasattr(mae, 'numpy') else float(mae),
                'mse_weight': mse_weight, 'mae_weight': mae_weight,
                'combined_loss': float(combined_loss.numpy()) if hasattr(combined_loss, 'numpy') else float(combined_loss)
            }

            state['history'].append(loss_info)
            return combined_loss
            
        except Exception as e:
            print(f"Warning: Adaptive loss computation failed: {e}")
            state['error_count'] += 1
            return mse_loss(y_true, y_pred)
    
    def update_state(epoch: int, accuracy: Optional[float] = None):
        """Update loss function state."""
        state['epoch'] = epoch

        if accuracy is not None: state['accuracy'] = accuracy

        if state['history']: state['prev_loss'] = state['history'][-1]['combined_loss']
    
    def get_current_info() -> str:
        """Get current strategy information."""
        mse_weight, mae_weight = compute_loss_weights(strategy, state['epoch'], state['accuracy'], state['prev_loss'])
        return f"{strategy} (MSE: {mse_weight:.3f}, MAE: {mae_weight:.3f})"
    
    def get_history() -> Dict[str, Any]:
        """Get complete loss history."""
        return {'losses': state['history'], 'strategy': strategy, 'error_count': state['error_count']}
    
    # Attach methods to function
    adaptive_loss.update_state = update_state
    adaptive_loss.get_current_info = get_current_info
    adaptive_loss.get_history = get_history
    return adaptive_loss


# ============================================================================
# DATA PROCESSING UTILITIES  
# ============================================================================

def generate_particle_data(num_particles: int = 10, save_to_file: bool = True) -> pd.DataFrame:
    """
    Generate synthetic particle simulation data with physics-based calculations.
    
    Args:
        num_particles: Number of particles to simulate
        save_to_file: Whether to save data to CSV
        
    Returns:
        DataFrame with particle simulation data
    """
    try:
        np.random.seed(42)
        # Generate input parameters
        data = {
            'particle_id': range(1, num_particles + 1),
            'mass': np.random.uniform(0.1, 10.0, num_particles),
            'initial_velocity_x': np.random.uniform(-5.0, 5.0, num_particles),
            'initial_velocity_y': np.random.uniform(-5.0, 5.0, num_particles),
            'initial_position_x': np.random.uniform(-10.0, 10.0, num_particles),
            'initial_position_y': np.random.uniform(-10.0, 10.0, num_particles),
            'charge': np.random.choice([-1, 0, 1], num_particles),
            'magnetic_field_strength': np.random.uniform(0.1, 2.0, num_particles),
            'simulation_time': np.random.uniform(1.0, 10.0, num_particles)
        }
        # Calculate physics outputs
        outputs = []

        for i in range(num_particles):
            t, vx0, vy0, x0, y0, m, q, B = [data[k][i] for k in 
                ['simulation_time', 'initial_velocity_x', 'initial_velocity_y',
                 'initial_position_x', 'initial_position_y', 'mass', 'charge', 'magnetic_field_strength']]
            
            if q != 0:  # Charged particle - circular motion
                omega = q * B / m
                vx_final = vx0 * np.cos(omega * t) - vy0 * np.sin(omega * t)
                vy_final = vx0 * np.sin(omega * t) + vy0 * np.cos(omega * t)
                x_final = x0 + (vx0 * np.sin(omega * t) + vy0 * (np.cos(omega * t) - 1)) / omega
                y_final = y0 + (-vx0 * (np.cos(omega * t) - 1) + vy0 * np.sin(omega * t)) / omega
            else:  # Neutral particle - linear motion
                vx_final, vy_final = vx0, vy0
                x_final, y_final = x0 + vx0 * t, y0 + vy0 * t
            
            # Add noise and calculate derived quantities
            noise = 0.05
            vx_final += np.random.normal(0, noise * abs(vx_final))
            vy_final += np.random.normal(0, noise * abs(vy_final))
            x_final += np.random.normal(0, noise * abs(x_final))
            y_final += np.random.normal(0, noise * abs(y_final))
            outputs.append([vx_final, vy_final, x_final, y_final,
                           0.5 * m * (vx_final**2 + vy_final**2),  # kinetic energy
                           np.sqrt((x_final - x0)**2 + (y_final - y0)**2)])  # trajectory length
        
        # Add outputs to data
        output_names = ['final_velocity_x', 'final_velocity_y', 'final_position_x',
                       'final_position_y', 'kinetic_energy', 'trajectory_length']
        
        for i, name in enumerate(output_names):
            data[name] = [row[i] for row in outputs]
        
        df = pd.DataFrame(data)
        
        if save_to_file:
            try:
                df.to_csv('particle_data.csv', index=False)
                print(f"Particle data saved to particle_data.csv ({num_particles} particles)")
            except Exception as e:
                print(f"Warning: Could not save particle data: {e}")
        
        return df  
    except Exception as e:
        print(f"Error generating particle data: {e}")
        # Return minimal fallback data
        return pd.DataFrame({col: [1.0] for col in ['mass', 'initial_velocity_x', 'initial_velocity_y',
            'initial_position_x', 'initial_position_y', 'charge', 'magnetic_field_strength', 'simulation_time',
            'final_velocity_x', 'final_velocity_y', 'final_position_x', 'final_position_y', 
            'kinetic_energy', 'trajectory_length']}).assign(particle_id=[1])


def load_and_validate_data(csv_path: str = 'particle_data.csv') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load particle data from CSV with comprehensive validation.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, validation_results)
    """
    try:
        # Load data
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded particle data from {csv_path} ({len(df)} particles)")
        else:
            print(f"CSV file {csv_path} not found. Generating synthetic data...")
            df = generate_particle_data(save_to_file=True)
        
        # Validate data integrity
        validation = {'is_valid': True, 'issues': [], 'recommendations': []}
        # Check for missing values
        missing = df.isnull().sum()

        if missing.any():
            validation['issues'].append(f"Missing values: {missing.to_dict()}")
            validation['recommendations'].append("Fill or remove missing values")
        
        # Check physical constraints
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == 'mass' and (df[col] <= 0).any():
                validation['issues'].append(f"Non-positive mass values found")
                validation['recommendations'].append("Mass values should be positive")
            
            if np.isinf(df[col]).any():
                validation['issues'].append(f"Infinite values in {col}")
                validation['recommendations'].append(f"Replace infinite values in {col}")
            
            if (np.abs(df[col]) > 1e10).any():
                validation['issues'].append(f"Very large values in {col}")
                validation['recommendations'].append(f"Consider scaling {col}")
        
        validation['is_valid'] = len(validation['issues']) == 0
        return df, validation 
    except Exception as e:
        print(f"Error loading/validating data: {e}")
        return generate_particle_data(save_to_file=True), {'is_valid': False, 'issues': [str(e)], 'recommendations': []}


def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive data summary with statistics.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with data summary and statistics
    """
    try:
        # Define feature categories
        input_features = ['mass', 'initial_velocity_x', 'initial_velocity_y', 'initial_position_x',
                         'initial_position_y', 'charge', 'magnetic_field_strength', 'simulation_time']
        output_features = ['final_velocity_x', 'final_velocity_y', 'final_position_x',
                          'final_position_y', 'kinetic_energy', 'trajectory_length']
        # Filter available features
        available_inputs = [f for f in input_features if f in df.columns]
        available_outputs = [f for f in output_features if f in df.columns]   
        # Calculate statistics
        stats = {}

        for feature in available_inputs + available_outputs:
            if df[feature].dtype in ['int64', 'float64']:
                stats[feature] = {
                    'mean': float(df[feature].mean()), 'std': float(df[feature].std()),
                    'min': float(df[feature].min()), 'max': float(df[feature].max()),
                    'missing': int(df[feature].isnull().sum())
                }
        
        summary = {
            'num_particles': len(df), 'input_features': available_inputs,
            'output_features': available_outputs, 'data_statistics': stats
        }
        
        # Save summary
        try:
            with open('data_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print("Data summary saved to data_summary.json")
        except Exception as e:
            print(f"Warning: Could not save data summary: {e}")
        
        return summary
    except Exception as e:
        print(f"Error creating data summary: {e}")
        return {'num_particles': 0, 'input_features': [], 'output_features': [], 'data_statistics': {}, 'error': str(e)}


def preprocess_for_training(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2,
                           random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Complete data preprocessing pipeline for neural network training.
    
    Args:
        df: Input DataFrame
        test_size: Test set proportion
        val_size: Validation set proportion (from remaining data)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        # Define features
        input_features = ['mass', 'initial_velocity_x', 'initial_velocity_y', 'initial_position_x',
                         'initial_position_y', 'charge', 'magnetic_field_strength', 'simulation_time']
        output_features = ['final_velocity_x', 'final_velocity_y', 'final_position_x',
                          'final_position_y', 'kinetic_energy', 'trajectory_length']
        # Filter available features
        available_inputs = [f for f in input_features if f in df.columns]
        available_outputs = [f for f in output_features if f in df.columns]
        
        if not available_inputs or not available_outputs:
            raise ValueError(f"Insufficient features: inputs={len(available_inputs)}, outputs={len(available_outputs)}")
        
        # Extract and clean data
        X, y = df[available_inputs].values, df[available_outputs].values
        
        # Handle missing values
        if np.any(pd.isnull(X)) or np.any(pd.isnull(y)):
            print("Warning: Filling missing values with column means...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
            y = pd.DataFrame(y).fillna(pd.DataFrame(y).mean()).values
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                          test_size=val_size, random_state=random_state)
        # Scale features
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled, X_test_scaled = scaler_X.transform(X_val), scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled, y_test_scaled = scaler_y.transform(y_val), scaler_y.transform(y_test)
        
        # Save scalers
        try:
            import joblib
            joblib.dump(scaler_X, 'scaler_X.pkl')
            joblib.dump(scaler_y, 'scaler_y.pkl')
            print("Scalers saved to scaler_X.pkl and scaler_y.pkl")
        except (ImportError, Exception) as e:
            print(f"Warning: Could not save scalers: {e}")
        
        print(f"Data preprocessing completed: Train={X_train_scaled.shape[0]}, "
              f"Val={X_val_scaled.shape[0]}, Test={X_test_scaled.shape[0]}, "
              f"Features={X_train_scaled.shape[1]}→{y_train_scaled.shape[1]}")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled  
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        # Return dummy data as fallback
        dummy_X, dummy_y = np.random.randn(10, 8), np.random.randn(10, 6)
        return dummy_X[:6], dummy_X[6:8], dummy_X[8:], dummy_y[:6], dummy_y[6:8], dummy_y[8:]


# ============================================================================
# INTEGRATED DATA PIPELINE
# ============================================================================

def complete_data_pipeline(csv_path: str = 'particle_data.csv', 
                          num_particles: int = 1000) -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]]:
    """
    Execute complete data loading, validation, and preprocessing pipeline.
    
    Args:
        csv_path: Path to CSV file (generates if missing)
        num_particles: Number of particles to generate if needed
        
    Returns:
        Tuple of ((X_train, X_val, X_test, y_train, y_val, y_test), pipeline_info)
    """
    try:
        print("=== ML Utils Data Pipeline ===")
        # Load and validate data
        df, validation = load_and_validate_data(csv_path)
        
        # Generate more data if needed
        if len(df) < num_particles:
            print(f"Generating additional data to reach {num_particles} particles...")
            df = generate_particle_data(num_particles, save_to_file=True)
        
        # Create summary
        summary = create_data_summary(df)
        # Preprocess for training
        data_splits = preprocess_for_training(df)
        # Compile pipeline information
        pipeline_info = {
            'data_summary': summary, 'validation': validation,
            'preprocessing_complete': True, 'total_samples': len(df),
            'train_samples': data_splits[0].shape[0], 'val_samples': data_splits[1].shape[0],
            'test_samples': data_splits[2].shape[0], 'input_features': data_splits[0].shape[1],
            'output_features': data_splits[3].shape[1]
        }
        print("Data pipeline completed successfully!")
        return data_splits, pipeline_info 
    except Exception as e:
        print(f"Error in data pipeline: {e}")
        # Return fallback data
        dummy_X, dummy_y = np.random.randn(100, 8), np.random.randn(100, 6)
        fallback_splits = (dummy_X[:60], dummy_X[60:80], dummy_X[80:], dummy_y[:60], dummy_y[60:80], dummy_y[80:])
        fallback_info = {'error': str(e), 'fallback_data': True, 'total_samples': 100}
        return fallback_splits, fallback_info
