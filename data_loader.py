"""
CSV Data Loading and Preprocessing for Particle Physics Simulations

This module handles loading and preprocessing of particle simulation data from CSV files,
creating synthetic data when needed, and preparing data for neural network training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import json


def generate_particle_data(num_particles: int = 10, save_to_file: bool = True) -> pd.DataFrame:
    """
    Generate synthetic particle simulation data.
    
    Args:
        num_particles: Number of particles to simulate
        save_to_file: Whether to save the data to CSV file
        
    Returns:
        DataFrame containing particle simulation data
    """
    try:
        np.random.seed(42)  # For reproducible results
        
        # Generate input parameters for particles
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
        
        # Calculate physics-based outputs (simplified particle motion)
        final_velocity_x = []
        final_velocity_y = []
        final_position_x = []
        final_position_y = []
        kinetic_energy = []
        trajectory_length = []
        
        for i in range(num_particles):
            # Simple physics simulation (simplified for demonstration)
            t = data['simulation_time'][i]
            vx0 = data['initial_velocity_x'][i]
            vy0 = data['initial_velocity_y'][i]
            x0 = data['initial_position_x'][i]
            y0 = data['initial_position_y'][i]
            m = data['mass'][i]
            q = data['charge'][i]
            B = data['magnetic_field_strength'][i]
            
            # Apply simplified Lorentz force (circular motion for charged particles)
            if q != 0:
                omega = q * B / m  # Cyclotron frequency
                
                # Circular motion in magnetic field
                vx_final = vx0 * np.cos(omega * t) - vy0 * np.sin(omega * t)
                vy_final = vx0 * np.sin(omega * t) + vy0 * np.cos(omega * t)
                
                # Position with circular motion
                x_final = x0 + (vx0 * np.sin(omega * t) + vy0 * (np.cos(omega * t) - 1)) / omega
                y_final = y0 + (-vx0 * (np.cos(omega * t) - 1) + vy0 * np.sin(omega * t)) / omega
            else:
                # No charge - linear motion
                vx_final = vx0
                vy_final = vy0
                x_final = x0 + vx0 * t
                y_final = y0 + vy0 * t
            
            # Add some noise to make it more realistic
            noise_factor = 0.05
            vx_final += np.random.normal(0, noise_factor * abs(vx_final))
            vy_final += np.random.normal(0, noise_factor * abs(vy_final))
            x_final += np.random.normal(0, noise_factor * abs(x_final))
            y_final += np.random.normal(0, noise_factor * abs(y_final))
            
            # Calculate derived quantities
            ke = 0.5 * m * (vx_final**2 + vy_final**2)
            trajectory_len = np.sqrt((x_final - x0)**2 + (y_final - y0)**2)
            
            final_velocity_x.append(vx_final)
            final_velocity_y.append(vy_final)
            final_position_x.append(x_final)
            final_position_y.append(y_final)
            kinetic_energy.append(ke)
            trajectory_length.append(trajectory_len)
        
        # Add outputs to data dictionary
        data.update({
            'final_velocity_x': final_velocity_x,
            'final_velocity_y': final_velocity_y,
            'final_position_x': final_position_x,
            'final_position_y': final_position_y,
            'kinetic_energy': kinetic_energy,
            'trajectory_length': trajectory_length
        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to file if requested
        if save_to_file:
            try:
                df.to_csv('particle_data.csv', index=False)
                print(f"Particle data saved to particle_data.csv ({num_particles} particles)")
            except Exception as e:
                print(f"Warning: Could not save particle data to CSV: {e}")
        
        return df
        
    except Exception as e:
        print(f"Error generating particle data: {e}")
        # Return minimal data if generation fails
        return pd.DataFrame({
            'particle_id': [1],
            'mass': [1.0],
            'initial_velocity_x': [1.0],
            'initial_velocity_y': [1.0],
            'initial_position_x': [0.0],
            'initial_position_y': [0.0],
            'charge': [0],
            'magnetic_field_strength': [1.0],
            'simulation_time': [1.0],
            'final_velocity_x': [1.0],
            'final_velocity_y': [1.0],
            'final_position_x': [1.0],
            'final_position_y': [1.0],
            'kinetic_energy': [1.0],
            'trajectory_length': [1.0]
        })


def load_particle_data(csv_path: str = 'particle_data.csv') -> pd.DataFrame:
    """
    Load particle simulation data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing particle data
    """
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded particle data from {csv_path} ({len(df)} particles)")
            return df
        else:
            print(f"CSV file {csv_path} not found. Generating synthetic data...")
            return generate_particle_data(save_to_file=True)
            
    except Exception as e:
        print(f"Error loading particle data from {csv_path}: {e}")
        print("Generating synthetic data as fallback...")
        return generate_particle_data(save_to_file=True)


def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary of the particle data.
    
    Args:
        df: DataFrame containing particle data
        
    Returns:
        Dictionary containing data summary statistics
    """
    try:
        summary = {
            'num_particles': len(df),
            'input_features': [],
            'output_features': [],
            'data_statistics': {}
        }
        
        # Define input and output features
        input_features = [
            'mass', 'initial_velocity_x', 'initial_velocity_y', 
            'initial_position_x', 'initial_position_y', 'charge',
            'magnetic_field_strength', 'simulation_time'
        ]
        
        output_features = [
            'final_velocity_x', 'final_velocity_y', 'final_position_x',
            'final_position_y', 'kinetic_energy', 'trajectory_length'
        ]
        
        # Filter features that exist in the dataframe
        available_input_features = [f for f in input_features if f in df.columns]
        available_output_features = [f for f in output_features if f in df.columns]
        
        summary['input_features'] = available_input_features
        summary['output_features'] = available_output_features
        
        # Calculate statistics for each feature
        for feature in available_input_features + available_output_features:
            if df[feature].dtype in ['int64', 'float64']:
                feature_stats = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'missing_values': int(df[feature].isnull().sum())
                }
                summary['data_statistics'][feature] = feature_stats
        
        # Save summary to JSON file
        try:
            with open('data_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print("Data summary saved to data_summary.json")
        except Exception as e:
            print(f"Warning: Could not save data summary: {e}")
        
        return summary
        
    except Exception as e:
        print(f"Error creating data summary: {e}")
        return {
            'num_particles': 0,
            'input_features': [],
            'output_features': [],
            'data_statistics': {},
            'error': str(e)
        }


def preprocess_data(df: pd.DataFrame, 
                   test_size: float = 0.2, 
                   val_size: float = 0.2,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess particle data for neural network training.
    
    Args:
        df: DataFrame containing particle data
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation (from remaining after test split)
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        # Define input and output features
        input_features = [
            'mass', 'initial_velocity_x', 'initial_velocity_y', 
            'initial_position_x', 'initial_position_y', 'charge',
            'magnetic_field_strength', 'simulation_time'
        ]
        
        output_features = [
            'final_velocity_x', 'final_velocity_y', 'final_position_x',
            'final_position_y', 'kinetic_energy', 'trajectory_length'
        ]
        
        # Filter features that exist in the dataframe
        available_input_features = [f for f in input_features if f in df.columns]
        available_output_features = [f for f in output_features if f in df.columns]
        
        if not available_input_features or not available_output_features:
            raise ValueError("Insufficient input or output features in the data")
        
        # Extract input and output data
        X = df[available_input_features].values
        y = df[available_output_features].values
        
        # Handle missing values
        if np.any(pd.isnull(X)) or np.any(pd.isnull(y)):
            print("Warning: Missing values detected. Filling with column means...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
            y = pd.DataFrame(y).fillna(pd.DataFrame(y).mean()).values
        
        # Split into train and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train_val into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        # Standardize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)
        
        # Save scalers for later use
        try:
            import joblib
            joblib.dump(scaler_X, 'scaler_X.pkl')
            joblib.dump(scaler_y, 'scaler_y.pkl')
            print("Scalers saved to scaler_X.pkl and scaler_y.pkl")
        except ImportError:
            print("Warning: joblib not available. Scalers not saved.")
        except Exception as e:
            print(f"Warning: Could not save scalers: {e}")
        
        print(f"Data preprocessing completed:")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Validation samples: {X_val_scaled.shape[0]}")
        print(f"  Test samples: {X_test_scaled.shape[0]}")
        print(f"  Input features: {X_train_scaled.shape[1]}")
        print(f"  Output features: {y_train_scaled.shape[1]}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
        
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        # Return minimal arrays if preprocessing fails
        dummy_X = np.random.randn(10, 8)
        dummy_y = np.random.randn(10, 6)
        
        return (dummy_X[:6], dummy_X[6:8], dummy_X[8:], 
                dummy_y[:6], dummy_y[6:8], dummy_y[8:])


def load_and_prepare_data(csv_path: str = 'particle_data.csv') -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                       np.ndarray, np.ndarray, np.ndarray,
                                                                       Dict[str, Any]]:
    """
    Complete data loading and preparation pipeline.
    
    Args:
        csv_path: Path to the CSV file containing particle data
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, data_summary)
    """
    try:
        print("=== Loading and Preparing Particle Data ===")
        
        # Load data
        df = load_particle_data(csv_path)
        
        # Create data summary
        data_summary = create_data_summary(df)
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
        
        print("Data loading and preparation completed successfully!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, data_summary
        
    except Exception as e:
        print(f"Error in data loading and preparation: {e}")
        # Return minimal data if everything fails
        dummy_X = np.random.randn(10, 8)
        dummy_y = np.random.randn(10, 6)
        
        dummy_summary = {
            'num_particles': 10,
            'input_features': ['feature_' + str(i) for i in range(8)],
            'output_features': ['output_' + str(i) for i in range(6)],
            'data_statistics': {},
            'error': 'Fallback data used due to loading error'
        }
        
        return (dummy_X[:6], dummy_X[6:8], dummy_X[8:], 
                dummy_y[:6], dummy_y[6:8], dummy_y[8:], dummy_summary)


def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the integrity and quality of the particle data.
    
    Args:
        df: DataFrame containing particle data
        
    Returns:
        Dictionary containing validation results
    """
    try:
        validation_results = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            validation_results['issues'].append(f"Missing values found: {missing_values.to_dict()}")
            validation_results['recommendations'].append("Consider filling missing values or removing incomplete records")
        
        # Check for duplicate particles
        if 'particle_id' in df.columns:
            duplicates = df['particle_id'].duplicated().sum()
            if duplicates > 0:
                validation_results['issues'].append(f"Duplicate particle IDs found: {duplicates}")
                validation_results['recommendations'].append("Remove or rename duplicate particle IDs")
        
        # Check for physical constraints
        if 'mass' in df.columns:
            negative_mass = (df['mass'] <= 0).sum()
            if negative_mass > 0:
                validation_results['issues'].append(f"Non-positive mass values found: {negative_mass}")
                validation_results['recommendations'].append("Mass values should be positive")
        
        # Check for infinite or very large values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            infinite_values = np.isinf(df[col]).sum()
            if infinite_values > 0:
                validation_results['issues'].append(f"Infinite values in {col}: {infinite_values}")
                validation_results['recommendations'].append(f"Replace infinite values in {col}")
            
            very_large_values = (np.abs(df[col]) > 1e10).sum()
            if very_large_values > 0:
                validation_results['issues'].append(f"Very large values in {col}: {very_large_values}")
                validation_results['recommendations'].append(f"Consider scaling values in {col}")
        
        # Set overall validity
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'issues': [f"Validation error: {e}"],
            'recommendations': ["Check data format and content"]
        }
