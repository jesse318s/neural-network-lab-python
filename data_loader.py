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
from typing import Tuple, Dict, Any, Optional, List
import json


def _get_complexity_parameters(complexity_level: str, num_particles: int) -> Dict[str, Any]:
    """
    Get simulation parameters based on complexity level and particle count.
    
    Args:
        complexity_level: Level of physics simulation complexity
        num_particles: Number of particles (affects some parameter ranges)
        
    Returns:
        Dictionary of simulation parameters
    """
    base_params = {
        'simple': {
            'mass_range': (0.5, 5.0),
            'velocity_range': 3.0,
            'position_range': 5.0,
            'charge_options': [-1, 0, 1],
            'charge_probabilities': [0.3, 0.4, 0.3],
            'field_range': (0.5, 1.5),
            'time_range': (2.0, 8.0),
            'noise_factor': 0.02
        },
        'medium': {
            'mass_range': (0.1, 10.0),
            'velocity_range': 5.0,
            'position_range': 10.0,
            'charge_options': [-1, 0, 1],
            'charge_probabilities': [0.25, 0.5, 0.25],
            'field_range': (0.1, 2.0),
            'time_range': (1.0, 10.0),
            'noise_factor': 0.05
        },
        'complex': {
            'mass_range': (0.01, 20.0),
            'velocity_range': 8.0,
            'position_range': 15.0,
            'charge_options': [-2, -1, 0, 1, 2],
            'charge_probabilities': [0.1, 0.25, 0.3, 0.25, 0.1],
            'field_range': (0.01, 3.0),
            'time_range': (0.5, 15.0),
            'noise_factor': 0.08
        },
        'extreme': {
            'mass_range': (0.001, 50.0),
            'velocity_range': 12.0,
            'position_range': 25.0,
            'charge_options': [-3, -2, -1, 0, 1, 2, 3],
            'charge_probabilities': [0.05, 0.15, 0.2, 0.2, 0.2, 0.15, 0.05],
            'field_range': (0.001, 5.0),
            'time_range': (0.1, 25.0),
            'noise_factor': 0.12
        }
    }
    
    params = base_params.get(complexity_level, base_params['medium'])
    
    # Scale complexity with particle count
    scale_factor = min(2.0, 1.0 + np.log10(max(1, num_particles / 100)))
    
    params['velocity_range'] *= scale_factor
    params['position_range'] *= scale_factor
    params['field_range'] = (params['field_range'][0], params['field_range'][1] * scale_factor)
    params['noise_factor'] *= min(1.5, scale_factor)
    
    return params


def _add_complex_features(num_particles: int, complexity_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Add additional complex features for advanced physics simulation.
    
    Args:
        num_particles: Number of particles
        complexity_params: Complexity parameters dictionary
        
    Returns:
        Dictionary of additional feature arrays
    """
    additional_features = {}
    
    # Electric field components
    additional_features['electric_field_x'] = np.random.uniform(
        -complexity_params['field_range'][1]/2, 
        complexity_params['field_range'][1]/2, 
        num_particles
    )
    additional_features['electric_field_y'] = np.random.uniform(
        -complexity_params['field_range'][1]/2, 
        complexity_params['field_range'][1]/2, 
        num_particles
    )
    
    # Particle interactions (simplified)
    additional_features['interaction_strength'] = np.random.exponential(0.1, num_particles)
    
    # Variable magnetic field direction
    additional_features['magnetic_field_angle'] = np.random.uniform(0, 2*np.pi, num_particles)
    
    # Material properties
    additional_features['medium_density'] = np.random.uniform(0.5, 2.0, num_particles)
    additional_features['friction_coefficient'] = np.random.uniform(0.01, 0.1, num_particles)
    
    # Temperature effects
    additional_features['temperature'] = np.random.uniform(200, 400, num_particles)
    
    return additional_features


def _calculate_final_velocities(data: Dict[str, Any], complexity_params: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Calculate final velocities using enhanced physics simulation."""
    num_particles = len(data['particle_id'])
    final_velocity_x = []
    final_velocity_y = []
    
    for i in range(num_particles):
        # Basic parameters
        t = data['simulation_time'][i]
        vx0 = data['initial_velocity_x'][i]
        vy0 = data['initial_velocity_y'][i]
        m = data['mass'][i]
        q = data['charge'][i]
        B = data['magnetic_field_strength'][i]
        
        # Enhanced physics with additional forces
        if q != 0:
            omega = q * B / m  # Cyclotron frequency
            
            # Circular motion in magnetic field
            vx_final = vx0 * np.cos(omega * t) - vy0 * np.sin(omega * t)
            vy_final = vx0 * np.sin(omega * t) + vy0 * np.cos(omega * t)
            
            # Add electric field effects if available
            if 'electric_field_x' in data and 'electric_field_y' in data:
                Ex = data['electric_field_x'][i]
                Ey = data['electric_field_y'][i]
                
                # Electric field acceleration
                ax = q * Ex / m
                ay = q * Ey / m
                
                vx_final += ax * t
                vy_final += ay * t
            
            # Add friction effects if available
            if 'friction_coefficient' in data:
                friction = data['friction_coefficient'][i]
                drag_factor = np.exp(-friction * t)
                vx_final *= drag_factor
                vy_final *= drag_factor
        else:
            # Neutral particle - linear motion with possible damping
            vx_final = vx0
            vy_final = vy0
            
            if 'friction_coefficient' in data:
                friction = data['friction_coefficient'][i]
                drag_factor = np.exp(-friction * t)
                vx_final *= drag_factor
                vy_final *= drag_factor
        
        # Add noise based on complexity
        noise_factor = complexity_params['noise_factor']
        vx_final += np.random.normal(0, noise_factor * abs(vx_final + 1e-6))
        vy_final += np.random.normal(0, noise_factor * abs(vy_final + 1e-6))
        
        final_velocity_x.append(vx_final)
        final_velocity_y.append(vy_final)
    
    return final_velocity_x, final_velocity_y


def _calculate_final_positions(data: Dict[str, Any], complexity_params: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Calculate final positions using enhanced physics simulation."""
    num_particles = len(data['particle_id'])
    final_position_x = []
    final_position_y = []
    
    for i in range(num_particles):
        # Basic parameters
        t = data['simulation_time'][i]
        x0 = data['initial_position_x'][i]
        y0 = data['initial_position_y'][i]
        vx0 = data['initial_velocity_x'][i]
        vy0 = data['initial_velocity_y'][i]
        m = data['mass'][i]
        q = data['charge'][i]
        B = data['magnetic_field_strength'][i]
        
        if q != 0:
            omega = q * B / m
            
            if omega != 0:
                # Circular motion trajectory
                x_final = x0 + (vx0 * np.sin(omega * t) + vy0 * (np.cos(omega * t) - 1)) / omega
                y_final = y0 + (-vx0 * (np.cos(omega * t) - 1) + vy0 * np.sin(omega * t)) / omega
            else:
                # Linear motion if omega is zero
                x_final = x0 + vx0 * t
                y_final = y0 + vy0 * t
        else:
            # Neutral particle - linear motion
            x_final = x0 + vx0 * t
            y_final = y0 + vy0 * t
        
        # Add noise based on complexity
        noise_factor = complexity_params['noise_factor']
        x_final += np.random.normal(0, noise_factor * abs(x_final + 1e-6))
        y_final += np.random.normal(0, noise_factor * abs(y_final + 1e-6))
        
        final_position_x.append(x_final)
        final_position_y.append(y_final)
    
    return final_position_x, final_position_y


def _calculate_kinetic_energies(data: Dict[str, Any], 
                              final_velocity_x: List[float], 
                              final_velocity_y: List[float]) -> List[float]:
    """Calculate kinetic energies."""
    kinetic_energy = []
    
    for i in range(len(data['particle_id'])):
        m = data['mass'][i]
        vx = final_velocity_x[i]
        vy = final_velocity_y[i]
        
        ke = 0.5 * m * (vx**2 + vy**2)
        kinetic_energy.append(ke)
    
    return kinetic_energy


def _calculate_trajectory_lengths(data: Dict[str, Any],
                                final_position_x: List[float],
                                final_position_y: List[float]) -> List[float]:
    """Calculate trajectory lengths."""
    trajectory_length = []
    
    for i in range(len(data['particle_id'])):
        x0 = data['initial_position_x'][i]
        y0 = data['initial_position_y'][i]
        x_final = final_position_x[i]
        y_final = final_position_y[i]
        
        length = np.sqrt((x_final - x0)**2 + (y_final - y0)**2)
        trajectory_length.append(length)
    
    return trajectory_length


def _calculate_additional_outputs(data: Dict[str, Any], 
                                complexity_params: Dict[str, Any],
                                final_velocity_x: List[float], 
                                final_velocity_y: List[float],
                                final_position_x: List[float], 
                                final_position_y: List[float]) -> Dict[str, List[float]]:
    """Calculate additional output features for complex simulations."""
    additional_outputs = {}
    num_particles = len(data['particle_id'])
    
    # Momentum components
    momentum_x = []
    momentum_y = []
    
    # Angular momentum (simplified)
    angular_momentum = []
    
    # Energy dissipation
    energy_dissipated = []
    
    for i in range(num_particles):
        m = data['mass'][i]
        vx = final_velocity_x[i]
        vy = final_velocity_y[i]
        x = final_position_x[i]
        y = final_position_y[i]
        
        # Momentum
        px = m * vx
        py = m * vy
        momentum_x.append(px)
        momentum_y.append(py)
        
        # Angular momentum (L = r × p)
        L = x * py - y * px
        angular_momentum.append(L)
        
        # Energy dissipation (simplified)
        if 'friction_coefficient' in data:
            friction = data['friction_coefficient'][i]
            initial_ke = 0.5 * m * (data['initial_velocity_x'][i]**2 + data['initial_velocity_y'][i]**2)
            final_ke = 0.5 * m * (vx**2 + vy**2)
            dissipated = max(0, initial_ke - final_ke)
            energy_dissipated.append(dissipated)
    
    additional_outputs.update({
        'final_momentum_x': momentum_x,
        'final_momentum_y': momentum_y,
        'angular_momentum': angular_momentum
    })
    
    if energy_dissipated:
        additional_outputs['energy_dissipated'] = energy_dissipated
    
    return additional_outputs


def generate_particle_data(num_particles: int = 10, save_to_file: bool = True, 
                          complexity_level: str = 'medium', random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic particle simulation data with variable complexity and particle counts.
    
    Args:
        num_particles: Number of particles to simulate
        save_to_file: Whether to save the data to CSV file
        complexity_level: Simulation complexity ('simple', 'medium', 'complex', 'extreme')
        random_seed: Random seed for reproducible results
        
    Returns:
        DataFrame containing particle simulation data
    """
    try:
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(42)  # Default for reproducibility
        
        # Adjust parameter ranges based on particle count and complexity
        complexity_params = _get_complexity_parameters(complexity_level, num_particles)
        
        # Generate input parameters for particles
        data = {
            'particle_id': range(1, num_particles + 1),
            'mass': np.random.uniform(
                complexity_params['mass_range'][0], 
                complexity_params['mass_range'][1], 
                num_particles
            ),
            'initial_velocity_x': np.random.uniform(
                -complexity_params['velocity_range'], 
                complexity_params['velocity_range'], 
                num_particles
            ),
            'initial_velocity_y': np.random.uniform(
                -complexity_params['velocity_range'], 
                complexity_params['velocity_range'], 
                num_particles
            ),
            'initial_position_x': np.random.uniform(
                -complexity_params['position_range'], 
                complexity_params['position_range'], 
                num_particles
            ),
            'initial_position_y': np.random.uniform(
                -complexity_params['position_range'], 
                complexity_params['position_range'], 
                num_particles
            ),
            'charge': np.random.choice(
                complexity_params['charge_options'], 
                num_particles, 
                p=complexity_params['charge_probabilities']
            ),
            'magnetic_field_strength': np.random.uniform(
                complexity_params['field_range'][0], 
                complexity_params['field_range'][1], 
                num_particles
            ),
            'simulation_time': np.random.uniform(
                complexity_params['time_range'][0], 
                complexity_params['time_range'][1], 
                num_particles
            )
        }
        
        # Add additional complexity features
        if complexity_level in ['complex', 'extreme']:
            data.update(_add_complex_features(num_particles, complexity_params))
        
        # Calculate physics-based outputs with enhanced complexity
        final_velocity_x, final_velocity_y = _calculate_final_velocities(data, complexity_params)
        final_position_x, final_position_y = _calculate_final_positions(data, complexity_params)
        kinetic_energy = _calculate_kinetic_energies(data, final_velocity_x, final_velocity_y)
        trajectory_length = _calculate_trajectory_lengths(data, final_position_x, final_position_y)
        
        # Add additional output features for complex simulations
        additional_outputs = {}
        if complexity_level in ['complex', 'extreme']:
            additional_outputs = _calculate_additional_outputs(data, complexity_params, 
                                                             final_velocity_x, final_velocity_y,
                                                             final_position_x, final_position_y)
        
        # Add outputs to data dictionary
        data.update({
            'final_velocity_x': final_velocity_x,
            'final_velocity_y': final_velocity_y,
            'final_position_x': final_position_x,
            'final_position_y': final_position_y,
            'kinetic_energy': kinetic_energy,
            'trajectory_length': trajectory_length
        })
        
        # Add additional outputs for complex simulations
        if additional_outputs:
            data.update(additional_outputs)
        
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
        # Define base input and output features
        base_input_features = [
            'mass', 'initial_velocity_x', 'initial_velocity_y', 
            'initial_position_x', 'initial_position_y', 'charge',
            'magnetic_field_strength', 'simulation_time'
        ]
        
        base_output_features = [
            'final_velocity_x', 'final_velocity_y', 'final_position_x',
            'final_position_y', 'kinetic_energy', 'trajectory_length'
        ]
        
        # Add complex features if available
        complex_input_features = [
            'electric_field_x', 'electric_field_y', 'interaction_strength',
            'magnetic_field_angle', 'medium_density', 'friction_coefficient', 'temperature'
        ]
        
        complex_output_features = [
            'final_momentum_x', 'final_momentum_y', 'angular_momentum', 'energy_dissipated'
        ]
        
        # Filter features that exist in the dataframe
        input_features = base_input_features + [f for f in complex_input_features if f in df.columns]
        output_features = base_output_features + [f for f in complex_output_features if f in df.columns]
        
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


def generate_enhanced_particle_data(num_particles: int, 
                                  complexity_level: str = 'medium',
                                  random_seed: Optional[int] = None,
                                  save_to_file: bool = False) -> pd.DataFrame:
    """
    Enhanced particle data generation with automatic complexity scaling based on particle count.
    
    Args:
        num_particles: Number of particles to simulate
        complexity_level: Base complexity level
        random_seed: Random seed for reproducibility
        save_to_file: Whether to save to CSV file
        
    Returns:
        DataFrame with enhanced particle simulation data
    """
    # Automatically adjust complexity based on particle count
    if num_particles <= 100:
        effective_complexity = 'simple' if complexity_level == 'simple' else 'medium'
    elif num_particles <= 1000:
        effective_complexity = complexity_level
    else:
        # For very large datasets, use complex features
        complexity_map = {'simple': 'medium', 'medium': 'complex', 'complex': 'extreme', 'extreme': 'extreme'}
        effective_complexity = complexity_map.get(complexity_level, 'complex')
    
    print(f"Generating {num_particles} particles with {effective_complexity} complexity")
    
    return generate_particle_data(
        num_particles=num_particles,
        save_to_file=save_to_file,
        complexity_level=effective_complexity,
        random_seed=random_seed
    )


def get_data_complexity_info(num_particles: int) -> Dict[str, Any]:
    """
    Get information about data complexity for different particle counts.
    
    Args:
        num_particles: Number of particles
        
    Returns:
        Dictionary with complexity information
    """
    if num_particles <= 100:
        complexity_info = {
            'suggested_complexity': 'simple',
            'expected_features': 8,
            'expected_outputs': 6,
            'estimated_generation_time': '< 1 second',
            'memory_usage': 'low'
        }
    elif num_particles <= 1000:
        complexity_info = {
            'suggested_complexity': 'medium',
            'expected_features': 8,
            'expected_outputs': 6,
            'estimated_generation_time': '1-5 seconds',
            'memory_usage': 'medium'
        }
    elif num_particles <= 5000:
        complexity_info = {
            'suggested_complexity': 'complex',
            'expected_features': 15,
            'expected_outputs': 10,
            'estimated_generation_time': '5-15 seconds',
            'memory_usage': 'medium-high'
        }
    else:
        complexity_info = {
            'suggested_complexity': 'extreme',
            'expected_features': 15,
            'expected_outputs': 10,
            'estimated_generation_time': '15+ seconds',
            'memory_usage': 'high'
        }
    
    return complexity_info
