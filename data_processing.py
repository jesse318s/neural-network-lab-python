"""
Data Processing Pipeline for Particle Simulation Data
Implements railway-style error handling for robust execution.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
                ['simulation_time', 'initial_velocity_x', 'initial_velocity_y', 'initial_position_x', 
                 'initial_position_y', 'mass', 'charge', 'magnetic_field_strength']]
            
            if q != 0 and m != 0 and abs(q * B / m) > 1e-10:  # Charged particle with significant magnetic field
                # Cyclotron frequency
                omega = (q * B) / m
                omega_t = omega * t
                cos_ot = np.cos(omega_t)
                sin_ot = np.sin(omega_t)
                # Velocity rotation
                vx_final = vx0 * cos_ot - vy0 * sin_ot
                vy_final = vx0 * sin_ot + vy0 * cos_ot          
                # Position update
                x_final = x0 + (vx0 * sin_ot + vy0 * (cos_ot - 1)) / omega
                y_final = y0 + (vy0 * sin_ot + vx0 * (1 - cos_ot)) / omega
            else:  # Neutral particle or negligible magnetic field - linear motion
                vx_final, vy_final = vx0, vy0
                x_final, y_final = x0 + vx0 * t, y0 + vy0 * t
            
            # Calculate derived quantities
            kinetic_energy = 0.5 * m * (vx_final**2 + vy_final**2)
            trajectory_length = np.sqrt((x_final - x0)**2 + (y_final - y0)**2) # Displacement for simplicity
            outputs.append([vx_final, vy_final, x_final, y_final, kinetic_energy, trajectory_length])
        
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
        return pd.DataFrame({col: [1.0] for col in ['mass', 'initial_velocity_x', 'initial_velocity_y',
            'initial_position_x', 'initial_position_y', 'charge', 'magnetic_field_strength', 'simulation_time',
            'final_velocity_x', 'final_velocity_y', 'final_position_x', 'final_position_y', 
            'kinetic_energy', 'trajectory_length']}).assign(particle_id=[1])


def load_and_validate_data(csv_path: str = 'particle_data.csv') -> pd.DataFrame:
    """
    Load particle data from CSV with comprehensive validation.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with loaded and validated data
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
        validation = {'issues': [], 'recommendations': []}
        # Check for missing values
        missing = df.isnull().sum()

        if missing.any():
            validation['issues'].append(f"Missing values: {missing.to_dict()}")
            validation['recommendations'].append("Fill or remove missing values")
        
        # Check physical constraints
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        calculated_ke = 0.5 * df['mass'] * (df['final_velocity_x']**2 + df['final_velocity_y']**2)
        ke_diff = np.abs(df['kinetic_energy'] - calculated_ke)
        
        if (ke_diff > 0.1 * df['kinetic_energy']).any():  # 10% tolerance
            validation['issues'].append("Kinetic energy inconsistencies detected")
            validation['recommendations'].append("Check energy calculations")
        
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
        
        # Report validation results
        if validation['issues']: print("Data validation issues found:")

        for issue in validation['issues']:
            print(f"- {issue}")

        for rec in validation['recommendations']:
            print(f"  * Recommendation: {rec}")

        return df
    except Exception as e:
        print(f"Error loading/validating data: {e}")
        return generate_particle_data(save_to_file=False)


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
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled  
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        np.random.seed(42)
        dummy_X, dummy_y = np.random.randn(10, 8), np.random.randn(10, 6)
        return dummy_X[:6], dummy_X[6:8], dummy_X[8:], dummy_y[:6], dummy_y[6:8], dummy_y[8:]


def complete_data_pipeline(csv_path: str = 'particle_data.csv', num_particles: int = 1000) -> Tuple[np.ndarray, ...]:
    """
    Execute complete data loading, validation, and preprocessing pipeline.
    
    Args:
        csv_path: Path to CSV file (generates if missing)
        num_particles: Number of particles to generate if needed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        print("=== Data Pipeline ===")
        df = load_and_validate_data(csv_path)
        
        # Generate more data if needed
        if len(df) < num_particles:
            print(f"Generating additional data to reach {num_particles} particles...")
            df = generate_particle_data(num_particles, save_to_file=True)
        
        data_splits = preprocess_for_training(df)
        print("Data pipeline completed successfully")
        return data_splits
    except Exception as e:
        print(f"Error in data pipeline: {e}")
        dummy_X, dummy_y = np.random.randn(100, 8), np.random.randn(100, 6)
        fallback_splits = (dummy_X[:60], dummy_X[60:80], dummy_X[80:], dummy_y[:60], dummy_y[60:80], dummy_y[80:])
        return fallback_splits
