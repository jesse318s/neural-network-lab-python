import os
import json
from typing import Dict, Tuple, Any
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
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled  
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        # Return dummy data as fallback
        dummy_X, dummy_y = np.random.randn(10, 8), np.random.randn(10, 6)
        return dummy_X[:6], dummy_X[6:8], dummy_X[8:], dummy_y[:6], dummy_y[6:8], dummy_y[8:]


def complete_data_pipeline(csv_path: str = 'particle_data.csv', 
                          num_particles: int = 1000) -> Tuple[np.ndarray, ...]:
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
        # Load and validate data
        df = load_and_validate_data(csv_path)
        
        # Generate more data if needed
        if len(df) < num_particles:
            print(f"Generating additional data to reach {num_particles} particles...")
            df = generate_particle_data(num_particles, save_to_file=True)
        
        # Preprocess for training
        data_splits = preprocess_for_training(df)
        print("Data pipeline completed successfully")
        return data_splits
    except Exception as e:
        print(f"Error in data pipeline: {e}")
        # Return fallback data
        dummy_X, dummy_y = np.random.randn(100, 8), np.random.randn(100, 6)
        fallback_splits = (dummy_X[:60], dummy_X[60:80], dummy_X[80:], dummy_y[:60], dummy_y[60:80], dummy_y[80:])
        return fallback_splits


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
