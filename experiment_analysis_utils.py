"""
Experiment Analysis Utilities

This module provides reusable functions for loading training artifacts,
analyzing results, and computing metrics in the neural-network-lab-python project.
Supports the experiment_analysis_framework.ipynb notebook with clean interfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from advanced_neural_network import AdvancedNeuralNetwork
from data_processing import complete_data_pipeline, load_and_validate_data


# Project constants
PROJECT_NAME = "neural-network-lab-python"

INPUT_FEATURES = [
    'mass', 'initial_velocity_x', 'initial_velocity_y', 'initial_position_x',
    'initial_position_y', 'charge', 'magnetic_field_strength', 'simulation_time',
    'initial_speed', 'initial_position_mag', 'initial_momentum_x', 'initial_momentum_y',
    'initial_momentum_mag', 'momentum_dot_position', 'charge_field_product', 'abs_charge',
    'cyclotron_frequency', 'cyclotron_phase', 'lorentz_force_mag', 'sim_time_field', 'time_squared',
    'sin_cyclotron_phase', 'cos_cyclotron_phase'
]

OUTPUT_TARGETS = [
    "final_velocity_x",
    "final_velocity_y",
    "final_position_x",
    "final_position_y",
    "kinetic_energy",
    "trajectory_length"
]


# Path resolution functions
def format_bytes(size: Optional[int]) -> Optional[str]:
    """Format raw byte counts into human readable text."""
    if size is None:
        return None

    threshold = 1024.0
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)

    for unit in units:
        if value < threshold or unit == units[-1]:
            return f"{value:.1f} {unit}"

        value /= threshold


def resolve_project_paths() -> Dict[str, Path]:
    """Resolve key project directories relative to notebook execution context."""
    root = Path.cwd()

    if root.name != PROJECT_NAME:
        for parent in root.parents:
            if parent.name == PROJECT_NAME:
                root = parent
                break

    config_dir = root / "ml_config"
    output_dir = root / "training_output"
    analysis_dir = output_dir / "analysis"
    figures_dir = analysis_dir / "figures"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": root,
        "config_dir": config_dir,
        "output_dir": output_dir,
        "analysis_dir": analysis_dir,
        "figures_dir": figures_dir,
        "data_path": root / "particle_data.csv",
        "scaler_X": root / "scaler_X.pkl",
        "scaler_y": root / "scaler_y.pkl",
        "weight_dir": root / "saved_weights/"
    }


def validate_required_artifacts(paths: Dict[str, Path]) -> pd.DataFrame:
    """Check presence and metadata of required artifacts."""
    required = {
        "model_config": paths["config_dir"] / "model_config.json",
        "training_config": paths["config_dir"] / "training_config.json",
        "loss_history": paths["output_dir"] / "loss_history.csv",
        "training_results": paths["output_dir"] / "training_results.csv",
        "configuration_log": paths["output_dir"] / "configuration_log.csv",
        "particle_data": paths["data_path"],
        "scaler_X": paths["scaler_X"],
        "scaler_y": paths["scaler_y"]
    }

    optional = {
        "analysis_dir": paths["analysis_dir"],
        "figures_dir": paths["figures_dir"]
    }

    notes = {
        "particle_data": "Regenerate via data pipeline if missing.",
        "scaler_X": "Rebuilt automatically through complete_data_pipeline.",
        "scaler_y": "Rebuilt automatically through complete_data_pipeline."
    }

    records: List[Dict[str, Any]] = []

    def append_record(label: str, path: Path, critical: bool) -> None:
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else None
        modified = pd.Timestamp(path.stat().st_mtime, unit="s") if exists else None

        records.append({
            "artifact": label,
            "critical": critical,
            "exists": exists,
            "path": str(path.relative_to(paths["project_root"])) if exists else str(path),
            "size_bytes": size,
            "size_readable": format_bytes(size),
            "modified": modified,
            "note": notes.get(label)
        })

    for label, path in required.items():
        append_record(label, path, True)

    for label, path in optional.items():
        append_record(label, path, False)

    status_df = pd.DataFrame(records)

    if status_df.empty:
        return status_df

    status_df = status_df.sort_values(["critical", "artifact"], ascending=[False, True]).reset_index(drop=True)

    return status_df


def list_checkpoint_weights(paths: Dict[str, Path]) -> pd.DataFrame:
    """List available weight checkpoints with epoch metadata."""
    pattern = "model_weights_epoch_*.weights.h5"
    checkpoint_files = sorted(paths["weight_dir"].glob(pattern))

    rows: List[Dict[str, Any]] = []

    for file_path in checkpoint_files:
        name = file_path.name
        parts = name.split("_")
        epoch_token = parts[3] if len(parts) > 3 else parts[-1]
        epoch = int(epoch_token.replace(".weights.h5", "")) if epoch_token else None

        rows.append({
            "epoch": epoch,
            "name": name,
            "path": str(file_path.relative_to(paths["project_root"])) if file_path.exists() else str(file_path),
            "modified": pd.Timestamp(file_path.stat().st_mtime, unit="s"),
            "size_bytes": file_path.stat().st_size
        })

    checkpoint_df = pd.DataFrame(rows)

    if checkpoint_df.empty:
        return checkpoint_df

    checkpoint_df = checkpoint_df.sort_values("epoch").reset_index(drop=True)
    latest_epoch = checkpoint_df["epoch"].max()
    checkpoint_df["size_readable"] = checkpoint_df["size_bytes"].apply(format_bytes)
    checkpoint_df["is_latest"] = checkpoint_df["epoch"] == latest_epoch

    return checkpoint_df


# Configuration and log loading functions
def load_configs(paths: Dict[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Load active configs and historical configuration snapshots with derived metrics."""
    model_config_path = paths["config_dir"] / "model_config.json"
    training_config_path = paths["config_dir"] / "training_config.json"

    with model_config_path.open() as handle:
        model_config = json.load(handle)

    with training_config_path.open() as handle:
        training_config = json.load(handle)

    snapshots: List[Dict[str, Any]] = []

    for config_path in sorted(paths["output_dir"].glob("training_config_*.json")):
        with config_path.open() as handle:
            payload = json.load(handle)

        combined: Dict[str, Any] = {
            "config_id": payload.get("config_id"),
            "timestamp": payload.get("timestamp")
        }

        model_payload = payload.get("model_config", {})
        for key, value in model_payload.items():
            combined[key] = value

        training_payload = payload.get("training_config", {})
        for key, value in training_payload.items():
            combined[f"train_{key}"] = value

        summary_payload = payload.get("performance_summary", {})
        combined["best_r2"] = summary_payload.get("best_r2")
        combined["final_r2"] = summary_payload.get("current_r2")
        combined["best_epoch"] = summary_payload.get("best_r2_epoch")
        combined["avg_epoch_time_logged"] = summary_payload.get("avg_epoch_time")
        combined["total_training_time"] = summary_payload.get("total_training_time")
        combined["weight_modifications_used"] = summary_payload.get("weight_modifications_used")
        combined["peak_memory_mb"] = summary_payload.get("peak_memory_mb")

        snapshots.append(combined)

    snapshots_df = pd.DataFrame(snapshots)

    if snapshots_df.empty:
        return model_config, training_config, snapshots_df

    snapshots_df["timestamp"] = pd.to_datetime(snapshots_df["timestamp"])

    if {"total_training_time", "train_epochs"}.issubset(snapshots_df.columns):
        snapshots_df["avg_epoch_time_calc"] = snapshots_df["total_training_time"] / snapshots_df["train_epochs"]

    snapshots_df["r2_delta"] = snapshots_df["best_r2"] - snapshots_df["final_r2"]
    snapshots_df = snapshots_df.sort_values("timestamp").reset_index(drop=True)

    return model_config, training_config, snapshots_df


def load_training_logs(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load loss history and training results with derived analytics."""
    loss_path = paths["output_dir"] / "loss_history.csv"
    results_path = paths["output_dir"] / "training_results.csv"

    loss_records = pd.read_csv(loss_path)
    loss_records = loss_records.sort_values(["epoch"]).reset_index(drop=True)
    loss_records["loss_ewm"] = loss_records["combined_loss"].ewm(alpha=0.15).mean()

    epoch_summary = (
        loss_records.groupby("epoch").agg(
            combined_loss_mean=("combined_loss", "mean"),
            combined_loss_std=("combined_loss", "std"),
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean")
        ).reset_index()
    )

    results_df = pd.read_csv(results_path)
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], format="mixed")
    results_df = results_df.sort_values("epoch").reset_index(drop=True)
    results_df["epoch"] = results_df["epoch"].astype(int)
    results_df["cumulative_time"] = results_df["epoch_time"].cumsum()
    results_df["val_loss_delta"] = results_df["val_loss"].diff()
    results_df["train_val_gap"] = results_df["val_loss"] - results_df["train_loss"]
    results_df["val_mae_delta"] = results_df["val_mae"].diff()
    results_df["epoch_time_rolling"] = results_df["epoch_time"].rolling(5, min_periods=1).mean()
    results_df["memory_headroom_mb"] = results_df["memory_mb"].max() - results_df["memory_mb"]

    merged_metrics = results_df.merge(epoch_summary, on="epoch", how="left")
    merged_metrics["val_loss_rolling"] = merged_metrics["val_loss"].rolling(5, min_periods=1).mean()
    merged_metrics["train_loss_rolling"] = merged_metrics["train_loss"].rolling(5, min_periods=1).mean()

    analytics = {
        "loss_records": loss_records,
        "epoch_summary": epoch_summary,
        "results": results_df,
        "merged_metrics": merged_metrics
    }

    return analytics


def load_scalers(paths: Dict[str, Path]) -> Tuple[Any, Any]:
    """Load cached scalers, regenerating them via training pipeline if missing."""
    scaler_X_path = paths["scaler_X"]
    scaler_y_path = paths["scaler_y"]
    pipeline_ran = False

    def ensure_pipeline() -> None:
        nonlocal pipeline_ran
        if pipeline_ran:
            return

        complete_data_pipeline(csv_path=str(paths["data_path"]))
        pipeline_ran = True

    try:
        scaler_X = joblib.load(scaler_X_path)
    except FileNotFoundError:
        ensure_pipeline()
        scaler_X = joblib.load(scaler_X_path)

    try:
        scaler_y = joblib.load(scaler_y_path)
    except FileNotFoundError:
        ensure_pipeline()
        scaler_y = joblib.load(scaler_y_path)

    return scaler_X, scaler_y


def load_particle_data(paths: Dict[str, Path]) -> pd.DataFrame:
    """Load particle simulation data with validation safeguards and derived features."""
    from data_processing import add_derived_features
    
    dataset = load_and_validate_data(csv_path=str(paths["data_path"]))
    
    # Add derived features for compatibility with training pipeline
    dataset = add_derived_features(dataset)

    if "particle_id" in dataset.columns:
        dataset = dataset.sort_values("particle_id").reset_index(drop=True)
    else:
        dataset = dataset.reset_index(drop=True)

    return dataset


# Model reconstruction and prediction functions
def build_model_from_config(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> tf.keras.Model:
    """Instantiate a compiled model that mirrors the training setup."""
    config_payload = dict(model_config)
    config_payload.update(training_config)
    config_payload.setdefault("enable_weight_oscillation_dampener", True)

    input_shape = (len(INPUT_FEATURES),)
    output_shape = len(OUTPUT_TARGETS)

    network = AdvancedNeuralNetwork(
        input_shape=input_shape,
        output_shape=output_shape,
        config=config_payload
    )
    network.compile_model()

    return network.model


def load_model_checkpoint(
    paths: Dict[str, Path],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    checkpoint_index: pd.DataFrame,
    checkpoint_name: Optional[str] = None
) -> Tuple[Optional[tf.keras.Model], Optional[Dict[str, Any]]]:
    """Load model weights from the selected checkpoint."""
    if checkpoint_index.empty:
        return None, None

    if checkpoint_name is None:
        selected_row = checkpoint_index.iloc[-1]
    else:
        if checkpoint_name not in checkpoint_index["name"].values:
            return None, None

        selected_row = checkpoint_index.loc[checkpoint_index["name"] == checkpoint_name].iloc[0]

    weights_path = paths["project_root"] / selected_row["path"]

    tf.keras.backend.clear_session()

    model = build_model_from_config(model_config=model_config, training_config=training_config)
    model.load_weights(weights_path)

    metadata = {
        "epoch": int(selected_row["epoch"]),
        "weights_path": str(weights_path.relative_to(paths["project_root"])),
        "size_bytes": int(selected_row["size_bytes"]),
        "size_readable": selected_row.get("size_readable"),
        "modified": selected_row["modified"],
        "parameter_count": int(model.count_params())
    }

    return model, metadata


def compute_predictions(
    model: Optional[tf.keras.Model],
    scaler_X: Any,
    scaler_y: Any,
    particle_df: pd.DataFrame,
    sample_size: int = 256
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate predictions and residual analytics using stored scalers."""
    if model is None:
        return pd.DataFrame(), {}

    # Add derived features if not present
    from data_processing import add_derived_features
    
    particle_df_with_features = add_derived_features(particle_df)
    
    # Check if all required features are available
    missing_features = [f for f in INPUT_FEATURES if f not in particle_df_with_features.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing features for prediction: {missing_features[:5]}")
        return pd.DataFrame(), {}
    
    feature_subset = particle_df_with_features[INPUT_FEATURES].copy()

    if sample_size and len(feature_subset) > sample_size:
        feature_subset = feature_subset.sample(sample_size, random_state=42).sort_index()

    scaled_inputs = scaler_X.transform(feature_subset.values) if scaler_X is not None else feature_subset.values
    predictions_scaled = model.predict(scaled_inputs, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled) if scaler_y is not None else predictions_scaled

    actual_outputs = particle_df_with_features.loc[feature_subset.index, OUTPUT_TARGETS].values
    residuals = predictions - actual_outputs

    residual_df = pd.DataFrame(index=feature_subset.index)

    if "particle_id" in particle_df_with_features.columns:
        residual_df["particle_id"] = particle_df_with_features.loc[feature_subset.index, "particle_id"]

    for idx, target in enumerate(OUTPUT_TARGETS):
        residual_df[f"actual_{target}"] = actual_outputs[:, idx]
        residual_df[f"pred_{target}"] = predictions[:, idx]
        residual_df[f"residual_{target}"] = residuals[:, idx]

    residual_df["residual_norm"] = np.linalg.norm(residuals, axis=1)

    residual_norm_mean = residual_df["residual_norm"].mean()
    residual_norm_std = residual_df["residual_norm"].std(ddof=0)

    if residual_norm_std and residual_norm_std > 0:
        residual_df["residual_norm_z"] = (residual_df["residual_norm"] - residual_norm_mean) / residual_norm_std

    mae_value = float(np.mean(np.abs(residuals)))
    rmse_value = float(np.sqrt(np.mean(np.square(residuals))))

    target_metrics: Dict[str, Dict[str, float]] = {}

    for idx, target in enumerate(OUTPUT_TARGETS):
        target_residuals = residuals[:, idx]
        target_metrics[target] = {
            "mae": float(np.mean(np.abs(target_residuals))),
            "rmse": float(np.sqrt(np.mean(np.square(target_residuals)))),
            "bias": float(np.mean(target_residuals))
        }

    metrics: Dict[str, Any] = {
        "samples": int(len(residual_df)),
        "mae": mae_value,
        "rmse": rmse_value,
        "residual_norm_median": float(residual_df["residual_norm"].median()),
        "residual_norm_p95": float(residual_df["residual_norm"].quantile(0.95)),
        "targets": target_metrics
    }

    return residual_df, metrics


# Summary and recommendation functions
def summarize_run_performance(
    results_df: pd.DataFrame,
    epoch_summary: pd.DataFrame
) -> pd.DataFrame:
    """Create a concise summary of key performance indicators."""
    if results_df.empty:
        return pd.DataFrame()

    best_epoch_idx = int(results_df["val_loss"].idxmin())
    best_row = results_df.loc[best_epoch_idx]
    final_row = results_df.iloc[-1]
    early_row = results_df.iloc[0]

    improvement = float(early_row["val_loss"] - best_row["val_loss"])
    consistency = float(epoch_summary["combined_loss_std"].tail(5).mean()) if not epoch_summary.empty else float("nan")
    best_r2_row = results_df.loc[results_df["r2_score"].idxmax()]

    summary = pd.DataFrame([
        {"metric": "Best validation loss", "value": best_row["val_loss"], "notes": f"Epoch {int(best_row['epoch'])}"},
        {"metric": "Final validation loss", "value": final_row["val_loss"], "notes": f"Train gap {final_row['train_val_gap']:.4f}"},
        {"metric": "Validation improvement", "value": improvement, "notes": "Drop from first to best epoch"},
        {"metric": "Validation stability (std last 5 epochs)", "value": consistency, "notes": "Lower is more stable"},
        {"metric": "Average epoch time (last 10 epochs)", "value": results_df["epoch_time"].tail(10).mean(), "notes": "Supports batch-size experiments"},
        {"metric": "Peak R²", "value": best_r2_row["r2_score"], "notes": f"Epoch {int(best_r2_row['epoch'])}"},
        {"metric": "Total recorded training time", "value": results_df["epoch_time"].sum(), "notes": "seconds"}
    ])

    return summary
