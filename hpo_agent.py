"""
Hyperparameter Optimization Agent

This agent performs hyperparameter searches across model presets and hyperparameter spaces,
training short runs and collecting metrics. Results are saved under training_output/analysis/.

Usage (PowerShell):
  python hpo_agent.py --max-trials 20 --presets baseline,deep_regularized --epochs 25 --batch-size 64

Notes:
- Uses existing data pipeline and AdvancedNeuralNetwork class.
- Each trial logs standard training artifacts; additionally, an aggregated CSV is produced.
"""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from advanced_neural_network import AdvancedNeuralNetwork
from data_processing import complete_data_pipeline


ROOT = Path(__file__).parent
CONFIG_DIR = ROOT / "ml_config"
PRESETS_DIR = CONFIG_DIR / "model_presets"
OUTPUT_DIR = ROOT / "training_output"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_preset(name: str) -> Dict[str, Any]:
    path = PRESETS_DIR / f"{name}.json"

    if not path.exists(): raise FileNotFoundError(f"Preset not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_hyperparams(rng: random.Random) -> Dict[str, Any]:
    """Randomly sample a set of hyperparameters within reasonable bounds."""
    activations = ["relu", "prelu"]
    optimizers = ["adam", "rmsprop", "sgd"]
    loss_strategies = ["r2_based", "loss_based", "combined", "physics_aware"]
    hidden_options: List[List[int]] = [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64],
        [256, 256, 128, 64]
    ]
    return {
        "activation": rng.choice(activations),
        "optimizer": rng.choice(optimizers),
        "learning_rate": 10 ** rng.uniform(-3.5, -3.0),  # ~[0.0003, 0.001]
        "dropout_rate": float(np.clip(rng.uniform(0.0, 0.25), 0.0, 0.5)),
        "use_batch_norm": rng.choice([True, False]),
        "batch_norm_momentum": float(np.clip(rng.uniform(0.85, 0.99), 0.85, 0.99)),
        "l2_regularization": 10 ** rng.uniform(-5.0, -3.5),
        "constraint_interval": int(rng.choice([1, 2, 3, 5])),
        "enable_weight_oscillation_dampener": rng.choice([True, False]),
        "use_adaptive_oscillation_dampener": rng.choice([True, False]),
        "loss_weighting_strategy": rng.choice(loss_strategies),
        "hidden_layers": rng.choice(hidden_options)
    }


def run_trial(preset: Dict[str, Any], overrides: Dict[str, Any],
              epochs: int, batch_size: int,
              data_cache: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    X_train, X_val, X_test, y_train, y_val, y_test = data_cache
    cfg = dict(preset)

    cfg.update(overrides)

    model = AdvancedNeuralNetwork((X_train.shape[1],), y_train.shape[1], cfg)
    model.compile_model()
    _ = model.train_with_custom_constraints(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    test_metrics = model.evaluate_model(X_test, y_test)
    perf = model.performance_tracker.get_summary() if model.performance_tracker else {}
    result = {
        "preset": overrides.get("_preset_name"),
        "overrides": {k: v for k, v in overrides.items() if not k.startswith("_")},
        "test_mse": test_metrics.get("mse"),
        "test_mae": test_metrics.get("mae"),
        "test_rmse": test_metrics.get("rmse"),
        "test_r2": test_metrics.get("r2_score"),
        "best_r2": perf.get("best_r2"),
        "final_r2": perf.get("current_r2"),
        "avg_epoch_time": perf.get("avg_epoch_time"),
        "total_epochs": perf.get("total_epochs")
    }
    return result


def save_results(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Agent")
    
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--presets", type=str, default="baseline,deep_regularized")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=str(ANALYSIS_DIR / "hpo_results.csv"))
    parser.add_argument("--benchmarking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    rng = random.Random(args.seed)
    data_splits = complete_data_pipeline(num_particles=3000)
    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    results: List[Dict[str, Any]] = []

    for trial in range(args.max_trials):
        preset_name = rng.choice(preset_names)
        preset = load_preset(preset_name)
        overrides = sample_hyperparams(rng)
        overrides["_preset_name"] = preset_name

        if args.benchmarking:
            overrides.clear()
            overrides.update(preset)
            overrides["_preset_name"] = preset_name

        print(f"\n=== Trial {trial+1}/{args.max_trials} | Preset={preset_name} ===")
        print(json.dumps({k: overrides[k] for k in sorted(overrides) if not k.startswith('_')}, indent=2))

        try:
            row = run_trial(preset, overrides, epochs=args.epochs, batch_size=args.batch_size, data_cache=data_splits)
            
            results.append(row)
        except Exception as e:
            print(f"Trial failed: {e}")
            continue

    out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, out_path)
    print(f"\nHPO results saved to: {out_path}")


if __name__ == "__main__":
    main()
