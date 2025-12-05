---
name: hyperparameter-tuner
description: Lightweight agent implemented in `hpo_agent.py` for exploring hyperparameters
target: vscode
---

## Goals
- Automate short training runs across model presets and hyperparameters
- Log metrics and produce a consolidated CSV for quick comparison
- Keep dependencies minimal and work headlessly

## How it works
- Data is prepared once via `complete_data_pipeline`.
- For each trial, the agent:
  - Picks a model preset from `ml_config/model_presets/`
  - Samples a hyperparameter configuration
  - Trains an `AdvancedNeuralNetwork` for N epochs
  - Records key metrics (best R², final R², test metrics)
- All trials are saved to `training_output/analysis/hpo_results.csv`.

## Presets
Presets are simple JSON files under `ml_config/model_presets/` that define a base model configuration.
Examples:
- `baseline.json` – small, fast network
- `fast_debug.json` – minimal layers for quick debugging
- `deep_regularized.json` – large network with dropout, batch norm, L2
- `high_precision.json` – medium-deep with tighter constraints

## Search space
Randomly sampled per trial:
- activation: {relu, prelu}
- optimizer: {adam, rmsprop, sgd}
- learning_rate: ~[3e-4, 1e-3]
- dropout_rate: [0.0, 0.25]
- use_batch_norm: {true, false}
- batch_norm_momentum: [0.85, 0.99]
- l2_regularization: ~[1e-5, 3e-4]
- constraint_interval: {1,2,3,5}
- enable_weight_oscillation_dampener: {true, false}
- use_adaptive_oscillation_dampener: {true, false}
- hidden_layers: one of [[64,32],[128,64,32],[256,128,64],[256,256,128,64]]

## Usage
Run trials from a terminal:

```powershell
python hpo_agent.py --max-trials 20 --presets baseline,deep_regularized --epochs 25 --batch-size 64
```

Outputs to:
- `training_output/analysis/hpo_results.csv`

## Tips
- Use smaller `--epochs` (10–25) for exploration; rerun best configs with higher epochs.
- Inspect `training_output/configuration_log.csv` for per-run details.
- Check `training_output/analysis/figures/` for animated training graphs produced during runs with >=3 epochs.
