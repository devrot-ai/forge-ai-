# Experiment Tracking Guide

Track, compare, and visualize your training experiments.

## Overview

EnvGen-LLM provides three methods for experiment tracking:

| Method | Description | Best For |
|--------|-------------|----------|
| JSON Logging | Built-in JSON-lines files | Simple tracking, no dependencies |
| MLflow | Full experiment management | Research teams, MLflow UI |
| Weights & Biases | Cloud-based tracking | Collaborative research |

## JSON Logging (Built-in)

### Basic Usage

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs
```

This creates a directory structure like:

```
logs/
└── run_20240101_120000/
    ├── metrics.jsonl      # Per-cycle metrics
    ├── config.json         # Training configuration
    └── summary.json        # Final summary
```

### Metrics File Format

Each line in `metrics.jsonl` is a JSON object:

```json
{
    "cycle": 1,
    "timestamp": "2024-01-01T12:00:00",
    "rewards": {"mean": 45.2, "std": 12.3},
    "skill_scores": {
        "navigation": 0.45,
        "planning": 0.32,
        "reasoning": 0.28,
        "memory": 0.35,
        "efficiency": 0.40
    },
    "difficulty": "EASY",
    "episodes": 10,
    "duration_seconds": 15.2
}
```

### Viewing with Dashboard

```bash
pip install -e ".[dashboard]"
python -m envgen dashboard --log-dir ./logs
```

Open http://localhost:8080 in your browser.

## MLflow Integration

### Installation

```bash
pip install -e ".[mlflow]"
```

### Basic Usage

```bash
python -m envgen train --config configs/default.yaml --mlflow
```

### Custom Tracking URI

```bash
python -m envgen train --config configs/default.yaml --mlflow --mlflow-tracking-uri ./mlruns
```

### Viewing MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 in your browser.

### What's Tracked

- **Parameters**: Agent type, environment type, hyperparameters
- **Metrics**: Rewards, skill scores, difficulty level per cycle
- **Artifacts**: Environment configs, checkpoints

## Weights & Biases Integration

### Installation

```bash
pip install wandb
```

### Basic Usage

```bash
# Login first
wandb login

# Train with W&B
python -m envgen train --config configs/default.yaml --wandb
```

### Custom Project Name

```bash
python -m envgen train --config configs/default.yaml --wandb --wandb-project my-experiments
```

## Comparing Experiments

### Using the Dashboard

```bash
# Run multiple experiments
python -m envgen train --config configs/default.yaml --agent ppo --log-dir ./logs
python -m envgen train --config configs/default.yaml --agent ppo_torch --log-dir ./logs
python -m envgen train --config configs/default.yaml --agent grpo_torch --log-dir ./logs

# View comparison
python -m envgen dashboard --log-dir ./logs
```

### Using Benchmark

```bash
# Compare agents on the same environment
python -m envgen benchmark --agent ppo_torch --episodes 50
python -m envgen benchmark --agent grpo_torch --episodes 50
```

## Organizing Experiments

### Directory Structure

```
experiments/
├── baseline/
│   ├── run_20240101_120000/
│   └── run_20240101_130000/
├── with_curriculum/
│   ├── run_20240102_120000/
│   └── run_20240102_130000/
└── with_ard/
    ├── run_20240103_120000/
    └── run_20240103_130000/
```

### Naming Convention

Use meaningful experiment names in your config:

```yaml
# configs/experiment_1.yaml
task:
  name: "ppo_on_maze_baseline"
  description: "Baseline PPO on maze environment"
```

## Reproducibility

### Fixed Seeds

```yaml
training:
  seed: 42
```

### Saving Full Config

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs
```

The full config is saved in `config.json` for each run.

### Reproducing a Run

```bash
# Use the saved config
python -m envgen train --config ./logs/run_20240101_120000/config.json
```

## Next Steps

- **[Docker Usage Guide](docker_usage.md)** — Containerized experiments
- **[LLM Integration Guide](llm_integration.md)** — Track LLM-generated experiments
- **[API Reference](../api/utils.md)** — Logger API documentation
