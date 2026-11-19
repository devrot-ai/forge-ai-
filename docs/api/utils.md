# Utilities API

Reference documentation for shared utilities.

## ExperimentLogger

JSON-lines experiment logging.

::: envgen.utils.logger.ExperimentLogger
    options:
      show_source: true
      show_root_heading: true

## MLflowLogger

MLflow experiment tracking.

::: envgen.utils.mlflow_logger.MLflowLogger
    options:
      show_source: true

## WandbLogger

Weights & Biases experiment tracking.

::: envgen.utils.wandb_logger.WandbLogger
    options:
      show_source: true

## EarlyStoppingMonitor

Auto-stop training when performance plateaus.

::: envgen.utils.early_stopping.EarlyStoppingMonitor
    options:
      show_source: true

## BenchmarkRunner

Standardized benchmark evaluation.

::: envgen.benchmark.runner.BenchmarkRunner
    options:
      show_source: true

## HyperparameterSearch

Grid search over hyperparameters.

::: envgen.benchmark.hyperparam_search.HyperparameterSearch
    options:
      show_source: true

## RewardOptimizer

Automated reward weight optimization.

::: envgen.benchmark.reward_optimizer.RewardOptimizer
    options:
      show_source: true

## PopulationBasedTrainer

Evolve agent population with hyperparameter mutation.

::: envgen.benchmark.pbt.PopulationBasedTrainer
    options:
      show_source: true

## ConfigLoader

YAML configuration loader.

::: envgen.config.loader.ConfigLoader
    options:
      show_source: true
