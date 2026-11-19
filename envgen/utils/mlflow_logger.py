"""Optional MLflow integration for experiment tracking.

This module wraps the ExperimentLogger and also sends metrics to MLflow,
enabling use of the MLflow UI for experiment management and visualization.

Requires MLflow: pip install mlflow

Example:
    logger = MLflowLogger(log_dir="./logs", tracking_uri=None)
    logger.log_cycle(cycle=1, rewards=[10, 15], skill_scores={"navigation": 0.8})
    logger.close()
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


class MLflowLogger:
    """Experiment logger that writes to both JSON files and MLflow.

    Wraps the ExperimentLogger functionality and additionally tracks
    params, metrics, and artifacts in MLflow for experiment management.

    Args:
        log_dir: Base directory for JSON log files
        run_name: Optional name for this run
        config: Optional pipeline config to log
        tracking_uri: Optional MLflow tracking URI (default: local)
        experiment_name: Name of the MLflow experiment (default: "envgen-llm")
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "envgen-llm",
    ):
        """Initialize the MLflow logger.

        Args:
            log_dir: Base directory for JSON log files
            run_name: Optional name for this run
            config: Optional pipeline config to log
            tracking_uri: Optional MLflow tracking URI
            experiment_name: Name of the MLflow experiment
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for MLflowLogger. " "Install it with: pip install mlflow")

        self.log_dir = log_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self._start_time = time.time()
        self._cycle_count = 0
        self._metrics: List[Dict[str, Any]] = []

        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Start run
        self.run = mlflow.start_run(run_id=None, run_name=self.run_name)

        # Log config as params
        if config:
            self._log_config(config)

        # Setup JSON logging
        self.run_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self._file = open(self.metrics_file, "w", encoding="utf-8")
        self._file.write("# Experiment metrics log - JSON lines format\n")
        self._file.flush()

        # Save config to JSON
        config_file = os.path.join(self.run_dir, "config.json")
        if config:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)

    def _log_config(self, config: Dict[str, Any]) -> None:
        """Log config as MLflow params."""
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}.{sub_key}", str(sub_value))
            else:
                mlflow.log_param(key, str(value))

    def log_cycle(
        self,
        cycle: int,
        rewards: Optional[List[float]] = None,
        skill_scores: Optional[Dict[str, float]] = None,
        difficulty: Optional[str] = None,
        episodes_trained: Optional[int] = None,
        steps_trained: Optional[int] = None,
        eval_result: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics for a completed training cycle.

        Args:
            cycle: Cycle number (1-indexed)
            rewards: List of episode rewards from this cycle
            skill_scores: Dict of skill name -> score (0-1)
            difficulty: Current difficulty level name
            episodes_trained: Number of episodes trained in this cycle
            steps_trained: Total steps taken in this cycle
            eval_result: Full evaluation result dict
            extra: Additional custom metrics to log
        """
        self._cycle_count = cycle
        elapsed = time.time() - self._start_time

        # Compute reward stats
        metrics: Dict[str, float] = {}
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)
            metrics["min_reward"] = min(rewards)
            metrics["max_reward"] = max(rewards)
            metrics["num_episodes"] = len(rewards)

        if episodes_trained is not None:
            metrics["episodes_trained"] = episodes_trained
        if steps_trained is not None:
            metrics["steps_trained"] = steps_trained

        # Log skill scores
        if skill_scores:
            for skill, score in skill_scores.items():
                metrics[f"skill.{skill}"] = score

        # Log eval result
        if eval_result:
            if "overall_score" in eval_result:
                metrics["overall_score"] = eval_result["overall_score"]

        # Log to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=cycle)

        # Log difficulty as param
        if difficulty:
            mlflow.log_param("difficulty", difficulty)

        # Write to JSON lines file
        json_metric: Dict[str, Any] = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "difficulty": difficulty,
            "episodes_trained": episodes_trained,
            "steps_trained": steps_trained,
            "rewards": metrics.get("mean_reward"),
            "skill_scores": skill_scores or {},
            "eval_result": eval_result,
        }
        self._file.write(json.dumps(json_metric) + "\n")
        self._file.flush()
        self._metrics.append(json_metric)

    def log_summary(self, final_skill_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Write a summary file and log final metrics to MLflow.

        Args:
            final_skill_scores: Final skill scores after all cycles

        Returns:
            Summary dictionary
        """
        elapsed = time.time() - self._start_time

        # Log final skill scores to MLflow
        if final_skill_scores:
            for skill, score in final_skill_scores.items():
                mlflow.log_metric(f"final_skill.{skill}", score)

        # Compute aggregate stats
        all_rewards = []
        for m in self._metrics:
            if isinstance(m.get("rewards"), (int, float)):
                all_rewards.append(m["rewards"])

        summary: Dict[str, Any] = {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "total_cycles": self._cycle_count,
            "total_time_seconds": round(elapsed, 2),
            "final_skill_scores": final_skill_scores or {},
            "aggregate": {
                "mean_reward_across_cycles": (round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else None),
                "best_cycle_reward": max(all_rewards) if all_rewards else None,
                "worst_cycle_reward": min(all_rewards) if all_rewards else None,
            },
        }

        # Save summary to JSON
        summary_file = os.path.join(self.run_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Log summary metrics to MLflow
        mlflow.log_metric("total_cycles", self._cycle_count)
        mlflow.log_metric("total_time_seconds", round(elapsed, 2))

        return summary

    def get_run_dir(self) -> str:
        """Return the path to this run's log directory."""
        return self.run_dir

    def close(self) -> None:
        """Close the logger and finalize files."""
        if hasattr(self, "_file") and self._file and not self._file.closed:
            self._file.write("# End of metrics log\n")
            self._file.close()
        if hasattr(self, "run") and self.run:
            mlflow.end_run()

    def __del__(self) -> None:
        """Ensure file is closed on deletion."""
        try:
            self.close()
        except Exception:
            pass
