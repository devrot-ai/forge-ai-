"""Optional Weights & Biases integration for experiment tracking.

This module wraps the ExperimentLogger and also sends metrics to W&B,
enabling use of the W&B UI for experiment management and visualization.

Requires Weights & Biases: pip install wandb

Example:
    logger = WandbLogger(log_dir="./logs", project="envgen-llm")
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
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


class WandbLogger:
    """Experiment logger that writes to both JSON files and Weights & Biases.

    Wraps the ExperimentLogger functionality and additionally tracks
    params, metrics, and artifacts in W&B for experiment management.

    Args:
        log_dir: Base directory for JSON log files
        run_name: Optional name for this run
        config: Optional pipeline config to log
        project: Name of the W&B project (default: "envgen-llm")
        entity: Optional W&B entity/team name
        run_id: Optional W&B run ID for resuming
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        project: str = "envgen-llm",
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize the W&B logger.

        Args:
            log_dir: Base directory for JSON log files
            run_name: Optional name for this run
            config: Optional pipeline config to log
            project: Name of the W&B project
            entity: Optional W&B entity/team name
            run_id: Optional W&B run ID for resuming
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Weights & Biases is required for WandbLogger. "
                "Install it with: pip install wandb"
            )

        self.log_dir = log_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project = project
        self._start_time = time.time()
        self._cycle_count = 0
        self._metrics: List[Dict[str, Any]] = []

        # Setup W&B
        wandb.init(
            project=project,
            entity=entity,
            name=self.run_name,
            id=run_id,
            resume="allow" if run_id else None,
            config=config or {},
        )

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

    def log_cycle(
        self,
        cycle: int,
        rewards: Optional[List[float]] = None,
        skill_scores: Optional[Dict[str, float]] = None,
        difficulty: Optional[str] = None,
        episode_count: Optional[int] = None,
        training_time: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log metrics for a training cycle to both JSON and W&B.

        Args:
            cycle: Current training cycle number
            rewards: List of episode rewards
            skill_scores: Dict of skill name to score (0-1)
            difficulty: Current difficulty level
            episode_count: Number of episodes in this cycle
            training_time: Time taken for this cycle in seconds
            extra_metrics: Additional metrics to log
        """
        self._cycle_count = cycle
        metrics: Dict[str, Any] = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
        }

        # Process rewards
        if rewards:
            metrics["reward_mean"] = sum(rewards) / len(rewards)
            metrics["reward_min"] = min(rewards)
            metrics["reward_max"] = max(rewards)
            metrics["reward_sum"] = sum(rewards)

        # Process skill scores
        if skill_scores:
            for skill, score in skill_scores.items():
                metrics[f"skill_{skill}"] = score

        if difficulty:
            metrics["difficulty"] = difficulty
        if episode_count is not None:
            metrics["episode_count"] = episode_count
        if training_time is not None:
            metrics["training_time"] = training_time
        if extra_metrics:
            metrics.update(extra_metrics)

        # Log to W&B
        wandb.log(metrics)

        # Log to JSON file
        self._metrics.append(metrics)
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log additional parameters to W&B.

        Args:
            params: Dict of parameter name to value
        """
        if wandb.run:
            for key, value in params.items():
                wandb.run.config[key] = value

    def log_artifact(self, filepath: str, artifact_name: str, artifact_type: str = "file") -> None:
        """Log a file as a W&B artifact.

        Args:
            filepath: Path to the file to log
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
        """
        if os.path.exists(filepath) and wandb.run:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged metrics.

        Returns:
            Dict with summary statistics
        """
        total_time = time.time() - self._start_time
        summary = {
            "run_name": self.run_name,
            "total_cycles": self._cycle_count,
            "total_time": total_time,
            "log_dir": self.run_dir,
        }

        if self._metrics:
            last_metrics = self._metrics[-1]
            summary["last_reward_mean"] = last_metrics.get("reward_mean")
            summary["last_skill_scores"] = {
                k: v
                for k, v in last_metrics.items()
                if k.startswith("skill_")
            }

        # Save summary to JSON
        summary_file = os.path.join(self.run_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def close(self) -> None:
        """Close the logger and finish the W&B run."""
        self.get_summary()
        self._file.close()
        if wandb.run:
            wandb.finish()
