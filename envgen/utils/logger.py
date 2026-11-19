"""Experiment logging for training runs.

This module provides structured logging of training metrics to JSON lines files,
enabling tracking and comparison of training runs over time. Each run gets its
own timestamped directory with a metrics log file.

No external dependencies required — uses only Python standard library.

Example:
    logger = ExperimentLogger(log_dir="./logs")
    logger.log_cycle(cycle=1, rewards=[10, 15, 20], skill_scores={"navigation": 0.8})
    logger.close()
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Structured experiment logger that writes JSON lines files.

    Tracks metrics after each training cycle including rewards, skill scores,
    difficulty level, episode stats, and timing. Auto-creates a run directory
    with a timestamp.

    Configuration:
    - log_dir: Base directory for logs (default: './logs')
    - run_name: Optional run name (default: auto-generated timestamp)

    Example:
        logger = ExperimentLogger(log_dir="./logs")
        logger.log_cycle(cycle=1, rewards=[10, 15], skill_scores={"navigation": 0.8})
        logger.close()
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the experiment logger.

        Args:
            log_dir: Base directory for log files
            run_name: Optional name for this run (default: timestamp-based)
            config: Optional pipeline config to log at start
        """
        cfg = config or {}
        self.log_dir: str = cfg.get("log_dir", log_dir)
        self.run_name: str = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir: str = os.path.join(self.log_dir, self.run_name)
        self.metrics_file: str = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_file: str = os.path.join(self.run_dir, "config.json")
        self.summary_file: str = os.path.join(self.run_dir, "summary.json")

        # Create run directory
        os.makedirs(self.run_dir, exist_ok=True)

        # Log initial config
        if config:
            self._save_config(config)

        # Track metrics
        self._metrics: List[Dict[str, Any]] = []
        self._start_time: float = time.time()
        self._cycle_count: int = 0

        # Open file handle for streaming writes
        self._file = open(self.metrics_file, "w", encoding="utf-8")
        self._file.write("# Experiment metrics log - JSON lines format\n")
        self._file.flush()

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
        reward_stats: Dict[str, float] = {}
        if rewards:
            reward_stats = {
                "mean_reward": sum(rewards) / len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "num_episodes": len(rewards),
            }

        metric: Dict[str, Any] = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "difficulty": difficulty,
            "episodes_trained": episodes_trained,
            "steps_trained": steps_trained,
            "rewards": reward_stats,
            "skill_scores": skill_scores or {},
            "eval_result": eval_result,
        }

        if extra:
            metric["extra"] = extra

        # Write to JSON lines file
        self._file.write(json.dumps(metric) + "\n")
        self._file.flush()

        # Keep in-memory copy
        self._metrics.append(metric)

    def log_curriculum_stage(
        self,
        cycle: int,
        stage: int,
        difficulty: str,
        stage_name: Optional[str] = None,
        criteria_met: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a curriculum stage change.

        Records when the curriculum advances to a new difficulty level,
        enabling visualization of difficulty progression over time.

        Args:
            cycle: Current cycle number
            stage: Curriculum stage number (0-indexed)
            difficulty: Difficulty level name (e.g., "easy", "medium")
            stage_name: Optional human-readable stage name
            criteria_met: Optional dict of criteria scores that triggered the change
        """
        metric = {
            "type": "curriculum_stage",
            "cycle": cycle,
            "stage": stage,
            "difficulty": difficulty,
            "stage_name": stage_name,
            "criteria_met": criteria_met or {},
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - self._start_time, 2),
        }
        self._file.write(json.dumps(metric) + "\n")
        self._file.flush()

    def log_episode(
        self,
        cycle: int,
        episode_idx: int,
        reward: float,
        steps: int,
        skills_demonstrated: Optional[List[str]] = None,
    ) -> None:
        """Log a single episode within a cycle.

        Args:
            cycle: Current cycle number
            episode_idx: Episode index within the cycle
            reward: Total reward for this episode
            steps: Number of steps in this episode
            skills_demonstrated: List of skills demonstrated
        """
        metric = {
            "type": "episode",
            "cycle": cycle,
            "episode": episode_idx,
            "reward": reward,
            "steps": steps,
            "skills": skills_demonstrated or [],
            "timestamp": datetime.now().isoformat(),
        }
        self._file.write(json.dumps(metric) + "\n")
        self._file.flush()

    def log_summary(self, final_skill_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Write a summary file with aggregate stats.

        Args:
            final_skill_scores: Final skill scores after all cycles

        Returns:
            Summary dictionary
        """
        elapsed = time.time() - self._start_time

        # Compute aggregate stats
        all_rewards = []
        for m in self._metrics:
            if "rewards" in m and "mean_reward" in m["rewards"]:
                all_rewards.append(m["rewards"]["mean_reward"])

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

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def get_run_dir(self) -> str:
        """Return the path to this run's log directory."""
        return self.run_dir

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save the pipeline config to a JSON file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    def close(self) -> None:
        """Close the logger and finalize files."""
        if self._file:
            self._file.write("# End of metrics log\n")
            self._file.close()

    def __del__(self) -> None:
        """Ensure file is closed on deletion."""
        try:
            if hasattr(self, "_file") and self._file and not self._file.closed:
                self.close()
        except Exception:
            pass
