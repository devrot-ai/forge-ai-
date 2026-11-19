"""Reward function visualization for experiment analysis.

Reads experiment logs (JSON-lines) and extracts reward components over time.
Shows how success_weight, progress_weight, step_penalty, and shaping contributions
evolve during training. Supports comparing multiple runs side by side.

Usage:
    python -m envgen visualize-rewards --log-dir ./logs
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardSnapshot:
    """A snapshot of reward components at a specific cycle.

    Attributes:
        cycle: Training cycle number.
        total_reward: Total reward received.
        success_reward: Reward from successful completions.
        progress_reward: Reward from progress made.
        step_penalty: Penalty for steps taken.
        shaping_reward: Reward from shaping functions.
        episode_count: Number of episodes in this cycle.
    """

    cycle: int
    total_reward: float = 0.0
    success_reward: float = 0.0
    progress_reward: float = 0.0
    step_penalty: float = 0.0
    shaping_reward: float = 0.0
    episode_count: int = 0


class RewardVisualizer:
    """Visualizes reward functions from experiment logs.

    Reads JSON-lines experiment logs and extracts reward components
    to show how rewards evolve over training cycles.

    Args:
        log_dir: Directory containing experiment log files.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.runs: Dict[str, List[RewardSnapshot]] = {}

    def load_run(self, run_path: str) -> List[RewardSnapshot]:
        """Load reward data from a single run's log files.

        Args:
            run_path: Path to a run directory containing metrics.jsonl.

        Returns:
            List of RewardSnapshot objects for each cycle.
        """
        metrics_file = os.path.join(run_path, "metrics.jsonl")
        if not os.path.exists(metrics_file):
            logger.warning(f"No metrics.jsonl found in {run_path}")
            return []

        snapshots = []
        with open(metrics_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snapshot = self._parse_metrics_line(data, line_num)
                    if snapshot is not None:
                        snapshots.append(snapshot)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {metrics_file}: {e}")

        run_name = os.path.basename(run_path)
        self.runs[run_name] = snapshots
        return snapshots

    def load_all_runs(self) -> Dict[str, List[RewardSnapshot]]:
        """Load all runs from the log directory.

        Returns:
            Dict mapping run names to their reward snapshots.
        """
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory does not exist: {self.log_dir}")
            return {}

        self.runs.clear()
        for entry in sorted(os.listdir(self.log_dir)):
            run_path = os.path.join(self.log_dir, entry)
            if os.path.isdir(run_path):
                snapshots = self.load_run(run_path)
                if snapshots:
                    logger.info(f"Loaded {len(snapshots)} cycles from {entry}")

        return self.runs

    def _parse_metrics_line(self, data: Dict[str, Any], default_cycle: int) -> Optional[RewardSnapshot]:
        """Parse a metrics JSON-lines entry into a RewardSnapshot.

        Args:
            data: Parsed JSON data from a log line.
            default_cycle: Default cycle number if not in data.

        Returns:
            RewardSnapshot or None if not a reward-related entry.
        """
        cycle = data.get("cycle", default_cycle)

        # Extract reward components from various possible field names
        total_reward = data.get(
            "total_reward",
            data.get("avg_reward", data.get("reward", 0.0)),
        )

        success_reward = data.get(
            "success_reward",
            data.get("success_weight", 0.0),
        )

        progress_reward = data.get(
            "progress_reward",
            data.get("progress_weight", 0.0),
        )

        step_penalty = data.get(
            "step_penalty",
            data.get("step_cost", 0.0),
        )

        shaping_reward = data.get(
            "shaping_reward",
            data.get("shaping_weight", 0.0),
        )

        episode_count = data.get("episode_count", data.get("episodes", 0))

        return RewardSnapshot(
            cycle=cycle,
            total_reward=float(total_reward),
            success_reward=float(success_reward),
            progress_reward=float(progress_reward),
            step_penalty=float(step_penalty),
            shaping_reward=float(shaping_reward),
            episode_count=int(episode_count),
        )

    def get_reward_breakdown(
        self, run_name: Optional[str] = None
    ) -> Dict[str, List[RewardSnapshot]]:
        """Get reward breakdown for specified run or all runs.

        Args:
            run_name: Optional run name to filter by.

        Returns:
            Dict mapping run names to reward snapshots.
        """
        if run_name:
            return {run_name: self.runs.get(run_name, [])}
        return dict(self.runs)

    def print_summary(self, run_name: Optional[str] = None) -> str:
        """Print a formatted summary of reward data.

        Args:
            run_name: Optional run name to filter by.

        Returns:
            Formatted summary string.
        """
        runs = self.get_reward_breakdown(run_name)
        if not runs:
            return "No reward data available."

        lines = []
        lines.append("=" * 70)
        lines.append("REWARD FUNCTION ANALYSIS")
        lines.append("=" * 70)

        for name, snapshots in runs.items():
            if not snapshots:
                continue

            lines.append(f"\nRun: {name}")
            lines.append("-" * 50)
            lines.append(
                f"{'Cycle':>6}  {'Total':>10}  {'Success':>10}  "
                f"{'Progress':>10}  {'StepPen':>10}  {'Episodes':>8}"
            )
            lines.append("-" * 50)

            for snap in snapshots:
                lines.append(
                    f"{snap.cycle:>6}  {snap.total_reward:>10.2f}  "
                    f"{snap.success_reward:>10.2f}  {snap.progress_reward:>10.2f}  "
                    f"{snap.step_penalty:>10.2f}  {snap.episode_count:>8}"
                )

            # Summary statistics
            if snapshots:
                total_rewards = [s.total_reward for s in snapshots]
                lines.append("-" * 50)
                lines.append(
                    f"  Avg total reward: {sum(total_rewards) / len(total_rewards):.2f}"
                )
                lines.append(
                    f"  Max total reward: {max(total_rewards):.2f}"
                )
                lines.append(
                    f"  Min total reward: {min(total_rewards):.2f}"
                )

                # Trend analysis
                if len(snapshots) >= 2:
                    first_half = total_rewards[: len(total_rewards) // 2]
                    second_half = total_rewards[len(total_rewards) // 2 :]
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    trend = "improving" if second_avg > first_avg else "declining" if second_avg < first_avg else "stable"
                    lines.append(f"  Trend: {trend} ({first_avg:.2f} -> {second_avg:.2f})")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def print_comparison(self) -> str:
        """Print a comparison of multiple runs.

        Returns:
            Formatted comparison string.
        """
        if len(self.runs) < 2:
            return self.print_summary()

        lines = []
        lines.append("=" * 70)
        lines.append("REWARD COMPARISON ACROSS RUNS")
        lines.append("=" * 70)

        # Collect stats per run
        stats = {}
        for name, snapshots in self.runs.items():
            if not snapshots:
                continue
            total_rewards = [s.total_reward for s in snapshots]
            stats[name] = {
                "avg": sum(total_rewards) / len(total_rewards),
                "max": max(total_rewards),
                "min": min(total_rewards),
                "cycles": len(snapshots),
                "final": total_rewards[-1] if total_rewards else 0,
            }

        if not stats:
            return "No data to compare."

        lines.append(f"\n{'Run':<25}  {'Avg':>8}  {'Max':>8}  {'Min':>8}  {'Final':>8}  {'Cycles':>7}")
        lines.append("-" * 70)

        for name, st in sorted(stats.items(), key=lambda x: x[1]["avg"], reverse=True):
            lines.append(
                f"{name:<25}  {st['avg']:>8.2f}  {st['max']:>8.2f}  "
                f"{st['min']:>8.2f}  {st['final']:>8.2f}  {st['cycles']:>7}"
            )

        # Find best run
        best_run = max(stats.keys(), key=lambda k: stats[k]["avg"])
        lines.append("-" * 70)
        lines.append(f"\nBest run by avg reward: {best_run} ({stats[best_run]['avg']:.2f})")
        lines.append("=" * 70)

        return "\n".join(lines)

    def get_components_over_time(
        self, run_name: str
    ) -> Dict[str, List[float]]:
        """Get reward components as time series for plotting.

        Args:
            run_name: Run name to extract data from.

        Returns:
            Dict mapping component names to lists of values over cycles.
        """
        snapshots = self.runs.get(run_name, [])
        if not snapshots:
            return {}

        return {
            "total_reward": [s.total_reward for s in snapshots],
            "success_reward": [s.success_reward for s in snapshots],
            "progress_reward": [s.progress_reward for s in snapshots],
            "step_penalty": [s.step_penalty for s in snapshots],
            "shaping_reward": [s.shaping_reward for s in snapshots],
            "cycles": [s.cycle for s in snapshots],
        }
