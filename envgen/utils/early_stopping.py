"""Early stopping monitor for training pipelines.

This module provides a monitor that tracks validation performance metrics
and automatically signals when training should stop because performance
has plateaued or is degrading. This prevents wasting compute on training
cycles that no longer improve the agent.

Usage:
    from envgen.utils.early_stopping import EarlyStoppingMonitor

    monitor = EarlyStoppingMonitor(
        patience=3,
        metric="mean_reward",
        mode="max",
        min_delta=0.01,
    )

    for cycle in range(max_cycles):
        # ... training ...
        should_stop = monitor.step(mean_reward=current_reward)
        if should_stop:
            print(f"Early stopping at cycle {cycle}")
            break
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class EarlyStoppingMonitor:
    """Monitors training metrics and signals when to stop early.

    Tracks a specified metric over training cycles and triggers early
    stopping when the metric hasn't improved for a configurable number
    of cycles (patience).

    Attributes:
        patience: Number of cycles without improvement before stopping.
        metric: Name of the metric to monitor.
        mode: "max" to maximize, "min" to minimize the metric.
        min_delta: Minimum change to qualify as an improvement.
        best_value: Best value seen so far for the monitored metric.
        stops_at: The cycle number at which early stopping will trigger.
    """

    def __init__(
        self,
        patience: int = 3,
        metric: str = "mean_reward",
        mode: str = "max",
        min_delta: float = 0.01,
    ):
        """Initialize the early stopping monitor.

        Args:
            patience: Number of cycles without improvement before stopping.
            metric: Name of the metric to monitor (e.g., "mean_reward",
                "navigation_score").
            mode: "max" to maximize the metric, "min" to minimize it.
            min_delta: Minimum absolute change to qualify as improvement.
                Changes smaller than this are ignored.

        Raises:
            ValueError: If mode is not "max" or "min".
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self._history: List[float] = []
        self._no_improve_count: int = 0
        self._best: Optional[float] = None
        self._cycle: int = 0

    @property
    def best_value(self) -> Optional[float]:
        """Best value seen so far for the monitored metric."""
        return self._best

    @property
    def stops_at(self) -> int:
        """Cycle number at which early stopping will trigger if not improved."""
        return self._cycle + (self.patience - self._no_improve_count)

    @property
    def history(self) -> List[float]:
        """History of all metric values recorded."""
        return self._history.copy()

    @property
    def no_improve_count(self) -> int:
        """Number of cycles since last improvement."""
        return self._no_improve_count

    @property
    def cycle(self) -> int:
        """Current cycle number."""
        return self._cycle

    def step(self, **metrics: float) -> bool:
        """Record a metric value and check if training should stop.

        Args:
            **metrics: Keyword arguments mapping metric names to values.
                The metric matching self.metric will be monitored.

        Returns:
            True if training should stop (patience exhausted), False otherwise.
        """
        value = metrics.get(self.metric)
        if value is None:
            # If the monitored metric isn't provided, don't trigger
            self._cycle += 1
            return False

        value = float(value)
        self._history.append(value)
        self._cycle += 1

        # Check if this is the first value
        if self._best is None:
            self._best = value
            self._no_improve_count = 0
            return False

        # Check if this is an improvement
        improved = self._is_improvement(value)

        if improved:
            self._best = value
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        # Check if patience is exhausted
        return self._no_improve_count >= self.patience

    def _is_improvement(self, value: float) -> bool:
        """Check if a new value is an improvement over the best.

        Args:
            value: The new metric value.

        Returns:
            True if the value is an improvement.
        """
        if self._best is None:
            return True

        if self.mode == "max":
            return value > self._best + self.min_delta
        else:  # mode == "min"
            return value < self._best - self.min_delta

    def reset(self):
        """Reset the monitor state. Useful for restarting training."""
        self._history.clear()
        self._no_improve_count = 0
        self._best = None
        self._cycle = 0

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the monitor.

        Returns:
            Dictionary with current monitor state.
        """
        return {
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "best_value": self._best,
            "no_improve_count": self._no_improve_count,
            "cycle": self._cycle,
            "history_length": len(self._history),
        }

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"EarlyStoppingMonitor(metric={status['metric']!r}, "
            f"mode={status['mode']!r}, patience={status['patience']}, "
            f"best={status['best_value']}, "
            f"no_improve={status['no_improve_count']}/{status['patience']})"
        )
