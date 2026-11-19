"""Multi-environment curriculum manager for rotating through environments.

This module provides a curriculum manager that rotates through multiple
environments during training. It supports different rotation strategies:
- round_robin: Cycle through environments in order
- skill_based: Focus on the environment where the agent performs worst
- random: Randomly select the next environment

Usage:
    from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

    manager = MultiEnvCurriculumManager(
        env_types=["puzzle", "maze", "resource"],
        rotation_strategy="round_robin",
    )
    next_env = manager.get_next_environment()
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiEnvCurriculumManager:
    """Manages rotation through multiple environments during training.

    Supports different rotation strategies for selecting which environment
    to train on next. Agent state and policy are shared across environments
    for generalist learning.

    Attributes:
        env_types: List of environment type strings (e.g., ["puzzle", "maze"]).
        rotation_strategy: Strategy for selecting next environment.
        current_index: Current position in round-robin rotation.
        skill_scores: Per-environment skill scores for skill-based rotation.
    """

    def __init__(
        self,
        env_types: List[str],
        rotation_strategy: str = "round_robin",
        episodes_per_env: Optional[int] = None,
    ):
        """Initialize the multi-environment curriculum manager.

        Args:
            env_types: List of environment type strings.
            rotation_strategy: One of "round_robin", "skill_based", "random".
            episodes_per_env: Fixed number of episodes per environment per cycle.
                If None, episodes are distributed evenly.
        """
        self.env_types = list(env_types)
        self.rotation_strategy = rotation_strategy
        self.episodes_per_env = episodes_per_env
        self.current_index = 0
        self.skill_scores: Dict[str, float] = {env: 0.0 for env in self.env_types}
        self.cycle_count = 0
        self._history: List[Dict[str, Any]] = []

    def get_next_environment(self) -> str:
        """Get the next environment type to train on.

        Returns:
            Environment type string.
        """
        if self.rotation_strategy == "round_robin":
            env = self.env_types[self.current_index % len(self.env_types)]
            self.current_index += 1
            return env
        elif self.rotation_strategy == "skill_based":
            # Pick the environment with the lowest skill score
            if self.skill_scores:
                env = min(self.skill_scores, key=self.skill_scores.get)
                return env
            return self.env_types[0]
        elif self.rotation_strategy == "random":
            return random.choice(self.env_types)
        else:
            raise ValueError(f"Unknown rotation strategy: {self.rotation_strategy}")

    def update_skill_scores(self, scores: Dict[str, float]) -> None:
        """Update skill scores for environments.

        Args:
            scores: Dictionary mapping environment types to skill scores.
        """
        for env_type, score in scores.items():
            if env_type in self.skill_scores:
                self.skill_scores[env_type] = score
        logger.debug(f"Updated skill scores: {self.skill_scores}")

    def distribute_episodes(self, total_episodes: int) -> Dict[str, int]:
        """Distribute episodes across environments.

        Args:
            total_episodes: Total number of episodes to distribute.

        Returns:
            Dictionary mapping environment types to number of episodes.
        """
        if self.episodes_per_env:
            return {env: self.episodes_per_env for env in self.env_types}

        num_envs = len(self.env_types)
        base = total_episodes // num_envs
        remainder = total_episodes % num_envs

        distribution = {}
        for i, env in enumerate(self.env_types):
            distribution[env] = base + (1 if i < remainder else 0)

        return distribution

    def advance_cycle(self) -> None:
        """Advance to the next training cycle."""
        self.cycle_count += 1
        if self.rotation_strategy == "round_robin":
            self.current_index = self.cycle_count % len(self.env_types)

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the curriculum.

        Returns:
            Dictionary with curriculum status information.
        """
        return {
            "env_types": self.env_types,
            "rotation_strategy": self.rotation_strategy,
            "current_index": self.current_index,
            "skill_scores": dict(self.skill_scores),
            "cycle_count": self.cycle_count,
        }

    def get_weakest_environment(self) -> Optional[str]:
        """Get the environment where the agent performs worst.

        Returns:
            Environment type string or None if no scores available.
        """
        if not self.skill_scores:
            return None
        return min(self.skill_scores, key=self.skill_scores.get)

    def get_strongest_environment(self) -> Optional[str]:
        """Get the environment where the agent performs best.

        Returns:
            Environment type string or None if no scores available.
        """
        if not self.skill_scores:
            return None
        return max(self.skill_scores, key=self.skill_scores.get)

    def log_cycle(self, cycle_results: Dict[str, Dict[str, Any]]) -> None:
        """Log the results of a training cycle.

        Args:
            cycle_results: Dictionary mapping environment types to their
                cycle results (skill scores, rewards, etc.).
        """
        entry = {
            "cycle": self.cycle_count,
            "results": cycle_results,
        }
        self._history.append(entry)

        # Update skill scores from results
        for env_type, result in cycle_results.items():
            if "mean_skill" in result:
                self.skill_scores[env_type] = result["mean_skill"]
