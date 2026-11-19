"""Nearest-neighbor agent for baseline comparison.

A non-parametric k-NN agent that stores (state, action, reward) tuples
and retrieves the k most similar past states to select actions with
the highest average reward. No training required—just experience accumulation.

This provides a simple baseline for comparing against learned agents like
PPO, GRPO, RNN, and Transformer agents.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from envgen.agents.base import Agent
from envgen.utils.types import Action, Observation, Transition

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single stored experience tuple.

    Attributes:
        state: Flattened observation as a list of floats.
        action_index: Integer action index.
        reward: Cumulative reward received for this action.
    """

    state: List[float]
    action_index: int
    reward: float


class NearestNeighborAgent(Agent):
    """k-Nearest Neighbors agent for baseline comparison.

    Stores flattened observations with actions and cumulative rewards.
    At inference, computes distances to stored experiences and selects
    the action with the best average reward among k nearest neighbors.

    Args:
        name: Agent name for identification.
        config: Optional configuration dict with keys:
            - k (int): Number of neighbors (default: 5)
            - distance_metric (str): 'euclidean' or 'cosine' (default: 'euclidean')
            - max_memory (int): Maximum experiences to store (default: 10000)
            - exploration_rate (float): Random action probability when memory is small (default: 0.3)
    """

    def __init__(self, name: str = "nn_agent", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        cfg = config or {}
        self.k = cfg.get("k", 5)
        self.distance_metric = cfg.get("distance_metric", "euclidean")
        self.max_memory = cfg.get("max_memory", 10000)
        self.exploration_rate = cfg.get("exploration_rate", 0.3)

        self.memory: List[Experience] = []
        self._action_space_size: Optional[int] = None
        self._obs_dim: Optional[int] = None
        self._episodes_trained = 0

    def _flatten_observation(self, observation: Observation) -> List[float]:
        """Flatten an observation dict into a list of floats.

        Args:
            observation: Observation dict from the environment.

        Returns:
            List of float values.
        """
        values = []
        for key in sorted(observation.keys()):
            val = observation[key]
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, (int, float)):
                        values.append(float(v))
            elif isinstance(val, str):
                # Simple hash-based encoding for strings
                values.append(hash(val) % 1000 / 1000.0)
            else:
                values.append(0.0)

        # Pad to consistent dimension
        if self._obs_dim is None and values:
            self._obs_dim = len(values)
        elif self._obs_dim and values:
            while len(values) < self._obs_dim:
                values.append(0.0)
            values = values[: self._obs_dim]

        return values if values else [0.0]

    def _compute_distance(self, obs1: List[float], obs2: List[float]) -> float:
        """Compute distance between two flattened observations.

        Args:
            obs1: First flattened observation.
            obs2: Second flattened observation.

        Returns:
            Distance value.
        """
        if self.distance_metric == "cosine":
            dot = sum(a * b for a, b in zip(obs1, obs2))
            mag1 = sum(a * a for a in obs1) ** 0.5
            mag2 = sum(b * b for b in obs2) ** 0.5
            if mag1 == 0 or mag2 == 0:
                return 1.0
            cosine_sim = dot / (mag1 * mag2)
            return 1.0 - cosine_sim
        else:  # euclidean (default)
            # Pad shorter list
            max_len = max(len(obs1), len(obs2))
            while len(obs1) < max_len:
                obs1.append(0.0)
            while len(obs2) < max_len:
                obs2.append(0.0)
            return sum((a - b) ** 2 for a, b in zip(obs1, obs2)) ** 0.5

    def _find_nearest_neighbors(
        self, query: List[float]
    ) -> List[Tuple[int, float, float]]:
        """Find k nearest neighbors in memory.

        Args:
            query: Flattened query observation.

        Returns:
            List of (action_index, reward, distance) tuples sorted by distance.
        """
        if not self.memory:
            return []

        distances = []
        for exp in self.memory:
            dist = self._compute_distance(query, exp.state)
            distances.append((exp.action_index, exp.reward, dist))

        distances.sort(key=lambda x: x[2])
        return distances[: self.k]

    def select_action(self, observation: Observation) -> Action:
        """Select an action based on nearest neighbor experience.

        If memory is too small, falls back to random exploration.

        Args:
            observation: Current observation from the environment (dict or Observation).

        Returns:
            Action to take.
        """
        obs_dict = observation if isinstance(observation, dict) else observation.state
        flattened = self._flatten_observation(obs_dict)

        # Explore when memory is small
        min_memory_for_exploitation = max(self.k, 3)
        if len(self.memory) < min_memory_for_exploitation or random.random() < self.exploration_rate:
            if self._action_space_size:
                action_idx = random.randint(0, self._action_space_size - 1)
            else:
                action_idx = 0
            return Action(action_type=f"action_{action_idx}", parameters={"action_index": action_idx})

        # Exploit: find nearest neighbors and select best action
        neighbors = self._find_nearest_neighbors(flattened)
        if not neighbors:
            return Action(action_type="action_0", parameters={"action_index": 0})

        # Group by action and compute average reward
        action_rewards: Dict[int, List[float]] = {}
        for action_idx, reward, _dist in neighbors:
            action_rewards.setdefault(action_idx, []).append(reward)

        # Select action with highest average reward
        best_action = max(action_rewards.keys(), key=lambda a: sum(action_rewards[a]) / len(action_rewards[a]))

        return Action(action_type=f"action_{best_action}", parameters={"action_index": best_action})

    def train(self, episodes: List[Any]) -> Dict[str, Any]:
        """Train by storing experiences from episodes.

        Extracts (state, action, reward) tuples from episodes and adds
        them to the memory buffer.

        Args:
            episodes: List of Episode objects with transitions.

        Returns:
            Training metrics dict.
        """
        experiences_added = 0
        for episode in episodes:
            if not hasattr(episode, "transitions"):
                continue

            for transition in episode.transitions:
                # Extract state, action, and cumulative reward
                obs = getattr(transition, "observation", {})
                action = getattr(transition, "action", None)
                reward = getattr(transition, "reward", 0.0)

                if action is None:
                    continue

                # Get action index
                if isinstance(action, dict):
                    action_idx = action.get("payload", {}).get("action_index", 0)
                elif hasattr(action, "payload"):
                    action_idx = getattr(action.payload, "action_index", 0) if hasattr(action, "payload") else 0
                else:
                    action_idx = 0

                # Store experience
                flattened = self._flatten_observation(obs if isinstance(obs, dict) else {})
                exp = Experience(
                    state=flattened,
                    action_index=int(action_idx),
                    reward=float(reward),
                )

                self.memory.append(exp)
                experiences_added += 1

                # Evict oldest if over capacity
                if len(self.memory) > self.max_memory:
                    self.memory.pop(0)

                # Track action space size
                if self._action_space_size is None:
                    self._action_space_size = max(1, action_idx + 1)

        self._episodes_trained += len(episodes)

        return {
            "experiences_added": experiences_added,
            "memory_size": len(self.memory),
            "episodes_trained": self._episodes_trained,
        }

    def save(self, path: str) -> None:
        """Save memory buffer to JSON file.

        Args:
            path: File path to save to.
        """
        data = {
            "memory": [
                {"state": exp.state, "action_index": exp.action_index, "reward": exp.reward}
                for exp in self.memory
            ],
            "config": {
                "k": self.k,
                "distance_metric": self.distance_metric,
                "max_memory": self.max_memory,
                "exploration_rate": self.exploration_rate,
            },
            "action_space_size": self._action_space_size,
            "obs_dim": self._obs_dim,
            "episodes_trained": self._episodes_trained,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved NN agent memory ({len(self.memory)} experiences) to {path}")

    def load(self, path: str) -> None:
        """Load memory buffer from JSON file.

        Args:
            path: File path to load from.
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.memory = [
            Experience(
                state=exp["state"],
                action_index=exp["action_index"],
                reward=exp["reward"],
            )
            for exp in data.get("memory", [])
        ]

        cfg = data.get("config", {})
        self.k = cfg.get("k", self.k)
        self.distance_metric = cfg.get("distance_metric", self.distance_metric)
        self.max_memory = cfg.get("max_memory", self.max_memory)
        self.exploration_rate = cfg.get("exploration_rate", self.exploration_rate)

        self._action_space_size = data.get("action_space_size")
        self._obs_dim = data.get("obs_dim")
        self._episodes_trained = data.get("episodes_trained", 0)

        logger.info(f"Loaded NN agent memory ({len(self.memory)} experiences) from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics.

        Returns:
            Dict with agent stats.
        """
        return {
            "name": self.name,
            "memory_size": len(self.memory),
            "k": self.k,
            "distance_metric": self.distance_metric,
            "episodes_trained": self._episodes_trained,
            "action_space_size": self._action_space_size,
        }
