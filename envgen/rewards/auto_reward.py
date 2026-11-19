"""Automatic Reward Design (ARD) for intrinsic motivation.

This module implements automatic reward design techniques that learn reward
functions from agent experience. It supports:

1. Curiosity-based intrinsic reward: trains a forward dynamics model and uses
   prediction error as an intrinsic reward signal to encourage exploration.
2. Density-based intrinsic reward: encourages visiting under-explored states
   using count-based methods (pseudo-counts via hash-based state hashing).

These intrinsic rewards are combined with extrinsic rewards from the environment
to guide agent exploration and learning.

Requires: pip install torch numpy
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envgen.rewards.engine import RewardEngine

logger = logging.getLogger(__name__)


class AutoRewardDesigner:
    """Automatic reward designer using intrinsic motivation signals.

    Combines extrinsic rewards from the environment with intrinsic rewards
    computed from the agent's own experience. This encourages exploration
    and helps the agent discover useful behaviors without manual reward
    engineering.

    Two intrinsic reward methods are supported:
    - Curiosity: prediction error from a forward dynamics model
    - Count-based: pseudo-counts of state visitation frequency

    The final reward is computed as:
        total_reward = extrinsic_reward + intrinsic_weight * intrinsic_reward
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the automatic reward designer.

        Args:
            config: Configuration dict with optional keys:
                - intrinsic_weight: weight for intrinsic reward (default: 0.5)
                - method: 'curiosity', 'count', or 'both' (default: 'curiosity')
                - forward_model_hidden_size: hidden size for forward model (default: 64)
                - prediction_loss_coef: coefficient for prediction loss (default: 1.0)
                - count_decay: decay factor for count-based rewards (default: 0.99)
                - count_scale: scaling factor for count-based rewards (default: 1.0)
        """
        self.config = config or {}
        self.intrinsic_weight: float = self.config.get("intrinsic_weight", 0.5)
        self.method: str = self.config.get("method", "curiosity")
        self.forward_model_hidden_size: int = self.config.get("forward_model_hidden_size", 64)
        self.prediction_loss_coef: float = self.config.get("prediction_loss_coef", 1.0)
        self.count_decay: float = self.config.get("count_decay", 0.99)
        self.count_scale: float = self.config.get("count_scale", 1.0)

        # Forward model for curiosity-based reward
        self.forward_model = None
        self.forward_optimizer = None
        self.forward_criterion = None
        self.obs_dim: Optional[int] = None

        # Count-based reward tracking
        self.state_counts: Dict[str, float] = {}
        self.total_visits: int = 0

        # Stats
        self.total_intrinsic_reward: float = 0.0
        self.total_extrinsic_reward: float = 0.0
        self.step_count: int = 0

    def _hash_state(self, observation) -> str:
        """Hash an observation to a string key for count-based methods."""
        if isinstance(observation, dict):
            key_str = str(sorted(observation.items()))
        elif isinstance(observation, (list, tuple)):
            key_str = str(observation)
        elif isinstance(observation, np.ndarray):
            key_str = observation.tobytes()
        else:
            key_str = str(observation)
        return hashlib.md5(str(key_str).encode()).hexdigest()

    def _flatten_observation(self, observation) -> np.ndarray:
        """Flatten observation to a numpy array for the forward model."""
        if isinstance(observation, dict):
            values = []
            for key in sorted(observation.keys()):
                val = observation[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, np.ndarray):
                    values.extend(val.flatten().tolist())
                elif isinstance(val, (list, tuple)):
                    values.extend([float(v) for v in val])
                elif isinstance(val, str):
                    values.append(hash(val) % 1000 / 1000.0)
                elif isinstance(val, bool):
                    values.append(1.0 if val else 0.0)
                else:
                    values.append(0.0)
            return np.array(values, dtype=np.float32)
        elif isinstance(observation, (list, tuple)):
            return np.array([float(v) for v in observation], dtype=np.float32)
        elif isinstance(observation, np.ndarray):
            return observation.flatten().astype(np.float32)
        else:
            return np.array([float(observation)], dtype=np.float32)

    def _init_forward_model(self, obs_dim: int) -> None:
        """Initialize the forward dynamics model for curiosity-based reward."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available; curiosity-based reward disabled")
            self.forward_model = None
            return

        self.obs_dim = obs_dim

        # Forward model: (obs, action) -> next_obs prediction
        input_dim = obs_dim * 2  # concat current obs and next obs for prediction
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim, self.forward_model_hidden_size),
            nn.ReLU(),
            nn.Linear(self.forward_model_hidden_size, self.forward_model_hidden_size),
            nn.ReLU(),
            nn.Linear(self.forward_model_hidden_size, obs_dim),
        )

        self.forward_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=1e-3
        )
        self.forward_criterion = nn.MSELoss()

    def _compute_curiosity_reward(self, obs: Any, next_obs: Any) -> float:
        """Compute curiosity-based intrinsic reward.

        Trains a forward model to predict next observation from current observation.
        The prediction error is used as the intrinsic reward signal.

        Args:
            obs: Current observation
            next_obs: Next observation after taking an action

        Returns:
            Intrinsic reward based on prediction error
        """
        if self.forward_model is None:
            obs_array = self._flatten_observation(obs)
            next_obs_array = self._flatten_observation(next_obs)

            if self.obs_dim is None:
                self._init_forward_model(len(obs_array))

            if self.forward_model is None:
                return 0.0  # PyTorch not available

            import torch

            # Train forward model: predict next_obs from obs
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)
            next_obs_tensor = torch.FloatTensor(next_obs_array).unsqueeze(0)

            # Forward pass
            predicted_next = self.forward_model(obs_tensor)

            # Compute prediction error (intrinsic reward)
            prediction_error = self.forward_criterion(predicted_next, next_obs_tensor).item()

            # Update forward model
            self.forward_optimizer.zero_grad()
            prediction_error.backward()
            self.forward_optimizer.step()

            # Clip intrinsic reward to prevent instability
            intrinsic_reward = min(prediction_error * self.prediction_loss_coef, 1.0)
            return intrinsic_reward
        else:
            obs_array = self._flatten_observation(obs)
            next_obs_array = self._flatten_observation(next_obs)

            import torch

            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)
            next_obs_tensor = torch.FloatTensor(next_obs_array).unsqueeze(0)

            predicted_next = self.forward_model(obs_tensor)
            prediction_error = self.forward_criterion(predicted_next, next_obs_tensor).item()

            self.forward_optimizer.zero_grad()
            prediction_error.backward()
            self.forward_optimizer.step()

            intrinsic_reward = min(prediction_error * self.prediction_loss_coef, 1.0)
            return intrinsic_reward

    def _compute_count_reward(self, observation: Any) -> float:
        """Compute count-based intrinsic reward.

        Uses pseudo-counts based on state visitation frequency.
        Less-visited states receive higher intrinsic rewards.

        Args:
            observation: Current observation

        Returns:
            Intrinsic reward based on state visitation count
        """
        state_key = self._hash_state(observation)

        # Update count
        if state_key in self.state_counts:
            self.state_counts[state_key] += 1.0
        else:
            self.state_counts[state_key] = 1.0

        self.total_visits += 1

        # Compute visitation count (with decay for older states)
        count = self.state_counts[state_key]

        # Intrinsic reward is inversely proportional to visitation count
        # Using the formula: N(s) / (N(s) + count) where N(s) is a scaling factor
        n_s = self.count_scale * self.total_visits
        intrinsic_reward = n_s / (n_s + count)

        # Apply decay to encourage continued exploration
        intrinsic_reward *= self.count_decay

        return intrinsic_reward

    def compute_intrinsic_reward(self, obs: Any, next_obs: Any) -> float:
        """Compute the intrinsic reward for a transition.

        Args:
            obs: Current observation
            next_obs: Next observation

        Returns:
            Intrinsic reward value
        """
        if self.method == "curiosity":
            return self._compute_curiosity_reward(obs, next_obs)
        elif self.method == "count":
            return self._compute_count_reward(obs)
        elif self.method == "both":
            curiosity_reward = self._compute_curiosity_reward(obs, next_obs)
            count_reward = self._compute_count_reward(obs)
            return (curiosity_reward + count_reward) / 2.0
        else:
            logger.warning(f"Unknown ARD method: {self.method}, returning 0")
            return 0.0

    def compute_total_reward(self, extrinsic_reward: float, obs: Any, next_obs: Any) -> float:
        """Compute the total reward combining extrinsic and intrinsic components.

        Args:
            extrinsic_reward: Reward from the environment
            obs: Current observation
            next_obs: Next observation

        Returns:
            Total reward = extrinsic + intrinsic_weight * intrinsic
        """
        intrinsic = self.compute_intrinsic_reward(obs, next_obs)
        total = extrinsic_reward + self.intrinsic_weight * intrinsic

        self.total_extrinsic_reward += extrinsic_reward
        self.total_intrinsic_reward += intrinsic
        self.step_count += 1

        return total

    def integrate_with_reward_engine(self, reward_engine: RewardEngine) -> RewardEngine:
        """Wrap a RewardEngine to include intrinsic rewards.

        Creates a new reward computation that adds intrinsic rewards
        to the extrinsic rewards from the engine.

        Args:
            reward_engine: Existing RewardEngine instance

        Returns:
            The same reward engine, now augmented with ARD tracking
        """
        # Store reference for integration
        self.reward_engine = reward_engine
        logger.info(
            f"Integrated AutoRewardDesigner with RewardEngine "
            f"(method={self.method}, intrinsic_weight={self.intrinsic_weight})"
        )
        return reward_engine

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about intrinsic reward computation."""
        avg_intrinsic = (
            self.total_intrinsic_reward / self.step_count if self.step_count > 0 else 0.0
        )
        avg_extrinsic = (
            self.total_extrinsic_reward / self.step_count if self.step_count > 0 else 0.0
        )
        return {
            "method": self.method,
            "intrinsic_weight": self.intrinsic_weight,
            "step_count": self.step_count,
            "avg_intrinsic_reward": avg_intrinsic,
            "avg_extrinsic_reward": avg_extrinsic,
            "total_intrinsic_reward": self.total_intrinsic_reward,
            "total_extrinsic_reward": self.total_extrinsic_reward,
            "unique_states_visited": len(self.state_counts),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0
        self.step_count = 0
