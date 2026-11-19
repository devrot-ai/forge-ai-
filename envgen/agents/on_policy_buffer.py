"""On-policy experience buffer for trajectory storage and mini-batch sampling.

This buffer stores complete trajectories (sequences of transitions) from
on-policy interactions and supports mini-batch sampling for PPO/GRPO
training. Unlike off-policy replay buffers, on-policy data is discarded
after each training epoch since the policy has changed.

Example:
    buffer = OnPolicyBuffer(max_size=4096)
    # Collect a trajectory
    for transition in trajectory:
        buffer.add(transition)
    buffer.complete_trajectory()
    # Sample mini-batches for training
    for batch in buffer.sample_batches(batch_size=64):
        train_on(batch)
    # Clear after training epochs
    buffer.clear()
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np


class OnPolicyBuffer:
    """Buffer for storing on-policy trajectories and sampling mini-batches.

    Stores transitions grouped by trajectory. Each trajectory represents
    a complete episode or rollout. After collecting enough trajectories,
    the buffer supports epoch-based mini-batch sampling where all data
    is seen before the buffer is cleared.

    Args:
        max_size: Maximum number of transitions to store.
        batch_size: Default batch size for sampling (used by get_all).
    """

    def __init__(self, max_size: int = 4096, batch_size: int = 64):
        """Initialize the on-policy buffer.

        Args:
            max_size: Maximum number of transitions to store.
            batch_size: Default batch size for sampling.
        """
        self.max_size = max_size
        self.batch_size = batch_size

        # Storage: list of trajectories, each trajectory is a list of transitions
        self.trajectories: List[List[Dict[str, Any]]] = []
        # Flat list of all transitions for sampling
        self.transitions: List[Dict[str, Any]] = []
        self._total_size = 0

    def add(self, transition: Dict[str, Any]) -> None:
        """Add a single transition to the current trajectory.

        Args:
            transition: Dict containing state, action, reward, next_state, done.
        """
        if not self.trajectories:
            self.trajectories.append([])

        self.trajectories[-1].append(transition)
        self.transitions.append(transition)
        self._total_size += 1

    def complete_trajectory(self) -> None:
        """Mark the current trajectory as complete and start a new one."""
        if self.trajectories and self.trajectories[-1]:
            self.trajectories.append([])
        elif not self.trajectories:
            self.trajectories.append([])

    def clear(self) -> None:
        """Clear all stored data. Called after training epochs are complete."""
        self.trajectories = []
        self.transitions = []
        self._total_size = 0

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._total_size

    def num_trajectories(self) -> int:
        """Return the number of complete trajectories."""
        return len([t for t in self.trajectories if t])

    def sample_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample. Defaults to self.batch_size.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones, log_probs.
            Each value is a numpy array of shape (batch_size, ...).
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self._total_size == 0:
            return {
                "states": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "next_states": np.array([]),
                "dones": np.array([]),
                "log_probs": np.array([]),
            }

        actual_batch_size = min(batch_size, self._total_size)
        indices = random.sample(range(self._total_size), actual_batch_size)

        batch = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
        }

        for idx in indices:
            t = self.transitions[idx]
            batch["states"].append(t.get("state", {}))
            batch["actions"].append(t.get("action", {}))
            batch["rewards"].append(t.get("reward", 0.0))
            batch["next_states"].append(t.get("next_state", {}))
            batch["dones"].append(t.get("done", False))
            batch["log_probs"].append(t.get("log_prob", 0.0))

        # Convert to numpy arrays where possible
        for key in batch:
            try:
                batch[key] = np.array(batch[key])
            except (TypeError, ValueError):
                pass

        return batch

    def sample_batches(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate all mini-batches from the stored transitions.

        Yields batches until all transitions are covered once (one epoch).

        Args:
            batch_size: Number of transitions per batch.

        Returns:
            List of batch dicts covering all transitions.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self._total_size == 0:
            return []

        # Shuffle indices
        indices = list(range(self._total_size))
        random.shuffle(indices)

        batches = []
        for start in range(0, self._total_size, batch_size):
            end = min(start + batch_size, self._total_size)
            batch_indices = indices[start:end]

            batch = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "log_probs": [],
            }

            for idx in batch_indices:
                t = self.transitions[idx]
                batch["states"].append(t.get("state", {}))
                batch["actions"].append(t.get("action", {}))
                batch["rewards"].append(t.get("reward", 0.0))
                batch["next_states"].append(t.get("next_state", {}))
                batch["dones"].append(t.get("done", False))
                batch["log_probs"].append(t.get("log_prob", 0.0))

            # Convert to numpy arrays where possible
            for key in batch:
                try:
                    batch[key] = np.array(batch[key])
                except (TypeError, ValueError):
                    pass

            batches.append(batch)

        return batches

    def get_all(self) -> Dict[str, Any]:
        """Return all stored transitions as a single batch.

        Returns:
            Dict with all transitions organized by field.
        """
        if self._total_size == 0:
            return {
                "states": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "next_states": np.array([]),
                "dones": np.array([]),
                "log_probs": np.array([]),
            }

        batch = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
        }

        for t in self.transitions:
            batch["states"].append(t.get("state", {}))
            batch["actions"].append(t.get("action", {}))
            batch["rewards"].append(t.get("reward", 0.0))
            batch["next_states"].append(t.get("next_state", {}))
            batch["dones"].append(t.get("done", False))
            batch["log_probs"].append(t.get("log_prob", 0.0))

        for key in batch:
            try:
                batch[key] = np.array(batch[key])
            except (TypeError, ValueError):
                pass

        return batch

    def compute_returns(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        values: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute returns and advantages for all transitions using GAE.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            values: Optional list of value estimates for each transition.

        Returns:
            List of (return, advantage) tuples for each transition.
        """
        if not self.transitions:
            return []

        rewards = [t.get("reward", 0.0) for t in self.transitions]
        dones = [t.get("done", False) for t in self.transitions]

        if values is None:
            values = [0.0] * len(self.transitions)

        # Compute GAE advantages
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, values)]

        return list(zip(returns, advantages))

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the buffer contents.

        Returns:
            Dict with buffer size, trajectory count, reward stats.
        """
        if not self.transitions:
            return {
                "size": 0,
                "num_trajectories": 0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }

        rewards = [t.get("reward", 0.0) for t in self.transitions]
        return {
            "size": self._total_size,
            "num_trajectories": self.num_trajectories(),
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }
