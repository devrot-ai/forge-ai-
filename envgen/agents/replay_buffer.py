"""Experience replay buffer for storing and sampling past transitions.

This module provides a replay buffer that stores past transitions and
supports both uniform and prioritized sampling. It enables off-policy
learning by allowing agents to learn from experience collected in the
past, improving sample efficiency.

Usage:
    from envgen.agents.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=10000)

    # Add transitions
    buffer.add(transition)

    # Sample a batch
    batch = buffer.sample(batch_size=64)

    # Update priorities after training
    for i, td_error in enumerate(td_errors):
        buffer.update_priority(i, abs(td_error) + 1e-5)
"""

from __future__ import annotations

import random
import threading
from typing import Any, Dict, List, Optional, Tuple

from envgen.utils.types import Transition


class ReplayBuffer:
    """Stores and samples past transitions for experience replay.

    Supports both uniform random sampling and prioritized sampling
    based on transition importance (typically TD error magnitude).

    Attributes:
        capacity: Maximum number of transitions to store.
        size: Current number of transitions in the buffer.
        epsilon: Minimum priority value to ensure all transitions
            have non-zero sampling probability.
        alpha: Priority exponent for prioritized sampling (0=uniform, 1=full).
    """

    def __init__(
        self,
        capacity: int = 10000,
        epsilon: float = 1e-5,
        alpha: float = 0.6,
    ):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
                Older transitions are evicted when full.
            epsilon: Minimum priority value to ensure all transitions
                have non-zero sampling probability.
            alpha: Priority exponent for prioritized sampling.
                0.0 = uniform sampling, 1.0 = full prioritized sampling.
                Values between 0 and 1 provide a smooth interpolation.
        """
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha

        self._buffer: List[Transition] = []
        self._priorities: List[float] = []
        self._pos: int = 0  # Current write position
        self._size: int = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Current number of transitions in the buffer."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return self._size >= self.capacity

    def add(self, transition: Transition, priority: Optional[float] = None):
        """Add a transition to the buffer.

        Args:
            transition: The transition to store.
            priority: Optional priority value. If None, uses the
                maximum current priority + epsilon.
        """
        with self._lock:
            if priority is None:
                if self._priorities:
                    priority = max(self._priorities) + self.epsilon
                else:
                    priority = 1.0 + self.epsilon

            # Apply alpha scaling
            priority = priority**self.alpha

            if self._size < self.capacity:
                # Buffer not full yet, append
                self._buffer.append(transition)
                self._priorities.append(priority)
                self._size += 1
            else:
                # Buffer full, overwrite oldest
                self._buffer[self._pos] = transition
                self._priorities[self._pos] = priority

            self._pos = (self._pos + 1) % self.capacity

    def add_batch(self, transitions: List[Transition], priorities: Optional[List[float]] = None):
        """Add multiple transitions to the buffer.

        Args:
            transitions: List of transitions to store.
            priorities: Optional list of priority values. If None,
                uses max priority + epsilon for all.
        """
        for i, transition in enumerate(transitions):
            priority = None
            if priorities is not None and i < len(priorities):
                priority = priorities[i]
            self.add(transition, priority)

    def sample(
        self,
        batch_size: int,
        prioritized: bool = True,
        beta: float = 0.4,
    ) -> Tuple[List[Transition], List[int], List[float]]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            prioritized: Whether to use prioritized sampling.
                If False, uses uniform random sampling.
            beta: Importance sampling weight exponent. Used to compute
                importance weights for correcting the sampling bias.

        Returns:
            Tuple of (transitions, indices, importance_weights).
            Importance weights are used to correct the bias introduced
            by prioritized sampling.

        Raises:
            ValueError: If batch_size is larger than buffer size.
        """
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) > buffer size ({self._size})")

        with self._lock:
            if prioritized and self._priorities:
                # Prioritized sampling
                priorities_array = self._priorities[: self._size]
                total_priority = sum(priorities_array)

                # Compute sampling probabilities
                probabilities = [p / total_priority for p in priorities_array]

                # Sample indices based on probabilities
                indices = random.choices(
                    range(self._size),
                    weights=probabilities,
                    k=batch_size,
                )

                # Compute importance weights
                weights = []
                for idx in indices:
                    prob = probabilities[idx]
                    weight = (self._size * prob) ** (-beta)
                    weight = weight / max(weights) if weights else 1.0
                    weights.append(weight)

                # Normalize weights
                max_weight = max(weights) if weights else 1.0
                weights = [w / max_weight for w in weights]

                transitions = [self._buffer[idx] for idx in indices]
            else:
                # Uniform random sampling
                indices = random.sample(range(self._size), batch_size)
                transitions = [self._buffer[idx] for idx in indices]
                weights = [1.0] * batch_size

            return transitions, indices, weights

    def update_priority(self, index: int, priority: float):
        """Update the priority of a transition.

        Typically called after computing TD errors during training.

        Args:
            index: Index of the transition in the buffer.
            priority: New priority value (e.g., |TD error| + epsilon).
        """
        with self._lock:
            if 0 <= index < self._size:
                self._priorities[index] = priority**self.alpha + self.epsilon

    def update_priorities_batch(self, indices: List[int], priorities: List[float]):
        """Update priorities for multiple transitions.

        Args:
            indices: List of buffer indices.
            priorities: List of new priority values.
        """
        for idx, priority in zip(indices, priorities):
            self.update_priority(idx, priority)

    def clear(self):
        """Clear all transitions from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._priorities.clear()
            self._pos = 0
            self._size = 0

    def get_all(self) -> List[Transition]:
        """Get all transitions in the buffer.

        Returns:
            List of all stored transitions.
        """
        with self._lock:
            return self._buffer[: self._size].copy()

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={self._size}/{self.capacity}, " f"alpha={self.alpha}, epsilon={self.epsilon})"
