"""Reward engineering and shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from envgen.utils.types import Episode, Observation, Transition


@dataclass
class RewardComponent:
    """A single component of a composite reward.

    Attributes:
        name: Name of this reward component.
        weight: Weight for combining with other components.
        compute_fn: Function that computes this reward from a transition.
    """

    name: str
    weight: float
    compute_fn: Callable[[Transition], float]


class RewardEngine:
    """Engine for computing and shaping rewards.

    The reward engine combines multiple reward components with configurable
    weights. It supports reward shaping, normalization, and adaptive weighting
    based on agent performance.

    Features:
    - Composite rewards from multiple components
    - Reward shaping with potential-based methods
    - Adaptive weighting based on skill progress
    - Reward normalization and clipping
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.components: List[RewardComponent] = []
        self.shaping_function: Optional[Callable[[float, int, int], float]] = None
        self.clip_range: tuple[float, float] = self.config.get("clip_range", (-10.0, 10.0))
        self.normalization: Dict[str, Dict[str, float]] = {}

    def add_component(
        self,
        name: str,
        weight: float,
        compute_fn: Callable[[Transition], float],
    ) -> None:
        """Add a reward component.

        Args:
            name: Component name.
            weight: Weight for this component.
            compute_fn: Function that takes a Transition and returns a float reward.
        """
        self.components.append(RewardComponent(name, weight, compute_fn))

    def compute(self, transition: Transition) -> float:
        """Compute the total reward for a transition.

        Args:
            transition: The MDP transition.

        Returns:
            The computed reward.
        """
        total = 0.0
        for component in self.components:
            reward = component.compute_fn(transition)
            total += component.weight * reward

        # Apply shaping if configured
        if self.shaping_function is not None:
            total = self.shaping_function(total, transition.reward, transition.observation.state.get("step", 0))

        # Clip reward
        total = max(self.clip_range[0], min(self.clip_range[1], total))

        return total

    def compute_episode_reward(self, episode: Episode) -> Dict[str, float]:
        """Compute per-component rewards for an entire episode.

        Args:
            episode: The completed episode.

        Returns:
            Dict mapping component name to total reward.
        """
        component_totals: Dict[str, float] = {c.name: 0.0 for c in self.components}
        component_totals["total"] = 0.0

        for transition in episode.transitions:
            for component in self.components:
                reward = component.compute_fn(transition)
                component_totals[component.name] += component.weight * reward
            component_totals["total"] += self.compute(transition)

        return component_totals

    def set_shaping_function(self, shaping_fn: Callable[[float, int, int], float]) -> None:
        """Set a reward shaping function.

        Args:
            shaping_fn: Function(reward, step, max_steps) -> shaped_reward
        """
        self.shaping_function = shaping_fn

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update component weights.

        Args:
            weights: Dict mapping component name to new weight.
        """
        for component in self.components:
            if component.name in weights:
                component.weight = weights[component.name]

    def get_weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return {c.name: c.weight for c in self.components}

    # --- Built-in reward functions ---

    @staticmethod
    def sparse_reward(
        success_key: str = "success",
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ) -> Callable[[Transition], float]:
        """Create a sparse reward function.

        Args:
            success_key: Key in transition.info indicating success.
            success_reward: Reward for success.
            failure_reward: Reward for non-success steps.

        Returns:
            Reward function.
        """

        def reward_fn(transition: Transition) -> float:
            if transition.info.get(success_key, False):
                return success_reward
            return failure_reward

        return reward_fn

    @staticmethod
    def density_shaping(
        potential_fn: Callable[[Observation], float],
        gamma: float = 0.99,
    ) -> Callable[[Transition], float]:
        """Create potential-based density reward shaping.

        Uses the potential-based reward shaping formula:
        F(s, a, s') = gamma * Phi(s') - Phi(s)

        Args:
            potential_fn: Function that maps observation to potential value.
            gamma: Discount factor.

        Returns:
            Shaping reward function.
        """

        def reward_fn(transition: Transition) -> float:
            phi_s = potential_fn(transition.observation)
            phi_s_next = potential_fn(transition.next_observation)
            return gamma * phi_s_next - phi_s

        return reward_fn

    @staticmethod
    def progress_reward(
        progress_key: str = "progress",
        scale: float = 1.0,
    ) -> Callable[[Transition], float]:
        """Create a progress-based reward function.

        Args:
            progress_key: Key in transition.info containing progress value.
            scale: Scale factor for the reward.

        Returns:
            Reward function.
        """

        def reward_fn(transition: Transition) -> float:
            progress = transition.info.get(progress_key, 0.0)
            return scale * progress

        return reward_fn

    @staticmethod
    def step_penalty(
        penalty: float = -0.01,
    ) -> Callable[[Transition], float]:
        """Create a per-step penalty to encourage efficiency.

        Args:
            penalty: Negative reward per step.

        Returns:
            Reward function.
        """

        def reward_fn(transition: Transition) -> float:
            return penalty

        return reward_fn

    @staticmethod
    def skill_specific_reward(
        skill_name: str,
        success_reward: float = 1.0,
    ) -> Callable[[Transition], float]:
        """Create a skill-specific reward function.

        Args:
            skill_name: The skill this reward targets.
            success_reward: Reward for demonstrating the skill.

        Returns:
            Reward function.
        """

        def reward_fn(transition: Transition) -> float:
            skills = transition.info.get("skills_demonstrated", [])
            if skill_name in skills:
                return success_reward
            return 0.0

        return reward_fn
