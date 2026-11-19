"""Gymnasium wrapper for EnvGen environments.

This wrapper makes any EnvGen `Environment` compatible with the
Gymnasium RL ecosystem, allowing it to be used with libraries like
Stable-Baselines3, Ray RLlib, and other Gymnasium-compatible trainers.

Requires: pip install gymnasium numpy
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, Observation

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore

logger = logging.getLogger(__name__)


# Dynamically create the base class so the module can be imported even without gymnasium.
# When gymnasium is available, GymnasiumEnvironment inherits from gym.Env for SB3 compatibility.
if gym is not None:
    _GymBase = gym.Env
else:
    _GymBase = object  # type: ignore[misc]


class GymnasiumEnvironment(_GymBase):
    """Wraps an EnvGen Environment into a Gymnasium-compatible interface.

    Maps EnvGen's Observation/Action types to Gymnasium spaces:
    - observation_space: Box (continuous feature vector)
    - action_space: Discrete (action index)

    Inherits from gymnasium.Env for compatibility with Stable-Baselines3
    and other Gymnasium-based libraries.

    Usage:
        env = GymnasiumEnvironment(PuzzleEnvironment(config))
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        environment: Environment,
        render_mode: Optional[str] = None,
    ):
        """Initialize the Gymnasium wrapper.

        Args:
            environment: An EnvGen Environment instance.
            render_mode: Optional render mode ('human' or 'ansi').
        """
        if gym is None:
            raise ImportError(
                "The 'gymnasium' package is required for GymnasiumEnvironment. "
                "Install it with: pip install gymnasium"
            )

        self._env = environment
        self._render_mode = render_mode
        self._np_random: Optional[np.random.Generator] = None
        self._current_obs: Optional[Observation] = None

        # Build observation space from feature vector
        self._sample_features = self._extract_features(
            Observation(state={"grid_size": 5, "agent_pos": [0, 0], "goal_pos": [0, 0]})
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._sample_features),),
            dtype=np.float32,
        )

        # Build action space from available actions
        # We use a fixed action set based on the environment's typical actions
        self._action_map = self._build_action_map()
        self.action_space = gym.spaces.Discrete(len(self._action_map))

        logger.info(
            f"GymnasiumEnvironment wrapped: obs_space={self.observation_space}, " f"action_space={self.action_space}"
        )

    @property
    def unwrapped(self) -> Environment:
        """Return the underlying EnvGen environment."""
        return self._env

    @property
    def spec(self):
        """Return the environment spec (Gymnasium compatibility)."""
        return None

    def _build_action_map(self) -> list:
        """Build a mapping from action indices to Action objects.

        Returns a list of common actions for grid-based environments.
        """
        return [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("collect", {}),
            Action("check_goal", {}),
            Action("noop", {}),
        ]

    def _extract_features(self, observation: Observation) -> np.ndarray:
        """Extract a fixed-size feature vector from an observation.

        This mirrors the feature extraction in PPOAgentTorch for consistency.
        """
        state = observation.state

        agent_pos = state.get("agent_pos", [0, 0])
        goal_pos = state.get("goal_pos", [0, 0])
        grid_size = state.get("grid_size", 5)

        max_pos = max(grid_size, 1)
        agent_x = agent_pos[0] / max_pos if len(agent_pos) > 0 else 0.0
        agent_y = agent_pos[1] / max_pos if len(agent_pos) > 1 else 0.0
        goal_x = goal_pos[0] / max_pos if len(goal_pos) > 0 else 0.0
        goal_y = goal_pos[1] / max_pos if len(goal_pos) > 1 else 0.0

        items_remaining = state.get("items_remaining", 0)
        items_collected = state.get("items_collected", 0)
        max_items = max(state.get("num_items", 3), 1)

        dist_to_goal = abs(agent_x - goal_x) + abs(agent_y - goal_y)

        visited = state.get("visited", 0)
        visited_ratio = visited / (grid_size * grid_size) if grid_size > 0 else 0.0

        current_step = state.get("current_step", 0)
        max_steps = max(state.get("max_steps", 50), 1)
        step_ratio = current_step / max_steps

        features = np.array(
            [
                agent_x,
                agent_y,
                goal_x,
                goal_y,
                items_remaining / max_items,
                items_collected / max_items,
                dist_to_goal,
                visited_ratio,
                step_ratio,
                1.0 / max(grid_size, 1),
            ],
            dtype=np.float32,
        )

        return features

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self.seed(seed)

        # Reset the underlying environment
        obs = self._env.reset()
        self._current_obs = obs

        # Extract features and build info dict
        features = self._extract_features(obs)
        info = {
            "raw_observation": obs.to_dict(),
            "step": 0,
        }

        return features, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action index (integer).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Map action index to EnvGen Action
        if action < 0 or action >= len(self._action_map):
            action = 0  # Default to first action

        env_action = self._action_map[action]

        # Step the underlying environment
        obs, reward, done, info = self._env.step(env_action)
        self._current_obs = obs

        # Extract features
        features = self._extract_features(obs)

        # Gymnasium uses terminated and truncated separately
        # We treat 'done' as terminated (natural end)
        terminated = done
        truncated = self._env.current_step >= self._env.max_steps

        # Add metadata to info
        info["raw_observation"] = obs.to_dict()
        info["action_type"] = env_action.action_type
        info["step"] = self._env.current_step

        return features, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the environment.

        Returns:
            String representation of the environment state.
        """
        if self._render_mode == "human" or self._render_mode is None:
            return self._env.render()
        elif self._render_mode == "ansi":
            return self._env.render()
        return None

    def close(self) -> None:
        """Close the environment."""
        pass

    def seed(self, seed: Optional[int] = None) -> list:
        """Set the random seed.

        Args:
            seed: Random seed.

        Returns:
            List containing the seed.
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            random.seed(seed)
            # Set seed on underlying environment if possible
            if hasattr(self._env, "_rng"):
                self._env._rng.seed(seed)
        return [seed] if seed is not None else []

    @property
    def config(self) -> EnvironmentConfig:
        """Return the underlying environment's config."""
        return self._env.config

    def get_available_actions(self) -> list:
        """Return the list of available actions (EnvGen compatibility)."""
        return self._action_map
