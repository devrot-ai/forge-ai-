"""Stable-Baselines3 wrapper agent.

This agent wraps any Stable-Baselines3 algorithm (PPO, SAC, A2C, DQN)
and makes it compatible with the EnvGen-LLM training pipeline. It uses
the GymnasiumEnvironment wrapper internally to bridge EnvGen environments
with SB3's training loop.

Requires: pip install stable-baselines3 gymnasium numpy
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation

logger = logging.getLogger(__name__)


class SB3Agent(Agent):
    """Stable-Baselines3 wrapper agent.

    Wraps any SB3 model (PPO, SAC, A2C, DQN) and provides the EnvGen
    Agent interface. Internally creates a GymnasiumEnvironment wrapper
    for training the SB3 model.

    Supported algorithms:
        - sb3_ppo: Proximal Policy Optimization
        - sb3_sac: Soft Actor-Critic
        - sb3_a2c: Advantage Actor-Critic
        - sb3_dqn: Deep Q-Network

    Usage:
        agent = SB3Agent(algorithm="ppo", config={"learning_rate": 3e-4})
        action = agent.select_action(observation, available_actions)
        metrics = agent.train(episodes)
    """

    # Mapping from algorithm name to SB3 class
    ALGORITHMS = {
        "ppo": "PPO",
        "sac": "SAC",
        "a2c": "A2C",
        "dqn": "DQN",
    }

    def __init__(
        self,
        algorithm: str = "ppo",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SB3 wrapper agent.

        Args:
            algorithm: SB3 algorithm name ('ppo', 'sac', 'a2c', 'dqn').
            name: Optional agent name (defaults to 'sb3_<algorithm>').
            config: Optional configuration dict for SB3 hyperparameters.
        """
        algo_lower = algorithm.lower().split("_")[-1] if "_" in algorithm.lower() else algorithm.lower()
        if algo_lower not in self.ALGORITHMS:
            raise ValueError(f"Unknown SB3 algorithm: {algorithm}. " f"Supported: {list(self.ALGORITHMS.keys())}")

        self.algorithm = algo_lower
        super().__init__(name or f"sb3_{algo_lower}", config)

        try:
            import stable_baselines3
        except ImportError:
            raise ImportError(
                "The 'stable-baselines3' package is required for SB3Agent. "
                "Install it with: pip install stable-baselines3"
            )

        try:
            import gymnasium
        except ImportError:
            raise ImportError(
                "The 'gymnasium' package is required for SB3Agent. " "Install it with: pip install gymnasium"
            )

        # SB3 hyperparameters
        self.sb3_config: Dict[str, Any] = {
            "learning_rate": self.config.get("learning_rate", 3e-4),
            "n_steps": self.config.get("n_steps", 2048),
            "batch_size": self.config.get("batch_size", 64),
            "gamma": self.config.get("discount_factor", self.config.get("gamma", 0.99)),
            "ent_coef": self.config.get("ent_coef", 0.01),
            "verbose": self.config.get("verbose", 1),
            "n_epochs": self.config.get("n_epochs", 10),
            "clip_range": self.config.get("clip_range", 0.2),
        }

        # Action mapping
        self._action_map: List[Action] = []
        self._reverse_action_map: Dict[str, int] = {}

        # SB3 model
        self.model = None
        self._gym_env = None

        # Training buffer
        self.buffer: List[Dict[str, Any]] = []
        self.max_buffer_size: int = self.config.get("max_buffer_size", 2048)

        # Stats
        self.update_count: int = 0
        self.total_timesteps: int = 0

    def _build_action_map(self, available_actions: List[Action]) -> List[Action]:
        """Build a mapping from action indices to Action objects.

        Args:
            available_actions: List of available actions from the environment.

        Returns:
            Sorted list of actions for consistent indexing.
        """
        if not available_actions:
            return [
                Action("move", {"direction": "up"}),
                Action("move", {"direction": "down"}),
                Action("move", {"direction": "left"}),
                Action("move", {"direction": "right"}),
                Action("collect", {}),
                Action("check_goal", {}),
                Action("noop", {}),
            ]

        # Sort actions for consistent indexing
        def action_key(a: Action) -> str:
            params = json.dumps(a.parameters, sort_keys=True) if a.parameters else ""
            return f"{a.action_type}:{params}"

        sorted_actions = sorted(available_actions, key=action_key)
        self._action_map = sorted_actions
        self._reverse_action_map = {action_key(a): i for i, a in enumerate(sorted_actions)}
        return sorted_actions

    def _get_action_index(self, action: Action) -> int:
        """Get the index of an action in the action map.

        Args:
            action: The action to find.

        Returns:
            The index of the action, or 0 if not found.
        """
        params = json.dumps(action.parameters, sort_keys=True) if action.parameters else ""
        key = f"{action.action_type}:{params}"
        return self._reverse_action_map.get(key, 0)

    def _extract_features(self, observation: Observation) -> np.ndarray:
        """Extract a fixed-size feature vector from an observation.

        This mirrors the feature extraction in the GymnasiumEnvironment wrapper.

        Args:
            observation: The observation to extract features from.

        Returns:
            A numpy array of features.
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

    def _init_model(self, gym_env) -> None:
        """Initialize the SB3 model.

        Args:
            gym_env: A Gymnasium-compatible environment.
        """
        import stable_baselines3 as sb3

        algo_class_name = self.ALGORITHMS[self.algorithm]

        # Get the algorithm class
        algo_class = getattr(sb3, algo_class_name, None)
        if algo_class is None:
            raise ImportError(f"SB3 algorithm '{algo_class_name}' not found")

        # Build policy keyword
        policy_kwargs = None
        obs_dim = 10  # Fixed feature vector size
        action_dim = len(self._action_map) if self._action_map else 7

        # Algorithm-specific config
        algo_config = dict(self.sb3_config)
        algo_config.pop("clip_range", None)  # PPO-specific
        algo_config.pop("n_epochs", None)  # PPO-specific

        if self.algorithm == "ppo":
            algo_config["clip_range"] = self.sb3_config.get("clip_range", 0.2)
            algo_config["n_epochs"] = self.sb3_config.get("n_epochs", 10)
        elif self.algorithm == "sac":
            # SAC doesn't use clip_range or n_epochs
            pass
        elif self.algorithm == "dqn":
            # DQN-specific
            algo_config["train_freq"] = self.config.get("train_freq", 4)
            algo_config["target_update_interval"] = self.config.get("target_update_interval", 1)

        self.model = algo_class(
            policy="MlpPolicy",
            env=gym_env,
            **algo_config,
        )

        logger.info(f"SB3Agent model initialized: algorithm={self.algorithm}, " f"action_dim={action_dim}")

    def select_action(self, observation: Observation, available_actions: List[Action]) -> Action:
        """Select an action using the SB3 model.

        If the model is not yet initialized, returns a random action.

        Args:
            observation: The current observation.
            available_actions: List of valid actions.

        Returns:
            The selected action.
        """
        # Build action map if needed
        if not self._action_map:
            self._build_action_map(available_actions)

        # If model not initialized, return random action
        if self.model is None:
            import random

            return random.choice(available_actions)

        # Extract features
        obs_vec = self._extract_features(observation)

        # Predict action using SB3 model
        action_idx, _ = self.model.predict(obs_vec, deterministic=False)
        action_idx = int(action_idx)

        # Map back to Action object
        if self._action_map and 0 <= action_idx < len(self._action_map):
            return self._action_map[action_idx]
        else:
            import random

            return random.choice(available_actions)

    def train(self, episodes: List[Episode]) -> Dict[str, float]:
        """Train the SB3 model on a batch of episodes.

        Collects transitions from episodes, stores them in a buffer,
        and uses SB3's learn() method for training.

        Args:
            episodes: List of completed episodes.

        Returns:
            Training metrics.
        """
        if not episodes:
            return {"num_updates": self.update_count, "total_timesteps": self.total_timesteps}

        # Collect transitions
        transitions = []
        for episode in episodes:
            for transition in episode.transitions:
                transitions.append(transition)

        if not transitions:
            return {"num_updates": self.update_count, "total_timesteps": self.total_timesteps}

        # Build action map from first transition
        if not self._action_map:
            # Use default action set
            self._build_action_map([])

        # Store transitions in buffer
        self.buffer.extend(
            [
                {
                    "obs": self._extract_features(t.observation),
                    "action": self._get_action_index(t.action),
                    "reward": t.reward,
                    "done": t.done,
                    "next_obs": self._extract_features(t.next_observation),
                }
                for t in transitions
            ]
        )

        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size :]

        # Create a temporary Gymnasium environment for SB3 training
        if self._gym_env is None:
            from envgen.environments.base import EnvironmentConfig
            from envgen.environments.gymnasium_wrapper import GymnasiumEnvironment
            from envgen.environments.puzzle_env import PuzzleEnvironment

            # Create a default environment for training
            default_cfg = EnvironmentConfig(
                name="sb3_training_env",
                description="SB3 training environment",
                difficulty="easy",
                parameters={"grid_size": 5, "num_items": 3, "num_obstacles": 2},
            )
            puzzle_env = PuzzleEnvironment(default_cfg)
            self._gym_env = GymnasiumEnvironment(puzzle_env)

        # Initialize model if needed
        if self.model is None:
            self._init_model(self._gym_env)

        # Calculate timesteps to train
        num_transitions = len(self.buffer)
        timesteps = max(num_transitions, 128)

        # Train using SB3's learn method
        try:
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            self.update_count += 1
            self.total_timesteps += timesteps
        except Exception as e:
            logger.warning(f"SB3 training error: {e}")
            return {"error": str(e), "num_updates": self.update_count}

        # Update stats
        self.total_episodes_trained += len(episodes)
        self.total_steps_trained += sum(ep.length for ep in episodes)

        # Get SB3 stats
        stats = {
            "num_updates": self.update_count,
            "total_timesteps": self.total_timesteps,
            "buffer_size": len(self.buffer),
        }

        # Try to get episode stats from SB3 model
        if hasattr(self.model, "episode_reward_mean"):
            stats["episode_reward_mean"] = float(self.model.episode_reward_mean)
        if hasattr(self.model, "episode_length_mean"):
            stats["episode_length_mean"] = float(self.model.episode_length_mean)

        return stats

    def save(self, path: str) -> None:
        """Save the agent's state including the SB3 model.

        Args:
            path: File path to save to.
        """
        # Save SB3 model
        if self.model is not None:
            model_path = path + "_model"
            self.model.save(model_path)

        # Save metadata
        save_data = {
            "name": self.name,
            "algorithm": self.algorithm,
            "config": self.config,
            "agent_id": self.agent_id,
            "total_episodes_trained": self.total_episodes_trained,
            "total_steps_trained": self.total_steps_trained,
            "update_count": self.update_count,
            "total_timesteps": self.total_timesteps,
            "sb3_config": self.sb3_config,
        }

        meta_path = path + ".json"
        os.makedirs(os.path.dirname(meta_path) if os.path.dirname(meta_path) else ".", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"SB3Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load the agent's state including the SB3 model.

        Args:
            path: File path to load from.
        """
        import stable_baselines3 as sb3

        # Load metadata
        meta_path = path + ".json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path) as f:
            save_data = json.load(f)

        # Restore config
        self.name = save_data.get("name", self.name)
        self.algorithm = save_data.get("algorithm", self.algorithm)
        self.config = save_data.get("config", self.config)
        self.total_episodes_trained = save_data.get("total_episodes_trained", 0)
        self.total_steps_trained = save_data.get("total_steps_trained", 0)
        self.update_count = save_data.get("update_count", 0)
        self.total_timesteps = save_data.get("total_timesteps", 0)
        self.sb3_config = save_data.get("sb3_config", self.sb3_config)

        # Load SB3 model
        model_path = path + "_model"
        if os.path.exists(model_path):
            algo_class_name = self.ALGORITHMS[self.algorithm]
            algo_class = getattr(sb3, algo_class_name, None)
            if algo_class is not None:
                self.model = algo_class.load(model_path)
                logger.info(f"SB3 model loaded from {model_path}")

        logger.info(f"SB3Agent loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics.

        Returns:
            Dictionary with agent stats.
        """
        stats = super().get_stats()
        stats.update(
            {
                "algorithm": self.algorithm,
                "update_count": self.update_count,
                "total_timesteps": self.total_timesteps,
                "buffer_size": len(self.buffer),
            }
        )
        return stats
