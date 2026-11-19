"""Base agent interface."""

from __future__ import annotations

import abc
import uuid
from typing import Any, Dict, List, Optional

from envgen.utils.types import Action, Episode, Observation


class Agent(abc.ABC):
    """Abstract base class for all agents.

    Agents interact with environments by selecting actions based on observations.
    This framework supports both RL-based agents (PPO, GRPO) and LLM-based agents.

    Subclasses should implement:
        - select_action(): Choose an action given an observation
        - train(): Update the agent's policy from experience
        - save()/load(): Persist and restore agent state
    """

    def __init__(self, name: str = "agent", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.agent_id = uuid.uuid4().hex[:12]
        self.total_episodes_trained: int = 0
        self.total_steps_trained: int = 0

    @abc.abstractmethod
    def select_action(self, observation: Observation, available_actions: List[Action]) -> Action:
        """Select an action given the current observation.

        Args:
            observation: The current observation from the environment.
            available_actions: List of valid actions.

        Returns:
            The selected action.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, episodes: List[Episode]) -> Dict[str, float]:
        """Train the agent on a batch of episodes.

        Args:
            episodes: List of completed episodes.

        Returns:
            Training metrics (e.g., loss, reward stats).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's state to disk.

        Args:
            path: File path to save to.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's state from disk.

        Args:
            path: File path to load from.
        """
        raise NotImplementedError

    def run_episode(self, environment: Any) -> Episode:
        """Run a single episode in the given environment.

        This is a convenience method that handles the interaction loop.
        Subclasses can override for custom behavior.

        Args:
            environment: The environment to interact with.

        Returns:
            The completed episode.
        """
        from envgen.utils.types import Episode, Transition

        obs = environment.reset()
        episode = Episode(environment_config=environment.config.to_dict())

        while not environment.is_terminal():
            available = environment.get_available_actions()
            action = self.select_action(obs, available)
            next_obs, reward, done, info = environment.step(action)

            from envgen.utils.types import Action as A
            from envgen.utils.types import Observation as O

            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                info=info,
            )
            episode.add_transition(transition)
            obs = next_obs
            environment.current_step += 1

        episode.metadata["agent_id"] = self.agent_id
        episode.metadata["agent_name"] = self.name
        return episode

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "total_episodes_trained": self.total_episodes_trained,
            "total_steps_trained": self.total_steps_trained,
        }
