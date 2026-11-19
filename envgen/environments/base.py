"""Base environment interface and configuration."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from envgen.utils.types import Action, DifficultyLevel, Observation


@dataclass
class EnvironmentConfig:
    """Configuration for creating and parameterizing an environment.

    This is the primary artifact produced by the LLM environment generator.
    Each config defines a specific training scenario with its parameters,
    difficulty, and target skills.

    Attributes:
        name: Human-readable name for this environment config.
        description: Detailed description of what this environment trains.
        difficulty: Difficulty level of the environment.
        parameters: Environment-specific parameters (e.g., terrain type, item spawn rates).
        target_skills: Skills this environment is designed to train.
        max_steps: Maximum steps per episode.
        reward_scale: Multiplier for rewards in this environment.
        seed: Random seed for reproducibility.
    """

    name: str
    description: str
    difficulty: DifficultyLevel = DifficultyLevel.EASY
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_skills: List[str] = field(default_factory=list)
    max_steps: int = 100
    reward_scale: float = 1.0
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "parameters": self.parameters,
            "target_skills": self.target_skills,
            "max_steps": self.max_steps,
            "reward_scale": self.reward_scale,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentConfig:
        data = data.copy()
        if "difficulty" in data and isinstance(data["difficulty"], str):
            data["difficulty"] = DifficultyLevel(data["difficulty"])
        return cls(**data)


class Environment(abc.ABC):
    """Abstract base class for all training environments.

    Environments define the problem space where agents learn. Each environment
    implements a specific task with configurable parameters. The LLM generator
    creates EnvironmentConfig instances that specify how to instantiate and
    parameterize environments.

    Subclasses should implement:
        - reset(): Initialize the environment state
        - step(): Execute an action and return the result
        - get_observation(): Return the current observation
        - get_available_actions(): Return valid actions from current state
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.current_step: int = 0
        self._done: bool = False
        self._rng: random.Random = random.Random(config.seed)
        self._state: Dict[str, Any] = {}
        self._episode_info: Dict[str, Any] = {}

    @abc.abstractmethod
    def reset(self) -> Observation:
        """Reset the environment to its initial state.

        Returns:
            The initial observation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action in the environment.

        Args:
            action: The action to execute.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        """Return the current observation.

        Returns:
            The current observation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_available_actions(self) -> List[Action]:
        """Return the list of valid actions from the current state.

        Returns:
            List of available actions.
        """
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self.current_step

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    def is_terminal(self) -> bool:
        """Check if the current state is terminal."""
        return self._done or self.current_step >= self.max_steps

    def get_state(self) -> Dict[str, Any]:
        """Return the current internal state (for debugging/logging)."""
        return self._state.copy()

    def render(self) -> str:
        """Return a text representation of the current state."""
        return str(self._state)

    def seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self._rng = random.Random(seed)

    def random_action(self) -> Action:
        """Return a random valid action."""
        actions = self.get_available_actions()
        if not actions:
            return Action(action_type="noop")
        return self._rng.choice(actions)
