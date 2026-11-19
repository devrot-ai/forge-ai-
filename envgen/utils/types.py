"""Common data types used across the framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DifficultyLevel(Enum):
    """Standard difficulty levels for environments and tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Observation:
    """Represents an observation from the environment.

    Attributes:
        state: The raw state dictionary from the environment.
        info: Additional metadata about the observation.
    """

    state: Dict[str, Any]
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "info": self.info}


@dataclass
class Action:
    """Represents an action taken by the agent.

    Attributes:
        action_type: The type/category of the action.
        parameters: Parameters for the action.
    """

    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type, "parameters": self.parameters}


@dataclass
class Transition:
    """A single MDP transition: (state, action, reward, next_state, done).

    Attributes:
        observation: The starting observation.
        action: The action taken.
        reward: The reward received.
        next_observation: The resulting observation.
        done: Whether the episode ended.
        info: Additional metadata.
    """

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A complete episode of interaction.

    Attributes:
        episode_id: Unique identifier for this episode.
        transitions: List of transitions in the episode.
        total_reward: Sum of rewards in the episode.
        environment_config: The environment config used.
        metadata: Additional episode metadata.
    """

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    environment_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)
        self.total_reward += transition.reward

    @property
    def length(self) -> int:
        return len(self.transitions)


@dataclass
class SkillScore:
    """Score for a specific skill dimension.

    Attributes:
        skill_name: Name of the skill.
        score: Normalized score in [0, 1].
        raw_value: Raw metric value before normalization.
        metadata: Additional info about the skill evaluation.
    """

    skill_name: str
    score: float
    raw_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating an agent on a set of environments.

    Attributes:
        agent_id: Identifier of the evaluated agent.
        skill_scores: Per-skill scores.
        overall_score: Aggregate score.
        episode_stats: Statistics over episodes.
        environment_configs: Environments used for evaluation.
    """

    agent_id: str
    skill_scores: List[SkillScore] = field(default_factory=list)
    overall_score: float = 0.0
    episode_stats: Dict[str, float] = field(default_factory=dict)
    environment_configs: List[Dict[str, Any]] = field(default_factory=list)

    def get_skill_score(self, skill_name: str) -> Optional[float]:
        for ss in self.skill_scores:
            if ss.skill_name == skill_name:
                return ss.score
        return None

    def get_weak_skills(self, threshold: float = 0.5, top_k: int = 3) -> List[SkillScore]:
        """Return the weakest skills below the threshold."""
        weak = [ss for ss in self.skill_scores if ss.score < threshold]
        weak.sort(key=lambda ss: ss.score)
        return weak[:top_k]

    def get_strong_skills(self, threshold: float = 0.7, top_k: int = 3) -> List[SkillScore]:
        """Return the strongest skills above the threshold."""
        strong = [ss for ss in self.skill_scores if ss.score >= threshold]
        strong.sort(key=lambda ss: ss.score, reverse=True)
        return strong[:top_k]
