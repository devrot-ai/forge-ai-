"""Base evaluator interface."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from envgen.utils.types import Episode, EvaluationResult


class Evaluator(abc.ABC):
    """Abstract base class for evaluating agent performance.

    Evaluators measure agent performance across multiple skill dimensions.
    They analyze episodes and produce structured evaluation results that
    feed back into the curriculum manager and environment generator.

    Subclasses should implement:
        - evaluate(): Produce evaluation results from episodes
        - get_skill_names(): Return the skills this evaluator measures
    """

    def __init__(self, name: str = "evaluator", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abc.abstractmethod
    def evaluate(self, episodes: List[Episode], agent_id: str) -> EvaluationResult:
        """Evaluate agent performance from a set of episodes.

        Args:
            episodes: Episodes to evaluate.
            agent_id: ID of the agent that generated the episodes.

        Returns:
            Structured evaluation result with skill scores.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_skill_names(self) -> List[str]:
        """Return the list of skill names this evaluator measures.

        Returns:
            List of skill name strings.
        """
        raise NotImplementedError

    def evaluate_batch(
        self,
        episodes_by_env: Dict[str, List[Episode]],
        agent_id: str,
    ) -> List[EvaluationResult]:
        """Evaluate episodes grouped by environment.

        Args:
            episodes_by_env: Dict mapping environment name to episodes.
            agent_id: ID of the agent.

        Returns:
            List of evaluation results, one per environment group.
        """
        results = []
        for env_name, episodes in episodes_by_env.items():
            if episodes:
                result = self.evaluate(episodes, agent_id)
                result.metadata = result.metadata or {}
                if hasattr(result, "metadata"):
                    pass
                results.append(result)
        return results
