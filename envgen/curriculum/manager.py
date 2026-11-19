"""Curriculum manager for progressive difficulty."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from envgen.utils.types import DifficultyLevel, EvaluationResult, SkillScore


@dataclass
class CurriculumStage:
    """A stage in the learning curriculum.

    Attributes:
        stage_id: Unique identifier.
        name: Human-readable name.
        difficulty: Difficulty level.
        skill_targets: Target skills for this stage.
        min_scores: Minimum skill scores to advance.
        environment_configs: Environment configs for this stage.
    """

    stage_id: int
    name: str
    difficulty: DifficultyLevel
    skill_targets: List[str] = field(default_factory=list)
    min_scores: Dict[str, float] = field(default_factory=dict)
    environment_configs: List[Dict[str, Any]] = field(default_factory=list)


class CurriculumManager:
    """Manages the learning curriculum with progressive difficulty.

    The curriculum manager tracks the agent's progress through stages of
    increasing difficulty. It uses evaluation results to determine when
    the agent is ready to advance and which skills need more focus.

    The curriculum follows these principles:
    1. Start easy, progress gradually
    2. Focus on weak skills before advancing
    3. Allow regression and re-visiting earlier stages
    4. Adapt difficulty based on performance trends
    """

    def __init__(
        self,
        name: str = "curriculum",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}
        self.stages: List[CurriculumStage] = []
        self.current_stage: int = 0
        self.history: List[Dict[str, Any]] = []
        self.skill_progress: Dict[str, List[float]] = {}

        # Thresholds
        self.advance_threshold: float = self.config.get("advance_threshold", 0.7)
        self.regression_threshold: float = self.config.get("regression_threshold", 0.4)
        self.min_improvement: float = self.config.get("min_improvement", 0.05)

    def add_stage(
        self,
        name: str,
        difficulty: DifficultyLevel,
        skill_targets: Optional[List[str]] = None,
        min_scores: Optional[Dict[str, float]] = None,
        environment_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Add a new stage to the curriculum.

        Args:
            name: Stage name.
            difficulty: Difficulty level.
            skill_targets: Skills targeted in this stage.
            min_scores: Minimum scores to advance past this stage.
            environment_configs: Environment configs for this stage.

        Returns:
            The stage ID.
        """
        stage_id = len(self.stages)
        stage = CurriculumStage(
            stage_id=stage_id,
            name=name,
            difficulty=difficulty,
            skill_targets=skill_targets or [],
            min_scores=min_scores or {},
            environment_configs=environment_configs or [],
        )
        self.stages.append(stage)
        return stage_id

    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get the current curriculum stage."""
        if 0 <= self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None

    def get_stage(self, stage_id: int) -> Optional[CurriculumStage]:
        """Get a specific stage by ID."""
        if 0 <= stage_id < len(self.stages):
            return self.stages[stage_id]
        return None

    def should_advance(self, result: EvaluationResult) -> bool:
        """Check if the agent should advance to the next stage.

        Args:
            result: Latest evaluation result.

        Returns:
            True if the agent should advance.
        """
        current = self.get_current_stage()
        if current is None or self.current_stage >= len(self.stages) - 1:
            return False

        # Check minimum scores for current stage
        for skill, min_score in current.min_scores.items():
            actual = result.get_skill_score(skill)
            if actual is None or actual < min_score:
                return False

        # Check overall score threshold
        if result.overall_score < self.advance_threshold:
            return False

        # Check that weak skills are improving
        weak_skills = result.get_weak_skills(threshold=self.advance_threshold)
        for ws in weak_skills:
            if ws.skill_name in self.skill_progress:
                history = self.skill_progress[ws.skill_name]
                if len(history) >= 2:
                    if ws.score < history[-1] - self.min_improvement:
                        return False

        return True

    def should_regress(self, result: EvaluationResult) -> bool:
        """Check if the agent should regress to an earlier stage.

        Args:
            result: Latest evaluation result.

        Returns:
            True if the agent should regress.
        """
        if self.current_stage <= 0:
            return False

        weak = result.get_weak_skills(threshold=self.regression_threshold)
        if len(weak) > len(result.skill_scores) // 2:
            return True

        if result.overall_score < self.regression_threshold:
            return True

        return False

    def update(self, result: EvaluationResult) -> Dict[str, Any]:
        """Update curriculum state based on evaluation result.

        Args:
            result: Latest evaluation result.

        Returns:
            Dict with action taken and reasoning.
        """
        action = "continue"
        reasoning = []

        # Record skill progress
        for ss in result.skill_scores:
            if ss.skill_name not in self.skill_progress:
                self.skill_progress[ss.skill_name] = []
            self.skill_progress[ss.skill_name].append(ss.score)

        # Decide action
        if self.should_advance(result):
            self.current_stage += 1
            action = "advance"
            reasoning.append(f"Advancing to stage {self.current_stage}")
        elif self.should_regress(result):
            self.current_stage = max(0, self.current_stage - 1)
            action = "regress"
            reasoning.append(f"Regressing to stage {self.current_stage}")
        else:
            reasoning.append(f"Staying at stage {self.current_stage}")

        # Record history
        record = {
            "stage": self.current_stage,
            "action": action,
            "overall_score": result.overall_score,
            "skill_scores": {ss.skill_name: ss.score for ss in result.skill_scores},
            "reasoning": reasoning,
        }
        self.history.append(record)

        return record

    def get_weak_skills_for_stage(self, result: EvaluationResult) -> List[SkillScore]:
        """Get weak skills that need focus in the current stage.

        Args:
            result: Latest evaluation result.

        Returns:
            List of weak skill scores.
        """
        current = self.get_current_stage()
        if current is None:
            return result.get_weak_skills()

        # Prioritize skills targeted by current stage
        targeted_weak = [
            ss
            for ss in result.skill_scores
            if ss.skill_name in current.skill_targets and ss.score < self.advance_threshold
        ]
        targeted_weak.sort(key=lambda ss: ss.score)

        # Add other weak skills
        other_weak = [
            ss
            for ss in result.skill_scores
            if ss.skill_name not in current.skill_targets and ss.score < self.advance_threshold
        ]
        other_weak.sort(key=lambda ss: ss.score)

        return targeted_weak + other_weak

    def get_progress_report(self) -> Dict[str, Any]:
        """Generate a progress report."""
        current = self.get_current_stage()
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "current_stage_name": current.name if current else None,
            "current_difficulty": current.difficulty.value if current else None,
            "skill_progress": {
                skill: {
                    "current": scores[-1] if scores else 0,
                    "best": max(scores) if scores else 0,
                    "history_length": len(scores),
                }
                for skill, scores in self.skill_progress.items()
            },
            "total_updates": len(self.history),
        }
