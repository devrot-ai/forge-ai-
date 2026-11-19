"""LLM-based curriculum manager for intelligent difficulty progression.

Uses an LLM to analyze agent performance and make curriculum decisions
(advance, stay, regress) with reasoning, instead of rule-based thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from envgen.curriculum.manager import CurriculumManager
from envgen.utils.types import DifficultyLevel, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class CurriculumDecision:
    """A curriculum decision made by the LLM.

    Attributes:
        action: One of 'advance', 'regress', 'stay'.
        reasoning: Human-readable explanation from the LLM.
        suggested_difficulty: Suggested next difficulty level.
        focus_skills: Skills that need focus in the next stage.
        confidence: Confidence score (0-1) in the decision.
    """

    action: str
    reasoning: str
    suggested_difficulty: DifficultyLevel = DifficultyLevel.EASY
    focus_skills: List[str] = field(default_factory=list)
    confidence: float = 0.5


class LLMCurriculumManager:
    """Uses an LLM to decide curriculum progression.

    Instead of fixed thresholds, the LLM receives evaluation results and
    returns a recommendation (advance/stay/regress) with reasoning.
    Falls back to rule-based CurriculumManager if LLM is unavailable.

    The LLM is prompted with:
    - Current stage and difficulty
    - Skill scores and trends
    - Weak and strong skills
    - Overall performance metrics

    Args:
        llm_client: Optional LLM client for curriculum decisions.
        fallback_manager: Rule-based manager to use if LLM fails.
        config: Optional configuration dict.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        fallback_manager: Optional[CurriculumManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.llm_client = llm_client
        self.fallback_manager = fallback_manager or CurriculumManager(
            name="fallback_curriculum",
            config=self.config.get("fallback", {}),
        )
        self.decision_history: List[Dict[str, Any]] = []
        self.current_stage = 0
        self.current_difficulty = DifficultyLevel.EASY
        self.skill_history: Dict[str, List[float]] = {}

    def initialize_stages(self, num_stages: int = 5) -> None:
        """Initialize curriculum stages in the fallback manager.

        Args:
            num_stages: Number of stages to create.
        """
        difficulties = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
            DifficultyLevel.EXPERT,
        ]

        for i in range(num_stages):
            diff = difficulties[min(i, len(difficulties) - 1)]
            self.fallback_manager.add_stage(
                name=f"stage_{i}_{diff.value}",
                difficulty=diff,
                min_scores={
                    "navigation": 0.5 + i * 0.1,
                    "planning": 0.5 + i * 0.1,
                    "reasoning": 0.5 + i * 0.1,
                },
            )

        self.current_stage = 0
        self.current_difficulty = DifficultyLevel.EASY

    def update(self, result: EvaluationResult) -> CurriculumDecision:
        """Update curriculum state using LLM-based decision.

        Args:
            result: Latest evaluation result.

        Returns:
            CurriculumDecision with action, reasoning, and metadata.
        """
        # Record skill progress
        for ss in result.skill_scores:
            if ss.skill_name not in self.skill_history:
                self.skill_history[ss.skill_name] = []
            self.skill_history[ss.skill_name].append(ss.score)

        # Try LLM-based decision first
        if self.llm_client is not None:
            try:
                decision = self._llm_decision(result)
                self._apply_decision(decision)
                return decision
            except Exception as e:
                logger.warning(f"LLM curriculum decision failed: {e}, falling back to rule-based")

        # Fall back to rule-based decision
        fallback_record = self.fallback_manager.update(result)
        decision = self._fallback_to_decision(fallback_record, result)
        self._apply_decision(decision)
        return decision

    def _llm_decision(self, result: EvaluationResult) -> CurriculumDecision:
        """Make a curriculum decision using the LLM.

        Args:
            result: Latest evaluation result.

        Returns:
            CurriculumDecision from the LLM.
        """
        prompt = self._build_prompt(result)

        # Try generate_json first
        try:
            response = self.llm_client.generate_json(prompt)
        except AttributeError:
            # Fallback to generate + parse
            response_text = self.llm_client.generate(prompt)
            response = self._parse_json_response(response_text)

        return self._parse_llm_decision(response, result)

    def _build_prompt(self, result: EvaluationResult) -> str:
        """Build the prompt for LLM curriculum decision.

        Args:
            result: Latest evaluation result.

        Returns:
            Prompt string for the LLM.
        """
        weak = result.get_weak_skills()
        strong = result.get_strong_skills()

        skill_scores_str = "\n".join(
            f"  - {ss.skill_name}: {ss.score:.3f}" for ss in result.skill_scores
        )

        weak_str = "\n".join(
            f"  - {ss.skill_name}: {ss.score:.3f}" for ss in weak
        ) if weak else "  (none)"

        strong_str = "\n".join(
            f"  - {ss.skill_name}: {ss.score:.3f}" for ss in strong
        ) if strong else "  (none)"

        # Skill trends
        trends_str = ""
        for skill, scores in self.skill_history.items():
            if len(scores) >= 2:
                trend = "improving" if scores[-1] > scores[-2] else "declining" if scores[-1] < scores[-2] else "stable"
                trends_str += f"  - {skill}: {trend} ({scores[-2]:.3f} -> {scores[-1]:.3f})\n"

        difficulties = [d.value for d in DifficultyLevel]

        return (
            f"You are a curriculum manager for an AI agent training system.\n"
            f"Based on the agent's latest performance, decide whether to:\n"
            f"1. ADVANCE - Increase difficulty (agent is ready for harder tasks)\n"
            f"2. STAY - Keep current difficulty (agent needs more practice)\n"
            f"3. REGRESS - Decrease difficulty (agent is struggling)\n\n"
            f"Current stage: {self.current_stage}\n"
            f"Current difficulty: {self.current_difficulty.value}\n"
            f"Overall score: {result.overall_score:.3f}\n\n"
            f"Skill scores:\n{skill_scores_str}\n\n"
            f"Weak skills (below threshold):\n{weak_str}\n\n"
            f"Strong skills (above threshold):\n{strong_str}\n\n"
            f"Skill trends (recent changes):\n{trends_str or '  (insufficient data)'}\n\n"
            f"Return a JSON object with:\n"
            f"- 'action': 'advance', 'stay', or 'regress'\n"
            f"- 'reasoning': Brief explanation (1-2 sentences)\n"
            f"- 'suggested_difficulty': One of {difficulties}\n"
            f"- 'focus_skills': List of skills to focus on next\n"
            f"- 'confidence': Float between 0 and 1\n"
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM text response.

        Args:
            text: Raw text response from LLM.

        Returns:
            Parsed JSON dict.
        """
        # Try to find JSON in the response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")

    def _parse_llm_decision(
        self, response: Dict[str, Any], result: EvaluationResult
    ) -> CurriculumDecision:
        """Parse LLM response into CurriculumDecision.

        Args:
            response: Parsed JSON response from LLM.
            result: Latest evaluation result (for fallback values).

        Returns:
            CurriculumDecision object.
        """
        action = response.get("action", "stay").lower()
        if action not in ("advance", "stay", "regress"):
            action = "stay"

        reasoning = response.get(
            "reasoning",
            f"LLM decided to {action} at stage {self.current_stage}",
        )

        # Parse difficulty
        diff_str = response.get("suggested_difficulty", self.current_difficulty.value)
        try:
            suggested_difficulty = DifficultyLevel(diff_str)
        except ValueError:
            suggested_difficulty = self.current_difficulty

        focus_skills = response.get("focus_skills", [])
        if not focus_skills:
            weak = result.get_weak_skills()
            focus_skills = [ss.skill_name for ss in weak]

        confidence = float(response.get("confidence", 0.5))

        decision = CurriculumDecision(
            action=action,
            reasoning=reasoning,
            suggested_difficulty=suggested_difficulty,
            focus_skills=focus_skills,
            confidence=confidence,
        )

        # Record decision
        self.decision_history.append({
            "stage": self.current_stage,
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "overall_score": result.overall_score,
            "skill_scores": {ss.skill_name: ss.score for ss in result.skill_scores},
        })

        return decision

    def _fallback_to_decision(
        self, record: Dict[str, Any], result: EvaluationResult
    ) -> CurriculumDecision:
        """Convert fallback manager record to CurriculumDecision.

        Args:
            record: Record from fallback CurriculumManager.
            result: Latest evaluation result.

        Returns:
            CurriculumDecision object.
        """
        action = record.get("action", "stay")
        reasoning = " ".join(record.get("reasoning", []))

        weak = result.get_weak_skills()
        focus_skills = [ss.skill_name for ss in weak]

        # Determine suggested difficulty based on action
        difficulties = list(DifficultyLevel)
        current_idx = difficulties.index(self.current_difficulty)

        if action == "advance":
            suggested = difficulties[min(current_idx + 1, len(difficulties) - 1)]
        elif action == "regress":
            suggested = difficulties[max(current_idx - 1, 0)]
        else:
            suggested = self.current_difficulty

        decision = CurriculumDecision(
            action=action,
            reasoning=reasoning or f"Rule-based decision: {action}",
            suggested_difficulty=suggested,
            focus_skills=focus_skills,
            confidence=0.7,
        )

        # Record decision in history
        self.decision_history.append({
            "stage": self.current_stage,
            "action": action,
            "reasoning": decision.reasoning,
            "confidence": 0.7,
            "overall_score": result.overall_score,
            "skill_scores": {ss.skill_name: ss.score for ss in result.skill_scores},
        })

        return decision

    def _apply_decision(self, decision: CurriculumDecision) -> None:
        """Apply a curriculum decision to update state.

        Args:
            decision: The curriculum decision to apply.
        """
        if decision.action == "advance":
            self.current_stage += 1
            self.current_difficulty = decision.suggested_difficulty
        elif decision.action == "regress":
            self.current_stage = max(0, self.current_stage - 1)
            difficulties = list(DifficultyLevel)
            idx = difficulties.index(self.current_difficulty)
            if idx > 0:
                self.current_difficulty = difficulties[idx - 1]
        # 'stay' keeps current stage and difficulty

    def get_current_difficulty(self) -> DifficultyLevel:
        """Get the current difficulty level.

        Returns:
            Current DifficultyLevel enum value.
        """
        return self.current_difficulty

    def get_current_stage(self) -> int:
        """Get the current stage number.

        Returns:
            Current stage index.
        """
        return self.current_stage

    def get_focus_skills(self) -> List[str]:
        """Get the skills that need focus.

        Returns:
            List of skill names to focus on.
        """
        if self.decision_history:
            return self.decision_history[-1].get("focus_skills", [])
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum manager statistics.

        Returns:
            Dict with stage, difficulty, decision counts, etc.
        """
        action_counts = {"advance": 0, "stay": 0, "regress": 0}
        for record in self.decision_history:
            action = record.get("action", "stay")
            if action in action_counts:
                action_counts[action] += 1

        return {
            "current_stage": self.current_stage,
            "current_difficulty": self.current_difficulty.value,
            "total_decisions": len(self.decision_history),
            "action_counts": action_counts,
            "skill_history_length": {
                k: len(v) for k, v in self.skill_history.items()
            },
        }
