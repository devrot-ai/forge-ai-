"""Curriculum learning management."""

from envgen.curriculum.manager import CurriculumManager, CurriculumStage

try:
    from envgen.curriculum.llm_curriculum import LLMCurriculumManager, CurriculumDecision
except ImportError:
    LLMCurriculumManager = None  # type: ignore
    CurriculumDecision = None  # type: ignore

__all__ = [
    "CurriculumManager",
    "CurriculumStage",
    "LLMCurriculumManager",
    "CurriculumDecision",
]
