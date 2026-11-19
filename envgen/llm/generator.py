"""LLM integration for environment generation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from envgen.environments.base import EnvironmentConfig
from envgen.utils.types import DifficultyLevel, EvaluationResult

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract interface for LLM backends.

    Subclasses implement the actual API calls to different LLM providers
    (OpenAI, Anthropic, local models, etc.).
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The LLM's response text.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            Parsed JSON response.
        """
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and demonstration.

    This client generates plausible environment configurations without
    actually calling an LLM API. It uses rule-based generation based on
    the task description and evaluation feedback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._call_count += 1
        logger.info(f"MockLLM call #{self._call_count}: {prompt[:100]}...")
        return json.dumps(self._generate_response(prompt))

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        self._call_count += 1
        logger.info(f"MockLLM JSON call #{self._call_count}: {prompt[:100]}...")
        return self._generate_response(prompt)

    def _generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a mock response based on prompt analysis."""
        prompt_lower = prompt.lower()

        # Detect if this is an adaptation request (contains evaluation feedback)
        is_adaptation = any(kw in prompt_lower for kw in ["weak", "improve", "feedback", "performance", "score"])

        if is_adaptation:
            return self._generate_adaptation(prompt)
        return self._generate_initial(prompt)

    def _generate_initial(self, prompt: str) -> Dict[str, Any]:
        """Generate initial environment configurations."""
        prompt_lower = prompt.lower()

        # Extract difficulty from prompt
        if "expert" in prompt_lower:
            difficulty = DifficultyLevel.EXPERT
        elif "hard" in prompt_lower:
            difficulty = DifficultyLevel.HARD
        elif "medium" in prompt_lower:
            difficulty = DifficultyLevel.MEDIUM
        else:
            difficulty = DifficultyLevel.EASY

        # Generate environments based on detected skills/tasks
        skills = self._extract_skills(prompt)

        environments = []
        for i, skill in enumerate(skills):
            env_config = {
                "name": f"{skill}_training_{i}",
                "description": f"Environment for training {skill} skill",
                "difficulty": difficulty.value,
                "parameters": {
                    "skill_focus": skill,
                    "complexity": (i + 1) * 0.2,
                    "noise_level": 0.1 * (i + 1),
                },
                "target_skills": [skill],
                "max_steps": 50 + i * 25,
                "reward_scale": 1.0,
            }
            environments.append(env_config)

        return {
            "environments": environments,
            "reasoning": f"Generated {len(environments)} environments targeting skills: {skills}",
        }

    def _generate_adaptation(self, prompt: str) -> Dict[str, Any]:
        """Generate adapted environments based on feedback."""
        weak_skills = self._extract_weak_skills(prompt)

        environments = []
        for skill in weak_skills:
            env_config = {
                "name": f"{skill}_improvement",
                "description": f"Adapted environment to improve {skill} skill based on feedback",
                "difficulty": "medium",
                "parameters": {
                    "skill_focus": skill,
                    "complexity": 0.3,
                    "scaffolding": True,
                    "hint_frequency": 0.3,
                },
                "target_skills": [skill],
                "max_steps": 75,
                "reward_scale": 1.5,
            }
            environments.append(env_config)

        return {
            "environments": environments,
            "reasoning": f"Adapted environments to focus on weak skills: {weak_skills}",
        }

    def _extract_skills(self, prompt: str) -> List[str]:
        """Extract skill names from prompt."""
        # Simple keyword-based extraction
        skill_keywords = {
            "navigation": "navigation",
            "planning": "planning",
            "reasoning": "reasoning",
            "memory": "memory",
            "tool": "tool_use",
            "search": "search",
            "solve": "problem_solving",
            "pattern": "pattern_recognition",
            "logic": "logical_reasoning",
            "math": "mathematical_reasoning",
        }
        prompt_lower = prompt.lower()
        skills = [skill for kw, skill in skill_keywords.items() if kw in prompt_lower]
        return skills if skills else ["general_task"]

    def _extract_weak_skills(self, prompt: str) -> List[str]:
        """Extract weak skills from feedback prompt."""
        # Look for patterns like "weak at X" or "score: X: 0.3"
        skills = []
        lines = prompt.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if "weak" in line_lower or "low" in line_lower:
                # Try to extract skill name
                parts = line_lower.split()
                for i, part in enumerate(parts):
                    if part in ("at", "is", "score"):
                        if i + 1 < len(parts):
                            skills.append(parts[i + 1].strip(",.:;"))
        return skills if skills else ["general_task"]

    @property
    def call_count(self) -> int:
        return self._call_count


class EnvironmentGenerator:
    """Generates training environments using an LLM.

    The generator prompts the LLM to create environment configurations
    based on task descriptions and agent performance feedback. It follows
    the EnvGen paradigm: generate → train → evaluate → adapt.

    Workflow:
    1. Initial generation: LLM creates diverse environments for all skills
    2. Training: Agent trains in generated environments
    3. Evaluation: Agent performance is measured per skill
    4. Adaptation: LLM adapts environments to focus on weak skills
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm_client = llm_client
        self.config = config or {}
        self.generated_configs: List[EnvironmentConfig] = []
        self.generation_history: List[Dict[str, Any]] = []

    def generate_initial(
        self,
        task_description: str,
        skills: List[str],
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        num_environments: int = 5,
    ) -> List[EnvironmentConfig]:
        """Generate initial set of training environments.

        Args:
            task_description: Description of the overall task.
            skills: List of skills to train.
            difficulty: Base difficulty level.
            num_environments: Number of environments to generate.

        Returns:
            List of environment configurations.
        """
        prompt = self._build_initial_prompt(task_description, skills, difficulty, num_environments)
        response = self.llm_client.generate_json(prompt)

        configs = self._parse_response(response)
        self.generated_configs = configs

        self.generation_history.append(
            {
                "type": "initial",
                "task": task_description,
                "skills": skills,
                "num_generated": len(configs),
                "response": response,
            }
        )

        logger.info(f"Generated {len(configs)} initial environments")
        return configs

    def adapt_environments(
        self,
        evaluation_result: EvaluationResult,
        current_configs: List[EnvironmentConfig],
        task_description: str,
    ) -> List[EnvironmentConfig]:
        """Adapt environments based on evaluation feedback.

        Args:
            evaluation_result: Latest evaluation results.
            current_configs: Current environment configurations.
            task_description: Description of the overall task.

        Returns:
            Updated environment configurations.
        """
        prompt = self._build_adaptation_prompt(evaluation_result, current_configs, task_description)
        response = self.llm_client.generate_json(prompt)

        new_configs = self._parse_response(response)

        # Merge with existing configs, keeping successful ones
        merged = self._merge_configs(current_configs, new_configs, evaluation_result)
        self.generated_configs = merged

        self.generation_history.append(
            {
                "type": "adaptation",
                "evaluation_summary": {
                    "overall_score": evaluation_result.overall_score,
                    "weak_skills": [f"{ss.skill_name}:{ss.score:.2f}" for ss in evaluation_result.get_weak_skills()],
                },
                "num_generated": len(new_configs),
                "num_merged": len(merged),
                "response": response,
            }
        )

        logger.info(f"Adapted environments: {len(current_configs)} -> {len(merged)}")
        return merged

    def _build_initial_prompt(
        self,
        task_description: str,
        skills: List[str],
        difficulty: DifficultyLevel,
        num_environments: int,
    ) -> str:
        """Build the initial generation prompt."""
        return (
            f"Generate {num_environments} training environments for the following task:\n\n"
            f"Task: {task_description}\n\n"
            f"Skills to train: {', '.join(skills)}\n"
            f"Difficulty level: {difficulty.value}\n\n"
            f"Create diverse environments that allow an agent to learn these skills in parallel.\n"
            f"Each environment should focus on specific skills while contributing to overall competence.\n"
            f"Return a JSON object with an 'environments' array."
        )

    def _build_adaptation_prompt(
        self,
        result: EvaluationResult,
        configs: List[EnvironmentConfig],
        task_description: str,
    ) -> str:
        """Build the adaptation prompt with performance feedback."""
        weak = result.get_weak_skills()
        strong = result.get_strong_skills()

        weak_str = "\n".join(f"  - {ss.skill_name}: {ss.score:.2f} (raw: {ss.raw_value:.2f})" for ss in weak)
        strong_str = "\n".join(f"  - {ss.skill_name}: {ss.score:.2f} (raw: {ss.raw_value:.2f})" for ss in strong)

        current_envs = "\n".join(
            f"  - {c.name} (difficulty: {c.difficulty.value}, skills: {c.target_skills})" for c in configs
        )

        return (
            f"Adapt training environments based on agent performance feedback.\n\n"
            f"Task: {task_description}\n\n"
            f"Overall score: {result.overall_score:.2f}\n\n"
            f"Weak skills (need improvement):\n{weak_str}\n\n"
            f"Strong skills:\n{strong_str}\n\n"
            f"Current environments:\n{current_envs}\n\n"
            f"Generate new or modified environments that specifically target the weak skills.\n"
            f"Consider:\n"
            f"1. Reducing difficulty for weak skill environments\n"
            f"2. Adding scaffolding or hints\n"
            f"3. Increasing reward density for weak skills\n"
            f"4. Creating focused micro-tasks for each weak skill\n\n"
            f"Return a JSON object with an 'environments' array."
        )

    def _parse_response(self, response: Dict[str, Any]) -> List[EnvironmentConfig]:
        """Parse LLM response into EnvironmentConfig objects."""
        configs = []
        env_data = response.get("environments", [])

        for item in env_data:
            try:
                config = EnvironmentConfig.from_dict(item)
                configs.append(config)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse environment config: {e}")
                continue

        return configs

    def _merge_configs(
        self,
        current: List[EnvironmentConfig],
        new: List[EnvironmentConfig],
        result: EvaluationResult,
    ) -> List[EnvironmentConfig]:
        """Merge current and new configs, keeping effective ones."""
        # Keep configs that target strong skills
        strong_skills = {ss.skill_name for ss in result.get_strong_skills()}
        kept = [c for c in current if any(s in strong_skills for s in c.target_skills)]

        # Add all new configs
        merged = kept + new

        # Deduplicate by name
        seen = set()
        deduplicated = []
        for c in merged:
            if c.name not in seen:
                seen.add(c.name)
                deduplicated.append(c)

        return deduplicated
