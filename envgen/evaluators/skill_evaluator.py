"""Skill-based evaluator for measuring agent performance.

This evaluator measures agent performance across multiple skill dimensions
by analyzing episodes and computing skill-specific scores.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from envgen.evaluators.base import Evaluator
from envgen.utils.types import Episode, EvaluationResult, SkillScore


class SkillEvaluator(Evaluator):
    """Evaluator that measures performance across skill dimensions.

    Skills measured:
    - navigation: How efficiently the agent moves through the environment
    - planning: How well the agent plans toward goals
    - reasoning: How well the agent makes correct decisions
    - memory: How well the agent remembers and uses information
    - efficiency: How few steps the agent uses to complete tasks

    Each skill is scored in [0, 1] based on episode analysis.
    """

    def __init__(self, name: str = "skill_evaluator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        self.skill_names = [
            "navigation",
            "planning",
            "reasoning",
            "memory",
            "efficiency",
        ]

        # Scoring parameters
        self.success_weight: float = self.config.get("success_weight", 0.4)
        self.progress_weight: float = self.config.get("progress_weight", 0.3)
        self.efficiency_weight: float = self.config.get("efficiency_weight", 0.3)

    def evaluate(self, episodes: List[Episode], agent_id: str) -> EvaluationResult:
        """Evaluate agent performance from episodes.

        Args:
            episodes: Episodes to evaluate.
            agent_id: ID of the agent.

        Returns:
            Evaluation result with skill scores.
        """
        if not episodes:
            return EvaluationResult(
                agent_id=agent_id,
                overall_score=0.0,
            )

        # Compute skill scores
        skill_scores = self._compute_skill_scores(episodes)

        # Compute overall score (weighted average)
        if skill_scores:
            overall_score = statistics.mean([ss.score for ss in skill_scores])
        else:
            overall_score = 0.0

        # Compute episode statistics
        episode_stats = self._compute_episode_stats(episodes)

        # Get environment configs
        env_configs = []
        for ep in episodes:
            if ep.environment_config and ep.environment_config not in env_configs:
                env_configs.append(ep.environment_config)

        return EvaluationResult(
            agent_id=agent_id,
            skill_scores=skill_scores,
            overall_score=overall_score,
            episode_stats=episode_stats,
            environment_configs=env_configs,
        )

    def get_skill_names(self) -> List[str]:
        """Return the list of skill names."""
        return self.skill_names.copy()

    def _compute_skill_scores(self, episodes: List[Episode]) -> List[SkillScore]:
        """Compute scores for each skill dimension."""
        scores = {}

        for skill in self.skill_names:
            skill_episodes = []
            for ep in episodes:
                score = self._score_skill(ep, skill)
                skill_episodes.append(score)

            if skill_episodes:
                avg_score = statistics.mean(skill_episodes)
                # Normalize to [0, 1]
                avg_score = max(0.0, min(1.0, avg_score))

                raw_value = sum(self._get_raw_skill_value(ep, skill) for ep in episodes) / len(episodes)

                scores[skill] = SkillScore(
                    skill_name=skill,
                    score=avg_score,
                    raw_value=raw_value,
                )

        return list(scores.values())

    def _score_skill(self, episode: Episode, skill: str) -> float:
        """Score a single episode for a specific skill."""
        if not episode.transitions:
            return 0.0

        if skill == "navigation":
            return self._score_navigation(episode)
        elif skill == "planning":
            return self._score_planning(episode)
        elif skill == "reasoning":
            return self._score_reasoning(episode)
        elif skill == "memory":
            return self._score_memory(episode)
        elif skill == "efficiency":
            return self._score_efficiency(episode)
        else:
            return 0.0

    def _score_navigation(self, episode: Episode) -> float:
        """Score navigation skill based on movement efficiency."""
        if not episode.transitions:
            return 0.0

        # Count successful moves vs total moves
        successful_moves = 0
        total_moves = 0
        skills = []

        for t in episode.transitions:
            if t.action.action_type == "move":
                total_moves += 1
                if t.info.get("moved", False):
                    successful_moves += 1
            skills.extend(t.info.get("skills_demonstrated", []))

        # Navigation skill demonstration
        nav_count = skills.count("navigation")

        if total_moves == 0:
            return 0.1

        move_success_rate = successful_moves / total_moves
        nav_skill_rate = min(1.0, nav_count / max(1, len(episode.transitions)))

        return 0.6 * move_success_rate + 0.4 * nav_skill_rate

    def _score_planning(self, episode: Episode) -> float:
        """Score planning skill based on goal-directed behavior."""
        if not episode.transitions:
            return 0.0

        skills = []
        for t in episode.transitions:
            skills.extend(t.info.get("skills_demonstrated", []))

        plan_count = skills.count("planning")
        plan_rate = min(1.0, plan_count / max(1, len(episode.transitions)))

        # Bonus for completing the task
        success_bonus = 0.0
        for t in episode.transitions:
            if t.info.get("goal_reached", False):
                success_bonus = 0.3
                break

        return min(1.0, plan_rate + success_bonus)

    def _score_reasoning(self, episode: Episode) -> float:
        """Score reasoning skill based on correct decisions."""
        if not episode.transitions:
            return 0.0

        skills = []
        for t in episode.transitions:
            skills.extend(t.info.get("skills_demonstrated", []))

        reasoning_count = skills.count("reasoning")
        reasoning_rate = min(1.0, reasoning_count / max(1, len(episode.transitions)))

        # Bonus for successful collections
        collect_success = 0
        collect_attempts = 0
        for t in episode.transitions:
            if t.action.action_type == "collect":
                collect_attempts += 1
                if t.info.get("collected"):
                    collect_success += 1

        if collect_attempts > 0:
            collect_rate = collect_success / collect_attempts
            return 0.5 * reasoning_rate + 0.5 * collect_rate

        return reasoning_rate

    def _score_memory(self, episode: Episode) -> float:
        """Score memory skill based on item collection and revisiting."""
        if not episode.transitions:
            return 0.0

        skills = []
        for t in episode.transitions:
            skills.extend(t.info.get("skills_demonstrated", []))

        memory_count = skills.count("memory")
        memory_rate = min(1.0, memory_count / max(1, len(episode.transitions)))

        # Check items collected
        items_collected = 0
        for t in episode.transitions:
            if t.info.get("collected"):
                items_collected += 1

        # Get max items from env config
        max_items = 3
        if episode.environment_config:
            max_items = episode.environment_config.get("parameters", {}).get("num_items", 3)

        collection_rate = items_collected / max(1, max_items)

        return 0.4 * memory_rate + 0.6 * min(1.0, collection_rate)

    def _score_efficiency(self, episode: Episode) -> float:
        """Score efficiency based on steps used vs optimal."""
        if not episode.transitions:
            return 0.0

        # Get grid size to estimate optimal path
        grid_size = 5
        if episode.environment_config:
            grid_size = episode.environment_config.get("parameters", {}).get("grid_size", 5)

        # Optimal steps ≈ grid_size * num_items (rough estimate)
        num_items = 3
        if episode.environment_config:
            num_items = episode.environment_config.get("parameters", {}).get("num_items", 3)

        optimal_steps = grid_size * num_items
        actual_steps = episode.length

        # Efficiency = optimal / actual (capped at 1.0)
        if actual_steps == 0:
            return 0.0

        efficiency = optimal_steps / max(1, actual_steps)
        return min(1.0, efficiency)

    def _get_raw_skill_value(self, episode: Episode, skill: str) -> float:
        """Get raw (unnormalized) value for a skill."""
        count = 0
        for t in episode.transitions:
            if skill in t.info.get("skills_demonstrated", []):
                count += 1
        return count

    def _compute_episode_stats(self, episodes: List[Episode]) -> Dict[str, float]:
        """Compute aggregate statistics over episodes."""
        if not episodes:
            return {}

        rewards = [ep.total_reward for ep in episodes]
        lengths = [ep.length for ep in episodes]

        # Count successes
        successes = 0
        for ep in episodes:
            for t in ep.transitions:
                if t.info.get("goal_reached", False):
                    successes += 1
                    break

        return {
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_length": statistics.mean(lengths),
            "success_rate": successes / len(episodes),
            "total_episodes": len(episodes),
        }
