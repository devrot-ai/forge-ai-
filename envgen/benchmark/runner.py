"""Benchmark runner for standardized agent evaluation.

This module provides a standardized way to evaluate agents across all environments
with fixed seeds, producing comparable performance scores. It runs each agent
through a fixed set of episodes in each environment type and difficulty level,
then aggregates the results into a summary table.

No external dependencies required — uses only the core framework.

Example:
    runner = BenchmarkRunner(agent, episodes_per_env=10)
    results = runner.run()
    runner.print_summary()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.utils.types import DifficultyLevel, Episode

# All available environment types and their factory functions
ENV_TYPES = [
    "puzzle",
    "maze",
    "resource",
    "text",
    "cartpole",
    "multi_agent",
]

DIFFICULTIES = [
    DifficultyLevel.EASY,
    DifficultyLevel.MEDIUM,
    DifficultyLevel.HARD,
]


def create_env(env_type: str, difficulty: DifficultyLevel, seed: int) -> Environment:
    """Create an environment of the specified type and difficulty.

    Args:
        env_type: Environment type name
        difficulty: Difficulty level
        seed: Random seed for reproducibility

    Returns:
        Configured environment instance
    """
    params = {"seed": seed}

    # Type-specific default parameters
    if env_type == "puzzle":
        params["grid_size"] = 5
        params["num_items"] = 3
        params["num_obstacles"] = 2
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters=params,
            max_steps=50,
        )
        from envgen.environments.puzzle_env import PuzzleEnvironment

        return PuzzleEnvironment(config)

    elif env_type == "maze":
        params["maze_width"] = 7
        params["maze_height"] = 7
        params["complexity"] = 0.5
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters=params,
            max_steps=50,
        )
        from envgen.environments.maze_env import MazeEnvironment

        return MazeEnvironment(config)

    elif env_type == "resource":
        params["num_resource_types"] = 3
        params["num_tasks"] = 3
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters=params,
            max_steps=30,
        )
        from envgen.environments.resource_env import ResourceEnvironment

        return ResourceEnvironment(config)

    elif env_type == "text":
        params["task_type"] = "mixed"
        params["num_choices"] = 4
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters=params,
            max_steps=10,
        )
        from envgen.environments.text_env import TextTaskEnvironment

        return TextTaskEnvironment(config)

    elif env_type == "cartpole":
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters={},
            max_steps=200,
        )
        from envgen.environments.cartpole_env import CartPoleEnvironment

        return CartPoleEnvironment(config)

    elif env_type == "multi_agent":
        params["grid_width"] = 8
        params["grid_height"] = 8
        params["num_items"] = 5
        params["target_count"] = 3
        config = EnvironmentConfig(
            name=f"benchmark_{env_type}_{difficulty.value}",
            description=f"Benchmark {env_type} at {difficulty.value}",
            difficulty=difficulty,
            parameters=params,
            max_steps=50,
        )
        from envgen.environments.multi_agent_env import MultiAgentEnvironment

        return MultiAgentEnvironment(config)

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


class BenchmarkResult:
    """Stores the results of a single benchmark run."""

    def __init__(
        self,
        env_type: str,
        difficulty: str,
        episodes: List[Episode],
        skill_scores: Dict[str, float],
        overall_score: float,
    ):
        self.env_type = env_type
        self.difficulty = difficulty
        self.episodes = episodes
        self.skill_scores = skill_scores
        self.overall_score = overall_score
        self.mean_reward = sum(ep.total_reward for ep in episodes) / len(episodes) if episodes else 0.0
        self.mean_steps = sum(ep.length for ep in episodes) / len(episodes) if episodes else 0.0
        self.success_rate = 0.0  # Approximated from rewards


class BenchmarkRunner:
    """Runs standardized benchmarks across environments.

    Evaluates an agent across all specified environment types and difficulty
    levels with fixed seeds for reproducibility.

    Configuration:
    - episodes_per_env: Number of episodes per environment (default: 10)
    - env_types: List of environment types to test (default: all)
    - difficulties: List of difficulty levels to test (default: EASY, MEDIUM, HARD)
    - seed: Base random seed (default: 42)

    Example:
        runner = BenchmarkRunner(agent, episodes_per_env=10)
        results = runner.run()
        runner.print_summary()
    """

    def __init__(
        self,
        agent: Agent,
        episodes_per_env: int = 10,
        env_types: Optional[List[str]] = None,
        difficulties: Optional[List[DifficultyLevel]] = None,
        seed: int = 42,
    ):
        """Initialize the benchmark runner.

        Args:
            agent: Agent to evaluate
            episodes_per_env: Episodes per environment type
            env_types: Environment types to test (default: all)
            difficulties: Difficulty levels to test (default: EASY, MEDIUM, HARD)
            seed: Base random seed
        """
        self.agent = agent
        self.episodes_per_env = episodes_per_env
        self.env_types = env_types or ENV_TYPES
        self.difficulties = difficulties or DIFFICULTIES
        self.seed = seed
        self.evaluator = SkillEvaluator()
        self.results: List[BenchmarkResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def run(self) -> List[BenchmarkResult]:
        """Run the benchmark across all environments.

        Returns:
            List of BenchmarkResult objects, one per environment-difficulty pair.
        """
        self.start_time = time.time()
        self.results = []

        for env_type in self.env_types:
            for difficulty in self.difficulties:
                result = self._run_single(env_type, difficulty)
                self.results.append(result)

        self.end_time = time.time()
        return self.results

    def _run_single(self, env_type: str, difficulty: DifficultyLevel) -> BenchmarkResult:
        """Run benchmark for a single environment type and difficulty.

        Args:
            env_type: Environment type name
            difficulty: Difficulty level

        Returns:
            BenchmarkResult for this configuration
        """
        seed = self.seed + hash(env_type) % 1000 + hash(difficulty.value) % 100
        env = create_env(env_type, difficulty, seed)
        episodes: List[Episode] = []

        for _ in range(self.episodes_per_env):
            env.reset()
            episode = self.agent.run_episode(env)
            episodes.append(episode)
            env.reset()

        # Evaluate
        if episodes:
            eval_result = self.evaluator.evaluate(episodes, self.agent.agent_id)
            skill_scores = {ss.skill_name: ss.score for ss in eval_result.skill_scores}
            overall = eval_result.overall_score
        else:
            skill_scores = {}
            overall = 0.0

        return BenchmarkResult(
            env_type=env_type,
            difficulty=difficulty.value,
            episodes=episodes,
            skill_scores=skill_scores,
            overall_score=overall,
        )

    def print_summary(self) -> str:
        """Print a formatted summary table of benchmark results.

        Returns:
            The summary as a string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("  EnvGen-LLM Benchmark Results")
        lines.append(f"  Agent: {self.agent.agent_id}")
        lines.append(f"  Episodes per env: {self.episodes_per_env}")
        lines.append("=" * 80)

        if not self.results:
            lines.append("  No results to display.")
            return "\n".join(lines)

        elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
        lines.append(f"  Total time: {elapsed:.1f}s")
        lines.append("")

        # Table header
        header = f"  {'Environment':<15} {'Difficulty':<10} {'Overall':>8} {'Mean Reward':>12} {'Mean Steps':>11}"
        lines.append(header)
        lines.append("  " + "-" * 70)

        for r in self.results:
            row = (
                f"  {r.env_type:<15} {r.difficulty:<10} "
                f"{r.overall_score:>8.3f} {r.mean_reward:>12.2f} {r.mean_steps:>11.1f}"
            )
            lines.append(row)

        lines.append("")

        # Aggregate scores by environment type
        lines.append("  Aggregate by Environment Type:")
        lines.append("  " + "-" * 40)
        env_totals: Dict[str, List[float]] = {}
        for r in self.results:
            env_totals.setdefault(r.env_type, []).append(r.overall_score)

        for env_type, scores in env_totals.items():
            avg = sum(scores) / len(scores)
            lines.append(f"  {env_type:<15} avg: {avg:.3f}")

        # Overall aggregate
        all_scores = [r.overall_score for r in self.results]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        lines.append("")
        lines.append(f"  Overall Average Score: {overall_avg:.3f}")
        lines.append("=" * 80)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def get_overall_score(self) -> float:
        """Get the overall average score across all benchmarks.

        Returns:
            Average overall score across all environment-difficulty pairs.
        """
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    def get_results_by_env(self, env_type: str) -> List[BenchmarkResult]:
        """Get results filtered by environment type.

        Args:
            env_type: Environment type to filter by.

        Returns:
            List of matching BenchmarkResult objects.
        """
        return [r for r in self.results if r.env_type == env_type]
