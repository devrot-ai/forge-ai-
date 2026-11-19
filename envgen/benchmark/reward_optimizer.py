"""Reward function optimizer for automated reward weight tuning.

This module provides automated grid search over reward function parameters
(success_weight, progress_weight, step_penalty, shaping_weight). It evaluates
each reward configuration using the benchmark protocol with fixed seeds,
then outputs ranked results showing the best reward configurations.

Usage:
    from envgen.benchmark.reward_optimizer import RewardOptimizer

    optimizer = RewardOptimizer(
        agent_type="ppo_torch",
        env_type="puzzle",
        reward_grid={
            "success_weight": [5, 10, 20],
            "progress_weight": [0.1, 0.5, 1.0],
            "step_penalty": [0.01, 0.1, 0.5],
        },
        episodes=20,
    )
    results = optimizer.run()
    optimizer.print_results()
"""

from __future__ import annotations

import itertools
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from envgen.agents.base import Agent
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import Action, DifficultyLevel, Episode, Observation, Transition

logger = logging.getLogger(__name__)

# Default reward parameter grids
DEFAULT_REWARD_GRIDS = {
    "simple": {
        "success_weight": [5, 10, 20],
        "progress_weight": [0.1, 0.5, 1.0],
        "step_penalty": [0.01, 0.1, 0.5],
    },
    "detailed": {
        "success_weight": [1, 5, 10, 20],
        "progress_weight": [0.1, 0.5, 1.0, 2.0],
        "step_penalty": [0.01, 0.05, 0.1, 0.5],
        "shaping_weight": [0.0, 0.5, 1.0],
    },
}


class RewardOptimizer:
    """Automated grid search over reward function parameters.

    Evaluates each reward parameter combination using a standardized training
    protocol and ranks results by agent performance metrics.

    Attributes:
        agent_type: Type of agent to use for evaluation.
        env_type: Environment type for evaluation.
        reward_grid: Dictionary mapping reward parameter names to lists of values.
        episodes: Number of training episodes per configuration.
        difficulty: Difficulty level for evaluation.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        agent_type: str = "ppo",
        env_type: str = "puzzle",
        reward_grid: Optional[Dict[str, List[Any]]] = None,
        episodes: int = 20,
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        seed: int = 42,
        agent_factory: Optional[Callable[..., Agent]] = None,
        env_factory: Optional[Callable[..., Environment]] = None,
    ):
        """Initialize the reward optimizer.

        Args:
            agent_type: Type of agent to use for evaluation.
            env_type: Environment type for evaluation.
            reward_grid: Dictionary mapping reward parameter names to lists
                of values to try. If None, uses default grid.
            episodes: Number of training episodes per configuration.
            difficulty: Difficulty level for evaluation.
            seed: Base random seed for reproducibility.
            agent_factory: Optional custom factory function for creating agents.
            env_factory: Optional custom factory function for creating environments.
        """
        self.agent_type = agent_type
        self.env_type = env_type
        self.reward_grid = reward_grid or DEFAULT_REWARD_GRIDS.get("simple")
        self.episodes = episodes
        self.difficulty = difficulty
        self.seed = seed
        self._agent_factory = agent_factory
        self._env_factory = env_factory

        self._results: List[Dict[str, Any]] = []

    def _create_agent(self, config: Optional[Dict[str, Any]] = None) -> Agent:
        """Create an agent instance."""
        if self._agent_factory:
            return self._agent_factory(self.agent_type, config or {})

        from envgen.config.loader import ConfigLoader

        loader_config = {"agent": {"type": self.agent_type}}
        if config:
            loader_config["agent"].update(config)
        loader = ConfigLoader()
        return loader.create_agent(loader_config)

    def _create_environment(self, seed: int) -> Environment:
        """Create an environment instance."""
        if self._env_factory:
            return self._env_factory(self.env_type, self.difficulty, seed)

        env_config = EnvironmentConfig(
            name=f"{self.env_type}_reward_opt",
            description=f"Reward optimization environment",
            difficulty=self.difficulty,
            parameters={
                "grid_size": 8,
                "num_items": 3,
                "num_obstacles": 5,
                "max_steps": 50,
            },
        )

        if self.env_type == "puzzle":
            from envgen.environments.puzzle_env import PuzzleEnvironment
            return PuzzleEnvironment(env_config)
        elif self.env_type == "maze":
            from envgen.environments.maze_env import MazeEnvironment
            return MazeEnvironment(env_config)
        elif self.env_type == "resource":
            from envgen.environments.resource_env import ResourceEnvironment
            return ResourceEnvironment(env_config)
        elif self.env_type == "text":
            from envgen.environments.text_env import TextTaskEnvironment
            return TextTaskEnvironment(env_config)
        elif self.env_type == "cartpole":
            from envgen.environments.cartpole_env import CartPoleEnvironment
            return CartPoleEnvironment(env_config)
        elif self.env_type == "multi_agent":
            from envgen.environments.multi_agent_env import MultiAgentEnvironment
            return MultiAgentEnvironment(env_config)
        else:
            from envgen.environments.puzzle_env import PuzzleEnvironment
            return PuzzleEnvironment(env_config)

    def _create_reward_engine(self, reward_params: Dict[str, float]) -> RewardEngine:
        """Create a reward engine with the given parameters."""
        reward_config = {
            "components": [
                {"name": "success", "weight": reward_params.get("success_weight", 10)},
                {"name": "progress", "weight": reward_params.get("progress_weight", 0.5)},
                {"name": "step_penalty", "weight": -reward_params.get("step_penalty", 0.1)},
            ],
            "clip_reward": True,
            "reward_shaping": reward_params.get("shaping_weight", 0.0) > 0,
        }
        return RewardEngine(reward_config)

    def _evaluate_config(
        self, reward_params: Dict[str, float], config_seed: int
    ) -> Dict[str, Any]:
        """Evaluate a single reward parameter configuration.

        Args:
            reward_params: Dictionary of reward parameters.
            config_seed: Random seed for this configuration.

        Returns:
            Dictionary with evaluation results.
        """
        random.seed(config_seed)

        agent = self._create_agent()
        env = self._create_environment(config_seed)
        reward_engine = self._create_reward_engine(reward_params)
        evaluator = SkillEvaluator()

        episodes: List[Episode] = []
        total_reward = 0.0

        for ep_idx in range(self.episodes):
            env.reset()
            observation = env.get_observation()
            transitions: List[Transition] = []
            ep_reward = 0.0

            step = 0
            done = False
            while not done and step < env.config.max_steps:
                available_actions = env.get_available_actions()
                action = agent.select_action(observation, available_actions)
                reward = env.step(action)
                next_observation = env.get_observation()
                done = env._done if hasattr(env, "_done") else False

                # Apply reward shaping
                transition_obj = Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                )
                shaped_reward = reward_engine.compute(transition_obj)

                transition = Transition(
                    observation=observation,
                    action=action,
                    reward=shaped_reward,
                    next_observation=next_observation,
                    done=done,
                )
                transitions.append(transition)
                ep_reward += shaped_reward
                observation = next_observation
                step += 1

            episode = Episode(
                transitions=transitions,
                total_reward=ep_reward,
            )
            episodes.append(episode)
            total_reward += ep_reward

            # Train agent on this episode
            agent.train([episode])

        # Evaluate agent skills
        eval_result = evaluator.evaluate(episodes, agent_id="reward_opt_agent")
        scores = [s.score for s in eval_result.skill_scores]
        mean_skill = sum(scores) / max(len(scores), 1)

        return {
            "reward_params": reward_params,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(self.episodes, 1),
            "mean_skill": mean_skill,
            "skill_scores": eval_result.skill_scores,
            "num_episodes": self.episodes,
        }

    def run(self) -> List[Dict[str, Any]]:
        """Run the reward optimization search.

        Returns:
            List of result dictionaries, sorted by mean_skill (descending).
        """
        param_names = sorted(self.reward_grid.keys())
        param_values = [self.reward_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))

        logger.info(
            f"Starting reward optimization: {len(combinations)} configurations, "
            f"{self.episodes} episodes each"
        )

        self._results = []
        start_time = time.time()

        for idx, combo in enumerate(combinations):
            reward_params = dict(zip(param_names, combo))
            config_seed = self.seed + idx
            result = self._evaluate_config(reward_params, config_seed)
            result["config_index"] = idx
            result["params_str"] = ", ".join(f"{k}={v}" for k, v in sorted(reward_params.items()))
            self._results.append(result)

            if (idx + 1) % 10 == 0 or idx == len(combinations) - 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress: {idx + 1}/{len(combinations)} configs, "
                    f"{elapsed:.1f}s elapsed"
                )

        # Sort by mean_skill descending
        self._results.sort(key=lambda r: r["mean_skill"], reverse=True)

        elapsed = time.time() - start_time
        logger.info(
            f"Reward optimization complete: {len(self._results)} configs in {elapsed:.1f}s"
        )

        return self._results

    def print_results(self, top_n: int = 10) -> str:
        """Print ranked results.

        Args:
            top_n: Number of top configurations to display.

        Returns:
            Formatted results string.
        """
        if not self._results:
            return "No results available. Run the optimizer first."

        lines = []
        lines.append("=" * 80)
        lines.append("REWARD OPTIMIZATION RESULTS")
        lines.append("=" * 80)
        lines.append(f"Agent: {self.agent_type} | Environment: {self.env_type}")
        lines.append(f"Difficulty: {self.difficulty.value} | Episodes: {self.episodes}")
        lines.append(f"Total configurations: {len(self._results)}")
        lines.append("-" * 80)

        results_to_show = self._results[:top_n]
        for rank, result in enumerate(results_to_show, 1):
            lines.append(f"\nRank #{rank}:")
            lines.append(f"  Parameters: {result['params_str']}")
            lines.append(f"  Mean skill: {result['mean_skill']:.4f}")
            lines.append(f"  Mean reward: {result['mean_reward']:.2f}")
            lines.append(f"  Total reward: {result['total_reward']:.2f}")

        lines.append("\n" + "=" * 80)

        output = "\n".join(lines)
        print(output)
        return output

    def get_best_config(self) -> Dict[str, Any]:
        """Get the best reward configuration found.

        Returns:
            Dictionary with the best reward parameters and results.
        """
        if not self._results:
            return {}
        return self._results[0]

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the Pareto front of reward configurations.

        Returns configurations that are not dominated by any other configuration
        in terms of both mean_skill and mean_reward.

        Returns:
            List of Pareto-optimal result dictionaries.
        """
        if not self._results:
            return []

        pareto = []
        for result in self._results:
            dominated = False
            for other in self._results:
                if other is result:
                    continue
                if (other["mean_skill"] >= result["mean_skill"] and
                    other["mean_reward"] >= result["mean_reward"] and
                    (other["mean_skill"] > result["mean_skill"] or
                     other["mean_reward"] > result["mean_reward"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(result)

        return pareto
