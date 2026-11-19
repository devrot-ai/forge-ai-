"""Hyperparameter search for automated agent tuning.

This module provides automated grid search over agent and environment
hyperparameters. It evaluates each parameter combination using the
benchmark protocol with fixed seeds, then outputs ranked results
showing the best configurations.

Usage:
    from envgen.benchmark.hyperparam_search import HyperparameterSearch

    search = HyperparameterSearch(
        agent_type="ppo_torch",
        env_type="puzzle",
        param_grid={
            "learning_rate": [0.001, 0.01, 0.1],
            "hidden_size": [64, 128],
        },
        episodes=10,
    )
    results = search.run()
    search.print_results()
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from envgen.agents.base import Agent
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.utils.types import Action, DifficultyLevel, Episode, Observation

# Default parameter grids for common agent types
DEFAULT_PARAM_GRIDS = {
    "ppo": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "initial_epsilon": [0.5, 1.0],
        "epsilon_decay": [0.9, 0.95, 0.99],
    },
    "ppo_torch": {
        "learning_rate": [0.0003, 0.001, 0.003],
        "hidden_size": [64, 128, 256],
        "gamma": [0.9, 0.95, 0.99],
    },
    "grpo": {
        "learning_rate": [0.01, 0.05, 0.1],
        "group_size": [4, 8, 16],
        "temperature": [0.5, 1.0, 1.5],
    },
    "grpo_torch": {
        "learning_rate": [0.0003, 0.001, 0.003],
        "group_size": [4, 8],
        "hidden_size": [64, 128],
    },
}


class HyperparameterSearch:
    """Automated grid search over agent hyperparameters.

    Evaluates each parameter combination using a standardized protocol
    and ranks results by performance metrics.

    Attributes:
        agent_type: Type of agent to tune (e.g., "ppo", "ppo_torch").
        env_type: Environment type for evaluation.
        param_grid: Dictionary mapping parameter names to lists of values.
        episodes: Number of training episodes per configuration.
        difficulty: Difficulty level for evaluation.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        agent_type: str = "ppo",
        env_type: str = "puzzle",
        param_grid: Optional[Dict[str, List[Any]]] = None,
        episodes: int = 10,
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        seed: int = 42,
        agent_factory: Optional[Callable[..., Agent]] = None,
        env_factory: Optional[Callable[..., Environment]] = None,
    ):
        """Initialize the hyperparameter search.

        Args:
            agent_type: Type of agent to tune. Used to select default
                parameter grid if none is provided.
            env_type: Environment type for evaluation.
            param_grid: Dictionary mapping parameter names to lists of
                values to try. If None, uses default grid for agent_type.
            episodes: Number of training episodes per configuration.
            difficulty: Difficulty level for evaluation.
            seed: Base random seed for reproducibility.
            agent_factory: Optional custom factory function for creating
                agents. Signature: (agent_type, config) -> Agent.
            env_factory: Optional custom factory function for creating
                environments. Signature: (env_type, difficulty, seed) -> Environment.
        """
        self.agent_type = agent_type
        self.env_type = env_type
        self.param_grid = param_grid or DEFAULT_PARAM_GRIDS.get(agent_type, {"learning_rate": [0.01, 0.1]})
        self.episodes = episodes
        self.difficulty = difficulty
        self.seed = seed
        self._agent_factory = agent_factory
        self._env_factory = env_factory

        self._results: List[Dict[str, Any]] = []
        self._evaluator = SkillEvaluator()

    def run(self) -> List[Dict[str, Any]]:
        """Run the hyperparameter search.

        Evaluates each parameter combination and stores the results.

        Returns:
            List of result dictionaries, sorted by mean_reward descending.
        """
        self._results = []

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))

        total = len(combinations)
        print(f"Hyperparameter Search: {self.agent_type} on {self.env_type}")
        print(f"Parameters: {param_names}")
        print(f"Total combinations: {total}")
        print(f"Episodes per config: {self.episodes}")
        print("=" * 70)

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Create agent with these parameters
            agent = self._create_agent(params)

            # Create environment
            env = self._create_env(i)

            # Train and evaluate
            start_time = time.time()
            episodes_list = self._train_agent(agent, env)
            elapsed = time.time() - start_time

            # Evaluate
            eval_result = self._evaluator.evaluate(episodes_list, self.agent_type)
            mean_reward = sum(ep.total_reward for ep in episodes_list) / max(1, len(episodes_list))

            skill_dict = {ss.skill_name: ss.score for ss in eval_result.skill_scores}

            result = {
                "params": params,
                "mean_reward": mean_reward,
                "skill_scores": skill_dict,
                "weak_skills": [ss.skill_name for ss in eval_result.get_weak_skills()],
                "strong_skills": [ss.skill_name for ss in eval_result.get_strong_skills()],
                "episodes_trained": len(episodes_list),
                "time_seconds": elapsed,
                "rank": 0,  # Filled in after sorting
            }
            self._results.append(result)

            # Progress indicator
            if (i + 1) % 5 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {params} -> reward={mean_reward:.2f}")

        # Sort by mean_reward descending
        self._results.sort(key=lambda r: r["mean_reward"], reverse=True)

        # Assign ranks
        for rank, result in enumerate(self._results, 1):
            result["rank"] = rank

        return self._results

    def _create_agent(self, params: Dict[str, Any]) -> Agent:
        """Create an agent with the given parameters.

        Args:
            params: Hyperparameter configuration.

        Returns:
            Configured agent instance.
        """
        if self._agent_factory:
            return self._agent_factory(self.agent_type, params)

        # Default factory
        config = {"name": f"hp_search_{self.agent_type}"}
        config.update(params)

        if self.agent_type == "ppo":
            from envgen.agents.ppo_agent import PPOAgent

            return PPOAgent(name=config.pop("name", "ppo"), config=config)
        elif self.agent_type == "ppo_torch":
            try:
                from envgen.agents.ppo_torch import PPOAgentTorch

                # PPOAgentTorch needs action_space; we'll set a default
                if "action_space" not in config:
                    config["action_space"] = 6
                return PPOAgentTorch(**config)
            except ImportError:
                raise ImportError("ppo_torch requires torch and numpy")
        elif self.agent_type == "grpo":
            from envgen.agents.grpo_agent import GRPOAgent

            return GRPOAgent(name=config.pop("name", "grpo"), config=config)
        elif self.agent_type == "grpo_torch":
            try:
                from envgen.agents.grpo_torch import GRPOAgentTorch

                if "action_space" not in config:
                    config["action_space"] = 6
                return GRPOAgentTorch(**config)
            except ImportError:
                raise ImportError("grpo_torch requires torch and numpy")
        else:
            # Fallback to PPOAgent
            from envgen.agents.ppo_agent import PPOAgent

            return PPOAgent(name=config.pop("name", "ppo"), config=config)

    def _create_env(self, seed_offset: int = 0) -> Environment:
        """Create an environment for evaluation.

        Args:
            seed_offset: Offset for the random seed.

        Returns:
            Configured environment instance.
        """
        if self._env_factory:
            return self._env_factory(self.env_type, self.difficulty, self.seed + seed_offset)

        seed = self.seed + seed_offset
        params = {"seed": seed}

        if self.env_type == "puzzle":
            params["grid_size"] = 5
            params["num_items"] = 3
            params["num_obstacles"] = 2
            config = EnvironmentConfig(
                name=f"hp_search_{self.env_type}",
                description="Hyperparameter search environment",
                difficulty=self.difficulty,
                parameters=params,
                max_steps=50,
            )
            from envgen.environments.puzzle_env import PuzzleEnvironment

            return PuzzleEnvironment(config)
        elif self.env_type == "maze":
            params["maze_width"] = 7
            params["maze_height"] = 7
            config = EnvironmentConfig(
                name=f"hp_search_{self.env_type}",
                description="Hyperparameter search environment",
                difficulty=self.difficulty,
                parameters=params,
                max_steps=50,
            )
            from envgen.environments.maze_env import MazeEnvironment

            return MazeEnvironment(config)
        elif self.env_type == "resource":
            params["num_resource_types"] = 3
            params["num_tasks"] = 3
            config = EnvironmentConfig(
                name=f"hp_search_{self.env_type}",
                description="Hyperparameter search environment",
                difficulty=self.difficulty,
                parameters=params,
                max_steps=30,
            )
            from envgen.environments.resource_env import ResourceEnvironment

            return ResourceEnvironment(config)
        elif self.env_type == "text":
            params["task_type"] = "mixed"
            config = EnvironmentConfig(
                name=f"hp_search_{self.env_type}",
                description="Hyperparameter search environment",
                difficulty=self.difficulty,
                parameters=params,
                max_steps=10,
            )
            from envgen.environments.text_env import TextTaskEnvironment

            return TextTaskEnvironment(config)
        else:
            # Default to puzzle
            config = EnvironmentConfig(
                name=f"hp_search_{self.env_type}",
                description="Hyperparameter search environment",
                difficulty=self.difficulty,
                parameters=params,
                max_steps=50,
            )
            from envgen.environments.puzzle_env import PuzzleEnvironment

            return PuzzleEnvironment(config)

    def _train_agent(
        self,
        agent: Agent,
        env: Environment,
    ) -> List[Episode]:
        """Train an agent in an environment for the specified episodes.

        Args:
            agent: The agent to train.
            env: The environment to train in.

        Returns:
            List of episodes collected during training.
        """
        episodes_list = []

        for _ in range(self.episodes):
            obs = env.reset()
            transitions = []

            while not env.done and env.current_step < env.max_steps:
                available = env.get_available_actions()
                action = agent.select_action(obs, available)

                next_obs, reward, done, info = env.step(action)

                from envgen.utils.types import Transition

                transition = Transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=done,
                )
                transitions.append(transition)
                obs = next_obs

                if done:
                    break

            episode = Episode(
                transitions=transitions,
                total_reward=sum(t.reward for t in transitions),
            )
            episodes_list.append(episode)
            agent.train([episode])

        return episodes_list

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best parameter configuration found.

        Returns:
            Dictionary with the best parameters and its results, or None
            if search hasn't been run.
        """
        if not self._results:
            return None
        return self._results[0]

    def print_results(self, top_n: int = 10) -> None:
        """Print a formatted table of the top results.

        Args:
            top_n: Number of top results to display.
        """
        if not self._results:
            print("No results yet. Run search first.")
            return

        results = self._results[:top_n]

        # Determine parameter names
        param_names = list(self.param_grid.keys())

        # Build header
        header = f"{'Rank':>4} {'Reward':>8}"
        for param in param_names:
            header += f" {param:>10}"
        header += f" {'Nav':>6} {'Eff':>6} {'Time':>6}"

        print()
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for result in results:
            line = f"{result['rank']:>4} {result['mean_reward']:>8.2f}"
            for param in param_names:
                val = result["params"].get(param, "")
                line += f" {str(val):>10}"
            nav = result["skill_scores"].get("navigation", 0.0)
            eff = result["skill_scores"].get("efficiency", 0.0)
            line += f" {nav:>6.3f} {eff:>6.3f} {result['time_seconds']:>6.2f}"
            print(line)

        print("=" * len(header))
        print(f"\nBest configuration: {results[0]['params']}")
        print(f"Best mean reward: {results[0]['mean_reward']:.2f}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results sorted by reward.

        Returns:
            List of all result dictionaries.
        """
        return self._results.copy()
