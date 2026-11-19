"""Multi-agent training pipeline for simultaneous agent training.

This pipeline trains multiple agents simultaneously in a shared environment,
supporting both competitive (zero-sum) and cooperative (shared reward) modes.
Each agent maintains its own experience buffer, policy updates, and evaluation.

Example:
    pipeline = MultiAgentPipeline(
        num_agents=2,
        mode="competitive",
        config=pipeline_config,
        ...
    )
    results = pipeline.train()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.base import Evaluator
from envgen.utils.types import (
    DifficultyLevel,
    Episode,
    EvaluationResult,
    Observation,
    Action,
    Transition,
)

logger = logging.getLogger(__name__)


class MultiAgentPipeline:
    """Training pipeline for multiple agents in shared environments.

    Trains N agents simultaneously, where each agent independently selects
    actions and receives rewards. Supports competitive modes (agents compete
    for resources) and cooperative modes (agents share rewards).

    Args:
        num_agents: Number of agents to train simultaneously.
        mode: Training mode — "competitive" or "cooperative".
        config: Pipeline configuration dict.
        agents: List of agent instances (one per agent slot).
        env_factory: Callable that creates environment instances.
        evaluator: Evaluator for measuring agent performance.
        curriculum_manager: Curriculum manager for difficulty progression.
        reward_engine: Reward engine for computing rewards.
        experiment_logger: Optional experiment logger.
    """

    def __init__(
        self,
        num_agents: int,
        mode: str,
        config: Dict[str, Any],
        agents: List[Agent],
        env_factory: Any,
        evaluator: Evaluator,
        curriculum_manager: Any,
        reward_engine: Any,
        experiment_logger: Any = None,
    ):
        """Initialize the multi-agent training pipeline.

        Args:
            num_agents: Number of agents to train.
            mode: "competitive" or "cooperative".
            config: Pipeline configuration.
            agents: List of agent instances.
            env_factory: Environment factory callable.
            evaluator: Performance evaluator.
            curriculum_manager: Curriculum manager.
            reward_engine: Reward computation engine.
            experiment_logger: Optional logger.
        """
        if len(agents) != num_agents:
            raise ValueError(
                f"Expected {num_agents} agents, got {len(agents)}"
            )

        self.num_agents = num_agents
        self.mode = mode
        self.config = config
        self.agents = agents
        self.env_factory = env_factory
        self.evaluator = evaluator
        self.curriculum = curriculum_manager
        self.reward_engine = reward_engine
        self.experiment_logger = experiment_logger

        # Per-agent episode storage
        self.agent_episodes: List[List[Episode]] = [[] for _ in range(num_agents)]

        # Training history
        self.history: List[Dict[str, Any]] = []

        # Current difficulty
        self.current_difficulty = config.get("initial_difficulty", "easy")

    def _create_environment(self) -> Environment:
        """Create a new environment instance."""
        return self.env_factory()

    def _collect_episode(
        self, env: Environment, max_steps: int = 100
    ) -> List[List[Transition]]:
        """Collect one episode of interactions from all agents.

        Args:
            env: Environment to interact with.
            max_steps: Maximum steps per episode.

        Returns:
            List of transition lists, one per agent.
        """
        env.reset()
        agent_transitions: List[List[Transition]] = [[] for _ in range(self.num_agents)]
        observations = [env.get_observation() for _ in range(self.num_agents)]
        available_actions = env.get_available_actions()

        done = False
        step = 0
        while not done and step < max_steps:
            step += 1

            # Each agent selects an action
            actions: List[Action] = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(observations[i], available_actions)
                actions.append(action)

            # Environment steps with the primary agent's action
            # In multi-agent mode, we combine actions or use the first one
            primary_action = actions[0]
            next_obs, reward, done, info = env.step(primary_action)

            # In cooperative mode, all agents share the reward
            # In competitive mode, rewards may differ
            next_observations = [next_obs for _ in range(self.num_agents)]
            available_actions = env.get_available_actions()

            for i, agent in enumerate(self.agents):
                agent_reward = reward
                if self.mode == "competitive" and i > 0:
                    # In competitive mode, other agents get inverse reward
                    agent_reward = -reward * 0.5

                transition = Transition(
                    observation=observations[i],
                    action=actions[i],
                    reward=agent_reward,
                    next_observation=next_observations[i],
                    done=done,
                )
                agent_transitions[i].append(transition)

            observations = next_observations

        return agent_transitions

    def train_cycle(self, episodes_per_cycle: int = 20) -> Dict[str, Any]:
        """Run one training cycle: collect episodes, train agents, evaluate.

        Args:
            episodes_per_cycle: Number of episodes to collect per cycle.

        Returns:
            Dict with training results for this cycle.
        """
        env = self._create_environment()
        cycle_start = time.time()

        # Collect episodes
        for ep_idx in range(episodes_per_cycle):
            agent_transitions = self._collect_episode(env)

            # Create episodes per agent
            for i, agent in enumerate(self.agents):
                transitions = agent_transitions[i]
                if not transitions:
                    continue

                total_reward = sum(t.reward for t in transitions)
                episode = Episode(transitions=transitions, total_reward=total_reward)
                self.agent_episodes[i].append(episode)

        # Train agents on collected experience
        for i, agent in enumerate(self.agents):
            if self.agent_episodes[i]:
                agent.train(self.agent_episodes[i])

        cycle_time = time.time() - cycle_start

        # Evaluate all agents
        eval_results = self._evaluate_all(env)

        # Log results
        result = {
            "cycle": len(self.history) + 1,
            "mode": self.mode,
            "difficulty": self.current_difficulty,
            "episodes_collected": episodes_per_cycle,
            "cycle_time": cycle_time,
            "agent_results": eval_results,
        }
        self.history.append(result)

        # Log to experiment logger if available
        if self.experiment_logger:
            rewards = [r.get("mean_reward", 0.0) for r in eval_results]
            self.experiment_logger.log_cycle(
                cycle=len(self.history),
                rewards=rewards,
                difficulty=self.current_difficulty,
                episode_count=episodes_per_cycle,
                training_time=cycle_time,
                extra_metrics={"mode": self.mode, "num_agents": self.num_agents},
            )

        return result

    def _evaluate_all(self, env: Environment) -> List[Dict[str, Any]]:
        """Evaluate all agents.

        Args:
            env: Environment to evaluate in.

        Returns:
            List of evaluation result dicts, one per agent.
        """
        eval_results = []

        for i, agent in enumerate(self.agents):
            if not self.agent_episodes[i]:
                eval_results.append({
                    "agent_index": i,
                    "agent_name": agent.name,
                    "mean_reward": 0.0,
                    "overall_score": 0.0,
                    "skill_scores": {},
                })
                continue

            episodes = self.agent_episodes[i][-10:]  # Last 10 episodes
            eval_result = self.evaluator.evaluate(episodes, agent.name)

            result = {
                "agent_index": i,
                "agent_name": agent.name,
                "mean_reward": eval_result.episode_stats.get("mean_reward", 0.0),
                "overall_score": eval_result.overall_score,
                "skill_scores": {s.skill_name: s.score for s in eval_result.skill_scores},
            }
            eval_results.append(result)

            # Clear old episodes for next cycle
            self.agent_episodes[i] = episodes

        return eval_results

    def train(
        self,
        max_cycles: int = 10,
        episodes_per_cycle: int = 20,
    ) -> List[Dict[str, Any]]:
        """Run full multi-agent training.

        Args:
            max_cycles: Maximum number of training cycles.
            episodes_per_cycle: Episodes per cycle.

        Returns:
            List of cycle results.
        """
        print(f"\n{'=' * 60}")
        print(f"  Multi-Agent Training ({self.mode} mode)")
        print(f"  Agents: {self.num_agents}")
        print(f"  Cycles: {max_cycles}")
        print(f"{'=' * 60}")

        for cycle in range(max_cycles):
            result = self.train_cycle(episodes_per_cycle)

            # Print cycle summary
            print(f"\n  Cycle {result['cycle']}/{max_cycles} "
                  f"(difficulty: {result['difficulty']})")
            for agent_result in result["agent_results"]:
                print(f"    Agent {agent_result['agent_index']} ({agent_result['agent_name']}): "
                      f"score={agent_result['overall_score']:.3f}, "
                      f"reward={agent_result['mean_reward']:.2f}")

        return self.history

    def get_agent_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all agents.

        Returns:
            List of stats dicts, one per agent.
        """
        stats = []
        for i, agent in enumerate(self.agents):
            agent_stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
            agent_stats["agent_index"] = i
            agent_stats["agent_name"] = agent.name
            agent_stats["episodes_trained"] = len(self.agent_episodes[i])
            stats.append(agent_stats)
        return stats

    def save_checkpoints(self, checkpoint_dir: str) -> List[str]:
        """Save checkpoints for all agents.

        Args:
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            List of checkpoint file paths.
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        paths = []
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'save'):
                path = os.path.join(checkpoint_dir, f"agent_{i}.json")
                agent.save(path)
                paths.append(path)
        return paths
