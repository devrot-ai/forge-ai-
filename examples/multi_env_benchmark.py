"""Multi-environment benchmark comparison script.

Trains a single agent across all 6 environments with fixed seeds and produces
a formatted comparison table with skill scores, rewards, and timing.

Usage:
    python examples/multi_env_benchmark.py --agent ppo --episodes 10 --cycles 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.WARNING)
logging.getLogger("envgen").setLevel(logging.WARNING)


ENV_TYPES = ["puzzle", "maze", "resource", "text", "cartpole", "multi_agent"]


def create_env_factory(env_type: str):
    """Create an environment factory for the specified environment type.

    Args:
        env_type: Environment type string.

    Returns:
        Callable that creates an environment instance.
    """
    from envgen.environments import EnvironmentConfig
    from envgen.utils.types import DifficultyLevel

    if env_type == "puzzle":
        from envgen.environments import PuzzleEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_puzzle",
                difficulty=DifficultyLevel.EASY,
                parameters={"grid_size": 8, "num_items": 3, "num_obstacles": 5},
            )
            return PuzzleEnvironment(config=config)

        return factory
    elif env_type == "maze":
        from envgen.environments import MazeEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_maze",
                difficulty=DifficultyLevel.EASY,
                parameters={"maze_size": 8, "complexity": 0.5},
            )
            return MazeEnvironment(config=config)

        return factory
    elif env_type == "resource":
        from envgen.environments import ResourceEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_resource",
                difficulty=DifficultyLevel.EASY,
                parameters={"num_resources": 3, "resource_rate": 1.0},
            )
            return ResourceEnvironment(config=config)

        return factory
    elif env_type == "text":
        from envgen.environments import TextTaskEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_text",
                difficulty=DifficultyLevel.EASY,
                parameters={"task_type": "mixed", "num_choices": 4},
            )
            return TextTaskEnvironment(config=config)

        return factory
    elif env_type == "cartpole":
        from envgen.environments import CartPoleEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_cartpole",
                difficulty=DifficultyLevel.EASY,
                parameters={"max_steps": 100},
            )
            return CartPoleEnvironment(config=config)

        return factory
    elif env_type == "multi_agent":
        from envgen.environments import MultiAgentEnvironment

        def factory(cfg=None):
            config = cfg or EnvironmentConfig(
                name="benchmark_multi_agent",
                difficulty=DifficultyLevel.EASY,
                parameters={"grid_size": 8, "num_items": 5, "cooperation_mode": False},
            )
            return MultiAgentEnvironment(config=config)

        return factory
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def create_agent(agent_type: str):
    """Create an agent instance by type.

    Args:
        agent_type: Agent type string.

    Returns:
        Agent instance.
    """
    if agent_type == "ppo":
        from envgen.agents import PPOAgent

        return PPOAgent(name="benchmark_agent", config={"learning_rate": 0.1, "epsilon": 0.2})
    elif agent_type == "grpo":
        from envgen.agents import GRPOAgent

        return GRPOAgent(name="benchmark_agent", config={"learning_rate": 0.1, "epsilon": 0.2})
    elif agent_type == "nn":
        from envgen.agents import NearestNeighborAgent

        return NearestNeighborAgent(name="benchmark_agent", config={"k": 5})
    elif agent_type in ("ppo_torch", "grpo_torch", "rnn", "transformer"):
        try:
            from envgen.config.loader import ConfigLoader

            loader = ConfigLoader()
            config = {"agent": {"type": agent_type}}
            return loader.create_agent(config)
        except ImportError as e:
            print(f"Skipping {agent_type}: {e}")
            return None
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def benchmark_single_env(
    agent,
    env_type: str,
    num_episodes: int,
    num_cycles: int,
) -> Dict[str, Any]:
    """Benchmark an agent on a single environment.

    Args:
        agent: Agent instance.
        env_type: Environment type string.
        num_episodes: Number of episodes per cycle.
        num_cycles: Number of training cycles.

    Returns:
        Dict with benchmark results.
    """
    from envgen.evaluators import SkillEvaluator

    env_factory = create_env_factory(env_type)
    evaluator = SkillEvaluator()

    start_time = time.time()
    total_reward = 0.0
    episodes_trained = 0

    for cycle in range(num_cycles):
        env = env_factory()
        episodes = []

        for _ in range(num_episodes):
            env.reset()
            episode_transitions = []
            episode_reward = 0.0
            done = False
            step = 0

            while not done and step < 100:
                obs = env.get_observation()
                action = agent.select_action(obs)
                reward, done, info = env.step(action)
                next_obs = env.get_observation()

                from envgen.utils.types import Transition

                transition = Transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=done,
                )
                episode_transitions.append(transition)
                episode_reward += reward
                step += 1

            from envgen.utils.types import Episode

            episode = Episode(transitions=episode_transitions, total_reward=episode_reward)
            episodes.append(episode)
            total_reward += episode_reward
            episodes_trained += 1

        # Train agent
        if episodes:
            agent.train(episodes)

    elapsed = time.time() - start_time

    # Evaluate
    env = env_factory()
    eval_episodes = []
    for _ in range(5):
        env.reset()
        transitions = []
        ep_reward = 0.0
        done = False
        step = 0
        while not done and step < 100:
            obs = env.get_observation()
            action = agent.select_action(obs)
            reward, done, info = env.step(action)
            next_obs = env.get_observation()
            from envgen.utils.types import Transition

            transitions.append(
                Transition(observation=obs, action=action, reward=reward, next_observation=next_obs, done=done)
            )
            ep_reward += reward
            step += 1
        from envgen.utils.types import Episode

        eval_episodes.append(Episode(transitions=transitions, total_reward=ep_reward))

    eval_result = evaluator.evaluate(eval_episodes)

    return {
        "env_type": env_type,
        "total_reward": total_reward,
        "avg_reward": total_reward / max(episodes_trained, 1),
        "episodes_trained": episodes_trained,
        "elapsed_seconds": round(elapsed, 2),
        "eval_success_rate": eval_result.success_rate if hasattr(eval_result, "success_rate") else 0.0,
        "eval_mean_skill": (
            eval_result.mean_skill if hasattr(eval_result, "mean_skill") else 0.0
        ),
    }


def run_benchmark(
    agent_type: str,
    env_types: List[str],
    num_episodes: int,
    num_cycles: int,
    output: Optional[str] = None,
):
    """Run benchmark across multiple environments.

    Args:
        agent_type: Agent type string.
        env_types: List of environment types to benchmark.
        num_episodes: Episodes per cycle.
        num_cycles: Number of cycles.
        output: Optional output JSON file path.
    """
    agent = create_agent(agent_type)
    if agent is None:
        print(f"Failed to create agent: {agent_type}")
        sys.exit(1)

    print("=" * 70)
    print("MULTI-ENVIRONMENT BENCHMARK")
    print("=" * 70)
    print(f"Agent: {agent_type}")
    print(f"Environments: {', '.join(env_types)}")
    print(f"Episodes per cycle: {num_episodes}")
    print(f"Cycles: {num_cycles}")
    print("=" * 70)

    results: List[Dict[str, Any]] = []

    for env_type in env_types:
        print(f"\nBenchmarking {env_type}...")
        try:
            result = benchmark_single_env(agent, env_type, num_episodes, num_cycles)
            results.append(result)
            print(f"  Avg reward: {result['avg_reward']:.2f}")
            print(f"  Time: {result['elapsed_seconds']:.2f}s")
            print(f"  Eval success rate: {result['eval_success_rate']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "env_type": env_type,
                    "error": str(e),
                    "avg_reward": 0.0,
                    "elapsed_seconds": 0.0,
                }
            )

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"{'Environment':<15}  {'Avg Reward':>12}  {'Time (s)':>10}  "
        f"{'Success Rate':>14}  {'Mean Skill':>12}"
    )
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['env_type']:<15}  {'ERROR':>12}  {'-':>10}  {'-':>14}  {'-':>12}")
        else:
            print(
                f"{r['env_type']:<15}  {r['avg_reward']:>12.2f}  {r['elapsed_seconds']:>10.2f}  "
                f"{r['eval_success_rate']:>14.2f}  {r['eval_mean_skill']:>12.2f}"
            )

    print("=" * 70)

    # Save to JSON if requested
    if output:
        with open(output, "w") as f:
            json.dump({"agent_type": agent_type, "results": results}, f, indent=2)
        print(f"\nResults saved to {output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-environment benchmark comparison")
    parser.add_argument(
        "--agent",
        default="ppo",
        choices=["ppo", "grpo", "nn", "ppo_torch", "grpo_torch", "rnn", "transformer"],
        help="Agent type to benchmark",
    )
    parser.add_argument(
        "--env-types",
        default=",".join(ENV_TYPES),
        help="Comma-separated list of environment types to benchmark",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Episodes per cycle"
    )
    parser.add_argument(
        "--cycles", type=int, default=2, help="Number of training cycles"
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON file path"
    )

    args = parser.parse_args()

    env_types = [e.strip() for e in args.env_types.split(",") if e.strip()]
    invalid = set(env_types) - set(ENV_TYPES)
    if invalid:
        print(f"Invalid environment types: {invalid}")
        print(f"Available: {', '.join(ENV_TYPES)}")
        sys.exit(1)

    run_benchmark(args.agent, env_types, args.episodes, args.cycles, args.output)


if __name__ == "__main__":
    main()
