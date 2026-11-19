"""Performance benchmarks for EnvGen-LLM.

This module provides timing-based benchmarks for comparing environment execution
speed and agent training performance. It measures episode collection speed for
single vs parallel environments and training speed across different agents.

Usage:
    python -m benchmarks.env_speed
    python benchmarks/env_speed.py
"""

import time
from typing import Any, Dict, List, Optional

from envgen.agents.ppo_agent import PPOAgent
from envgen.environments.base import EnvironmentConfig
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.utils.types import Action, DifficultyLevel, Episode, Observation


def benchmark_env_speed(
    env_type: str = "puzzle",
    episodes: int = 50,
    workers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Benchmark environment execution speed.

    Compares single environment vs parallel environment execution speed.

    Args:
        env_type: Type of environment to benchmark ('puzzle', 'maze', 'resource', 'text').
        episodes: Number of episodes to run per configuration.
        workers: List of worker counts to test. Defaults to [1, 2, 4].

    Returns:
        Dictionary with benchmark results including time per episode and episodes per second.
    """
    if workers is None:
        workers = [1, 2, 4]

    results: Dict[str, Any] = {}

    # Create environment config
    config = EnvironmentConfig(
        name=f"benchmark_{env_type}",
        description="Benchmark environment",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "grid_size": 5,
            "num_items": 2,
            "num_obstacles": 1,
            "complexity": 0.2,
            "noise_level": 0.0,
        },
        target_skills=["navigation"],
        max_steps=30,
        seed=42,
    )

    # Create agent
    agent = PPOAgent(name="benchmark_agent")

    # Benchmark single environment
    print(f"\n{'=' * 60}")
    print(f"Environment Speed Benchmark: {env_type}")
    print(f"{'=' * 60}")

    for num_workers in workers:
        if num_workers == 1:
            # Single environment
            env = PuzzleEnvironment(config)
            start = time.perf_counter()

            for _ in range(episodes):
                obs = env.reset()
                done = False
                while not done:
                    actions = env.get_available_actions()
                    action = actions[hash(str(obs.state)) % len(actions)]
                    obs, _, done, _ = env.step(action)

            elapsed = time.perf_counter() - start
            env.close() if hasattr(env, "close") else None
        else:
            # Parallel environment
            try:
                from envgen.environments.parallel_env import ParallelEnvironment

                def env_factory():
                    return PuzzleEnvironment(config)

                with ParallelEnvironment(env_factory=env_factory, num_workers=num_workers, seed=42) as parallel_env:
                    start = time.perf_counter()

                    for _ in range(episodes):
                        obs_list = parallel_env.reset()
                        dones = [False] * num_workers

                        while not all(dones):
                            action_list = []
                            for i, obs in enumerate(obs_list):
                                actions = [Action("move", {"direction": "right"})]
                                action_list.append(actions[0])

                            obs_list, _, dones, _ = parallel_env.step(action_list)

                    elapsed = time.perf_counter() - start
            except ImportError:
                print(f"  Skipping {num_workers} workers: ParallelEnvironment not available")
                continue

        eps_per_sec = episodes / elapsed if elapsed > 0 else 0
        time_per_ep = elapsed / episodes if episodes > 0 else 0

        results[str(num_workers)] = {
            "workers": num_workers,
            "episodes": episodes,
            "elapsed_seconds": round(elapsed, 4),
            "episodes_per_second": round(eps_per_sec, 2),
            "time_per_episode_ms": round(time_per_ep * 1000, 2),
        }

        print(f"  Workers: {num_workers:>2} | Time: {elapsed:>8.4f}s | "
              f"Episodes/sec: {eps_per_sec:>6.2f} | Time/ep: {time_per_ep * 1000:>6.2f}ms")

    return results


def benchmark_agent_speed(
    agent_types: Optional[List[str]] = None,
    episodes: int = 20,
) -> Dict[str, Any]:
    """Benchmark agent training speed.

    Compares training speed across different agent types.

    Args:
        agent_types: List of agent types to benchmark. Defaults to ['ppo', 'ppo_torch', 'grpo'].
        episodes: Number of episodes for training.

    Returns:
        Dictionary with benchmark results for each agent type.
    """
    if agent_types is None:
        agent_types = ["ppo", "ppo_torch", "grpo"]

    results: Dict[str, Any] = {}

    # Create sample episodes
    config = EnvironmentConfig(
        name="benchmark",
        description="Benchmark",
        difficulty=DifficultyLevel.EASY,
        parameters={"grid_size": 5, "num_items": 2},
        target_skills=["navigation"],
        max_steps=20,
        seed=42,
    )

    env = PuzzleEnvironment(config)
    sample_episodes: List[Episode] = []

    for _ in range(episodes):
        obs = env.reset()
        episode = Episode()
        done = False
        while not done:
            actions = env.get_available_actions()
            action = actions[hash(str(obs.state)) % len(actions)]
            next_obs, reward, done, _ = env.step(action)
            from envgen.utils.types import Transition

            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            episode.add_transition(transition)
            obs = next_obs
        episode.environment_config = config.to_dict()
        sample_episodes.append(episode)

    env.close() if hasattr(env, "close") else None

    print(f"\n{'=' * 60}")
    print(f"Agent Training Speed Benchmark")
    print(f"{'=' * 60}")

    for agent_type in agent_types:
        agent = None

        try:
            if agent_type == "ppo":
                agent = PPOAgent(name=f"benchmark_{agent_type}")
            elif agent_type == "ppo_torch":
                try:
                    from envgen.agents.ppo_torch import PPOAgentTorch

                    agent = PPOAgentTorch(action_space=6, config={"hidden_size": 64})
                except ImportError:
                    print(f"  Skipping {agent_type}: PyTorch not available")
                    continue
            elif agent_type == "grpo":
                from envgen.agents.grpo_agent import GRPOAgent

                agent = GRPOAgent(name=f"benchmark_{agent_type}")
            elif agent_type == "grpo_torch":
                try:
                    from envgen.agents.grpo_torch import GRPOAgentTorch

                    agent = GRPOAgentTorch(action_space=6, config={"hidden_size": 64})
                except ImportError:
                    print(f"  Skipping {agent_type}: PyTorch not available")
                    continue
            else:
                print(f"  Unknown agent type: {agent_type}")
                continue

            start = time.perf_counter()
            agent.train(sample_episodes)
            elapsed = time.perf_counter() - start

            results[agent_type] = {
                "agent_type": agent_type,
                "episodes": episodes,
                "elapsed_seconds": round(elapsed, 4),
                "episodes_per_second": round(episodes / elapsed if elapsed > 0 else 0, 2),
            }

            print(f"  Agent: {agent_type:>12} | Time: {elapsed:>8.4f}s | "
                  f"Episodes/sec: {episodes / elapsed if elapsed > 0 else 0:>6.2f}")

        except Exception as e:
            print(f"  Error benchmarking {agent_type}: {e}")

    return results


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and return combined results."""
    print("\n" + "=" * 60)
    print("EnvGen-LLM Performance Benchmarks")
    print("=" * 60)

    env_results = benchmark_env_speed()
    agent_results = benchmark_agent_speed()

    print(f"\n{'=' * 60}")
    print("Benchmark Complete")
    print("=" * 60)

    return {
        "environment_speed": env_results,
        "agent_speed": agent_results,
    }


if __name__ == "__main__":
    results = run_all_benchmarks()
