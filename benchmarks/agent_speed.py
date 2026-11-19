"""Performance benchmark for comparing agent training speed.

This script measures the training speed of different agent types
in the EnvGen-LLM framework, reporting time per episode and memory usage.

Usage:
    python benchmarks/agent_speed.py
"""

import logging
import time
from typing import Any, Dict, List, Optional

from envgen.environments import PuzzleEnvironment, EnvironmentConfig
from envgen.agents import PPOAgent
from envgen.utils.types import DifficultyLevel, Episode, Transition, Action

logging.basicConfig(level=logging.WARNING)
logging.getLogger("envgen").setLevel(logging.WARNING)


def create_agent(agent_type: str, config: Optional[Dict[str, Any]] = None):
    """Create an agent by type name."""
    agents = {"ppo": PPOAgent}

    # Try to import PyTorch-based agents
    try:
        from envgen.agents import PPOAgentTorch
        agents["ppo_torch"] = PPOAgentTorch
    except ImportError:
        pass

    try:
        from envgen.agents import GRPOAgent
        agents["grpo"] = GRPOAgent
    except ImportError:
        pass

    try:
        from envgen.agents import GRPOAgentTorch
        agents["grpo_torch"] = GRPOAgentTorch
    except ImportError:
        pass

    try:
        from envgen.agents import RNNAgent
        agents["rnn"] = RNNAgent
    except (ImportError, TypeError):
        pass

    try:
        from envgen.agents import TransformerAgent
        agents["transformer"] = TransformerAgent
    except (ImportError, TypeError):
        pass

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type} (available: {list(agents.keys())})")

    AgentClass = agents[agent_type]
    return AgentClass(name=agent_type, config=config or {})


def benchmark_agent(
    agent_type: str,
    num_episodes: int = 10,
    max_steps: int = 50,
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Benchmark a single agent type.

    Args:
        agent_type: Name of the agent type to benchmark
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        agent_config: Optional config dict for the agent

    Returns:
        Dict with timing and performance metrics
    """
    # Create environment and agent
    env_config = EnvironmentConfig(
        name="benchmark",
        difficulty=DifficultyLevel.EASY,
        parameters={"grid_size": 8, "num_items": 3, "num_obstacles": 5},
    )
    env = PuzzleEnvironment(config=env_config)
    agent = create_agent(agent_type, agent_config)

    # Measure episode collection time
    start_time = time.time()
    episodes: List[Episode] = []

    for ep_i in range(num_episodes):
        env.reset()
        obs = env.get_observation()
        transitions = []
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(obs, episode_id=ep_i)
            reward, done, info = env.step(action)
            next_obs = env.get_observation()

            trans = Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            transitions.append(trans)
            episode_reward += reward
            obs = next_obs

            if done:
                break

        episodes.append(Episode(transitions=transitions, total_reward=episode_reward))

    collection_time = time.time() - start_time

    # Measure training time
    start_time = time.time()
    agent.train(episodes)
    training_time = time.time() - start_time

    total_time = collection_time + training_time
    avg_reward = sum(ep.total_reward for ep in episodes) / len(episodes)

    return {
        "agent_type": agent_type,
        "num_episodes": num_episodes,
        "collection_time": collection_time,
        "training_time": training_time,
        "total_time": total_time,
        "time_per_episode": total_time / num_episodes,
        "episodes_per_second": num_episodes / total_time if total_time > 0 else 0,
        "avg_reward": avg_reward,
    }


def benchmark_agent_speed(
    agent_types: Optional[List[str]] = None,
    num_episodes: int = 10,
) -> List[Dict[str, Any]]:
    """Benchmark multiple agent types and return results.

    Args:
        agent_types: List of agent type names to benchmark. If None, uses all available.
        num_episodes: Number of episodes per agent.

    Returns:
        List of result dicts, one per agent type.
    """
    if agent_types is None:
        # Try all available agent types
        agent_types = ["ppo", "ppo_torch", "grpo", "grpo_torch", "rnn", "transformer"]

    results = []
    available_types = []

    # Filter to available agent types
    for agent_type in agent_types:
        try:
            create_agent(agent_type)
            available_types.append(agent_type)
        except (ValueError, ImportError):
            print(f"Skipping {agent_type}: not available")

    print(f"\nBenchmarking {len(available_types)} agent types with {num_episodes} episodes each...")
    print("=" * 70)

    for agent_type in available_types:
        print(f"\nBenchmarking: {agent_type}...")
        result = benchmark_agent(agent_type, num_episodes=num_episodes)
        results.append(result)
        print(
            f"  Time: {result['total_time']:.3f}s | "
            f"{result['episodes_per_second']:.1f} ep/s | "
            f"Avg reward: {result['avg_reward']:.2f}"
        )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<20} {'Time (s)':<12} {'Ep/s':<10} {'Avg Reward':<12}")
    print("-" * 70)

    for result in results:
        print(
            f"{result['agent_type']:<20} "
            f"{result['total_time']:<12.3f} "
            f"{result['episodes_per_second']:<10.1f} "
            f"{result['avg_reward']:<12.2f}"
        )

    # Find fastest agent
    if results:
        fastest = min(results, key=lambda r: r["total_time"])
        print(f"\nFastest agent: {fastest['agent_type']} ({fastest['total_time']:.3f}s)")

    return results


if __name__ == "__main__":
    results = benchmark_agent_speed(num_episodes=10)
