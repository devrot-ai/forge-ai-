"""Multi-agent training example.

Demonstrates competitive and cooperative multi-agent training using the
MultiAgentPipeline. Two agents are trained simultaneously on a shared
environment, competing or cooperating to collect items.

Usage:
    python examples/multi_agent_training.py
"""

import logging
import sys

# Reduce noise
logging.basicConfig(level=logging.WARNING)
logging.getLogger("envgen").setLevel(logging.WARNING)


def train_competitive():
    """Train two agents in competitive mode."""
    print("=" * 60)
    print("COMPETITIVE MULTI-AGENT TRAINING")
    print("=" * 60)

    from envgen.agents import PPOAgent
    from envgen.curriculum import CurriculumManager
    from envgen.evaluators import SkillEvaluator
    from envgen.environments import MultiAgentEnvironment, EnvironmentConfig
    from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
    from envgen.rewards import RewardEngine
    from envgen.utils.types import DifficultyLevel

    # Create environment config
    env_config = EnvironmentConfig(
        name="competitive_training",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "grid_size": 8,
            "num_items": 5,
            "cooperation_mode": False,
        },
    )

    def env_factory():
        return MultiAgentEnvironment(config=env_config)

    # Create two agents
    agent1 = PPOAgent(name="agent_1", config={"learning_rate": 0.1, "epsilon": 0.2})
    agent2 = PPOAgent(name="agent_2", config={"learning_rate": 0.1, "epsilon": 0.2})

    # Create pipeline components
    curriculum = CurriculumManager()
    curriculum.add_stage("easy", DifficultyLevel.EASY)
    curriculum.add_stage("medium", DifficultyLevel.MEDIUM)
    curriculum.add_stage("hard", DifficultyLevel.HARD)

    evaluator = SkillEvaluator()
    reward_engine = RewardEngine()

    # Create multi-agent pipeline
    pipeline = MultiAgentPipeline(
        num_agents=2,
        mode="competitive",
        config={"num_cycles": 3, "num_episodes": 10},
        agents=[agent1, agent2],
        env_factory=env_factory,
        evaluator=evaluator,
        curriculum_manager=curriculum,
        reward_engine=reward_engine,
    )

    # Train
    results = pipeline.train()

    # Print results
    print("\n--- Competitive Training Results ---")
    for i, agent_result in enumerate(results.get("agent_results", [])):
        print(f"\nAgent {i + 1}:")
        print(f"  Avg reward: {agent_result.get('avg_reward', 0):.2f}")
        stats = agent_result.get("stats", {})
        print(f"  Stats: {stats}")

    return results


def train_cooperative():
    """Train two agents in cooperative mode."""
    print("\n" + "=" * 60)
    print("COOPERATIVE MULTI-AGENT TRAINING")
    print("=" * 60)

    from envgen.agents import PPOAgent
    from envgen.curriculum import CurriculumManager
    from envgen.evaluators import SkillEvaluator
    from envgen.environments import MultiAgentEnvironment, EnvironmentConfig
    from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
    from envgen.rewards import RewardEngine
    from envgen.utils.types import DifficultyLevel

    # Create environment config
    env_config = EnvironmentConfig(
        name="cooperative_training",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "grid_size": 8,
            "num_items": 5,
            "cooperation_mode": True,
        },
    )

    def env_factory():
        return MultiAgentEnvironment(config=env_config)

    # Create two agents
    agent1 = PPOAgent(name="coop_agent_1", config={"learning_rate": 0.1, "epsilon": 0.2})
    agent2 = PPOAgent(name="coop_agent_2", config={"learning_rate": 0.1, "epsilon": 0.2})

    # Create pipeline components
    curriculum = CurriculumManager()
    curriculum.add_stage("easy", DifficultyLevel.EASY)
    curriculum.add_stage("medium", DifficultyLevel.MEDIUM)

    evaluator = SkillEvaluator()
    reward_engine = RewardEngine()

    # Create multi-agent pipeline
    pipeline = MultiAgentPipeline(
        num_agents=2,
        mode="cooperative",
        config={"num_cycles": 3, "num_episodes": 10},
        agents=[agent1, agent2],
        env_factory=env_factory,
        evaluator=evaluator,
        curriculum_manager=curriculum,
        reward_engine=reward_engine,
    )

    # Train
    results = pipeline.train()

    # Print results
    print("\n--- Cooperative Training Results ---")
    for i, agent_result in enumerate(results.get("agent_results", [])):
        print(f"\nCooperative Agent {i + 1}:")
        print(f"  Avg reward: {agent_result.get('avg_reward', 0):.2f}")
        stats = agent_result.get("stats", {})
        print(f"  Stats: {stats}")

    return results


def compare_modes():
    """Compare competitive vs cooperative training."""
    print("\n" + "=" * 60)
    print("MODE COMPARISON")
    print("=" * 60)

    comp_results = train_competitive()
    coop_results = train_cooperative()

    print("\n--- Comparison Summary ---")
    comp_rewards = [
        r.get("avg_reward", 0) for r in comp_results.get("agent_results", [])
    ]
    coop_rewards = [
        r.get("avg_reward", 0) for r in coop_results.get("agent_results", [])
    ]

    if comp_rewards and coop_rewards:
        comp_avg = sum(comp_rewards) / len(comp_rewards)
        coop_avg = sum(coop_rewards) / len(coop_rewards)
        print(f"Competitive avg reward: {comp_avg:.2f}")
        print(f"Cooperative avg reward: {coop_avg:.2f}")

        if coop_avg > comp_avg:
            print("Cooperative mode yielded higher average rewards.")
        else:
            print("Competitive mode yielded higher average rewards.")


if __name__ == "__main__":
    try:
        compare_modes()
    except ImportError as e:
        print(f"Multi-agent training requires additional dependencies: {e}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print("Multi-agent training demo completed successfully.")
    print("=" * 60)
