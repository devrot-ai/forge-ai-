"""LLM-generated environments example.

This example demonstrates how to use an LLM to automatically generate training
environments. The LLM creates environments with specific characteristics based
on a task description, then adapts them based on the agent's performance.

The example shows:
1. **Initial generation** — LLM creates environments from a task description
2. **Performance-based adaptation** — LLM adjusts environments based on weak skills
3. **Difficulty progression** — environments get harder as the agent improves

Run with:
    python examples/llm_env_generation.py              # uses mock LLM
    python examples/llm_env_generation.py --llm openai  # uses OpenAI (requires OPENAI_API_KEY)
    python examples/llm_env_generation.py --llm ollama  # uses Ollama (requires running Ollama)
"""

import argparse
import json
import sys
from pathlib import Path

# isort: off
sys.path.insert(0, str(Path(__file__).parent.parent))
# isort: on

from envgen.agents.ppo_agent import PPOAgent
from envgen.environments.base import EnvironmentConfig
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator, MockLLMClient
from envgen.utils.types import DifficultyLevel, Episode, Transition


def create_llm_client(llm_type: str):
    """Create an LLM client based on the specified type.

    Args:
        llm_type: Type of LLM backend ('mock', 'openai', or 'ollama').

    Returns:
        An LLMClient instance.
    """
    if llm_type == "mock":
        return MockLLMClient()
    elif llm_type == "openai":
        from envgen.llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(model="gpt-4o-mini")
    elif llm_type == "ollama":
        from envgen.llm.ollama_client import OllamaLLMClient

        return OllamaLLMClient({"model": "llama3.2"})
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def demonstrate_initial_generation(llm_type: str = "mock"):
    """Demonstrate initial environment generation from a task description.

    Args:
        llm_type: Type of LLM backend to use.
    """
    print("=" * 70)
    print("Step 1: Initial Environment Generation")
    print("=" * 70)
    print()

    llm_client = create_llm_client(llm_type)
    generator = EnvironmentGenerator(llm_client)

    # Generate initial environments
    task_description = (
        "Create a grid-world navigation task where the agent must collect "
        "items and reach a goal position. The agent should learn pathfinding "
        "and item collection strategies."
    )

    skills = ["navigation", "planning", "reasoning"]

    print(f"Task description: {task_description}")
    print(f"Skills to train: {skills}")
    print()

    env_configs = generator.generate_initial(task_description, skills, DifficultyLevel.EASY, num_environments=3)

    print(f"Generated {len(env_configs)} environment(s):\n")
    for i, config in enumerate(env_configs, 1):
        print(f"  Environment {i}: {config.name}")
        print(f"    Difficulty: {config.difficulty}")
        print(f"    Parameters: {json.dumps(config.parameters, indent=6)}")
        print(f"    Target skills: {config.target_skills}")
        print()

        # Create and test the environment
        env = PuzzleEnvironment(config)
        obs = env.reset()
        print(f"    Grid size: {obs.state.get('grid_size', 'N/A')}")
        print(f"    Available actions: {len(env.get_available_actions())}")
        print()

    return env_configs


def demonstrate_adaptive_generation(llm_type: str = "mock"):
    """Demonstrate adaptive environment generation based on agent performance.

    The LLM generates new environments targeting the agent's weak skills.

    Args:
        llm_type: Type of LLM backend to use.
    """
    print("=" * 70)
    print("Step 2: Adaptive Environment Generation")
    print("=" * 70)
    print()

    llm_client = create_llm_client(llm_type)
    generator = EnvironmentGenerator(llm_client)

    # Simulate agent training and evaluation
    env_config = EnvironmentConfig(
        name="adaptive_test",
        description="Test environment for adaptive generation",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "grid_size": 8,
            "num_items": 3,
            "num_obstacles": 5,
            "complexity": 0.3,
            "noise_level": 0.1,
        },
        max_steps=50,
    )
    env = PuzzleEnvironment(env_config)

    agent = PPOAgent(
        name="llm_demo_agent",
        config={"learning_rate": 0.1, "initial_epsilon": 1.0, "epsilon_decay": 0.95},
    )

    # Train briefly
    episodes = []
    for _ in range(10):
        obs = env.reset()
        transitions = []

        while not env.done and env.current_step < env.max_steps:
            available = env.get_available_actions()
            action = agent.select_action(obs, available)

            next_obs, reward, done, info = env.step(action)
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

        episode = Episode(transitions=transitions, total_reward=sum(t.reward for t in transitions))
        episodes.append(episode)
        agent.train([episode])

    # Evaluate agent
    evaluator = SkillEvaluator()
    eval_result = evaluator.evaluate(episodes, agent_id="llm_demo_agent")

    weak = [ss.skill_name for ss in eval_result.get_weak_skills()]
    strong = [ss.skill_name for ss in eval_result.get_strong_skills()]
    print("Agent evaluation results:")
    print(f"  Overall score: {eval_result.overall_score:.3f}")
    print(f"  Weak skills: {', '.join(weak) if weak else 'none'}")
    print(f"  Strong skills: {', '.join(strong) if strong else 'none'}")
    print()

    # Generate initial configs first, then adapt
    initial_configs = generator.generate_initial(
        "Grid-world navigation task", ["navigation", "planning"], DifficultyLevel.EASY
    )

    # Adapt based on evaluation
    adapted_configs = generator.adapt_environments(eval_result, initial_configs, "Grid-world navigation task")

    print(f"Adapted to {len(adapted_configs)} environment(s):\n")
    for i, config in enumerate(adapted_configs, 1):
        print(f"  Environment {i}: {config.name}")
        print(f"    Difficulty: {config.difficulty}")
        print(f"    Target skills: {config.target_skills}")
        print(f"    Parameters: {json.dumps(config.parameters, indent=6)}")
        print()


def demonstrate_difficulty_progression(llm_type: str = "mock"):
    """Demonstrate environment generation across difficulty levels.

    Args:
        llm_type: Type of LLM backend to use.
    """
    print("=" * 70)
    print("Step 3: Difficulty Progression")
    print("=" * 70)
    print()

    llm_client = create_llm_client(llm_type)
    generator = EnvironmentGenerator(llm_client)

    difficulties = [
        (DifficultyLevel.EASY, "Simple navigation with few obstacles"),
        (DifficultyLevel.MEDIUM, "Moderate complexity with more items"),
        (DifficultyLevel.HARD, "Complex maze-like environment"),
        (DifficultyLevel.EXPERT, "Expert-level challenge with noise and limited visibility"),
    ]

    task = "Grid-world navigation and item collection task"
    skills = ["navigation", "planning"]

    for difficulty, description in difficulties:
        print(f"Difficulty: {difficulty.value} — {description}")

        env_configs = generator.generate_initial(task, skills, difficulty, num_environments=1)
        if env_configs:
            config = env_configs[0]
            print(f"  Name: {config.name}")
            print(f"  Grid size: {config.parameters.get('grid_size', 'N/A')}")
            print(f"  Obstacles: {config.parameters.get('num_obstacles', 'N/A')}")
            print(f"  Items: {config.parameters.get('num_items', 'N/A')}")
            print(f"  Noise: {config.parameters.get('noise_level', 'N/A')}")
        print()


def main():
    """Run the LLM environment generation demonstration."""
    parser = argparse.ArgumentParser(description="LLM-Generated Environments Demo")
    parser.add_argument(
        "--llm",
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM backend to use (default: mock)",
    )
    args = parser.parse_args()

    print()
    print("=" * 72)
    print(" LLM-Generated Environments Demo".center(72))
    print("=" * 72)
    print()
    print(f"Using LLM backend: {args.llm}")
    print()

    # Run demonstrations
    demonstrate_initial_generation(args.llm)
    demonstrate_adaptive_generation(args.llm)
    demonstrate_difficulty_progression(args.llm)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The LLM environment generator enables:")
    print("  1. Automatic environment creation from natural language descriptions")
    print("  2. Adaptive environments that target the agent's weak skills")
    print("  3. Progressive difficulty that scales with agent ability")
    print()
    print("With a real LLM (OpenAI/Ollama), the generated environments are more")
    print("diverse and creative than the mock LLM's template-based approach.")
    print()


if __name__ == "__main__":
    main()
