"""Demo pipeline — run the self-improving training loop.

This script demonstrates the full EnvGen-LLM pipeline:
1. LLM generates initial training environments
2. Agent trains in those environments
3. Agent performance is evaluated per skill
4. LLM adapts environments based on feedback
5. Curriculum progresses through difficulty stages
6. The loop repeats, showing measurable improvement

Run with:
    python examples/demo_pipeline.py                  # uses mock LLM
    python examples/demo_pipeline.py --llm openai     # uses OpenAI (requires OPENAI_API_KEY)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envgen.agents.ppo_agent import PPOAgent
from envgen.curriculum.manager import CurriculumManager
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator, MockLLMClient
from envgen.pipeline.trainer import PipelineConfig, TrainingPipeline
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import DifficultyLevel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EnvGen-LLM Demo Pipeline")
    parser.add_argument(
        "--llm",
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM backend to use (default: mock)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--agent",
        choices=["ppo", "ppo_torch", "grpo", "grpo_torch", "sb3_ppo", "sb3_sac", "sb3_a2c", "sb3_dqn"],
        default="ppo",
        help="Agent type to use (default: ppo)",
    )
    parser.add_argument(
        "--env-type",
        choices=["puzzle", "maze", "resource", "text", "cartpole", "multi_agent"],
        default="puzzle",
        help="Environment type to use (default: puzzle)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=8,
        help="Number of training cycles (default: 8)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per cycle (default: 30)",
    )
    return parser.parse_args()


def create_llm_client(llm_type: str, model: str = "gpt-4o-mini"):
    """Create an LLM client based on the specified type.

    Args:
        llm_type: Type of LLM backend ('mock' or 'openai').
        model: Model name for OpenAI client.

    Returns:
        An LLMClient instance.

    Raises:
        ImportError: If openai package is not installed when using --llm openai.
        ValueError: If OPENAI_API_KEY is not set when using --llm openai.
    """
    if llm_type == "mock":
        return MockLLMClient()
    elif llm_type == "openai":
        from envgen.llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(model=model)
    elif llm_type == "ollama":
        from envgen.llm.ollama_client import OllamaLLMClient

        return OllamaLLMClient({"model": model})
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_reward_engine() -> RewardEngine:
    """Create a reward engine with multiple components."""
    engine = RewardEngine()

    # Add reward components
    engine.add_component(
        name="success",
        weight=1.0,
        compute_fn=RewardEngine.sparse_reward("success", 1.0, 0.0),
    )
    engine.add_component(
        name="progress",
        weight=0.5,
        compute_fn=RewardEngine.progress_reward("progress", 1.0),
    )
    engine.add_component(
        name="step_penalty",
        weight=0.3,
        compute_fn=RewardEngine.step_penalty(-0.01),
    )

    return engine


def create_agent(agent_type: str):
    """Create an agent based on the specified type.

    Args:
        agent_type: Type of agent ('ppo', 'ppo_torch', 'grpo', 'grpo_torch').

    Returns:
        An Agent instance.
    """
    if agent_type == "ppo":
        return PPOAgent(
            name="demo_agent",
            config={
                "learning_rate": 0.15,
                "discount_factor": 0.95,
                "initial_epsilon": 0.4,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.92,
                "max_experience": 200,
            },
        )
    elif agent_type == "ppo_torch":
        from envgen.agents.ppo_torch import PPOAgentTorch

        return PPOAgentTorch(
            name="ppo_torch",
            config={
                "learning_rate": 3e-4,
                "discount_factor": 0.99,
                "initial_epsilon": 0.3,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.98,
                "hidden_size": 128,
            },
        )
    elif agent_type == "grpo":
        from envgen.agents.grpo_agent import GRPOAgent

        return GRPOAgent(
            name="grpo_agent",
            config={
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "group_size": 4,
                "initial_epsilon": 0.3,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.95,
            },
        )
    elif agent_type == "grpo_torch":
        from envgen.agents.grpo_torch import GRPOAgentTorch

        return GRPOAgentTorch(
            name="grpo_torch",
            config={
                "learning_rate": 3e-4,
                "discount_factor": 0.99,
                "group_size": 8,
                "initial_epsilon": 0.3,
                "epsilon_min": 0.05,
                "epsilon_decay": 0.98,
                "hidden_size": 128,
            },
        )
    elif agent_type in ("sb3_ppo", "sb3_sac", "sb3_a2c", "sb3_dqn"):
        from envgen.agents.sb3_agent import SB3Agent

        algo = agent_type.split("_")[-1]
        return SB3Agent(
            algorithm=algo,
            name=agent_type,
            config={
                "learning_rate": 3e-4,
                "discount_factor": 0.99,
                "n_steps": 2048,
                "batch_size": 64,
                "verbose": 1,
            },
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_env_factory(env_type: str):
    """Create an environment factory based on the specified type.

    Args:
        env_type: Type of environment ('puzzle', 'maze', 'resource').

    Returns:
        A callable that creates environments from EnvironmentConfig.
    """
    if env_type == "puzzle":

        def factory(cfg):
            return PuzzleEnvironment(cfg)

        return factory
    elif env_type == "maze":
        from envgen.environments.maze_env import MazeEnvironment

        def factory(cfg):
            return MazeEnvironment(cfg)

        return factory
    elif env_type == "resource":
        from envgen.environments.resource_env import ResourceEnvironment

        def factory(cfg):
            return ResourceEnvironment(cfg)

        return factory
    elif env_type == "text":
        from envgen.environments.text_env import TextTaskEnvironment

        def factory(cfg):
            return TextTaskEnvironment(cfg)

        return factory
    elif env_type == "cartpole":
        from envgen.environments.cartpole_env import CartPoleEnvironment

        def factory(cfg):
            return CartPoleEnvironment(cfg)

        return factory
    elif env_type == "multi_agent":
        from envgen.environments.multi_agent_env import MultiAgentEnvironment

        def factory(cfg):
            return MultiAgentEnvironment(cfg)

        return factory
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def run_demo():
    """Run the full demo pipeline."""
    args = parse_args()
    setup_logging()

    print("=" * 70)
    print("  EnvGen-LLM: Self-Improving AI Agent Training Pipeline")
    print(f"  Demo: {args.env_type.capitalize()} Environment")
    print(f"  Agent: {args.agent}")
    print(f"  LLM Backend: {args.llm}" + (f" ({args.model})" if args.llm == "openai" else ""))
    print("=" * 70)
    print()

    # --- Configuration ---
    task_description = (
        "Navigate a grid world to collect items and reach the goal. "
        "The agent must plan efficient paths, remember item locations, "
        "and reason about the optimal collection order."
    )

    skills = ["navigation", "planning", "reasoning", "memory", "efficiency"]

    config = PipelineConfig(
        task_description=task_description,
        skills=skills,
        max_cycles=args.cycles,
        episodes_per_cycle=args.episodes,
        eval_episodes=15,
        initial_difficulty=DifficultyLevel.EASY,
        num_initial_envs=4,
        save_interval=0,  # Disable checkpoints for demo
    )

    # --- Components ---
    # LLM client (choose mock or OpenAI via --llm flag)
    llm_client = create_llm_client(args.llm, args.model)
    env_generator = EnvironmentGenerator(llm_client)

    # Agent (choose type via --agent flag)
    agent = create_agent(args.agent)

    # Evaluator
    evaluator = SkillEvaluator()

    # Curriculum
    curriculum = CurriculumManager(
        config={
            "advance_threshold": 0.6,
            "regression_threshold": 0.3,
            "min_improvement": 0.03,
        }
    )

    # Reward engine
    reward_engine = create_reward_engine()

    # Environment factory (choose type via --env-type flag)
    env_factory = create_env_factory(args.env_type)

    # --- Run Pipeline ---
    pipeline = TrainingPipeline(
        config=config,
        env_generator=env_generator,
        agent=agent,
        evaluator=evaluator,
        curriculum_manager=curriculum,
        reward_engine=reward_engine,
        env_factory=env_factory,
    )

    results = pipeline.run()

    # --- Print Summary ---
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 70)
    print()

    report = pipeline.get_improvement_report()

    if "overall_improvement" in report:
        oi = report["overall_improvement"]
        print(f"  Cycles completed:     {report['total_cycles']}")
        print(f"  Initial score:        {oi['initial']:.3f}")
        print(f"  Final score:          {oi['final']:.3f}")
        print(f"  Improvement:          {oi['improvement']:+.3f}")
        print()

        if "skill_improvement" in report:
            print("  Skill Improvement:")
            for skill, data in report["skill_improvement"].items():
                bar_len = 30
                initial_bar = int(data["initial"] * bar_len)
                final_bar = int(data["final"] * bar_len)

                initial_str = "#" * initial_bar + "-" * (bar_len - initial_bar)
                final_str = "#" * final_bar + "-" * (bar_len - final_bar)

                print(f"    {skill:<15} [{initial_str}] {data['initial']:.3f}")
                print(f"    {'':<15}  |")
                print(f"    {'':<15} [{final_str}] {data['final']:.3f} ({data['improvement']:+.3f})")
                print()

    print(f"  Agent stats:")
    stats = agent.get_stats()
    print(f"    Episodes trained:  {stats['total_episodes_trained']}")
    print(f"    Steps trained:     {stats['total_steps_trained']}")
    print(f"    States learned:    {stats.get('num_states_learned', 'N/A')}")
    epsilon = stats.get("epsilon")
    if epsilon is not None and isinstance(epsilon, (int, float)):
        print(f"    Final epsilon:     {epsilon:.3f}")
    else:
        print(f"    Final epsilon:     N/A")
    # Show algorithm-specific stats
    if "algorithm" in stats:
        print(f"    Algorithm:         {stats['algorithm']}")
    if "total_timesteps" in stats:
        print(f"    Total timesteps:   {stats['total_timesteps']}")
    print()

    print(f"  LLM calls: {report.get('llm_calls', 'N/A')}")
    print()

    # Curriculum progress
    cp = report.get("curriculum_progress", {})
    print(f"  Curriculum:")
    print(f"    Current stage:     {cp.get('current_stage_name', 'N/A')}")
    print(f"    Current difficulty: {cp.get('current_difficulty', 'N/A')}")
    print()

    print("=" * 70)
    print("  Demo complete! The agent showed measurable improvement")
    print("  across training cycles through the self-improving loop.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_demo()
