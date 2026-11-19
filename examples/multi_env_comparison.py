"""Compare the same agent across different environment types.

This example trains a single agent across multiple environment types
and compares the results to see which environments are easiest/hardest.

Usage:
    python examples/multi_env_comparison.py
"""

import logging

from envgen.agents.ppo_agent import PPOAgent
from envgen.config.loader import ConfigLoader
from envgen.curriculum.manager import CurriculumManager
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator, MockLLMClient
from envgen.pipeline.trainer import PipelineConfig, TrainingPipeline
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import DifficultyLevel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    """Run multi-environment comparison."""
    env_types = ["puzzle", "maze", "resource", "text", "cartpole"]

    results = {}
    for env_type in env_types:
        print(f"\n{'=' * 60}")
        print(f"  Training on {env_type} environment")
        print(f"{'=' * 60}")

        # Create components
        llm_client = MockLLMClient()
        env_generator = EnvironmentGenerator(llm_client)
        agent = PPOAgent(name="ppo_agent")
        evaluator = SkillEvaluator()
        curriculum = CurriculumManager()
        reward_engine = RewardEngine()
        reward_engine.add_component(
            name="success", weight=1.0, compute_fn=RewardEngine.sparse_reward("success", 1.0, 0.0)
        )
        reward_engine.add_component(
            name="progress", weight=0.5, compute_fn=RewardEngine.progress_reward("progress", 1.0)
        )
        reward_engine.add_component(name="step_penalty", weight=0.1, compute_fn=RewardEngine.step_penalty(-0.01))

        # Create environment factory
        loader = ConfigLoader()
        config = {"environment": {"type": env_type}}
        env_factory = loader.create_environment_factory(config)

        # Build pipeline
        pipeline_cfg = PipelineConfig(
            task_description=f"Train on {env_type} environment",
            skills=["navigation", "planning", "reasoning", "memory", "efficiency"],
            max_cycles=3,
            episodes_per_cycle=15,
            eval_episodes=5,
            initial_difficulty=DifficultyLevel.EASY,
            num_initial_envs=3,
        )

        pipeline = TrainingPipeline(
            config=pipeline_cfg,
            env_generator=env_generator,
            agent=agent,
            evaluator=evaluator,
            curriculum_manager=curriculum,
            reward_engine=reward_engine,
            env_factory=env_factory,
        )

        history = pipeline.run()
        if history:
            final = history[-1]
            results[env_type] = {
                "overall_score": final.overall_score,
                "skill_scores": {ss.skill_name: ss.score for ss in final.skill_scores},
            }

    # Print comparison
    print(f"\n{'=' * 60}")
    print("  Multi-Environment Comparison Results")
    print(f"{'=' * 60}")
    print(f"{'Environment':<15} {'Score':>10} {'Navigation':>12} {'Planning':>12} {'Reasoning':>12}")
    print("-" * 61)
    for env_type, result in results.items():
        skills = result["skill_scores"]
        print(
            f"{env_type:<15} {result['overall_score']:>10.3f} "
            f"{skills.get('navigation', 0):>12.3f} "
            f"{skills.get('planning', 0):>12.3f} "
            f"{skills.get('reasoning', 0):>12.3f}"
        )


if __name__ == "__main__":
    main()
