"""CLI evaluate command — evaluate a trained agent from a checkpoint.

This command loads an agent from a checkpoint file and runs evaluation
episodes across specified environments with fixed seeds, outputting
skill scores and performance metrics.

Example:
    python -m envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json --episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from envgen.config.loader import ConfigLoader
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.utils.types import DifficultyLevel, Episode


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint from a JSON file.

    Args:
        checkpoint_path: Path to the checkpoint JSON file.

    Returns:
        Dict containing checkpoint data.
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_env_from_config(config: Dict[str, Any], env_type: str, difficulty: str):
    """Create an environment from config.

    Args:
        config: Full config dict.
        env_type: Environment type string.
        difficulty: Difficulty level string.

    Returns:
        Environment instance.
    """
    loader = ConfigLoader()
    config["environment"]["type"] = env_type
    factory = loader.create_environment_factory(config)
    env = factory()
    return env


def run_evaluation(
    agent: Any,
    env: Any,
    num_episodes: int,
    seed: int = 42,
) -> List[Episode]:
    """Run evaluation episodes and collect results.

    Args:
        agent: Trained agent to evaluate.
        env: Environment to evaluate in.
        num_episodes: Number of evaluation episodes.
        seed: Random seed for reproducibility.

    Returns:
        List of completed episodes.
    """
    episodes: List[Episode] = []

    for ep_idx in range(num_episodes):
        env.reset()
        observation = env.get_observation()
        transitions = []
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(observation)
            reward, done, info = env.step(action)
            next_observation = env.get_observation()

            from envgen.utils.types import Transition

            transition = Transition(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                done=done,
            )
            transitions.append(transition)
            total_reward += reward
            observation = next_observation

        episode = Episode(transitions=transitions, total_reward=total_reward)
        episodes.append(episode)

    return episodes


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation from CLI arguments.

    Args:
        args: Parsed command-line arguments.
    """
    setup_logging(args.verbose)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    # Extract agent info from checkpoint
    agent_type = checkpoint.get("agent_type", "ppo")
    agent_config = checkpoint.get("agent_config", {})

    # Load agent
    from envgen.config.loader import ConfigLoader

    loader = ConfigLoader()
    config = {"agent": {"type": agent_type, **agent_config}}
    agent = loader.create_agent(config)

    # Load agent weights from checkpoint
    agent_state = checkpoint.get("agent_state", {})
    if agent_state:
        agent.load_state_dict(agent_state)
        print(f"  Agent state loaded (type: {agent_type})")

    # Determine environments to evaluate on
    env_types = args.env_types.split(",") if args.env_types else ["puzzle"]
    difficulties = args.difficulties.split(",") if args.difficulties else ["easy", "medium", "hard"]

    # Run evaluation
    evaluator = SkillEvaluator()
    results: Dict[str, Dict[str, Any]] = {}

    print("=" * 60)
    print("  EnvGen-LLM Agent Evaluation")
    print(f"  Agent: {agent_type}")
    print(f"  Episodes per env: {args.episodes}")
    print("=" * 60)

    for env_type in env_types:
        for difficulty in difficulties:
            key = f"{env_type}/{difficulty}"
            print(f"\nEvaluating: {key}")

            # Create environment
            config = {
                "environment": {
                    "type": env_type,
                    "difficulty": difficulty,
                }
            }
            factory = loader.create_environment_factory(config)
            env = factory()

            # Run episodes
            start_time = time.time()
            episodes = run_evaluation(
                agent=agent,
                env=env,
                num_episodes=args.episodes,
                seed=args.seed,
            )
            elapsed = time.time() - start_time

            # Evaluate
            eval_result = evaluator.evaluate(episodes)

            results[key] = {
                "overall_score": eval_result.overall_score,
                "skill_scores": {s.skill: s.score for s in eval_result.skill_scores},
                "mean_reward": eval_result.mean_reward,
                "episodes": args.episodes,
                "time": elapsed,
            }

            # Print results
            print(f"  Overall score: {eval_result.overall_score:.3f}")
            print(f"  Mean reward: {eval_result.mean_reward:.2f}")
            for skill in eval_result.skill_scores:
                print(f"    {skill.skill}: {skill.score:.3f}")
            print(f"  Time: {elapsed:.2f}s")

    # Print summary table
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"{'Environment':<20} {'Score':<10} {'Reward':<10} {'Time':<10}")
    print("-" * 50)
    for key, result in results.items():
        print(
            f"{key:<20} {result['overall_score']:<10.3f} "
            f"{result['mean_reward']:<10.2f} {result['time']:<10.2f}"
        )

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def main():
    """Main entry point for the evaluate CLI."""
    parser = argparse.ArgumentParser(
        prog="envgen evaluate",
        description="Evaluate a trained agent from a checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to checkpoint JSON file",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=20,
        help="Number of evaluation episodes per environment (default: 20)",
    )
    parser.add_argument(
        "--env-types",
        default=None,
        help="Comma-separated list of environment types (default: puzzle)",
    )
    parser.add_argument(
        "--difficulties",
        default=None,
        help="Comma-separated list of difficulty levels (default: easy,medium,hard)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
