"""CLI command for reward optimization.

Usage:
    python -m envgen optimize-rewards --agent ppo --episodes 10
    python -m envgen optimize-rewards --agent ppo_torch --env-type maze --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from envgen.utils.types import DifficultyLevel


def create_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Create argument parser for optimize-rewards command."""
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="envgen optimize-rewards",
            description="Optimize reward function weights via grid search.",
        )

    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        help="Agent type to use for evaluation (default: ppo)",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="puzzle",
        help="Environment type for evaluation (default: puzzle)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of training episodes per configuration (default: 20)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "expert"],
        default="easy",
        help="Difficulty level for evaluation (default: easy)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="simple",
        choices=["simple", "detailed"],
        help="Reward parameter grid preset (default: simple)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser


def run(args=None) -> None:
    """Run the reward optimization command."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from envgen.benchmark.reward_optimizer import RewardOptimizer, DEFAULT_REWARD_GRIDS

    difficulty = DifficultyLevel(parsed.difficulty)
    reward_grid = DEFAULT_REWARD_GRIDS.get(parsed.grid, DEFAULT_REWARD_GRIDS["simple"])

    optimizer = RewardOptimizer(
        agent_type=parsed.agent,
        env_type=parsed.env_type,
        reward_grid=reward_grid,
        episodes=parsed.episodes,
        difficulty=difficulty,
        seed=parsed.seed,
    )

    print(f"Starting reward optimization...")
    print(f"  Agent: {parsed.agent}")
    print(f"  Environment: {parsed.env_type}")
    print(f"  Difficulty: {parsed.difficulty}")
    print(f"  Episodes per config: {parsed.episodes}")
    print(f"  Grid preset: {parsed.grid}")
    print()

    results = optimizer.run()
    optimizer.print_results(top_n=parsed.top_n)

    best = optimizer.get_best_config()
    if best:
        print(f"\nBest configuration: {best['params_str']}")
        print(f"  Mean skill: {best['mean_skill']:.4f}")
        print(f"  Mean reward: {best['mean_reward']:.2f}")
