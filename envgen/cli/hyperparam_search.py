"""CLI command for hyperparameter search.

Usage:
    python -m envgen hyperparam-search --agent ppo --episodes 10
    python -m envgen hyperparam-search --agent ppo_torch --env-type maze --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from envgen.benchmark.hyperparam_search import HyperparameterSearch
from envgen.utils.types import DifficultyLevel


def create_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Create the argument parser for hyperparameter search.

    Args:
        parser: Existing parser to add arguments to. If None, creates a new one.

    Returns:
        Configured argument parser.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="envgen hyperparam-search",
            description="Run hyperparameter grid search over agent configurations.",
        )

    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "ppo_torch", "grpo", "grpo_torch"],
        help="Agent type to tune (default: ppo)",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="puzzle",
        choices=["puzzle", "maze", "resource", "text"],
        help="Environment type for evaluation (default: puzzle)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of training episodes per configuration (default: 10)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="EASY",
        choices=["EASY", "MEDIUM", "HARD", "EXPERT"],
        help="Difficulty level for evaluation (default: EASY)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to display (default: 10)",
    )
    parser.add_argument(
        "--param",
        type=str,
        action="append",
        metavar="NAME=VAL1,VAL2,...",
        help="Custom parameter grid entry, e.g., --param learning_rate=0.001,0.01,0.1 "
        "(can be specified multiple times; overrides default grid)",
    )

    return parser


def run(args: Optional[argparse.Namespace] = None) -> None:
    """Run the hyperparameter search command.

    Args:
        args: Parsed command-line arguments. If None, parses from sys.argv.
    """
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    # Parse custom parameter grid if provided
    param_grid = None
    if args.param:
        param_grid = {}
        for param_str in args.param:
            name, values = param_str.split("=", 1)
            name = name.strip()
            # Try to parse as numbers, fall back to strings
            parsed_values = []
            for v in values.split(","):
                v = v.strip()
                try:
                    parsed_values.append(int(v))
                except ValueError:
                    try:
                        parsed_values.append(float(v))
                    except ValueError:
                        parsed_values.append(v)
            param_grid[name] = parsed_values

    difficulty = DifficultyLevel(args.difficulty)

    search = HyperparameterSearch(
        agent_type=args.agent,
        env_type=args.env_type,
        param_grid=param_grid,
        episodes=args.episodes,
        difficulty=difficulty,
        seed=args.seed,
    )

    results = search.run()
    search.print_results(top_n=args.top_n)

    # Return best config
    best = search.get_best_config()
    if best:
        print(f"\nRecommended config: {best['params']}")


if __name__ == "__main__":
    run()
