"""CLI benchmark command — run standardized agent evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from envgen.config.loader import ConfigLoader
from envgen.utils.types import DifficultyLevel


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def benchmark(args: argparse.Namespace) -> None:
    """Run benchmark from CLI arguments.

    Args:
        args: Parsed command-line arguments.
    """
    setup_logging(args.verbose)

    # Load config if provided
    loader = ConfigLoader()
    if args.config:
        config = loader.load(args.config)
        if args.agent:
            config.setdefault("agent", {})["type"] = args.agent
    else:
        # Minimal config
        config = {"agent": {"type": args.agent or "ppo"}}

    # Create agent
    agent = loader.create_agent(config)

    # Parse environment types
    env_types: Optional[List[str]] = None
    if args.env_types:
        env_types = args.env_types.split(",")

    # Parse difficulties
    difficulties: Optional[List[DifficultyLevel]] = None
    if args.difficulties:
        diff_map = {
            "easy": DifficultyLevel.EASY,
            "medium": DifficultyLevel.MEDIUM,
            "hard": DifficultyLevel.HARD,
            "expert": DifficultyLevel.EXPERT,
        }
        difficulties = [diff_map.get(d.strip(), DifficultyLevel.EASY) for d in args.difficulties.split(",")]

    # Run benchmark
    from envgen.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        agent=agent,
        episodes_per_env=args.episodes,
        env_types=env_types,
        difficulties=difficulties,
        seed=args.seed,
    )

    print(f"Running benchmark: {agent.agent_id}")
    print(f"  Episodes per env: {args.episodes}")
    print(f"  Environment types: {env_types or 'all'}")
    print(f"  Difficulties: {difficulties or 'easy, medium, hard'}")
    print()

    results = runner.run()
    runner.print_summary()

    overall = runner.get_overall_score()
    print(f"\nBenchmark complete! Overall score: {overall:.3f}")


def main():
    """Main entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="envgen",
        description="EnvGen-LLM: Self-Improving AI Agent Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run standardized benchmark")
    bench_parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to YAML config file (optional)",
    )
    bench_parser.add_argument(
        "--agent",
        "-a",
        default="ppo",
        choices=["ppo", "ppo_torch", "grpo", "grpo_torch", "sb3_ppo", "sb3_sac", "sb3_a2c", "sb3_dqn"],
        help="Agent type to benchmark",
    )
    bench_parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=10,
        help="Episodes per environment (default: 10)",
    )
    bench_parser.add_argument(
        "--env-types",
        default=None,
        help="Comma-separated list of environment types (default: all)",
    )
    bench_parser.add_argument(
        "--difficulties",
        default=None,
        help="Comma-separated list of difficulty levels (default: easy,medium,hard)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    bench_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
