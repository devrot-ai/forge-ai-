"""EnvGen-LLM entry point.

Run with:
    python -m envgen train --config configs/default.yaml
    python -m envgen benchmark --agent ppo --episodes 10
    python -m envgen hyperparam-search --agent ppo --episodes 10
    python -m envgen dashboard --log-dir ./logs
    python -m envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json
    python -m envgen optimize-rewards --agent ppo --episodes 10
"""

import argparse
import sys


def main():
    """Main entry point that dispatches to sub-commands."""
    parser = argparse.ArgumentParser(
        prog="envgen",
        description="EnvGen-LLM: Self-Improving AI Agent Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    subparsers.add_parser("train", help="Run training pipeline")
    # Benchmark command
    subparsers.add_parser("benchmark", help="Run standardized benchmark")
    # Hyperparameter search command
    subparsers.add_parser("hyperparam-search", help="Run hyperparameter grid search")
    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch web dashboard")
    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate a trained agent from checkpoint")
    # Optimize rewards command
    subparsers.add_parser("optimize-rewards", help="Optimize reward function weights")

    args, _ = parser.parse_known_args()

    if args.command == "train":
        from envgen.cli.train import main as train_main

        train_main()
    elif args.command == "benchmark":
        from envgen.cli.benchmark import main as benchmark_main

        benchmark_main()
    elif args.command == "hyperparam-search":
        from envgen.cli.hyperparam_search import run as hyperparam_run

        hyperparam_run()
    elif args.command == "dashboard":
        from envgen.cli.dashboard import main as dashboard_main

        dashboard_main()
    elif args.command == "evaluate":
        from envgen.cli.evaluate import main as evaluate_main

        evaluate_main()
    elif args.command == "optimize-rewards":
        from envgen.cli.optimize_rewards import run as optimize_rewards_run

        optimize_rewards_run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
