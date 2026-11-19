"""CLI train command — run training from YAML config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from envgen.config.loader import ConfigLoader
from envgen.curriculum.manager import CurriculumManager
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator
from envgen.pipeline.trainer import PipelineConfig, TrainingPipeline
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import DifficultyLevel

try:
    from envgen.utils.logger import ExperimentLogger
except ImportError:
    ExperimentLogger = None  # type: ignore

try:
    from envgen.utils.mlflow_logger import MLflowLogger
except ImportError:
    MLflowLogger = None  # type: ignore

try:
    from envgen.utils.wandb_logger import WandbLogger
except ImportError:
    WandbLogger = None  # type: ignore

try:
    from envgen.utils.early_stopping import EarlyStoppingMonitor
except ImportError:
    EarlyStoppingMonitor = None  # type: ignore

try:
    from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
except ImportError:
    MultiAgentPipeline = None  # type: ignore


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_reward_engine(components):
    """Create a reward engine from config components."""
    engine = RewardEngine()

    for comp in components:
        name = comp.get("name", "default")
        weight = comp.get("weight", 1.0)
        comp_type = comp.get("type", "sparse")

        if comp_type == "sparse":
            fn = RewardEngine.sparse_reward(name, 1.0, 0.0)
        elif comp_type == "progress":
            fn = RewardEngine.progress_reward(name, 1.0)
        elif comp_type == "step_penalty":
            penalty = comp.get("penalty", -0.01)
            fn = RewardEngine.step_penalty(penalty)
        else:
            fn = RewardEngine.sparse_reward(name, 1.0, 0.0)

        engine.add_component(name=name, weight=weight, compute_fn=fn)

    return engine


def train(args: argparse.Namespace) -> None:
    """Run training from CLI arguments.

    Args:
        args: Parsed command-line arguments.
    """
    setup_logging(args.verbose)

    # Load config
    loader = ConfigLoader()
    config = loader.load(args.config)

    # Override with CLI args
    if args.cycles:
        config.setdefault("training", {})["max_cycles"] = args.cycles
    if args.episodes:
        config.setdefault("training", {})["episodes_per_cycle"] = args.episodes
    if args.agent:
        config.setdefault("agent", {})["type"] = args.agent
    if args.env_type:
        config.setdefault("environment", {})["type"] = args.env_type
    if args.llm:
        config.setdefault("llm", {})["client"] = args.llm
    if args.log_dir:
        config.setdefault("training", {})["log_dir"] = args.log_dir
    if args.mlflow:
        config.setdefault("training", {})["use_mlflow"] = True
    if args.mlflow_tracking_uri:
        config.setdefault("training", {})["mlflow_tracking_uri"] = args.mlflow_tracking_uri
    if args.early_stopping:
        config.setdefault("training", {})["use_early_stopping"] = True
    if args.patience:
        config.setdefault("training", {})["early_stopping_patience"] = args.patience
    if args.early_stopping_metric:
        config.setdefault("training", {})["early_stopping_metric"] = args.early_stopping_metric
    if args.wandb:
        config.setdefault("training", {})["use_wandb"] = True
    if args.wandb_project:
        config.setdefault("training", {})["wandb_project"] = args.wandb_project

    # Extract configs
    task_cfg = loader.get_task_config(config)
    training_cfg = loader.get_training_config(config)
    curriculum_cfg = loader.get_curriculum_config(config)
    reward_cfg = loader.get_reward_config(config)

    # Map difficulty string to enum
    diff_map = {
        "easy": DifficultyLevel.EASY,
        "medium": DifficultyLevel.MEDIUM,
        "hard": DifficultyLevel.HARD,
        "expert": DifficultyLevel.EXPERT,
    }
    initial_diff = diff_map.get(training_cfg["initial_difficulty"], DifficultyLevel.EASY)

    # Build pipeline config
    pipeline_cfg = PipelineConfig(
        task_description=task_cfg["description"],
        skills=task_cfg["skills"],
        max_cycles=training_cfg["max_cycles"],
        episodes_per_cycle=training_cfg["episodes_per_cycle"],
        eval_episodes=training_cfg["eval_episodes"],
        initial_difficulty=initial_diff,
        num_initial_envs=training_cfg["num_initial_envs"],
        save_interval=training_cfg["save_interval"],
        log_dir=training_cfg.get("log_dir", ""),
        use_mlflow=training_cfg.get("use_mlflow", False),
        mlflow_tracking_uri=training_cfg.get("mlflow_tracking_uri", ""),
    )

    # Build components
    llm_client = loader.create_llm_client(config)
    env_generator = EnvironmentGenerator(llm_client)
    agent = loader.create_agent(config)
    evaluator = SkillEvaluator()
    curriculum = CurriculumManager(config=curriculum_cfg)
    reward_engine = create_reward_engine(reward_cfg["components"])
    env_factory = loader.create_environment_factory(config)

    # Setup early stopping monitor
    early_stopping = None
    use_early_stopping = training_cfg.get("use_early_stopping", False)
    if use_early_stopping and EarlyStoppingMonitor is not None:
        patience = training_cfg.get("early_stopping_patience", 3)
        metric = training_cfg.get("early_stopping_metric", "mean_reward")
        early_stopping = EarlyStoppingMonitor(
            patience=patience,
            metric=metric,
            mode="max",
        )
        print(f"  Early stopping enabled (patience={patience}, metric={metric})")

    # Setup experiment logger
    experiment_logger = None
    log_dir = training_cfg.get("log_dir", "")
    use_mlflow = training_cfg.get("use_mlflow", False)
    mlflow_tracking_uri = training_cfg.get("mlflow_tracking_uri", "")

    use_wandb = training_cfg.get("use_wandb", False)
    wandb_project = training_cfg.get("wandb_project", "envgen-llm")

    if use_mlflow and MLflowLogger is not None and log_dir:
        experiment_logger = MLflowLogger(
            log_dir=log_dir,
            config=config,
            tracking_uri=mlflow_tracking_uri or None,
        )
        print(f"  MLflow logging enabled: {experiment_logger.get_run_dir()}")
    elif use_wandb and WandbLogger is not None and log_dir:
        experiment_logger = WandbLogger(
            log_dir=log_dir,
            config=config,
            project=wandb_project,
        )
        print(f"  W&B logging enabled: {experiment_logger.run_dir}")
    elif log_dir and ExperimentLogger is not None:
        experiment_logger = ExperimentLogger(log_dir=log_dir)
        print(f"  Experiment logging enabled: {experiment_logger.get_run_dir()}")

    # Print config summary
    print("=" * 60)
    print("  EnvGen-LLM Training Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Agent: {loader.get_agent_config(config)['type']}")
    print(f"  Environment: {loader.get_environment_config(config)['type']}")
    print(f"  LLM: {loader.get_llm_config(config)['client']}")
    print(f"  Cycles: {training_cfg['max_cycles']}")
    print("=" * 60)

    # Check for multi-agent mode
    use_multi_agent = getattr(args, 'multi_agent', False)
    if use_multi_agent and MultiAgentPipeline is not None:
        num_agents = getattr(args, 'num_agents', 2)
        agent_mode = getattr(args, 'agent_mode', 'competitive')

        print(f"  Multi-agent mode: {agent_mode} ({num_agents} agents)")

        # Create multiple agents
        agents = []
        for i in range(num_agents):
            a = loader.create_agent(config)
            a.name = f"{a.name}_agent_{i}"
            agents.append(a)

        # Run multi-agent pipeline
        ma_pipeline = MultiAgentPipeline(
            num_agents=num_agents,
            mode=agent_mode,
            config={
                "initial_difficulty": str(initial_diff),
            },
            agents=agents,
            env_factory=env_factory,
            evaluator=evaluator,
            curriculum_manager=curriculum,
            reward_engine=reward_engine,
            experiment_logger=experiment_logger,
        )

        results = ma_pipeline.train(
            max_cycles=training_cfg["max_cycles"],
            episodes_per_cycle=training_cfg["episodes_per_cycle"],
        )
        print("\nMulti-agent training complete!")

    else:
        # Run single-agent pipeline
        pipeline = TrainingPipeline(
            config=pipeline_cfg,
            env_generator=env_generator,
            agent=agent,
            evaluator=evaluator,
            curriculum_manager=curriculum,
            reward_engine=reward_engine,
            env_factory=env_factory,
            experiment_logger=experiment_logger,
            early_stopping=early_stopping,
        )

        results = pipeline.run()
        if results:
            final = results[-1]
            print(f"\nTraining complete! Final score: {final.overall_score:.3f}")
        else:
            print("\nTraining complete!")

        if early_stopping and early_stopping.no_improve_count > 0:
            print(f"  Early stopping triggered after {early_stopping.cycle} cycles")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="envgen",
        description="EnvGen-LLM: Self-Improving AI Agent Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config file",
    )
    train_parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Override max training cycles",
    )
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per cycle",
    )
    train_parser.add_argument(
        "--agent",
        choices=["ppo", "ppo_torch", "grpo", "grpo_torch", "sb3_ppo", "sb3_sac", "sb3_a2c", "sb3_dqn", "rnn", "transformer", "nn"],
        default=None,
        help="Override agent type",
    )
    train_parser.add_argument(
        "--env-type",
        choices=["puzzle", "maze", "resource", "text", "cartpole", "multi_agent"],
        default=None,
        help="Override environment type",
    )
    train_parser.add_argument(
        "--llm",
        choices=["mock", "openai", "ollama", "anthropic"],
        default=None,
        help="Override LLM client type",
    )
    train_parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for experiment logs (e.g., ./logs)",
    )
    train_parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking (requires --log-dir)",
    )
    train_parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (default: local)",
    )
    train_parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping when performance plateaus",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of cycles without improvement before early stopping (default: 3)",
    )
    train_parser.add_argument(
        "--early-stopping-metric",
        default=None,
        choices=["mean_reward", "navigation", "planning", "reasoning", "memory", "efficiency"],
        help="Metric to monitor for early stopping (default: mean_reward)",
    )
    train_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking (requires --log-dir)",
    )
    train_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (default: envgen-llm)",
    )
    train_parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Enable multi-agent training mode",
    )
    train_parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents to train simultaneously (default: 2)",
    )
    train_parser.add_argument(
        "--agent-mode",
        choices=["competitive", "cooperative"],
        default="competitive",
        help="Multi-agent mode: competitive or cooperative (default: competitive)",
    )
    train_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
