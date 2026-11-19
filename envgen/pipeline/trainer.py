"""Main training pipeline — the self-improving loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.base import Evaluator
from envgen.utils.types import (
    DifficultyLevel,
    Episode,
    EvaluationResult,
)

try:
    from envgen.utils.logger import ExperimentLogger
except ImportError:
    ExperimentLogger = None  # type: ignore

try:
    from envgen.utils.mlflow_logger import MLflowLogger
except ImportError:
    MLflowLogger = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline.

    Attributes:
        task_description: Description of the overall task.
        skills: List of skills to train.
        max_cycles: Maximum number of training-adaptation cycles.
        episodes_per_cycle: Episodes to collect per cycle.
        eval_episodes: Episodes for evaluation per cycle.
        episodes_per_env: Episodes per environment per cycle.
        initial_difficulty: Starting difficulty level.
        num_initial_envs: Number of initial environments to generate.
        save_interval: Save checkpoint every N cycles.
        log_dir: Directory for experiment logs (empty = disabled).
        use_mlflow: Enable MLflow tracking (default: False).
        mlflow_tracking_uri: Optional MLflow tracking URI.
    """

    task_description: str
    skills: List[str] = field(default_factory=list)
    max_cycles: int = 10
    episodes_per_cycle: int = 50
    eval_episodes: int = 20
    episodes_per_env: int = 10
    initial_difficulty: DifficultyLevel = DifficultyLevel.EASY
    num_initial_envs: int = 5
    save_interval: int = 5
    log_dir: str = ""
    use_mlflow: bool = False
    mlflow_tracking_uri: str = ""


class TrainingPipeline:
    """The main self-improving training pipeline.

    This is the core orchestrator that implements the EnvGen loop:

    ```
    ┌─────────────────────────────────────────────────────┐
    │              Training Pipeline Loop                 │
    │                                                     │
    │  1. Generate environments (LLM)                     │
    │  2. Train agent in environments                     │
    │  3. Evaluate agent performance                      │
    │  4. Adapt environments based on feedback (LLM)      │
    │  5. Update curriculum                               │
    │  6. Repeat until convergence or max cycles          │
    └─────────────────────────────────────────────────────┘
    ```

    The pipeline coordinates:
    - EnvironmentGenerator: Creates and adapts training environments
    - Agent: Learns to solve tasks in environments
    - Evaluator: Measures performance per skill
    - CurriculumManager: Manages progressive difficulty
    - RewardEngine: Computes and shapes rewards
    """

    def __init__(
        self,
        config: PipelineConfig,
        env_generator: Any,  # EnvironmentGenerator
        agent: Agent,
        evaluator: Evaluator,
        curriculum_manager: Any,  # CurriculumManager
        reward_engine: Any,  # RewardEngine
        env_factory: Any,  # Callable[[EnvironmentConfig], Environment]
        experiment_logger: Any = None,  # ExperimentLogger or None
        early_stopping: Any = None,  # EarlyStoppingMonitor or None
    ):
        self.config = config
        self.env_generator = env_generator
        self.agent = agent
        self.evaluator = evaluator
        self.curriculum = curriculum_manager
        self.reward_engine = reward_engine
        self.env_factory = env_factory
        self.experiment_logger = experiment_logger
        self.early_stopping = early_stopping

        # State
        self.current_configs: List[EnvironmentConfig] = []
        self.evaluation_history: List[EvaluationResult] = []
        self.cycle_stats: List[Dict[str, Any]] = []
        self.current_cycle: int = 0

    def run(self) -> List[EvaluationResult]:
        """Run the full training pipeline.

        Returns:
            List of evaluation results from each cycle.
        """
        logger.info(f"Starting training pipeline: {self.config.task_description}")
        logger.info(f"Skills: {self.config.skills}")
        logger.info(f"Max cycles: {self.config.max_cycles}")

        # Phase 1: Initial environment generation
        logger.info("=" * 60)
        logger.info("PHASE 1: Initial Environment Generation")
        logger.info("=" * 60)
        self.current_configs = self.env_generator.generate_initial(
            task_description=self.config.task_description,
            skills=self.config.skills,
            difficulty=self.config.initial_difficulty,
            num_environments=self.config.num_initial_envs,
        )
        logger.info(f"Generated {len(self.current_configs)} initial environments")
        for cfg in self.current_configs:
            logger.info(f"  - {cfg.name} ({cfg.difficulty.value}): {cfg.target_skills}")

        # Setup curriculum with initial stages
        self._setup_curriculum()

        # Phase 2: Training loop
        logger.info("=" * 60)
        logger.info("PHASE 2: Training Loop")
        logger.info("=" * 60)

        for cycle in range(self.config.max_cycles):
            self.current_cycle = cycle + 1
            logger.info(f"\n--- Cycle {self.current_cycle}/{self.config.max_cycles} ---")

            # Train
            episodes = self._train_cycle()

            # Evaluate
            result = self._evaluate()
            self.evaluation_history.append(result)

            # Log results
            self._log_cycle_results(result)

            # Log to experiment logger
            self._log_to_experiment(result, episodes)

            # Check early stopping
            if self.early_stopping:
                mean_reward = sum(ep.total_reward for ep in episodes) / max(1, len(episodes))
                should_stop = self.early_stopping.step(mean_reward=mean_reward)
                if should_stop:
                    logger.info(
                        f"Early stopping triggered at cycle {self.current_cycle} "
                        f"(no improvement for {self.early_stopping.patience} cycles)"
                    )
                    break

            # Update curriculum
            curriculum_update = self.curriculum.update(result)
            logger.info(f"Curriculum: {curriculum_update['action']} - {curriculum_update['reasoning']}")

            # Adapt environments (after first cycle)
            if cycle > 0:
                self.current_configs = self.env_generator.adapt_environments(
                    evaluation_result=result,
                    current_configs=self.current_configs,
                    task_description=self.config.task_description,
                )
                logger.info(f"Adapted to {len(self.current_configs)} environments")

            # Save checkpoint
            if self.config.save_interval > 0 and (cycle + 1) % self.config.save_interval == 0:
                self._save_checkpoint()

        # Final evaluation
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        final_result = self._evaluate()
        self._log_cycle_results(final_result, prefix="FINAL")

        # Write experiment summary
        if self.experiment_logger:
            final_scores = {ss.skill_name: ss.score for ss in final_result.skill_scores}
            summary = self.experiment_logger.log_summary(final_skill_scores=final_scores)
            logger.info(f"Experiment summary saved to {self.experiment_logger.get_run_dir()}")
            self.experiment_logger.close()

        return self.evaluation_history

    def _setup_curriculum(self) -> None:
        """Setup curriculum stages based on config."""
        difficulties = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
            DifficultyLevel.EXPERT,
        ]

        for i, diff in enumerate(difficulties):
            self.curriculum.add_stage(
                name=f"{diff.value}_stage",
                difficulty=diff,
                skill_targets=self.config.skills,
                min_scores={skill: 0.3 + i * 0.15 for skill in self.config.skills},
            )

    def _train_cycle(self) -> List[Episode]:
        """Run one training cycle.

        Returns:
            List of episodes collected during training.
        """
        all_episodes: List[Episode] = []

        # Distribute episodes across environments
        num_envs = len(self.current_configs)
        if num_envs == 0:
            logger.warning("No environments to train in")
            return all_episodes

        episodes_per_env = max(1, self.config.episodes_per_cycle // num_envs)

        for cfg in self.current_configs:
            env = self.env_factory(cfg)
            for _ in range(episodes_per_env):
                episode = self.agent.run_episode(env)
                all_episodes.append(episode)
                env.reset()

        # Train agent on collected episodes
        if all_episodes:
            metrics = self.agent.train(all_episodes)
            self.agent.total_episodes_trained += len(all_episodes)
            self.agent.total_steps_trained += sum(e.length for e in all_episodes)

            logger.info(
                f"Trained on {len(all_episodes)} episodes, "
                f"total steps: {sum(e.length for e in all_episodes)}, "
                f"avg reward: {sum(e.total_reward for e in all_episodes) / len(all_episodes):.3f}"
            )

            # Record cycle stats
            self.cycle_stats.append(
                {
                    "cycle": self.current_cycle,
                    "num_episodes": len(all_episodes),
                    "total_steps": sum(e.length for e in all_episodes),
                    "avg_reward": sum(e.total_reward for e in all_episodes) / len(all_episodes),
                    "training_metrics": metrics,
                }
            )

        return all_episodes

    def _evaluate(self) -> EvaluationResult:
        """Evaluate agent performance.

        Returns:
            Evaluation result with skill scores.
        """
        eval_episodes: List[Episode] = []

        # Run evaluation episodes across current environments
        num_envs = len(self.current_configs)
        if num_envs == 0:
            # Return empty result
            from envgen.utils.types import EvaluationResult

            return EvaluationResult(
                agent_id=self.agent.agent_id,
                overall_score=0.0,
            )

        eval_per_env = max(1, self.config.eval_episodes // num_envs)

        for cfg in self.current_configs:
            env = self.env_factory(cfg)
            for _ in range(eval_per_env):
                episode = self.agent.run_episode(env)
                eval_episodes.append(episode)
                env.reset()

        # Evaluate
        result = self.evaluator.evaluate(eval_episodes, self.agent.agent_id)

        return result

    def _log_cycle_results(self, result: EvaluationResult, prefix: str = "") -> None:
        """Log evaluation results for a cycle."""
        p = f"[{prefix}] " if prefix else ""

        logger.info(f"{p}Overall score: {result.overall_score:.3f}")
        logger.info(f"{p}Episode stats: {result.episode_stats}")

        for ss in result.skill_scores:
            bar = "█" * int(ss.score * 20) + "░" * (20 - int(ss.score * 20))
            logger.info(f"{p}  [{bar}] {ss.skill_name}: {ss.score:.3f}")

        weak = result.get_weak_skills()
        if weak:
            logger.info(f"{p}Weak skills: " f"{', '.join(f'{ss.skill_name}({ss.score:.2f})' for ss in weak)}")

    def _log_to_experiment(self, result: EvaluationResult, episodes: List[Episode]) -> None:
        """Log cycle metrics to the experiment logger if available."""
        if not self.experiment_logger:
            return

        rewards = [ep.total_reward for ep in episodes]
        skill_scores = {ss.skill_name: ss.score for ss in result.skill_scores}
        difficulty = (
            self.curriculum.get_current_stage().difficulty.value if self.curriculum.get_current_stage() else "unknown"
        )

        self.experiment_logger.log_cycle(
            cycle=self.current_cycle,
            rewards=rewards,
            skill_scores=skill_scores,
            difficulty=difficulty,
            episodes_trained=len(episodes),
            steps_trained=sum(ep.length for ep in episodes),
            eval_result={
                "overall_score": result.overall_score,
                "weak_skills": [ss.skill_name for ss in result.get_weak_skills()],
                "strong_skills": [ss.skill_name for ss in result.get_strong_skills()],
            },
        )

    def _save_checkpoint(self, checkpoint_dir: str = "checkpoints") -> str:
        """Save a checkpoint of the current pipeline state.

        Saves agent weights, curriculum state, environment configs,
        evaluation history, and cycle stats.

        Args:
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Path to the saved checkpoint file.
        """
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save agent state separately
        agent_path = os.path.join(checkpoint_dir, f"agent_cycle_{self.current_cycle}")
        self.agent.save(agent_path)

        checkpoint = {
            "cycle": self.current_cycle,
            "agent_stats": self.agent.get_stats(),
            "agent_path": agent_path,
            "num_environments": len(self.current_configs),
            "environment_configs": [c.to_dict() for c in self.current_configs],
            "curriculum_progress": self.curriculum.get_progress_report(),
            "evaluation_history": [
                {
                    "overall_score": r.overall_score,
                    "skill_scores": {ss.skill_name: ss.score for ss in r.skill_scores},
                }
                for r in self.evaluation_history
            ],
            "cycle_stats": self.cycle_stats,
        }

        path = os.path.join(checkpoint_dir, f"checkpoint_cycle_{self.current_cycle}.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint to restore pipeline state.

        Args:
            checkpoint_path: Path to the checkpoint JSON file.
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Restore agent state
        agent_path = checkpoint.get("agent_path", "")
        if agent_path and os.path.exists(agent_path + ".json"):
            self.agent.load(agent_path)
            logger.info(f"Loaded agent from {agent_path}")

        # Restore cycle
        self.current_cycle = checkpoint.get("cycle", 0)

        # Restore environment configs
        env_configs_data = checkpoint.get("environment_configs", [])
        if env_configs_data:
            from envgen.environments.base import EnvironmentConfig

            self.current_configs = [EnvironmentConfig.from_dict(cfg) for cfg in env_configs_data]

        logger.info(f"Loaded checkpoint from cycle {self.current_cycle}")

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate a report showing improvement over cycles."""
        if len(self.evaluation_history) < 2:
            return {"message": "Need at least 2 cycles to compute improvement"}

        first = self.evaluation_history[0]
        last = self.evaluation_history[-1]

        skill_improvement = {}
        all_skills = {ss.skill_name for ss in first.skill_scores} | {ss.skill_name for ss in last.skill_scores}

        for skill in all_skills:
            first_score = first.get_skill_score(skill) or 0.0
            last_score = last.get_skill_score(skill) or 0.0
            skill_improvement[skill] = {
                "initial": first_score,
                "final": last_score,
                "improvement": last_score - first_score,
            }

        return {
            "total_cycles": len(self.evaluation_history),
            "overall_improvement": {
                "initial": first.overall_score,
                "final": last.overall_score,
                "improvement": last.overall_score - first.overall_score,
            },
            "skill_improvement": skill_improvement,
            "curriculum_progress": self.curriculum.get_progress_report(),
            "llm_calls": getattr(self.env_generator.llm_client, "call_count", "unknown"),
        }
