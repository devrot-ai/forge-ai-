"""Transfer Learning Pipeline for fine-tuning pre-trained agents.

This module enables loading a pre-trained agent from a checkpoint and
fine-tuning it on a new environment with minimal additional training.
It supports:
- Full weight transfer: load all weights and fine-tune with reduced learning rate
- Partial weight transfer: only transfer shared layers (e.g., observation encoder)
- Layer freezing: freeze certain layers during fine-tuning

Usage:
    python -m envgen train --transfer-from <checkpoint> --env-type <new_env>
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.curriculum.manager import CurriculumManager
from envgen.environments.base import EnvironmentConfig
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.pipeline.trainer import TrainingPipeline
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import DifficultyLevel, EvaluationResult

logger = logging.getLogger(__name__)


class TransferLearningPipeline(TrainingPipeline):
    """Training pipeline that fine-tunes a pre-trained agent on a new environment.

    Loads an agent from a checkpoint path, optionally adjusts the learning rate
    and freezes certain layers, then trains on a new environment. This enables
    transfer learning where knowledge from one environment is reused for another.

    The pipeline supports:
    - Full weight transfer: all weights are loaded and fine-tuned
    - Partial weight transfer: only shared layers are transferred
    - Layer freezing: specified number of layers are frozen during fine-tuning
    """

    def __init__(
        self,
        agent: Agent,
        env_factory,
        reward_engine: Optional[RewardEngine] = None,
        curriculum_manager: Optional[CurriculumManager] = None,
        evaluator: Optional[SkillEvaluator] = None,
        checkpoint_path: Optional[str] = None,
        transfer_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the transfer learning pipeline.

        Args:
            agent: Agent to fine-tune
            env_factory: Factory function to create environments
            reward_engine: Reward engine for computing rewards
            curriculum_manager: Curriculum manager for difficulty progression
            evaluator: Skill evaluator for agent evaluation
            checkpoint_path: Path to checkpoint to load weights from
            transfer_config: Config dict with optional keys:
                - transfer_learning_rate: learning rate for fine-tuning (default: 0.0001)
                - freeze_layers: number of layers to freeze (default: 0)
                - fine_tune_epochs: number of fine-tune epochs (default: 10)
            **kwargs: Additional arguments passed to TrainingPipeline
        """
        super().__init__(
            agent=agent,
            env_factory=env_factory,
            reward_engine=reward_engine,
            curriculum_manager=curriculum_manager,
            evaluator=evaluator,
            **kwargs,
        )

        self.checkpoint_path = checkpoint_path
        self.transfer_config = transfer_config or {}
        self.transfer_learning_rate: float = self.transfer_config.get("transfer_learning_rate", 1e-4)
        self.freeze_layers: int = self.transfer_config.get("freeze_layers", 0)
        self.fine_tune_epochs: int = self.transfer_config.get("fine_tune_epochs", 10)

    def load_pretrained_agent(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load a pre-trained agent from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory.
                           If None, uses the checkpoint_path from __init__.

        Returns:
            True if loading was successful, False otherwise
        """
        path = checkpoint_path or self.checkpoint_path
        if not path or not os.path.exists(path):
            logger.warning(f"Checkpoint path not found: {path}")
            return False

        try:
            # Try to load the agent directly
            self.agent.load(path)
            logger.info(f"Loaded pre-trained agent from {path}")

            # Adjust learning rate for fine-tuning
            if hasattr(self.agent, 'learning_rate'):
                old_lr = self.agent.learning_rate
                self.agent.learning_rate = self.transfer_learning_rate
                logger.info(
                    f"Adjusted learning rate for fine-tuning: "
                    f"{old_lr} -> {self.transfer_learning_rate}"
                )

            # Freeze layers if requested
            if self.freeze_layers > 0:
                self._freeze_layers(self.freeze_layers)

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the specified number of layers in the agent's network.

        Args:
            num_layers: Number of layers to freeze from the beginning
        """
        try:
            import torch

            # Handle PyTorch-based agents
            if hasattr(self.agent, 'policy_net') and self.agent.policy_net is not None:
                # Freeze layers in policy network
                if isinstance(self.agent.policy_net, torch.nn.Sequential):
                    for i, module in enumerate(self.agent.policy_net):
                        if i < num_layers:
                            for param in module.parameters():
                                param.requires_grad = False
                            logger.info(f"Froze layer {i} in policy network")

            if hasattr(self.agent, 'value_net') and self.agent.value_net is not None:
                # Freeze layers in value network
                if isinstance(self.agent.value_net, torch.nn.Sequential):
                    for i, module in enumerate(self.agent.value_net):
                        if i < num_layers:
                            for param in module.parameters():
                                param.requires_grad = False
                            logger.info(f"Froze layer {i} in value network")

            # Handle LSTM-based agents
            if hasattr(self.agent, 'lstm') and self.agent.lstm is not None:
                if num_layers > 0:
                    for param in self.agent.lstm.parameters():
                        param.requires_grad = False
                    logger.info("Froze LSTM layer")

            # Handle Transformer-based agents
            if hasattr(self.agent, 'transformer_encoder') and self.agent.transformer_encoder is not None:
                if num_layers > 0:
                    for param in self.agent.transformer_encoder.parameters():
                        param.requires_grad = False
                    logger.info("Froze Transformer encoder")

        except ImportError:
            logger.warning("PyTorch not available; layer freezing skipped")

    def fine_tune(
        self,
        num_cycles: int = 5,
        episodes_per_cycle: int = 20,
    ) -> List[EvaluationResult]:
        """Fine-tune the pre-trained agent on the new environment.

        Args:
            num_cycles: Number of training cycles
            episodes_per_cycle: Number of episodes per cycle

        Returns:
            List of evaluation results after each cycle
        """
        # Load pre-trained weights if not already loaded
        if self.checkpoint_path:
            self.load_pretrained_agent()

        results: List[EvaluationResult] = []

        for cycle in range(num_cycles):
            logger.info(
                f"Fine-tuning cycle {cycle + 1}/{num_cycles} "
                f"(learning_rate={self.agent.learning_rate if hasattr(self.agent, 'learning_rate') else 'N/A'})"
            )

            # Run training cycle
            cycle_result = self._train_cycle(episodes_per_cycle)
            results.append(cycle_result)

            # Log progress
            avg_reward = cycle_result.avg_reward if hasattr(cycle_result, 'avg_reward') else 0.0
            logger.info(f"Cycle {cycle + 1} completed, avg_reward: {avg_reward:.2f}")

        return results

    def _train_cycle(self, num_episodes: int) -> EvaluationResult:
        """Run a single fine-tuning cycle.

        Args:
            num_episodes: Number of episodes to train

        Returns:
            Evaluation result for this cycle
        """
        # Create environment with current difficulty
        env_config = self._get_current_env_config()
        environment = self.env_factory(env_config)

        # Run episodes
        episodes = []
        for _ in range(num_episodes):
            episode = self._run_episode(environment)
            episodes.append(episode)

        # Train agent
        if episodes:
            self.agent.train(episodes)

        # Evaluate
        result = self._evaluate_agent(environment)
        return result

    def _get_current_env_config(self) -> EnvironmentConfig:
        """Get the current environment config based on curriculum difficulty."""
        if self.curriculum_manager:
            difficulty = self.curriculum_manager.get_current_difficulty()
        else:
            difficulty = DifficultyLevel.EASY

        return EnvironmentConfig(
            name="transfer_env",
            difficulty=difficulty,
            parameters={},
            target_skills=[],
        )

    def _run_episode(self, environment) -> Any:
        """Run a single episode in the environment.

        Args:
            environment: Environment instance

        Returns:
            Episode object
        """
        from envgen.utils.types import Episode, Transition

        observation = environment.reset()
        transitions = []
        total_reward = 0.0
        done = False

        while not done:
            action = self.agent.select_action(observation)
            next_observation, reward, done, info = environment.step(action)

            # Apply reward engineering if available
            if self.reward_engine:
                reward = self.reward_engine.compute_reward(
                    observation, action, next_observation, reward, done
                )

            transition = Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
            transitions.append(transition)
            total_reward += reward
            observation = next_observation

        return Episode(transitions=transitions, total_reward=total_reward)

    def _evaluate_agent(self, environment) -> EvaluationResult:
        """Evaluate the agent in the environment.

        Args:
            environment: Environment instance

        Returns:
            Evaluation result
        """
        if self.evaluator:
            # Run evaluation episodes
            eval_episodes = []
            for _ in range(5):
                episode = self._run_episode(environment)
                eval_episodes.append(episode)

            result = self.evaluator.evaluate(eval_episodes)
            return result
        else:
            # Simple evaluation without skill evaluator
            episode = self._run_episode(environment)
            return EvaluationResult(
                skill_scores={},
                avg_reward=episode.total_reward,
                weak_skills=[],
                strong_skills=[],
            )

    def save_finetuned_agent(self, path: str) -> None:
        """Save the fine-tuned agent to a checkpoint.

        Args:
            path: Directory path to save the checkpoint
        """
        os.makedirs(path, exist_ok=True)
        self.agent.save(path)

        # Save transfer config
        config_path = os.path.join(path, "transfer_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "transfer_learning_rate": self.transfer_learning_rate,
                "freeze_layers": self.freeze_layers,
                "fine_tune_epochs": self.fine_tune_epochs,
                "source_checkpoint": self.checkpoint_path,
            }, f, indent=2)

        logger.info(f"Saved fine-tuned agent to {path}")
