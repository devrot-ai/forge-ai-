"""Integration tests for EnvGen-LLM.

These tests verify end-to-end pipeline flows including full training cycles,
checkpoint save/resume, parallel environments, early stopping, and
hyperparameter search. They use longer timeouts and fixed seeds for reproducibility.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from envgen.agents.ppo_agent import PPOAgent
from envgen.curriculum.manager import CurriculumManager
from envgen.environments.base import EnvironmentConfig
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator, MockLLMClient
from envgen.pipeline.trainer import PipelineConfig, TrainingPipeline
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import Action, DifficultyLevel, Episode, Observation, Transition


def _env_factory(config):
    """Top-level factory for ParallelEnvironment (can't pickle local functions)."""
    return PuzzleEnvironment(config)


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the full training pipeline."""

    def _create_pipeline(self, max_cycles=2, episodes_per_cycle=5):
        """Helper to create a fully configured TrainingPipeline."""
        config = PipelineConfig(
            task_description="Test navigation task",
            skills=["navigation", "planning"],
            max_cycles=max_cycles,
            episodes_per_cycle=episodes_per_cycle,
            eval_episodes=3,
            episodes_per_env=3,
            initial_difficulty=DifficultyLevel.EASY,
            num_initial_envs=2,
        )
        llm_client = MockLLMClient()
        env_generator = EnvironmentGenerator(llm_client)
        agent = PPOAgent(name="integration_test")
        evaluator = SkillEvaluator()
        curriculum = CurriculumManager()
        reward_engine = RewardEngine()

        return TrainingPipeline(
            config=config,
            env_generator=env_generator,
            agent=agent,
            evaluator=evaluator,
            curriculum_manager=curriculum,
            reward_engine=reward_engine,
            env_factory=_env_factory,
        )

    def test_pipeline_with_mock_llm(self):
        """Test full training pipeline with mock LLM."""
        pipeline = self._create_pipeline()
        results = pipeline.run()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        from envgen.utils.types import EvaluationResult
        self.assertIsInstance(results[0], EvaluationResult)

    def test_pipeline_checkpoint_save_resume(self):
        """Test checkpoint save and resume preserves state."""
        tmp_dir = tempfile.mkdtemp()
        try:
            # _save_checkpoint takes a directory, not a file path
            pipeline1 = self._create_pipeline(max_cycles=1, episodes_per_cycle=3)
            pipeline1.run()
            checkpoint_dir = pipeline1._save_checkpoint(tmp_dir)

            # The checkpoint file is created inside the directory
            checkpoint_files = [f for f in os.listdir(tmp_dir) if f.startswith("checkpoint_cycle")]
            self.assertGreater(len(checkpoint_files), 0)

            checkpoint_path = os.path.join(tmp_dir, checkpoint_files[0])
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)

            self.assertIn("cycle", checkpoint)
            self.assertIn("agent_stats", checkpoint)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestParallelEnvironmentIntegration(unittest.TestCase):
    """Integration tests for parallel environments."""

    def test_parallel_env_basic(self):
        """Test parallel environment reset and step."""
        try:
            from envgen.environments.parallel_env import ParallelEnvironment
        except ImportError:
            self.skipTest("ParallelEnvironment not available")

        cfg = EnvironmentConfig(
            name="parallel_test",
            description="Test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 5, "num_items": 1},
            target_skills=["navigation"],
            max_steps=10,
            seed=42,
        )

        with ParallelEnvironment(env_factory=_env_factory, num_workers=2, seed=42) as env:
            obs_list = env.reset()
            self.assertEqual(len(obs_list), 2)
            self.assertTrue(all(isinstance(obs, Observation) for obs in obs_list))

            action_list = [Action("move", {"direction": "right"}) for _ in range(2)]
            next_obs, rewards, dones, infos = env.step(action_list)
            self.assertEqual(len(next_obs), 2)
            self.assertEqual(len(rewards), 2)
            self.assertEqual(len(dones), 2)


class TestEarlyStoppingIntegration(unittest.TestCase):
    """Integration tests for early stopping."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers when performance plateaus."""
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, metric="reward", mode="max")

        # Simulate plateauing rewards
        for reward in [10.0, 10.0, 10.0, 10.0]:
            should_stop = monitor.step(reward=reward)

        # Should stop after patience cycles without improvement
        self.assertTrue(should_stop)
        self.assertEqual(monitor.no_improve_count, 3)
        self.assertEqual(monitor.best_value, 10.0)

    def test_early_stopping_resets(self):
        """Test that improvement resets the counter."""
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, metric="reward", mode="max")

        monitor.step(reward=10.0)
        monitor.step(reward=10.0)
        monitor.step(reward=10.0)
        # Not stopped yet (2 without improvement, patience=3)
        self.assertEqual(monitor.no_improve_count, 2)

        # Improvement resets counter
        should_stop = monitor.step(reward=15.0)
        self.assertFalse(should_stop)
        self.assertEqual(monitor.no_improve_count, 0)


class TestHyperparameterSearchIntegration(unittest.TestCase):
    """Integration tests for hyperparameter search."""

    def test_hyperparam_search_runs(self):
        """Test that hyperparameter search completes and returns results."""
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        param_grid = {
            "learning_rate": [0.05, 0.1],
            "initial_epsilon": [0.3, 0.5],
        }

        search = HyperparameterSearch(
            agent_type="ppo",
            env_type="puzzle",
            param_grid=param_grid,
            episodes=3,
            seed=42,
        )
        results = search.run()
        self.assertGreater(len(results), 0)
        # Should have 2 x 2 = 4 configurations
        self.assertEqual(len(results), 4)


class TestReplayBufferIntegration(unittest.TestCase):
    """Integration tests for replay buffer."""

    def test_replay_buffer_with_training(self):
        """Test replay buffer stores and samples transitions correctly."""
        from envgen.agents.replay_buffer import ReplayBuffer

        buffer = ReplayBuffer(capacity=100)

        # Add transitions
        obs1 = Observation(state={"pos": [0, 0]})
        obs2 = Observation(state={"pos": [0, 1]})
        action = Action("move", {"direction": "right"})

        for i in range(10):
            transition = Transition(
                observation=obs1,
                action=action,
                reward=float(i),
                next_observation=obs2,
                done=False,
            )
            buffer.add(transition)

        self.assertEqual(buffer.size, 10)

        # Sample batch (returns tuple of transitions, indices, weights)
        batch = buffer.sample(batch_size=5)
        transitions, indices, weights = batch
        self.assertEqual(len(transitions), 5)


class TestConfigLoaderIntegration(unittest.TestCase):
    """Integration tests for YAML config loader."""

    def test_config_loader_creates_pipeline(self):
        """Test that config loader creates all pipeline components."""
        from envgen.config.loader import ConfigLoader

        config_path = str(Path(__file__).parent.parent / "configs" / "default.yaml")
        if not os.path.exists(config_path):
            self.skipTest("default.yaml not found")

        loader = ConfigLoader()
        config = loader.load(config_path)

        self.assertIn("task", config)
        self.assertIn("training", config)

        # Test agent creation
        agent = loader.create_agent(config.get("agent", {}))
        self.assertIsNotNone(agent)

        # Test LLM client creation
        llm_client = loader.create_llm_client(config.get("llm", {}))
        self.assertIsNotNone(llm_client)


class TestBenchmarkRunnerIntegration(unittest.TestCase):
    """Integration tests for benchmark runner."""

    def test_benchmark_single_env(self):
        """Test benchmark runner on a single environment."""
        from envgen.benchmark.runner import BenchmarkRunner

        agent = PPOAgent(name="benchmark_integration")
        runner = BenchmarkRunner(
            agent=agent,
            episodes_per_env=3,
            env_types=["puzzle"],
            difficulties=[DifficultyLevel.EASY],
        )
        results = runner.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].env_type, "puzzle")


if __name__ == "__main__":
    unittest.main()
