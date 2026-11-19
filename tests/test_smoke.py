"""Smoke tests for EnvGen-LLM v1.0.

Validates that all CLI commands, agent types, environment types, and LLM
clients can be imported and created without errors. These are lightweight
integration checks, not full functional tests.
"""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestAgentImports(unittest.TestCase):
    """Verify all agent types can be imported."""

    def test_import_ppo_agent(self):
        from envgen.agents import PPOAgent  # noqa: F401
        self.assertTrue(True)

    def test_import_grpo_agent(self):
        from envgen.agents import GRPOAgent  # noqa: F401
        self.assertTrue(True)

    def test_import_nearest_neighbor_agent(self):
        from envgen.agents import NearestNeighborAgent  # noqa: F401
        self.assertTrue(True)

    def test_import_ppo_torch_agent(self):
        try:
            from envgen.agents import PPOAgentTorch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_import_grpo_torch_agent(self):
        try:
            from envgen.agents import GRPOAgentTorch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_import_rnn_agent(self):
        try:
            from envgen.agents import RNNAgent  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_import_transformer_agent(self):
        try:
            from envgen.agents import TransformerAgent  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")


class TestEnvironmentImports(unittest.TestCase):
    """Verify all environment types can be imported."""

    def test_import_puzzle_environment(self):
        from envgen.environments import PuzzleEnvironment  # noqa: F401
        self.assertTrue(True)

    def test_import_maze_environment(self):
        from envgen.environments import MazeEnvironment  # noqa: F401
        self.assertTrue(True)

    def test_import_resource_environment(self):
        from envgen.environments import ResourceEnvironment  # noqa: F401
        self.assertTrue(True)

    def test_import_text_task_environment(self):
        from envgen.environments import TextTaskEnvironment  # noqa: F401
        self.assertTrue(True)

    def test_import_cartpole_environment(self):
        from envgen.environments import CartPoleEnvironment  # noqa: F401
        self.assertTrue(True)

    def test_import_multi_agent_environment(self):
        from envgen.environments import MultiAgentEnvironment  # noqa: F401
        self.assertTrue(True)


class TestLLMImports(unittest.TestCase):
    """Verify all LLM clients can be imported."""

    def test_import_mock_llm(self):
        from envgen.llm import MockLLMClient  # noqa: F401
        self.assertTrue(True)

    def test_import_openai_llm(self):
        try:
            from envgen.llm import OpenAILLMClient  # noqa: F401
        except ImportError:
            self.skipTest("openai not installed")

    def test_import_ollama_llm(self):
        from envgen.llm import OllamaLLMClient  # noqa: F401
        self.assertTrue(True)

    def test_import_anthropic_llm(self):
        try:
            from envgen.llm import AnthropicLLMClient  # noqa: F401
        except ImportError:
            self.skipTest("anthropic not installed")


class TestAgentCreation(unittest.TestCase):
    """Verify all agents can be instantiated."""

    def test_create_ppo_agent(self):
        from envgen.agents import PPOAgent
        agent = PPOAgent(name="test", config={"learning_rate": 0.1})
        self.assertEqual(agent.name, "test")

    def test_create_grpo_agent(self):
        from envgen.agents import GRPOAgent
        agent = GRPOAgent(name="test", config={"learning_rate": 0.1})
        self.assertEqual(agent.name, "test")

    def test_create_nn_agent(self):
        from envgen.agents import NearestNeighborAgent
        agent = NearestNeighborAgent(name="test", config={"k": 5})
        self.assertEqual(agent.name, "test")


class TestEnvironmentCreation(unittest.TestCase):
    """Verify all environments can be instantiated."""

    def test_create_puzzle_environment(self):
        from envgen.environments import PuzzleEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test puzzle", difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 5, "num_items": 1, "num_obstacles": 1},
        )
        env = PuzzleEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())

    def test_create_maze_environment(self):
        from envgen.environments import MazeEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test maze", difficulty=DifficultyLevel.EASY,
            parameters={"maze_size": 5, "complexity": 0.3},
        )
        env = MazeEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())

    def test_create_resource_environment(self):
        from envgen.environments import ResourceEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test resource", difficulty=DifficultyLevel.EASY,
            parameters={"num_resources": 2, "resource_rate": 1.0},
        )
        env = ResourceEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())

    def test_create_text_environment(self):
        from envgen.environments import TextTaskEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test text", difficulty=DifficultyLevel.EASY,
            parameters={"task_type": "arithmetic", "num_choices": 4},
        )
        env = TextTaskEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())

    def test_create_cartpole_environment(self):
        from envgen.environments import CartPoleEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test cartpole", difficulty=DifficultyLevel.EASY,
            parameters={"max_steps": 50},
        )
        env = CartPoleEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())

    def test_create_multi_agent_environment(self):
        from envgen.environments import MultiAgentEnvironment, EnvironmentConfig
        from envgen.utils.types import DifficultyLevel
        config = EnvironmentConfig(
            name="test", description="test multi agent", difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 5, "num_items": 2, "cooperation_mode": False},
        )
        env = MultiAgentEnvironment(config=config)
        env.reset()
        self.assertIsNotNone(env.get_observation())


class TestPipelineImports(unittest.TestCase):
    """Verify pipeline components can be imported."""

    def test_import_training_pipeline(self):
        from envgen.pipeline.trainer import TrainingPipeline  # noqa: F401
        self.assertTrue(True)

    def test_import_pipeline_config(self):
        from envgen.pipeline.trainer import PipelineConfig  # noqa: F401
        self.assertTrue(True)

    def test_import_skill_evaluator(self):
        from envgen.evaluators.skill_evaluator import SkillEvaluator  # noqa: F401
        self.assertTrue(True)

    def test_import_curriculum_manager(self):
        from envgen.curriculum.manager import CurriculumManager  # noqa: F401
        self.assertTrue(True)

    def test_import_reward_engine(self):
        from envgen.rewards.engine import RewardEngine  # noqa: F401
        self.assertTrue(True)

    def test_import_environment_generator(self):
        from envgen.llm.generator import EnvironmentGenerator  # noqa: F401
        self.assertTrue(True)


class TestConfigLoader(unittest.TestCase):
    """Verify config loader works."""

    def test_config_loader_import(self):
        from envgen.config.loader import ConfigLoader  # noqa: F401
        self.assertTrue(True)

    def test_create_agent_via_loader(self):
        from envgen.config.loader import ConfigLoader
        loader = ConfigLoader()
        agent = loader.create_agent({"type": "ppo", "learning_rate": 0.1})
        self.assertIsNotNone(agent)


class TestVersionExport(unittest.TestCase):
    """Verify version is properly exported."""

    def test_version_exported(self):
        import envgen
        self.assertTrue(hasattr(envgen, "__version__"))
        self.assertIsInstance(envgen.__version__, str)
        self.assertTrue(len(envgen.__version__) > 0)


if __name__ == "__main__":
    unittest.main()
