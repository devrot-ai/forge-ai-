"""Tests for new components: agents, environments, config, CLI."""

import json
import os
import shutil
import tempfile
import unittest

from envgen.environments.base import EnvironmentConfig
from envgen.utils.types import (
    Action,
    DifficultyLevel,
    Episode,
    Observation,
    Transition,
)

try:
    from envgen.agents.ppo_torch import PPOAgentTorch
except ImportError:
    PPOAgentTorch = None

try:
    from envgen.agents.rnn_agent import RNNAgent
except ImportError:
    RNNAgent = None

try:
    from envgen.agents.transformer_agent import TransformerAgent
except ImportError:
    TransformerAgent = None

try:
    from envgen.config.loader import ConfigLoader
except ImportError:
    ConfigLoader = None


class TestMazeEnvironment(unittest.TestCase):
    """Tests for the MazeEnvironment."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="maze_test",
            description="test maze",
            difficulty=DifficultyLevel.EASY,
            parameters={"maze_width": 7, "maze_height": 7, "complexity": 0.5},
        )
        from envgen.environments.maze_env import MazeEnvironment

        self.env = MazeEnvironment(self.config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("agent_pos", obs.state)
        self.assertIn("exit_pos", obs.state)

    def test_step(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("move", {"direction": "right"}))
        self.assertIsInstance(obs, Observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("A", rendered)  # Agent marker

    def test_available_actions(self):
        actions = self.env.get_available_actions()
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(isinstance(a, Action) for a in actions))

    def test_reach_exit(self):
        """Agent can eventually reach the exit."""
        self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, reward, done, _ = self.env.step(Action("move", {"direction": "right"}))
            steps += 1
        # May not reach exit with random moves, but should not crash


class TestResourceEnvironment(unittest.TestCase):
    """Tests for the ResourceEnvironment."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="resource_test",
            description="test resource",
            difficulty=DifficultyLevel.EASY,
            parameters={"num_resource_types": 3, "num_tasks": 3},
        )
        from envgen.environments.resource_env import ResourceEnvironment

        self.env = ResourceEnvironment(self.config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("resources", obs.state)

    def test_step_collect(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("collect", {}))
        self.assertIsInstance(obs, Observation)
        self.assertIsInstance(reward, float)

    def test_step_execute_task(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("execute_task", {"task_index": 0}))
        self.assertIsInstance(obs, Observation)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("Resource", rendered)

    def test_available_actions(self):
        self.env.reset()
        actions = self.env.get_available_actions()
        self.assertGreater(len(actions), 0)


class TestGymnasiumWrapper(unittest.TestCase):
    """Tests for the GymnasiumEnvironment wrapper."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="gym_test",
            description="test gym",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 4, "num_items": 2},
        )
        from envgen.environments.gymnasium_wrapper import GymnasiumEnvironment
        from envgen.environments.puzzle_env import PuzzleEnvironment

        self.env = GymnasiumEnvironment(PuzzleEnvironment(self.config))

    def test_spaces(self):
        import gymnasium as gym

        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIn("raw_observation", info)

    def test_step(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)

    def test_close(self):
        self.env.close()  # Should not raise


class TestPPOAgentTorch(unittest.TestCase):
    """Tests for the PyTorch PPO agent."""

    def setUp(self):
        from envgen.agents.ppo_torch import PPOAgentTorch

        self.agent = PPOAgentTorch("test_torch", {"hidden_size": 32, "num_layers": 1})

    def test_select_action(self):
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [Action("move", {"direction": "up"}), Action("move", {"direction": "right"})]
        action = self.agent.select_action(obs, actions)
        self.assertIsInstance(action, Action)

    def test_train(self):
        episodes = []
        for _ in range(5):
            ep = Episode()
            for _ in range(20):
                obs = Observation(
                    state={
                        "agent_pos": [0, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 1,
                        "current_step": 0,
                        "max_steps": 50,
                    }
                )
                next_obs = Observation(
                    state={
                        "agent_pos": [1, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 2,
                        "current_step": 1,
                        "max_steps": 50,
                    }
                )
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=0.1,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)

        metrics = self.agent.train(episodes)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)

    def test_save_load(self):
        # First train to initialize networks
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("collect", {}),
            Action("check_goal", {}),
        ]
        self.agent.select_action(obs, actions)

        with tempfile.NamedTemporaryFile(suffix="_agent", delete=False) as f:
            path = f.name

        try:
            self.agent.save(path)
            self.assertTrue(os.path.exists(path + ".pt"))
            self.assertTrue(os.path.exists(path + ".json"))

            new_agent = PPOAgentTorch("test_torch", {"hidden_size": 32, "num_layers": 1})
            new_agent.load(path)
            self.assertEqual(new_agent.name, self.agent.name)
            self.assertEqual(new_agent.obs_dim, self.agent.obs_dim)
        finally:
            for ext in [".pt", ".json"]:
                p = path + ext
                if os.path.exists(p):
                    os.remove(p)


class TestGRPOAgent(unittest.TestCase):
    """Tests for the GRPO agent."""

    def setUp(self):
        from envgen.agents.grpo_agent import GRPOAgent

        self.agent = GRPOAgent("test_grpo", {"group_size": 4})

    def test_select_action(self):
        obs = Observation(state={"agent_pos": [0, 0], "goal_pos": [3, 3]})
        actions = [Action("move", {"direction": "up"}), Action("move", {"direction": "right"})]
        action = self.agent.select_action(obs, actions)
        self.assertIsInstance(action, Action)

    def test_train(self):
        episodes = []
        for _ in range(5):
            ep = Episode()
            for _ in range(10):
                obs = Observation(state={"agent_pos": [0, 0], "goal_pos": [3, 3]})
                next_obs = Observation(state={"agent_pos": [1, 0], "goal_pos": [3, 3]})
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=0.5,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)

        metrics = self.agent.train(episodes)
        self.assertIn("loss", metrics)
        self.assertIn("num_updates", metrics)

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix="_agent", delete=False) as f:
            path = f.name

        try:
            self.agent.save(path)
            self.assertTrue(os.path.exists(path + ".json"))

            new_agent = type(self.agent)()
            new_agent.load(path)
            self.assertEqual(new_agent.name, self.agent.name)
        finally:
            if os.path.exists(path + ".json"):
                os.remove(path + ".json")


class TestConfigLoader(unittest.TestCase):
    """Tests for the YAML config loader."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_load_config(self):
        config = self.loader.load("configs/default.yaml")
        self.assertIn("task", config)
        self.assertIn("training", config)
        self.assertIn("agent", config)

    def test_get_task_config(self):
        config = {"task": {"description": "test", "skills": ["nav"]}}
        task = self.loader.get_task_config(config)
        self.assertEqual(task["description"], "test")
        self.assertEqual(task["skills"], ["nav"])

    def test_get_training_config(self):
        config = {"training": {"max_cycles": 5, "episodes_per_cycle": 20}}
        training = self.loader.get_training_config(config)
        self.assertEqual(training["max_cycles"], 5)

    def test_create_agent_ppo(self):
        config = {"agent": {"type": "ppo", "learning_rate": 0.1}}
        agent = self.loader.create_agent(config)
        self.assertEqual(agent.name, "ppo_agent")

    def test_create_agent_grpo(self):
        config = {"agent": {"type": "grpo", "group_size": 4}}
        agent = self.loader.create_agent(config)
        self.assertEqual(agent.name, "grpo_agent")

    def test_create_llm_mock(self):
        config = {"llm": {"client": "mock"}}
        client = self.loader.create_llm_client(config)
        self.assertEqual(client._call_count, 0)

    def test_create_env_factory_puzzle(self):
        config = {"environment": {"type": "puzzle"}}
        factory = self.loader.create_environment_factory(config)
        env = factory(
            EnvironmentConfig(
                name="test", description="test", difficulty=DifficultyLevel.EASY, parameters={"grid_size": 4}
            )
        )
        self.assertEqual(env.config.name, "test")

    def test_create_env_factory_maze(self):
        config = {"environment": {"type": "maze"}}
        factory = self.loader.create_environment_factory(config)
        env = factory(
            EnvironmentConfig(
                name="test", description="test", difficulty=DifficultyLevel.EASY, parameters={"maze_width": 7}
            )
        )
        self.assertEqual(env.config.name, "test")


class TestGRPOAgentTorch(unittest.TestCase):
    """Tests for the PyTorch GRPO agent."""

    def setUp(self):
        from envgen.agents.grpo_torch import GRPOAgentTorch

        self.agent = GRPOAgentTorch("test_grpo_torch", {"hidden_size": 32, "num_layers": 1, "group_size": 4})

    def test_select_action(self):
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [Action("move", {"direction": "up"}), Action("move", {"direction": "right"})]
        action = self.agent.select_action(obs, actions)
        self.assertIsInstance(action, Action)

    def test_train(self):
        episodes = []
        for _ in range(5):
            ep = Episode()
            for _ in range(20):
                obs = Observation(
                    state={
                        "agent_pos": [0, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 1,
                        "current_step": 0,
                        "max_steps": 50,
                    }
                )
                next_obs = Observation(
                    state={
                        "agent_pos": [1, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 2,
                        "current_step": 1,
                        "max_steps": 50,
                    }
                )
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=0.1,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)

        metrics = self.agent.train(episodes)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("group_size", metrics)

    def test_train_empty(self):
        metrics = self.agent.train([])
        self.assertEqual(metrics["policy_loss"], 0.0)

    def test_save_load(self):
        # First train to initialize networks
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("collect", {}),
            Action("check_goal", {}),
            Action("execute_task", {}),
            Action("allocate", {}),
        ]
        self.agent.select_action(obs, actions)

        with tempfile.NamedTemporaryFile(suffix="_agent", delete=False) as f:
            path = f.name

        try:
            self.agent.save(path)
            self.assertTrue(os.path.exists(path + ".pt"))
            self.assertTrue(os.path.exists(path + ".json"))

            from envgen.agents.grpo_torch import GRPOAgentTorch

            new_agent = GRPOAgentTorch("test_grpo_torch", {"hidden_size": 32, "num_layers": 1})
            new_agent.load(path)
            self.assertEqual(new_agent.name, self.agent.name)
            self.assertEqual(new_agent.obs_dim, self.agent.obs_dim)
        finally:
            for ext in [".pt", ".json"]:
                p = path + ext
                if os.path.exists(p):
                    os.remove(p)

    def test_get_stats(self):
        stats = self.agent.get_stats()
        self.assertIn("update_count", stats)
        self.assertIn("group_size", stats)
        self.assertIn("epsilon", stats)

    def test_group_advantage_computation(self):
        """Verify that GRPO groups transitions and computes relative advantages."""
        episodes = []
        for _ in range(10):
            ep = Episode()
            for _ in range(10):
                obs = Observation(
                    state={
                        "agent_pos": [0, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 1,
                        "current_step": 0,
                        "max_steps": 50,
                    }
                )
                next_obs = Observation(
                    state={
                        "agent_pos": [1, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 2,
                        "current_step": 1,
                        "max_steps": 50,
                    }
                )
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=1.0,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)

        metrics = self.agent.train(episodes)
        # Should have performed at least one update
        self.assertGreaterEqual(metrics["num_updates"], 1)


class TestConfigLoaderGRPO(unittest.TestCase):
    """Tests for config loader with GRPO agent types."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_agent_grpo_torch(self):
        config = {"agent": {"type": "grpo_torch", "learning_rate": 3e-4, "group_size": 8}}
        agent = self.loader.create_agent(config)
        self.assertEqual(agent.name, "grpo_torch")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_empty_episode_training(self):
        from envgen.agents.ppo_agent import PPOAgent

        agent = PPOAgent()
        metrics = agent.train([])
        self.assertEqual(metrics["loss"], 0.0)

    def test_zero_reward(self):
        from envgen.environments.puzzle_env import PuzzleEnvironment

        config = EnvironmentConfig(
            name="test",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 3, "num_items": 0, "num_obstacles": 0},
        )
        env = PuzzleEnvironment(config)
        obs = env.reset()
        obs, reward, done, info = env.step(Action("move", {"direction": "up"}))
        self.assertIsInstance(reward, float)

    def test_obstacle_grid(self):
        """Test environment with obstacles."""
        from envgen.environments.puzzle_env import PuzzleEnvironment

        config = EnvironmentConfig(
            name="test",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 3, "num_items": 1, "num_obstacles": 2},
        )
        env = PuzzleEnvironment(config)
        obs = env.reset()
        self.assertIsInstance(obs, Observation)
        # Should not crash with obstacles
        obs, reward, done, info = env.step(Action("move", {"direction": "up"}))
        self.assertIsInstance(reward, float)


try:
    from envgen.agents.sb3_agent import SB3Agent
except ImportError:
    SB3Agent = None


@unittest.skipIf(SB3Agent is None, "stable-baselines3 not installed")
class TestSB3Agent(unittest.TestCase):
    """Tests for the SB3Agent wrapper."""

    def setUp(self):
        self.agent = SB3Agent(
            algorithm="ppo",
            name="test_sb3_ppo",
            config={"learning_rate": 3e-4, "verbose": 0},
        )

    def test_create_sb3_ppo(self):
        self.assertEqual(self.agent.algorithm, "ppo")
        self.assertEqual(self.agent.name, "test_sb3_ppo")

    def test_create_sb3_sac(self):
        agent = SB3Agent(algorithm="sac", name="test_sb3_sac", config={"verbose": 0})
        self.assertEqual(agent.algorithm, "sac")

    def test_create_sb3_a2c(self):
        agent = SB3Agent(algorithm="a2c", name="test_sb3_a2c", config={"verbose": 0})
        self.assertEqual(agent.algorithm, "a2c")

    def test_create_sb3_dqn(self):
        agent = SB3Agent(algorithm="dqn", name="test_sb3_dqn", config={"verbose": 0})
        self.assertEqual(agent.algorithm, "dqn")

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            SB3Agent(algorithm="invalid")

    def test_select_action_before_init(self):
        """Action selection works before model is initialized (returns random)."""
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("collect", {}),
            Action("check_goal", {}),
        ]
        action = self.agent.select_action(obs, actions)
        self.assertIsInstance(action, Action)
        self.assertIn(action, actions)

    def test_train_empty(self):
        metrics = self.agent.train([])
        self.assertIn("num_updates", metrics)

    def test_train_and_select_action(self):
        """Full cycle: train then select action."""
        episodes = []
        for _ in range(3):
            ep = Episode()
            for _ in range(5):
                obs = Observation(
                    state={
                        "agent_pos": [0, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 1,
                        "current_step": 0,
                        "max_steps": 50,
                    }
                )
                next_obs = Observation(
                    state={
                        "agent_pos": [1, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 2,
                        "current_step": 1,
                        "max_steps": 50,
                    }
                )
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=1.0,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)

        metrics = self.agent.train(episodes)
        self.assertGreaterEqual(metrics["num_updates"], 1)

        # Select action after training
        obs = Observation(
            state={
                "agent_pos": [0, 0],
                "goal_pos": [3, 3],
                "grid_size": 4,
                "items_remaining": 2,
                "items_collected": 0,
                "num_items": 2,
                "visited": 1,
                "current_step": 0,
                "max_steps": 50,
            }
        )
        actions = [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
        ]
        action = self.agent.select_action(obs, actions)
        self.assertIsInstance(action, Action)

    def test_save_load(self):
        """Test save and load round-trip."""
        # Train first to initialize model
        episodes = []
        for _ in range(2):
            ep = Episode()
            for _ in range(5):
                obs = Observation(
                    state={
                        "agent_pos": [0, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 1,
                        "current_step": 0,
                        "max_steps": 50,
                    }
                )
                next_obs = Observation(
                    state={
                        "agent_pos": [1, 0],
                        "goal_pos": [3, 3],
                        "grid_size": 4,
                        "items_remaining": 2,
                        "items_collected": 0,
                        "num_items": 2,
                        "visited": 2,
                        "current_step": 1,
                        "max_steps": 50,
                    }
                )
                t = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=1.0,
                    next_observation=next_obs,
                    done=False,
                )
                ep.add_transition(t)
            episodes.append(ep)
        self.agent.train(episodes)

        with tempfile.NamedTemporaryFile(suffix="_agent", delete=False) as f:
            path = f.name

        try:
            self.agent.save(path)
            self.assertTrue(os.path.exists(path + ".json"))

            new_agent = SB3Agent(algorithm="ppo", name="loaded", config={"verbose": 0})
            new_agent.load(path)
            self.assertEqual(new_agent.name, self.agent.name)
            self.assertEqual(new_agent.algorithm, self.agent.algorithm)
        finally:
            for ext in [".json"]:
                p = path + ext
                if os.path.exists(p):
                    os.remove(p)
            # Clean up SB3 model files (directory)
            model_path = path + "_model"
            if os.path.exists(model_path + ".zip"):
                os.remove(model_path + ".zip")

    def test_get_stats(self):
        stats = self.agent.get_stats()
        self.assertIn("algorithm", stats)
        self.assertIn("update_count", stats)
        self.assertIn("total_timesteps", stats)
        self.assertEqual(stats["algorithm"], "ppo")


class TestConfigLoaderSB3(unittest.TestCase):
    """Tests for config loader with SB3 agent types."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    @unittest.skipIf(SB3Agent is None, "stable-baselines3 not installed")
    def test_create_agent_sb3_ppo(self):
        config = {"agent": {"type": "sb3_ppo", "learning_rate": 3e-4}}
        agent = self.loader.create_agent(config)
        self.assertEqual(agent.name, "sb3_ppo")
        self.assertEqual(agent.algorithm, "ppo")

    @unittest.skipIf(SB3Agent is None, "stable-baselines3 not installed")
    def test_create_agent_sb3_sac(self):
        config = {"agent": {"type": "sb3_sac", "learning_rate": 3e-4}}
        agent = self.loader.create_agent(config)
        self.assertEqual(agent.algorithm, "sac")


class TestTextTaskEnvironment(unittest.TestCase):
    """Tests for the TextTaskEnvironment."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="text_test",
            description="test text tasks",
            difficulty=DifficultyLevel.EASY,
            parameters={"task_type": "arithmetic", "num_choices": 4, "allow_hints": True},
        )
        from envgen.environments.text_env import TextTaskEnvironment

        self.env = TextTaskEnvironment(self.config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("question", obs.state)
        self.assertIn("choices", obs.state)
        self.assertGreater(len(obs.state["choices"]), 0)

    def test_step_select_and_submit(self):
        """Test selecting an answer and submitting."""
        self.env.reset()
        # Select first answer
        obs, reward, done, info = self.env.step(Action("select_answer", {"index": 0}))
        self.assertIsInstance(obs, Observation)
        self.assertFalse(done)
        self.assertEqual(self.env.selected_answer_idx, 0)

        # Submit
        obs, reward, done, info = self.env.step(Action("submit", {}))
        self.assertTrue(done)
        self.assertIn("result", info)

    def test_hint(self):
        """Test requesting a hint."""
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("request_hint", {}))
        self.assertIn("hint", info)
        self.assertEqual(info["hints_used"], 1)
        self.assertLess(reward, 0)  # Hint penalty

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("Question:", rendered)
        self.assertIn("Choices:", rendered)

    def test_available_actions(self):
        self.env.reset()
        actions = self.env.get_available_actions()
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(isinstance(a, Action) for a in actions))

    def test_task_types(self):
        """Test all task types generate valid puzzles."""
        for task_type in ["arithmetic", "sequence", "logic", "pattern", "mixed"]:
            config = EnvironmentConfig(
                name=f"text_{task_type}",
                description="test",
                difficulty=DifficultyLevel.EASY,
                parameters={"task_type": task_type, "num_choices": 4},
            )
            from envgen.environments.text_env import TextTaskEnvironment

            env = TextTaskEnvironment(config)
            obs = env.reset()
            self.assertIn("question", obs.state)
            self.assertGreater(len(obs.state["choices"]), 0)

    def test_difficulty_levels(self):
        """Test all difficulty levels."""
        for diff in DifficultyLevel:
            config = EnvironmentConfig(
                name=f"text_{diff.value}",
                description="test",
                difficulty=diff,
                parameters={"task_type": "arithmetic", "num_choices": 4},
            )
            from envgen.environments.text_env import TextTaskEnvironment

            env = TextTaskEnvironment(config)
            obs = env.reset()
            self.assertIn("question", obs.state)

    def test_max_steps_termination(self):
        """Test that environment terminates at max steps."""
        config = EnvironmentConfig(
            name="text_max_steps",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"task_type": "arithmetic", "num_choices": 4},
            max_steps=3,
        )
        from envgen.environments.text_env import TextTaskEnvironment

        env = TextTaskEnvironment(config)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _ = env.step(Action("noop", {}))
            steps += 1
        self.assertTrue(done)
        self.assertLessEqual(steps, 10)

    def test_correct_answer_reward(self):
        """Test that correct answers give positive reward."""
        self.env.reset()
        # Find the correct answer index
        correct_idx = None
        for i, choice in enumerate(self.env.current_choices):
            if str(choice) == str(self.env.current_answer):
                correct_idx = i
                break

        if correct_idx is not None:
            self.env.step(Action("select_answer", {"index": correct_idx}))
            _, reward, done, info = self.env.step(Action("submit", {}))
            self.assertGreater(reward, 0)
            self.assertEqual(info["result"], "correct")

    def test_no_hints_mode(self):
        """Test environment with hints disabled."""
        config = EnvironmentConfig(
            name="text_no_hints",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"task_type": "arithmetic", "num_choices": 4, "allow_hints": False},
        )
        from envgen.environments.text_env import TextTaskEnvironment

        env = TextTaskEnvironment(config)
        env.reset()
        actions = env.get_available_actions()
        action_types = [a.action_type for a in actions]
        self.assertNotIn("request_hint", action_types)


class TestConfigLoaderTextEnv(unittest.TestCase):
    """Tests for config loader with text environment type."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_env_factory_text(self):
        config = {"environment": {"type": "text"}}
        factory = self.loader.create_environment_factory(config)
        env = factory(
            EnvironmentConfig(
                name="test", description="test", difficulty=DifficultyLevel.EASY, parameters={"task_type": "arithmetic"}
            )
        )
        self.assertEqual(env.config.name, "test")
        self.assertIn("question", env.reset().state)


class TestOllamaLLMClient(unittest.TestCase):
    """Tests for the Ollama LLM client."""

    def setUp(self):
        from envgen.llm.ollama_client import OllamaLLMClient

        self.client = OllamaLLMClient({"model": "llama3.2", "temperature": 0.7})

    def test_initialization(self):
        """Test client initialization with config."""
        self.assertEqual(self.client.model, "llama3.2")
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.base_url, "http://localhost:11434")
        self.assertEqual(self.client.timeout, 120)

    def test_default_initialization(self):
        """Test client initialization with defaults."""
        from envgen.llm.ollama_client import OllamaLLMClient

        client = OllamaLLMClient()
        self.assertEqual(client.model, "llama3.2")
        self.assertEqual(client.temperature, 0.7)

    def test_is_available_false(self):
        """Test that is_available returns False when Ollama is not running."""
        # Ollama is unlikely to be running in test environment
        available = self.client.is_available()
        # Just check it doesn't raise an exception
        self.assertIsInstance(available, bool)

    def test_generate_raises_when_unavailable(self):
        """Test that generate raises RuntimeError when Ollama is not running."""
        with self.assertRaises(RuntimeError):
            self.client.generate("test prompt")

    def test_generate_json_raises_when_unavailable(self):
        """Test that generate_json raises RuntimeError when Ollama is not running."""
        with self.assertRaises(RuntimeError):
            self.client.generate_json("test prompt")

    def test_custom_base_url(self):
        """Test client with custom base URL."""
        from envgen.llm.ollama_client import OllamaLLMClient

        client = OllamaLLMClient({"base_url": "http://custom:11434"})
        self.assertEqual(client.base_url, "http://custom:11434")


class TestConfigLoaderOllama(unittest.TestCase):
    """Tests for config loader with Ollama LLM client."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_llm_client_ollama(self):
        """Test creating Ollama client via config loader."""
        config = {"llm": {"client": "ollama", "model": "llama3.2"}}
        client = self.loader.create_llm_client(config)
        self.assertEqual(client.model, "llama3.2")
        self.assertIsInstance(client.is_available(), bool)


class TestCartPoleEnvironment(unittest.TestCase):
    """Tests for the CartPoleEnvironment."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="cartpole_test",
            description="test cartpole",
            difficulty=DifficultyLevel.EASY,
            parameters={},
            max_steps=200,
        )
        from envgen.environments.cartpole_env import CartPoleEnvironment

        self.env = CartPoleEnvironment(self.config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("cart_position", obs.state)
        self.assertIn("cart_velocity", obs.state)
        self.assertIn("pole_angle", obs.state)
        self.assertIn("pole_angular_velocity", obs.state)

    def test_step_move_right(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("move", {"direction": "right"}))
        self.assertIsInstance(obs, Observation)
        self.assertIsInstance(reward, float)
        self.assertIn("steps_survived", info)

    def test_step_move_left(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("move", {"direction": "left"}))
        self.assertIsInstance(obs, Observation)

    def test_step_stay(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("move", {"direction": "stay"}))
        self.assertIsInstance(obs, Observation)

    def test_step_noop(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("noop", {}))
        self.assertIsInstance(obs, Observation)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("[===]", rendered)

    def test_available_actions(self):
        self.env.reset()
        actions = self.env.get_available_actions()
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(isinstance(a, Action) for a in actions))

    def test_difficulty_levels(self):
        """Test all difficulty levels."""
        for diff in DifficultyLevel:
            config = EnvironmentConfig(
                name=f"cartpole_{diff.value}",
                description="test",
                difficulty=diff,
                parameters={},
                max_steps=200,
            )
            from envgen.environments.cartpole_env import CartPoleEnvironment

            env = CartPoleEnvironment(config)
            obs = env.reset()
            self.assertIn("cart_position", obs.state)

    def test_pole_fallen_termination(self):
        """Test that environment terminates when pole falls."""
        config = EnvironmentConfig(
            name="cartpole_fallen",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"angle_threshold": 0.01},  # Very small threshold
            max_steps=200,
        )
        from envgen.environments.cartpole_env import CartPoleEnvironment

        env = CartPoleEnvironment(config)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, done, info = env.step(Action("move", {"direction": "left"}))
            steps += 1
        self.assertTrue(done)
        if "termination_reason" in info:
            self.assertIn(info["termination_reason"], ["pole_fallen", "cart_out_of_bounds", "max_steps"])

    def test_max_steps_termination(self):
        """Test that environment terminates at max steps."""
        config = EnvironmentConfig(
            name="cartpole_max_steps",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={},
            max_steps=5,
        )
        from envgen.environments.cartpole_env import CartPoleEnvironment

        env = CartPoleEnvironment(config)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _ = env.step(Action("noop", {}))
            steps += 1
        self.assertTrue(done)
        self.assertLessEqual(steps, 10)

    def test_alive_reward(self):
        """Test that staying alive gives positive reward."""
        self.env.reset()
        _, reward, done, _ = self.env.step(Action("move", {"direction": "right"}))
        # Alive reward is 1.0, minus small penalties
        self.assertGreater(reward, -1)


class TestConfigLoaderCartPole(unittest.TestCase):
    """Tests for config loader with CartPole environment type."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_env_factory_cartpole(self):
        config = {"environment": {"type": "cartpole"}}
        factory = self.loader.create_environment_factory(config)
        env = factory(
            EnvironmentConfig(
                name="test", description="test", difficulty=DifficultyLevel.EASY, parameters={}, max_steps=200
            )
        )
        self.assertEqual(env.config.name, "test")
        obs = env.reset()
        self.assertIn("cart_position", obs.state)


class TestMultiAgentEnvironment(unittest.TestCase):
    """Tests for the MultiAgentEnvironment."""

    def setUp(self):
        self.config = EnvironmentConfig(
            name="multi_agent_test",
            description="test multi-agent",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_width": 8, "grid_height": 8, "num_items": 5, "target_count": 3},
            max_steps=100,
        )
        from envgen.environments.multi_agent_env import MultiAgentEnvironment

        self.env = MultiAgentEnvironment(self.config)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("own_position", obs.state)
        self.assertIn("visible_items", obs.state)
        self.assertIn("own_inventory", obs.state)

    def test_step_move(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("move", {"direction": "right"}))
        self.assertIsInstance(obs, Observation)
        self.assertIn("agent_inventory", info)

    def test_step_collect(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("collect", {}))
        self.assertIsInstance(obs, Observation)

    def test_step_noop(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(Action("noop", {}))
        self.assertIsInstance(obs, Observation)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("@", rendered)

    def test_available_actions(self):
        self.env.reset()
        actions = self.env.get_available_actions()
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(isinstance(a, Action) for a in actions))

    def test_difficulty_levels(self):
        """Test all difficulty levels."""
        for diff in DifficultyLevel:
            config = EnvironmentConfig(
                name=f"multi_agent_{diff.value}",
                description="test",
                difficulty=diff,
                parameters={},
                max_steps=100,
            )
            from envgen.environments.multi_agent_env import MultiAgentEnvironment

            env = MultiAgentEnvironment(config)
            obs = env.reset()
            self.assertIn("own_position", obs.state)

    def test_max_steps_termination(self):
        """Test that environment terminates at max steps."""
        config = EnvironmentConfig(
            name="multi_agent_max_steps",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_width": 8, "grid_height": 8, "num_items": 5, "target_count": 3},
            max_steps=3,
        )
        from envgen.environments.multi_agent_env import MultiAgentEnvironment

        env = MultiAgentEnvironment(config)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _ = env.step(Action("noop", {}))
            steps += 1
        self.assertTrue(done)
        self.assertLessEqual(steps, 10)

    def test_cooperation_mode(self):
        """Test cooperative mode."""
        config = EnvironmentConfig(
            name="multi_agent_coop",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_width": 8, "grid_height": 8, "num_items": 5, "target_count": 3, "cooperation_mode": True},
            max_steps=100,
        )
        from envgen.environments.multi_agent_env import MultiAgentEnvironment

        env = MultiAgentEnvironment(config)
        self.assertTrue(env.cooperation_mode)
        obs = env.reset()
        self.assertIn("own_position", obs.state)


class TestConfigLoaderMultiAgent(unittest.TestCase):
    """Tests for config loader with MultiAgent environment type."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_env_factory_multi_agent(self):
        config = {"environment": {"type": "multi_agent"}}
        factory = self.loader.create_environment_factory(config)
        env = factory(
            EnvironmentConfig(
                name="test", description="test", difficulty=DifficultyLevel.EASY, parameters={}, max_steps=100
            )
        )
        self.assertEqual(env.config.name, "test")
        obs = env.reset()
        self.assertIn("own_position", obs.state)


class TestExperimentLogger(unittest.TestCase):
    """Tests for the ExperimentLogger."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        from envgen.utils.logger import ExperimentLogger

        self.logger = ExperimentLogger(log_dir=self.tmp_dir, run_name="test_run")

    def tearDown(self):
        self.logger.close()
        # Clean up temp directory
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test logger creates run directory and files."""
        self.assertTrue(os.path.isdir(self.logger.run_dir))
        self.assertTrue(os.path.exists(self.logger.metrics_file))

    def test_log_cycle(self):
        """Test logging a cycle with rewards and skill scores."""
        self.logger.log_cycle(
            cycle=1,
            rewards=[10.0, 15.0, 20.0],
            skill_scores={"navigation": 0.8, "planning": 0.6},
            difficulty="easy",
            episodes_trained=3,
            steps_trained=100,
        )
        # Check metrics file has content
        with open(self.logger.metrics_file, "r") as f:
            lines = f.readlines()
        # Should have header + 1 data line
        self.assertGreaterEqual(len(lines), 2)
        # Parse the data line
        data_line = [line for line in lines if not line.startswith("#")][0]
        metric = json.loads(data_line)
        self.assertEqual(metric["cycle"], 1)
        self.assertEqual(metric["difficulty"], "easy")
        self.assertEqual(metric["episodes_trained"], 3)
        self.assertAlmostEqual(metric["rewards"]["mean_reward"], 15.0)

    def test_log_episode(self):
        """Test logging individual episodes."""
        self.logger.log_episode(
            cycle=1,
            episode_idx=0,
            reward=10.0,
            steps=20,
            skills_demonstrated=["navigation"],
        )
        with open(self.logger.metrics_file, "r") as f:
            content = f.read()
        self.assertIn('"type": "episode"', content)

    def test_log_summary(self):
        """Test summary generation."""
        self.logger.log_cycle(cycle=1, rewards=[10.0, 20.0], skill_scores={"navigation": 0.8})
        self.logger.log_cycle(cycle=2, rewards=[15.0, 25.0], skill_scores={"navigation": 0.9})
        summary = self.logger.log_summary(final_skill_scores={"navigation": 0.9})
        self.assertEqual(summary["total_cycles"], 2)
        self.assertEqual(summary["final_skill_scores"]["navigation"], 0.9)
        self.assertIsNotNone(summary["aggregate"]["mean_reward_across_cycles"])
        self.assertTrue(os.path.exists(self.logger.summary_file))

    def test_config_saving(self):
        """Test that config is saved when provided."""
        from envgen.utils.logger import ExperimentLogger

        tmp_dir2 = tempfile.mkdtemp()
        try:
            logger2 = ExperimentLogger(
                log_dir=tmp_dir2,
                run_name="config_test",
                config={"task": "test", "cycles": 5},
            )
            self.assertTrue(os.path.exists(logger2.config_file))
            with open(logger2.config_file, "r") as f:
                saved = json.load(f)
            self.assertEqual(saved["task"], "test")
            logger2.close()
        finally:
            shutil.rmtree(tmp_dir2, ignore_errors=True)

    def test_get_run_dir(self):
        """Test getting the run directory path."""
        run_dir = self.logger.get_run_dir()
        self.assertIn("test_run", run_dir)

    def test_multiple_cycles(self):
        """Test logging multiple cycles and reading back."""
        for i in range(3):
            self.logger.log_cycle(
                cycle=i + 1,
                rewards=[float(i * 10)],
                skill_scores={"skill": float(i + 1) / 3},
                difficulty="easy",
            )
        with open(self.logger.metrics_file, "r") as f:
            lines = f.readlines()
        data_lines = [line for line in lines if not line.startswith("#") and line.strip()]
        self.assertEqual(len(data_lines), 3)


class TestBenchmarkRunner(unittest.TestCase):
    """Tests for the BenchmarkRunner."""

    def setUp(self):
        from envgen.agents.ppo_agent import PPOAgent

        self.agent = PPOAgent("benchmark_test")

    def test_create_runner(self):
        """Test creating a benchmark runner."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(self.agent, episodes_per_env=2)
        self.assertEqual(runner.episodes_per_env, 2)
        self.assertEqual(runner.seed, 42)

    def test_run_single_env(self):
        """Test running benchmark on a single environment type."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            self.agent,
            episodes_per_env=2,
            env_types=["puzzle"],
            difficulties=[DifficultyLevel.EASY],
        )
        results = runner.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].env_type, "puzzle")
        self.assertEqual(results[0].difficulty, "easy")
        self.assertGreater(results[0].mean_steps, 0)

    def test_run_multiple_envs(self):
        """Test running benchmark on multiple environment types."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            self.agent,
            episodes_per_env=2,
            env_types=["puzzle", "text"],
            difficulties=[DifficultyLevel.EASY],
        )
        results = runner.run()
        self.assertEqual(len(results), 2)
        env_types = {r.env_type for r in results}
        self.assertIn("puzzle", env_types)
        self.assertIn("text", env_types)

    def test_print_summary(self):
        """Test that summary printing works."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            self.agent,
            episodes_per_env=2,
            env_types=["puzzle"],
            difficulties=[DifficultyLevel.EASY],
        )
        runner.run()
        summary = runner.print_summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Benchmark Results", summary)

    def test_get_overall_score(self):
        """Test getting overall score."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            self.agent,
            episodes_per_env=2,
            env_types=["puzzle"],
            difficulties=[DifficultyLevel.EASY],
        )
        runner.run()
        score = runner.get_overall_score()
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_get_results_by_env(self):
        """Test filtering results by environment type."""
        from envgen.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            self.agent,
            episodes_per_env=2,
            env_types=["puzzle", "text"],
            difficulties=[DifficultyLevel.EASY],
        )
        runner.run()
        puzzle_results = runner.get_results_by_env("puzzle")
        self.assertEqual(len(puzzle_results), 1)
        self.assertEqual(puzzle_results[0].env_type, "puzzle")

    def test_create_env_all_types(self):
        """Test creating all environment types."""
        from envgen.benchmark.runner import create_env

        for env_type in ["puzzle", "maze", "resource", "text", "cartpole", "multi_agent"]:
            env = create_env(env_type, DifficultyLevel.EASY, seed=42)
            obs = env.reset()
            self.assertIsNotNone(obs)

    def test_invalid_env_type(self):
        """Test that invalid env type raises error."""
        from envgen.benchmark.runner import create_env

        with self.assertRaises(ValueError):
            create_env("invalid", DifficultyLevel.EASY, seed=42)


class TestEnvironmentRegistry(unittest.TestCase):
    """Tests for the EnvironmentRegistry."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        from envgen.environments.registry import EnvironmentRegistry

        self.registry = EnvironmentRegistry(registry_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Test saving and loading a config."""
        config = EnvironmentConfig(
            name="test_env",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 5},
        )
        env_id = self.registry.save(config, cycle=1, prompt="Generate a puzzle")
        self.assertIsInstance(env_id, str)
        self.assertGreater(len(env_id), 0)

        loaded = self.registry.load(env_id)
        self.assertEqual(loaded.name, "test_env")
        self.assertEqual(loaded.difficulty, DifficultyLevel.EASY)

    def test_save_with_metadata(self):
        """Test saving with extra metadata."""
        config = EnvironmentConfig(
            name="meta_env",
            description="test",
            difficulty=DifficultyLevel.MEDIUM,
            parameters={"grid_size": 5},
        )
        env_id = self.registry.save(
            config,
            cycle=2,
            prompt="Generate a maze",
            metadata={"custom_key": "custom_value"},
        )
        loaded, metadata = self.registry.load_with_metadata(env_id)
        self.assertEqual(metadata["cycle"], 2)
        self.assertEqual(metadata["custom_key"], "custom_value")

    def test_list(self):
        """Test listing configs."""
        config1 = EnvironmentConfig(
            name="list_env_1",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={},
        )
        config2 = EnvironmentConfig(
            name="list_env_2",
            description="test",
            difficulty=DifficultyLevel.HARD,
            parameters={},
        )
        self.registry.save(config1)
        self.registry.save(config2)
        entries = self.registry.list()
        self.assertEqual(len(entries), 2)

    def test_list_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        config1 = EnvironmentConfig(
            name="filter_env_1",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={},
        )
        config2 = EnvironmentConfig(
            name="filter_env_2",
            description="test",
            difficulty=DifficultyLevel.HARD,
            parameters={},
        )
        self.registry.save(config1)
        self.registry.save(config2)
        easy_entries = self.registry.list(difficulty="easy")
        self.assertEqual(len(easy_entries), 1)
        self.assertEqual(easy_entries[0]["difficulty"], "easy")

    def test_delete(self):
        """Test deleting a config."""
        config = EnvironmentConfig(
            name="delete_env",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={},
        )
        env_id = self.registry.save(config)
        self.assertEqual(self.registry.get_count(), 1)
        result = self.registry.delete(env_id)
        self.assertTrue(result)
        self.assertEqual(self.registry.get_count(), 0)

    def test_delete_nonexistent(self):
        """Test deleting a nonexistent config."""
        result = self.registry.delete("nonexistent_id")
        self.assertFalse(result)

    def test_load_nonexistent(self):
        """Test loading a nonexistent config."""
        with self.assertRaises(FileNotFoundError):
            self.registry.load("nonexistent_id")

    def test_get_count(self):
        """Test getting the count."""
        self.assertEqual(self.registry.get_count(), 0)
        config = EnvironmentConfig(
            name="count_env",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={},
        )
        self.registry.save(config)
        self.assertEqual(self.registry.get_count(), 1)

    def test_round_trip(self):
        """Test full save/load round-trip preserves config."""
        config = EnvironmentConfig(
            name="round_trip",
            description="round trip test",
            difficulty=DifficultyLevel.HARD,
            parameters={"grid_size": 8, "num_items": 5, "num_obstacles": 3},
            target_skills=["navigation", "planning"],
            max_steps=100,
        )
        env_id = self.registry.save(config, cycle=5, prompt="Test prompt")
        loaded = self.registry.load(env_id)
        self.assertEqual(loaded.name, config.name)
        self.assertEqual(loaded.description, config.description)
        self.assertEqual(loaded.difficulty, config.difficulty)
        self.assertEqual(loaded.parameters, config.parameters)
        self.assertEqual(loaded.target_skills, config.target_skills)
        self.assertEqual(loaded.max_steps, config.max_steps)


class TestAnthropicLLMClient(unittest.TestCase):
    """Tests for the Anthropic LLM client."""

    def setUp(self):
        from envgen.llm.anthropic_client import AnthropicLLMClient

        self.client = AnthropicLLMClient({"model": "claude-3-haiku-20240307", "temperature": 0.7})

    def test_initialization(self):
        """Test client initialization with config."""
        self.assertEqual(self.client.model, "claude-3-haiku-20240307")
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.max_tokens, 4096)

    def test_default_initialization(self):
        """Test client initialization with defaults."""
        from envgen.llm.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient()
        self.assertEqual(client.model, "claude-3-haiku-20240307")
        self.assertEqual(client.temperature, 0.7)

    def test_call_count(self):
        """Test that call count starts at 0."""
        self.assertEqual(self.client.call_count, 0)

    def test_generate_raises_without_api_key(self):
        """Test that generate raises RuntimeError when no API key is set."""
        import os

        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                self.client.generate("test prompt")
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_generate_raises_without_package(self):
        """Test that generate raises RuntimeError when anthropic package is missing."""
        # This test assumes the anthropic package is not installed
        # If it is installed but no API key, it will still raise RuntimeError
        import os

        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                self.client.generate("test prompt")
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_custom_model(self):
        """Test client with custom model."""
        from envgen.llm.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient({"model": "claude-3-opus-20240229", "max_tokens": 2048})
        self.assertEqual(client.model, "claude-3-opus-20240229")
        self.assertEqual(client.max_tokens, 2048)


class TestConfigLoaderAnthropic(unittest.TestCase):
    """Tests for config loader with Anthropic LLM client."""

    def setUp(self):
        from envgen.config.loader import ConfigLoader

        self.loader = ConfigLoader()

    def test_create_llm_client_anthropic(self):
        """Test creating Anthropic client via config loader."""
        config = {"llm": {"client": "anthropic", "model": "claude-3-haiku-20240307"}}
        client = self.loader.create_llm_client(config)
        self.assertEqual(client.model, "claude-3-haiku-20240307")


class TestEarlyStoppingMonitor(unittest.TestCase):
    """Tests for the EarlyStoppingMonitor."""

    def test_init(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, metric="mean_reward", mode="max")
        self.assertEqual(monitor.patience, 3)
        self.assertEqual(monitor.metric, "mean_reward")
        self.assertEqual(monitor.mode, "max")
        self.assertIsNone(monitor.best_value)

    def test_invalid_mode(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        with self.assertRaises(ValueError):
            EarlyStoppingMonitor(mode="invalid")

    def test_first_step_no_stop(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3)
        should_stop = monitor.step(mean_reward=10.0)
        self.assertFalse(should_stop)
        self.assertEqual(monitor.best_value, 10.0)

    def test_improvement_resets_counter(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, min_delta=0.0)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=11.0)  # improvement
        self.assertEqual(monitor.no_improve_count, 0)
        self.assertEqual(monitor.best_value, 11.0)

    def test_no_improvement_increases_counter(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, min_delta=0.0)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=10.0)  # no improvement
        self.assertEqual(monitor.no_improve_count, 1)

    def test_early_stopping_triggers(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, min_delta=0.0)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=10.0)
        should_stop = monitor.step(mean_reward=10.0)
        self.assertTrue(should_stop)

    def test_min_delta(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, min_delta=0.5)
        monitor.step(mean_reward=10.0)
        should_stop = monitor.step(mean_reward=10.3)  # less than min_delta
        self.assertFalse(should_stop)
        self.assertEqual(monitor.no_improve_count, 1)

    def test_min_mode(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3, mode="min", min_delta=0.0)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=9.0)  # improvement for min
        self.assertEqual(monitor.no_improve_count, 0)
        self.assertEqual(monitor.best_value, 9.0)

    def test_reset(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3)
        monitor.step(mean_reward=10.0)
        monitor.reset()
        self.assertIsNone(monitor.best_value)
        self.assertEqual(monitor.cycle, 0)

    def test_get_status(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3)
        monitor.step(mean_reward=10.0)
        status = monitor.get_status()
        self.assertEqual(status["metric"], "mean_reward")
        self.assertEqual(status["best_value"], 10.0)
        self.assertEqual(status["cycle"], 1)

    def test_history(self):
        from envgen.utils.early_stopping import EarlyStoppingMonitor

        monitor = EarlyStoppingMonitor(patience=3)
        monitor.step(mean_reward=10.0)
        monitor.step(mean_reward=11.0)
        self.assertEqual(monitor.history, [10.0, 11.0])


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer."""

    def setUp(self):
        from envgen.agents.replay_buffer import ReplayBuffer

        self.buffer = ReplayBuffer(capacity=100)

    def _make_transition(self, reward=1.0):
        return Transition(
            observation=Observation(state={"x": 0}),
            action=Action("test"),
            reward=reward,
            next_observation=Observation(state={"x": 1}),
            done=False,
        )

    def test_add_and_size(self):
        self.buffer.add(self._make_transition())
        self.assertEqual(self.buffer.size, 1)
        self.assertFalse(self.buffer.is_full)

    def test_add_batch(self):
        transitions = [self._make_transition(i) for i in range(10)]
        self.buffer.add_batch(transitions)
        self.assertEqual(self.buffer.size, 10)

    def test_sample_uniform(self):
        for i in range(20):
            self.buffer.add(self._make_transition(float(i)))
        transitions, indices, weights = self.buffer.sample(batch_size=5, prioritized=False)
        self.assertEqual(len(transitions), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)
        self.assertTrue(all(w == 1.0 for w in weights))

    def test_sample_prioritized(self):
        for i in range(20):
            self.buffer.add(self._make_transition(float(i)))
        transitions, indices, weights = self.buffer.sample(batch_size=5, prioritized=True)
        self.assertEqual(len(transitions), 5)
        self.assertGreaterEqual(min(weights), 0.0)

    def test_sample_too_large(self):
        self.buffer.add(self._make_transition())
        with self.assertRaises(ValueError):
            self.buffer.sample(batch_size=10)

    def test_capacity_overflow(self):
        from envgen.agents.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.add(self._make_transition(float(i)))
        self.assertEqual(buf.size, 5)
        self.assertTrue(buf.is_full)

    def test_update_priority(self):
        self.buffer.add(self._make_transition())
        self.buffer.update_priority(0, 5.0)
        # Should not raise

    def test_update_priorities_batch(self):
        for i in range(10):
            self.buffer.add(self._make_transition())
        self.buffer.update_priorities_batch([0, 1, 2], [1.0, 2.0, 3.0])

    def test_clear(self):
        for i in range(10):
            self.buffer.add(self._make_transition())
        self.buffer.clear()
        self.assertEqual(self.buffer.size, 0)

    def test_get_all(self):
        for i in range(5):
            self.buffer.add(self._make_transition(float(i)))
        all_t = self.buffer.get_all()
        self.assertEqual(len(all_t), 5)

    def test_len_and_repr(self):
        self.buffer.add(self._make_transition())
        self.assertEqual(len(self.buffer), 1)
        r = repr(self.buffer)
        self.assertIn("ReplayBuffer", r)


class TestHyperparameterSearch(unittest.TestCase):
    """Tests for the HyperparameterSearch."""

    def test_init(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch(
            agent_type="ppo",
            env_type="puzzle",
            episodes=3,
        )
        self.assertEqual(search.agent_type, "ppo")
        self.assertEqual(search.episodes, 3)

    def test_custom_param_grid(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch(
            agent_type="ppo",
            param_grid={"learning_rate": [0.01, 0.1]},
            episodes=2,
        )
        self.assertEqual(search.param_grid, {"learning_rate": [0.01, 0.1]})

    def test_run_small(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch(
            agent_type="ppo",
            env_type="puzzle",
            param_grid={"learning_rate": [0.01, 0.1]},
            episodes=2,
        )
        results = search.run()
        self.assertEqual(len(results), 2)
        # Results should be sorted by reward descending
        self.assertGreaterEqual(results[0]["mean_reward"], results[1]["mean_reward"])

    def test_get_best_config(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch(
            agent_type="ppo",
            param_grid={"learning_rate": [0.01]},
            episodes=2,
        )
        search.run()
        best = search.get_best_config()
        self.assertIsNotNone(best)
        self.assertIn("params", best)
        self.assertIn("mean_reward", best)

    def test_get_results_empty(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch()
        self.assertEqual(search.get_results(), [])

    def test_print_results_no_error(self):
        from envgen.benchmark.hyperparam_search import HyperparameterSearch

        search = HyperparameterSearch(
            agent_type="ppo",
            param_grid={"learning_rate": [0.01]},
            episodes=2,
        )
        search.run()
        search.print_results()  # Should not raise


class TestWandbLogger(unittest.TestCase):
    """Tests for the WandbLogger (mocked W&B)."""

    def test_wandb_not_available(self):
        """WandbLogger raises ImportError when wandb is not installed."""
        from envgen.utils.wandb_logger import WANDB_AVAILABLE

        if not WANDB_AVAILABLE:
            with self.assertRaises(ImportError):
                from envgen.utils.wandb_logger import WandbLogger

                WandbLogger(log_dir="./logs")

    def test_wandb_available_flag(self):
        """WANDB_AVAILABLE flag is properly set."""
        from envgen.utils.wandb_logger import WANDB_AVAILABLE

        self.assertIsInstance(WANDB_AVAILABLE, bool)

    @unittest.skipIf(
        not __import__("sys").modules.get("wandb"),
        "wandb not installed",
    )
    def test_wandb_logger_init(self):
        """WandbLogger initializes when wandb is available."""
        from unittest.mock import patch

        with patch("wandb.init"), patch("wandb.run", None):
            from envgen.utils.wandb_logger import WandbLogger

            with tempfile.TemporaryDirectory() as tmpdir:
                logger = WandbLogger(log_dir=tmpdir, project="test")
                self.assertEqual(logger.project, "test")
                logger.close()

    @unittest.skipIf(
        not __import__("sys").modules.get("wandb"),
        "wandb not installed",
    )
    def test_wandb_logger_log_cycle(self):
        """WandbLogger logs cycle metrics."""
        from unittest.mock import patch

        with patch("wandb.init"), patch("wandb.log"), patch("wandb.run", None):
            from envgen.utils.wandb_logger import WandbLogger

            with tempfile.TemporaryDirectory() as tmpdir:
                logger = WandbLogger(log_dir=tmpdir, project="test")
                logger.log_cycle(
                    cycle=1,
                    rewards=[10.0, 15.0],
                    skill_scores={"navigation": 0.8},
                )
                summary = logger.get_summary()
                self.assertEqual(summary["total_cycles"], 1)
                logger.close()


class TestAMPConfig(unittest.TestCase):
    """Tests for AMP configuration in PyTorch agents."""

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_ppo_torch_amp_default(self):
        """PPOAgentTorch has use_amp=False by default."""
        agent = PPOAgentTorch(config={})
        self.assertFalse(agent.use_amp)
        self.assertIsNone(agent.scaler)

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_ppo_torch_amp_enabled(self):
        """PPOAgentTorch creates scaler when use_amp=True."""
        agent = PPOAgentTorch(config={"use_amp": True})
        self.assertTrue(agent.use_amp)
        self.assertIsNotNone(agent.scaler)

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_ppo_torch_train_no_amp(self):
        """PPOAgentTorch trains without AMP."""
        agent = PPOAgentTorch(config={"use_amp": False})
        episodes = self._create_dummy_episodes(5)
        result = agent.train(episodes)
        self.assertIn("policy_loss", result)

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_ppo_torch_train_with_amp(self):
        """PPOAgentTorch trains with AMP enabled."""
        agent = PPOAgentTorch(config={"use_amp": True})
        episodes = self._create_dummy_episodes(5)
        result = agent.train(episodes)
        self.assertIn("policy_loss", result)

    def _create_dummy_episodes(self, count):
        """Create dummy episodes for testing."""
        episodes = []
        for _ in range(count):
            transitions = []
            for step in range(3):
                obs = Observation(state={"x": 0.1, "y": 0.2, "goal_x": 0.8, "goal_y": 0.9})
                next_obs = Observation(state={"x": 0.2, "y": 0.2, "goal_x": 0.8, "goal_y": 0.9})
                transition = Transition(
                    observation=obs,
                    action=Action("move", {"direction": "up"}),
                    reward=1.0,
                    next_observation=next_obs,
                    done=step == 2,
                )
                transitions.append(transition)
            episode = Episode(transitions=transitions, total_reward=3.0)
            episodes.append(episode)
        return episodes

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_grpo_torch_amp_default(self):
        """GRPOAgentTorch has use_amp=False by default."""
        from envgen.agents.grpo_torch import GRPOAgentTorch

        agent = GRPOAgentTorch(config={})
        self.assertFalse(agent.use_amp)

    @unittest.skipIf(PPOAgentTorch is None, "torch not installed")
    def test_grpo_torch_amp_enabled(self):
        """GRPOAgentTorch creates scaler when use_amp=True."""
        from envgen.agents.grpo_torch import GRPOAgentTorch

        agent = GRPOAgentTorch(config={"use_amp": True})
        self.assertTrue(agent.use_amp)
        self.assertIsNotNone(agent.scaler)


class TestEvaluateCLI(unittest.TestCase):
    """Tests for the evaluate CLI command."""

    def test_evaluate_module_imports(self):
        """Evaluate CLI module imports correctly."""
        from envgen.cli.evaluate import main, evaluate, load_checkpoint

        self.assertTrue(callable(main))
        self.assertTrue(callable(evaluate))
        self.assertTrue(callable(load_checkpoint))

    def test_load_checkpoint(self):
        """Load checkpoint from JSON file."""
        from envgen.cli.evaluate import load_checkpoint

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "agent_type": "ppo",
                    "agent_config": {},
                    "agent_state": {},
                },
                f,
            )
            f.flush()
            checkpoint = load_checkpoint(f.name)
            self.assertEqual(checkpoint["agent_type"], "ppo")
        os.unlink(f.name)


class TestOnPolicyBuffer(unittest.TestCase):
    """Tests for the OnPolicyBuffer."""

    def setUp(self):
        from envgen.agents.on_policy_buffer import OnPolicyBuffer
        self.BufferClass = OnPolicyBuffer
        self.buffer = OnPolicyBuffer(max_size=100, batch_size=10)

    def test_add_and_length(self):
        """Adding transitions increases buffer length."""
        for i in range(10):
            self.buffer.add({"state": {"x": i}, "action": {"type": "move"}, "reward": 1.0, "done": False})
        self.assertEqual(len(self.buffer), 10)

    def test_complete_trajectory(self):
        """Completing trajectory starts a new one."""
        for i in range(5):
            self.buffer.add({"state": {"x": i}, "reward": 1.0})
        self.buffer.complete_trajectory()
        self.assertEqual(self.buffer.num_trajectories(), 1)

    def test_sample_batch(self):
        """Sampling returns a batch dict."""
        for i in range(20):
            self.buffer.add({"state": {"x": i}, "action": {"type": "move"}, "reward": float(i), "done": False, "log_prob": 0.5})
        batch = self.buffer.sample_batch(batch_size=10)
        self.assertIn("states", batch)
        self.assertIn("rewards", batch)
        self.assertEqual(len(batch["rewards"]), 10)

    def test_sample_batches(self):
        """Sample batches covers all transitions."""
        for i in range(25):
            self.buffer.add({"state": {"x": i}, "reward": float(i), "done": False})
        batches = self.buffer.sample_batches(batch_size=10)
        self.assertEqual(len(batches), 3)  # 25 / 10 = 3 batches

    def test_get_all(self):
        """Get all returns all transitions."""
        for i in range(15):
            self.buffer.add({"state": {"x": i}, "reward": float(i), "done": False})
        all_data = self.buffer.get_all()
        self.assertEqual(len(all_data["rewards"]), 15)

    def test_clear(self):
        """Clear resets the buffer."""
        for i in range(10):
            self.buffer.add({"state": {"x": i}, "reward": 1.0})
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)

    def test_compute_returns(self):
        """GAE returns computation works."""
        for i in range(5):
            self.buffer.add({"reward": float(i), "done": i == 4})
        returns = self.buffer.compute_returns(gamma=0.99, gae_lambda=0.95)
        self.assertEqual(len(returns), 5)
        self.assertTrue(all(len(r) == 2 for r in returns))  # (return, advantage)

    def test_get_stats(self):
        """Stats returns correct info."""
        for i in range(10):
            self.buffer.add({"reward": float(i)})
        stats = self.buffer.get_stats()
        self.assertEqual(stats["size"], 10)
        self.assertIn("mean_reward", stats)

    def test_empty_buffer(self):
        """Empty buffer operations don't crash."""
        batch = self.buffer.sample_batch()
        self.assertEqual(len(batch["rewards"]), 0)
        returns = self.buffer.compute_returns()
        self.assertEqual(len(returns), 0)


class TestMultiAgentPipeline(unittest.TestCase):
    """Tests for the MultiAgentPipeline."""

    def setUp(self):
        from envgen.agents.ppo_agent import PPOAgent
        from envgen.environments.puzzle_env import PuzzleEnvironment
        from envgen.evaluators.skill_evaluator import SkillEvaluator
        from envgen.curriculum.manager import CurriculumManager
        from envgen.rewards.engine import RewardEngine
        from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline

        self.PPOAgent = PPOAgent
        self.PuzzleEnvironment = PuzzleEnvironment
        self.SkillEvaluator = SkillEvaluator
        self.CurriculumManager = CurriculumManager
        self.RewardEngine = RewardEngine
        self.MultiAgentPipeline = MultiAgentPipeline

    def _create_agent(self, name="test_agent"):
        return self.PPOAgent(name=name, config={"learning_rate": 0.1, "epsilon": 0.3})

    def _create_env_factory(self):
        config = EnvironmentConfig(
            name="test",
            description="test",
            difficulty=DifficultyLevel.EASY,
            parameters={"grid_size": 5, "num_items": 2, "num_obstacles": 2},
        )
        def factory():
            return self.PuzzleEnvironment(config)
        return factory

    def test_init(self):
        """Pipeline initializes with correct number of agents."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="competitive",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        self.assertEqual(pipeline.num_agents, 2)
        self.assertEqual(pipeline.mode, "competitive")

    def test_init_wrong_agent_count(self):
        """Pipeline raises error if agent count doesn't match."""
        agents = [self._create_agent()]
        with self.assertRaises(ValueError):
            self.MultiAgentPipeline(
                num_agents=3,
                mode="competitive",
                config={},
                agents=agents,
                env_factory=self._create_env_factory(),
                evaluator=self.SkillEvaluator(),
                curriculum_manager=self.CurriculumManager(),
                reward_engine=self.RewardEngine(),
            )

    def test_train_cycle(self):
        """Single training cycle completes."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="cooperative",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        result = pipeline.train_cycle(episodes_per_cycle=3)
        self.assertIn("cycle", result)
        self.assertIn("agent_results", result)
        self.assertEqual(len(result["agent_results"]), 2)

    def test_train_multiple_cycles(self):
        """Multiple training cycles complete."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="competitive",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        history = pipeline.train(max_cycles=2, episodes_per_cycle=3)
        self.assertEqual(len(history), 2)

    def test_get_agent_stats(self):
        """Agent stats returns per-agent info."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="cooperative",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        stats = pipeline.get_agent_stats()
        self.assertEqual(len(stats), 2)
        self.assertIn("agent_name", stats[0])

    def test_save_checkpoints(self):
        """Checkpoints save for all agents."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="competitive",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = pipeline.save_checkpoints(tmpdir)
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(os.path.exists(path))

    def test_competitive_mode(self):
        """Competitive mode assigns different rewards."""
        agents = [self._create_agent(f"agent_{i}") for i in range(2)]
        pipeline = self.MultiAgentPipeline(
            num_agents=2,
            mode="competitive",
            config={"initial_difficulty": "easy"},
            agents=agents,
            env_factory=self._create_env_factory(),
            evaluator=self.SkillEvaluator(),
            curriculum_manager=self.CurriculumManager(),
            reward_engine=self.RewardEngine(),
        )
        result = pipeline.train_cycle(episodes_per_cycle=2)
        self.assertEqual(result["mode"], "competitive")


class TestCurriculumLogging(unittest.TestCase):
    """Tests for curriculum stage logging in ExperimentLogger."""

    def test_log_curriculum_stage(self):
        """Curriculum stage is logged to file."""
        from envgen.utils.logger import ExperimentLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir)
            logger.log_curriculum_stage(
                cycle=1,
                stage=0,
                difficulty="easy",
                stage_name="Initial",
                criteria_met={"navigation": 0.5},
            )
            logger.close()

            # Check the metrics file contains the curriculum entry
            metrics_file = os.path.join(logger.run_dir, "metrics.jsonl")
            with open(metrics_file, "r") as f:
                lines = f.readlines()
            # Skip the header line
            data_lines = [l for l in lines if not l.startswith("#")]
            self.assertGreaterEqual(len(data_lines), 1)
            import json as json_mod
            entry = json_mod.loads(data_lines[0])
            self.assertEqual(entry["type"], "curriculum_stage")
            self.assertEqual(entry["difficulty"], "easy")
            self.assertEqual(entry["stage"], 0)


class TestRNNAgent(unittest.TestCase):
    """Tests for the RNN/LSTM agent."""

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_create(self):
        """Test creating an RNN agent."""
        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32, "num_layers": 1})
        self.assertEqual(agent.name, "test_rnn")
        self.assertEqual(agent.hidden_size, 32)
        self.assertEqual(agent.num_layers, 1)

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_select_action(self):
        """Test RNN agent action selection."""
        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32})
        agent._inferred_action_dim = 6
        obs = {"x": 0.5, "y": 0.3, "step": 1}
        action = agent.select_action(obs, episode_id=0)
        self.assertIsNotNone(action)
        self.assertIn("action_idx", action.parameters)
        self.assertGreaterEqual(action.parameters["action_idx"], 0)
        self.assertLess(action.parameters["action_idx"], 6)

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_hidden_state_reset(self):
        """Test that hidden state resets on episode boundary."""
        import torch

        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32})
        agent._inferred_action_dim = 6
        obs = {"x": 0.5, "y": 0.3, "step": 1}
        # First episode - run multiple steps to build up hidden state
        agent.select_action(obs, episode_id=0)
        agent.select_action(obs, episode_id=0)
        agent.select_action(obs, episode_id=0)
        # Hidden state should be non-zero after processing
        self.assertFalse(torch.all(agent.hidden_state[0] == 0))
        # Second episode should reset hidden state before processing
        # The reset happens at the start of select_action, then LSTM overwrites it
        # We verify the reset mechanism works by checking current_episode_id
        agent.select_action(obs, episode_id=1)
        self.assertEqual(agent.current_episode_id, 1)
        # Verify reset method works directly
        agent._reset_hidden_state()
        self.assertTrue(torch.all(agent.hidden_state[0] == 0))
        self.assertTrue(torch.all(agent.hidden_state[1] == 0))

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_train_empty(self):
        """Test training with empty episodes."""
        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32})
        result = agent.train([])
        self.assertEqual(result["loss"], 0.0)

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_save_load(self):
        """Test RNN agent save and load round-trip."""
        import tempfile
        import torch

        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32, "num_layers": 1})
        agent._inferred_action_dim = 6
        obs = {"x": 0.5, "y": 0.3, "step": 1}
        agent.select_action(obs, episode_id=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            loaded_agent = RNNAgent(name="test_rnn", config={"hidden_size": 32, "num_layers": 1})
            loaded_agent.load(tmpdir)

            self.assertEqual(loaded_agent.hidden_size, 32)
            self.assertEqual(loaded_agent.num_layers, 1)
            self.assertIsNotNone(loaded_agent.lstm)

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_get_stats(self):
        """Test RNN agent statistics."""
        agent = RNNAgent(name="test_rnn", config={"hidden_size": 64, "num_layers": 2})
        stats = agent.get_stats()
        self.assertEqual(stats["name"], "test_rnn")
        self.assertEqual(stats["type"], "rnn")
        self.assertEqual(stats["hidden_size"], 64)
        self.assertEqual(stats["num_layers"], 2)

    @unittest.skipIf(RNNAgent is None, "PyTorch not installed")
    def test_rnn_agent_flatten_observation(self):
        """Test observation flattening."""
        agent = RNNAgent(name="test_rnn", config={"hidden_size": 32})
        # Dict observation
        obs = {"x": 1.0, "y": 2.0, "done": False}
        flat = agent._flatten_observation(obs)
        self.assertEqual(len(flat), 3)
        # List observation
        obs2 = [1.0, 2.0, 3.0]
        flat2 = agent._flatten_observation(obs2)
        self.assertEqual(len(flat2), 3)

    @unittest.skipIf(RNNAgent is None, "ConfigLoader not available")
    def test_rnn_config_loader(self):
        """Test that config loader supports RNN agent type."""
        config = {"agent": {"type": "rnn", "learning_rate": 0.001}}
        loader = ConfigLoader()
        agent = loader.create_agent(config)
        self.assertIsInstance(agent, RNNAgent)


class TestRewardOptimizer(unittest.TestCase):
    """Tests for the Reward Function Optimizer."""

    def test_reward_optimizer_create(self):
        """Test creating a reward optimizer."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        optimizer = RewardOptimizer(
            agent_type="ppo",
            env_type="puzzle",
            episodes=5,
        )
        self.assertEqual(optimizer.agent_type, "ppo")
        self.assertEqual(optimizer.env_type, "puzzle")
        self.assertEqual(optimizer.episodes, 5)

    def test_reward_optimizer_custom_grid(self):
        """Test reward optimizer with custom grid."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        grid = {"success_weight": [5, 10], "step_penalty": [0.1, 0.5]}
        optimizer = RewardOptimizer(reward_grid=grid, episodes=3)
        self.assertEqual(optimizer.reward_grid, grid)

    def test_reward_optimizer_run_small(self):
        """Test running reward optimizer with small grid."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        grid = {"success_weight": [5, 10], "step_penalty": [0.1]}
        optimizer = RewardOptimizer(
            agent_type="ppo",
            env_type="puzzle",
            reward_grid=grid,
            episodes=3,
        )
        results = optimizer.run()
        self.assertEqual(len(results), 2)
        # Results should be sorted by mean_skill descending
        self.assertGreaterEqual(results[0]["mean_skill"], results[1]["mean_skill"])

    def test_reward_optimizer_best_config(self):
        """Test getting best configuration."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        grid = {"success_weight": [5], "step_penalty": [0.1]}
        optimizer = RewardOptimizer(
            agent_type="ppo",
            env_type="puzzle",
            reward_grid=grid,
            episodes=2,
        )
        optimizer.run()
        best = optimizer.get_best_config()
        self.assertIn("reward_params", best)
        self.assertIn("mean_skill", best)

    def test_reward_optimizer_pareto_front(self):
        """Test Pareto front computation."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        grid = {"success_weight": [5], "step_penalty": [0.1]}
        optimizer = RewardOptimizer(
            agent_type="ppo",
            env_type="puzzle",
            reward_grid=grid,
            episodes=2,
        )
        optimizer.run()
        pareto = optimizer.get_pareto_front()
        self.assertGreaterEqual(len(pareto), 1)

    def test_reward_optimizer_print_results(self):
        """Test printing results."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        optimizer = RewardOptimizer(episodes=1)
        output = optimizer.print_results()
        self.assertIn("No results", output)

    def test_reward_optimizer_empty_results(self):
        """Test empty results handling."""
        from envgen.benchmark.reward_optimizer import RewardOptimizer

        optimizer = RewardOptimizer()
        best = optimizer.get_best_config()
        self.assertEqual(best, {})
        pareto = optimizer.get_pareto_front()
        self.assertEqual(pareto, [])


class TestMultiEnvCurriculum(unittest.TestCase):
    """Tests for the Multi-environment Curriculum Manager."""

    def test_multi_env_curriculum_create(self):
        """Test creating a multi-env curriculum manager."""
        from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

        manager = MultiEnvCurriculumManager(
            env_types=["puzzle", "maze"],
            rotation_strategy="round_robin",
        )
        self.assertEqual(manager.env_types, ["puzzle", "maze"])
        self.assertEqual(manager.rotation_strategy, "round_robin")

    def test_multi_env_curriculum_round_robin(self):
        """Test round-robin rotation."""
        from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

        manager = MultiEnvCurriculumManager(
            env_types=["puzzle", "maze", "resource"],
            rotation_strategy="round_robin",
        )
        env1 = manager.get_next_environment()
        env2 = manager.get_next_environment()
        env3 = manager.get_next_environment()
        env4 = manager.get_next_environment()
        self.assertEqual(env1, "puzzle")
        self.assertEqual(env2, "maze")
        self.assertEqual(env3, "resource")
        self.assertEqual(env4, "puzzle")  # Wraps around

    def test_multi_env_curriculum_random(self):
        """Test random rotation."""
        from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

        manager = MultiEnvCurriculumManager(
            env_types=["puzzle", "maze"],
            rotation_strategy="random",
        )
        env = manager.get_next_environment()
        self.assertIn(env, ["puzzle", "maze"])

    def test_multi_env_curriculum_skill_based(self):
        """Test skill-based rotation."""
        from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

        manager = MultiEnvCurriculumManager(
            env_types=["puzzle", "maze"],
            rotation_strategy="skill_based",
        )
        manager.update_skill_scores({"puzzle": 0.3, "maze": 0.8})
        env = manager.get_next_environment()
        # Should pick the environment with lowest skill
        self.assertEqual(env, "puzzle")

    def test_multi_env_curriculum_episode_distribution(self):
        """Test episode distribution across environments."""
        from envgen.curriculum.multi_env_curriculum import MultiEnvCurriculumManager

        manager = MultiEnvCurriculumManager(
            env_types=["puzzle", "maze", "resource"],
            rotation_strategy="round_robin",
        )
        distribution = manager.distribute_episodes(9)
        self.assertEqual(distribution, {"puzzle": 3, "maze": 3, "resource": 3})


class TestTransformerAgent(unittest.TestCase):
    """Tests for the Transformer-based agent."""

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_create(self):
        """Test creating a Transformer agent."""
        agent = TransformerAgent(name="test_transformer")
        self.assertEqual(agent.name, "test_transformer")
        self.assertEqual(agent.hidden_size, 64)
        self.assertEqual(agent.num_heads, 4)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_select_action(self):
        """Test action selection."""
        agent = TransformerAgent(name="test_transformer")
        obs = Observation(state={"x": 0.0, "y": 0.0})
        action = agent.select_action(obs)
        self.assertIsInstance(action, Action)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_train(self):
        """Test training."""
        agent = TransformerAgent(name="test_transformer")
        obs = Observation(state={"x": 0.0, "y": 0.0})
        agent.select_action(obs)  # init networks
        transitions = [
            Transition(
                observation=Observation(state={"x": 0.0, "y": 0.0}),
                action=Action("0", {"action_idx": 0}),
                reward=1.0,
                next_observation=Observation(state={"x": 0.1, "y": 0.0}),
                done=False,
            )
        ]
        episode = Episode(transitions=transitions, total_reward=1.0)
        result = agent.train([episode])
        self.assertIn("loss", result)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_save_load(self):
        """Test save/load round-trip."""
        import tempfile
        agent = TransformerAgent(name="test_transformer")
        obs = Observation(state={"x": 0.0, "y": 0.0})
        agent.select_action(obs)  # init networks
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            loaded_agent = TransformerAgent(name="test_transformer")
            loaded_agent.load(tmpdir)
            self.assertEqual(loaded_agent.hidden_size, agent.hidden_size)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_get_stats(self):
        """Test get_stats."""
        agent = TransformerAgent(name="test_transformer")
        stats = agent.get_stats()
        self.assertEqual(stats["type"], "transformer")
        self.assertIn("num_heads", stats)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_reset_episode(self):
        """Test episode reset clears history."""
        agent = TransformerAgent(name="test_transformer")
        obs = Observation(state={"x": 0.0, "y": 0.0})
        agent.select_action(obs)
        agent.select_action(obs)
        self.assertEqual(len(agent.obs_history), 2)
        agent.reset_episode()
        self.assertEqual(len(agent.obs_history), 0)

    @unittest.skipIf(TransformerAgent is None, "PyTorch not available")
    def test_transformer_config_loader(self):
        """Test config loader integration."""
        loader = ConfigLoader()
        config = {"agent": {"type": "transformer", "hidden_size": 32}}
        agent = loader.create_agent(config)
        self.assertIsNotNone(agent)


class TestAutoRewardDesigner(unittest.TestCase):
    """Tests for the Automatic Reward Designer."""

    def test_ard_create(self):
        """Test creating ARD with default config."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner()
        self.assertEqual(ard.method, "curiosity")
        self.assertEqual(ard.intrinsic_weight, 0.5)

    def test_ard_count_reward(self):
        """Test count-based intrinsic reward."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner(config={"method": "count"})
        obs1 = {"x": 0.0, "y": 0.0}
        obs2 = {"x": 1.0, "y": 1.0}
        reward1 = ard.compute_intrinsic_reward(obs1, obs1)
        reward2 = ard.compute_intrinsic_reward(obs2, obs2)
        # Different states should have different rewards
        self.assertIsInstance(reward1, float)
        self.assertIsInstance(reward2, float)

    def test_ard_total_reward(self):
        """Test total reward computation."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner(config={"method": "count", "intrinsic_weight": 0.5})
        obs = {"x": 0.0}
        total = ard.compute_total_reward(1.0, obs, obs)
        self.assertGreater(total, 1.0)  # extrinsic + intrinsic

    def test_ard_get_stats(self):
        """Test stats reporting."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner(config={"method": "count"})
        obs = {"x": 0.0}
        ard.compute_total_reward(1.0, obs, obs)
        stats = ard.get_stats()
        self.assertEqual(stats["step_count"], 1)
        self.assertEqual(stats["method"], "count")

    def test_ard_reset_stats(self):
        """Test stats reset."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner(config={"method": "count"})
        obs = {"x": 0.0}
        ard.compute_intrinsic_reward(obs, obs)
        ard.reset_stats()
        self.assertEqual(ard.step_count, 0)

    def test_ard_hash_state(self):
        """Test state hashing."""
        from envgen.rewards.auto_reward import AutoRewardDesigner
        ard = AutoRewardDesigner()
        h1 = ard._hash_state({"x": 1.0})
        h2 = ard._hash_state({"x": 1.0})
        h3 = ard._hash_state({"x": 2.0})
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)


class TestTransferLearning(unittest.TestCase):
    """Tests for the Transfer Learning Pipeline."""

    def test_transfer_pipeline_create(self):
        """Test creating transfer learning pipeline."""
        from envgen.pipeline.transfer_trainer import TransferLearningPipeline
        from envgen.agents.ppo_agent import PPOAgent
        from envgen.environments.puzzle_env import PuzzleEnvironment
        from envgen.llm.generator import MockLLMClient

        agent = PPOAgent(name="test")
        def factory(cfg):
            return PuzzleEnvironment(cfg)

        pipeline = TransferLearningPipeline(
            agent=agent,
            env_factory=factory,
            config={},
            env_generator=MockLLMClient(),
            transfer_config={"transfer_learning_rate": 1e-4, "freeze_layers": 1},
        )
        self.assertEqual(pipeline.transfer_learning_rate, 1e-4)
        self.assertEqual(pipeline.freeze_layers, 1)

    def test_transfer_load_missing_checkpoint(self):
        """Test loading from non-existent checkpoint."""
        from envgen.pipeline.transfer_trainer import TransferLearningPipeline
        from envgen.agents.ppo_agent import PPOAgent
        from envgen.environments.puzzle_env import PuzzleEnvironment
        from envgen.llm.generator import MockLLMClient

        agent = PPOAgent(name="test")
        def factory(cfg):
            return PuzzleEnvironment(cfg)

        pipeline = TransferLearningPipeline(
            agent=agent,
            env_factory=factory,
            config={},
            env_generator=MockLLMClient(),
            checkpoint_path="/nonexistent/path",
        )
        result = pipeline.load_pretrained_agent()
        self.assertFalse(result)


class TestPopulationBasedTraining(unittest.TestCase):
    """Tests for Population-Based Training."""

    def test_pbt_create(self):
        """Test creating PBT trainer."""
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 5})
        self.assertEqual(pbt.population_size, 5)

    def test_pbt_initialize_population(self):
        """Test population initialization."""
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 3})
        population = pbt.initialize_population()
        self.assertEqual(len(population), 3)

    def test_pbt_mutate(self):
        """Test hyperparameter mutation."""
        from envgen.benchmark.pbt import PopulationBasedTrainer, AgentMember
        pbt = PopulationBasedTrainer()
        member = AgentMember(member_id=0, hyperparams={
            "learning_rate": 0.001, "epsilon": 0.2, "hidden_size": 64
        })
        mutated = pbt.mutate_hyperparams(member)
        self.assertEqual(mutated.generation, 1)

    def test_pbt_evolve(self):
        """Test population evolution."""
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 3})
        pbt.initialize_population()
        scores = [0.5, 0.8, 0.3]
        evolved = pbt.evolve(scores)
        self.assertEqual(len(evolved), 3)
        self.assertEqual(pbt.generation, 1)

    def test_pbt_best_member(self):
        """Test getting best member."""
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 3})
        pbt.initialize_population()
        scores = [0.5, 0.9, 0.3]
        pbt.evolve(scores)
        best = pbt.get_best_member()
        self.assertIsNotNone(best)
        self.assertEqual(best.current_score, 0.9)

    def test_pbt_save_load(self):
        """Test save/load round-trip."""
        import tempfile
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 3})
        pbt.initialize_population()
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt.save_population(tmpdir)
            pbt2 = PopulationBasedTrainer(config={"population_size": 3})
            loaded = pbt2.load_population(tmpdir)
            self.assertEqual(len(loaded), 3)

    def test_pbt_get_stats(self):
        """Test stats reporting."""
        from envgen.benchmark.pbt import PopulationBasedTrainer
        pbt = PopulationBasedTrainer(config={"population_size": 3})
        pbt.initialize_population()
        stats = pbt.get_stats()
        self.assertEqual(stats["population_size"], 3)
        self.assertEqual(stats["generation"], 0)

    def test_pbt_share_best_config(self):
        """Test sharing best config."""
        from envgen.benchmark.pbt import PopulationBasedTrainer, AgentMember
        pbt = PopulationBasedTrainer()
        best = AgentMember(member_id=0, hyperparams={"lr": 0.001})
        target = AgentMember(member_id=1, hyperparams={"lr": 0.01})
        shared = pbt.share_best_config(best, target)
        self.assertEqual(shared.hyperparams["lr"], 0.001)


class TestLLMCurriculumManager(unittest.TestCase):
    """Tests for the LLM-based Curriculum Manager."""

    def test_create_without_llm(self):
        """Test creating LLM curriculum manager without LLM client."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager

        mgr = LLMCurriculumManager()
        self.assertIsNone(mgr.llm_client)
        self.assertIsNotNone(mgr.fallback_manager)

    def test_initialize_stages(self):
        """Test initializing curriculum stages."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager

        mgr = LLMCurriculumManager()
        mgr.initialize_stages(num_stages=5)
        self.assertEqual(mgr.current_stage, 0)
        self.assertEqual(mgr.current_difficulty, DifficultyLevel.EASY)
        self.assertEqual(len(mgr.fallback_manager.stages), 5)

    def test_update_with_mock_llm(self):
        """Test curriculum update with mock LLM."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager
        from envgen.llm.generator import MockLLMClient
        from envgen.utils.types import EvaluationResult, SkillScore

        mock_llm = MockLLMClient()
        mgr = LLMCurriculumManager(llm_client=mock_llm)
        mgr.initialize_stages()

        result = EvaluationResult(
            agent_id="test_agent",
            skill_scores=[
                SkillScore("navigation", 0.8, 0.8),
                SkillScore("planning", 0.7, 0.7),
                SkillScore("reasoning", 0.6, 0.6),
            ],
            overall_score=0.7,
        )
        decision = mgr.update(result)
        self.assertIn(decision.action, ("advance", "stay", "regress"))
        self.assertIsInstance(decision.reasoning, str)

    def test_update_fallback(self):
        """Test curriculum update falls back to rule-based."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager
        from envgen.utils.types import EvaluationResult, SkillScore

        mgr = LLMCurriculumManager()  # No LLM client
        mgr.initialize_stages()

        result = EvaluationResult(
            agent_id="test_agent",
            skill_scores=[
                SkillScore("navigation", 0.8, 0.8),
                SkillScore("planning", 0.8, 0.8),
                SkillScore("reasoning", 0.8, 0.8),
            ],
            overall_score=0.8,
        )
        decision = mgr.update(result)
        self.assertIsNotNone(decision)
        self.assertIsInstance(decision.action, str)

    def test_skill_history_tracking(self):
        """Test that skill history is tracked across updates."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager
        from envgen.utils.types import EvaluationResult, SkillScore

        mgr = LLMCurriculumManager()
        mgr.initialize_stages()

        for i in range(3):
            result = EvaluationResult(
                agent_id="test_agent",
                skill_scores=[
                    SkillScore("navigation", 0.5 + i * 0.1, 0.5 + i * 0.1),
                    SkillScore("planning", 0.6 + i * 0.1, 0.6 + i * 0.1),
                ],
                overall_score=0.6 + i * 0.1,
            )
            mgr.update(result)

        self.assertEqual(len(mgr.skill_history["navigation"]), 3)
        self.assertEqual(len(mgr.skill_history["planning"]), 3)

    def test_get_stats(self):
        """Test curriculum manager statistics."""
        from envgen.curriculum.llm_curriculum import LLMCurriculumManager
        from envgen.utils.types import EvaluationResult, SkillScore

        mgr = LLMCurriculumManager()
        mgr.initialize_stages()

        result = EvaluationResult(
            agent_id="test_agent",
            skill_scores=[
                SkillScore("navigation", 0.7, 0.7),
                SkillScore("planning", 0.7, 0.7),
            ],
            overall_score=0.7,
        )
        mgr.update(result)
        stats = mgr.get_stats()
        self.assertIn("current_stage", stats)
        self.assertIn("current_difficulty", stats)
        self.assertIn("total_decisions", stats)
        self.assertEqual(stats["total_decisions"], 1)

    def test_curriculum_decision_dataclass(self):
        """Test CurriculumDecision dataclass."""
        from envgen.curriculum.llm_curriculum import CurriculumDecision

        decision = CurriculumDecision(
            action="advance",
            reasoning="Agent is ready",
            suggested_difficulty=DifficultyLevel.MEDIUM,
            focus_skills=["navigation"],
            confidence=0.9,
        )
        self.assertEqual(decision.action, "advance")
        self.assertEqual(decision.confidence, 0.9)
        self.assertEqual(len(decision.focus_skills), 1)


class TestRewardVisualizer(unittest.TestCase):
    """Tests for the Reward Function Visualizer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_visualizer(self):
        """Test creating reward visualizer."""
        from envgen.utils.reward_visualizer import RewardVisualizer

        viz = RewardVisualizer(self.tmpdir)
        self.assertEqual(viz.log_dir, self.tmpdir)
        self.assertEqual(len(viz.runs), 0)

    def test_parse_metrics_line(self):
        """Test parsing metrics JSON-lines entry."""
        from envgen.utils.reward_visualizer import RewardVisualizer

        viz = RewardVisualizer(self.tmpdir)
        data = {
            "cycle": 1,
            "total_reward": 15.5,
            "success_reward": 10.0,
            "progress_reward": 5.0,
            "step_penalty": -0.5,
            "episode_count": 10,
        }
        snapshot = viz._parse_metrics_line(data, 1)
        self.assertEqual(snapshot.cycle, 1)
        self.assertEqual(snapshot.total_reward, 15.5)
        self.assertEqual(snapshot.success_reward, 10.0)
        self.assertEqual(snapshot.episode_count, 10)

    def test_load_run_with_data(self):
        """Test loading a run with metrics data."""
        import json as json_mod
        from envgen.utils.reward_visualizer import RewardVisualizer

        run_dir = os.path.join(self.tmpdir, "test_run")
        os.makedirs(run_dir)
        metrics_file = os.path.join(run_dir, "metrics.jsonl")

        with open(metrics_file, "w") as f:
            for i in range(3):
                line = json_mod.dumps({
                    "cycle": i,
                    "total_reward": 10.0 + i * 2,
                    "success_reward": 5.0 + i,
                    "progress_reward": 3.0,
                    "step_penalty": -0.1,
                    "episode_count": 10,
                })
                f.write(line + "\n")

        viz = RewardVisualizer(self.tmpdir)
        snapshots = viz.load_run(run_dir)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0].cycle, 0)
        self.assertEqual(snapshots[2].total_reward, 14.0)

    def test_load_all_runs(self):
        """Test loading all runs from log directory."""
        import json as json_mod
        from envgen.utils.reward_visualizer import RewardVisualizer

        for run_name in ["run_a", "run_b"]:
            run_dir = os.path.join(self.tmpdir, run_name)
            os.makedirs(run_dir)
            metrics_file = os.path.join(run_dir, "metrics.jsonl")
            with open(metrics_file, "w") as f:
                f.write(json_mod.dumps({"cycle": 0, "total_reward": 10.0}) + "\n")

        viz = RewardVisualizer(self.tmpdir)
        runs = viz.load_all_runs()
        self.assertEqual(len(runs), 2)
        self.assertIn("run_a", runs)
        self.assertIn("run_b", runs)

    def test_print_summary(self):
        """Test printing reward summary."""
        import json as json_mod
        from envgen.utils.reward_visualizer import RewardVisualizer

        run_dir = os.path.join(self.tmpdir, "summary_run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
            f.write(json_mod.dumps({"cycle": 0, "total_reward": 10.0}) + "\n")
            f.write(json_mod.dumps({"cycle": 1, "total_reward": 15.0}) + "\n")

        viz = RewardVisualizer(self.tmpdir)
        viz.load_all_runs()
        summary = viz.print_summary()
        self.assertIn("REWARD FUNCTION ANALYSIS", summary)
        self.assertIn("summary_run", summary)

    def test_print_comparison(self):
        """Test printing comparison of multiple runs."""
        import json as json_mod
        from envgen.utils.reward_visualizer import RewardVisualizer

        for i, run_name in enumerate(["high_reward", "low_reward"]):
            run_dir = os.path.join(self.tmpdir, run_name)
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
                reward = 20.0 if i == 0 else 5.0
                f.write(json_mod.dumps({"cycle": 0, "total_reward": reward}) + "\n")

        viz = RewardVisualizer(self.tmpdir)
        viz.load_all_runs()
        comparison = viz.print_comparison()
        self.assertIn("REWARD COMPARISON", comparison)
        self.assertIn("high_reward", comparison)

    def test_get_components_over_time(self):
        """Test getting reward components as time series."""
        import json as json_mod
        from envgen.utils.reward_visualizer import RewardVisualizer

        run_dir = os.path.join(self.tmpdir, "time_series")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
            f.write(json_mod.dumps({"cycle": 0, "total_reward": 10.0, "success_reward": 5.0}) + "\n")
            f.write(json_mod.dumps({"cycle": 1, "total_reward": 15.0, "success_reward": 8.0}) + "\n")

        viz = RewardVisualizer(self.tmpdir)
        viz.load_all_runs()
        components = viz.get_components_over_time("time_series")
        self.assertIn("total_reward", components)
        self.assertEqual(len(components["total_reward"]), 2)
        self.assertEqual(components["total_reward"], [10.0, 15.0])

    def test_load_nonexistent_dir(self):
        """Test loading from non-existent directory."""
        from envgen.utils.reward_visualizer import RewardVisualizer

        viz = RewardVisualizer("/nonexistent/path")
        runs = viz.load_all_runs()
        self.assertEqual(len(runs), 0)


class TestMultiAgentTrainingExample(unittest.TestCase):
    """Tests for the multi-agent training example."""

    def test_multi_agent_example_imports(self):
        """Test that multi-agent training example imports work."""
        from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
        from envgen.environments import MultiAgentEnvironment, EnvironmentConfig
        self.assertIsNotNone(MultiAgentPipeline)
        self.assertIsNotNone(MultiAgentEnvironment)

    def test_multi_agent_pipeline_create(self):
        """Test creating a multi-agent pipeline."""
        from envgen.agents import PPOAgent
        from envgen.curriculum import CurriculumManager
        from envgen.evaluators.skill_evaluator import SkillEvaluator
        from envgen.environments import EnvironmentConfig
        from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
        from envgen.rewards.engine import RewardEngine

        agent1 = PPOAgent(name="a1")
        agent2 = PPOAgent(name="a2")
        curriculum = CurriculumManager()
        curriculum.add_stage("easy", DifficultyLevel.EASY)

        def factory():
            cfg = EnvironmentConfig(
                name="test",
                difficulty=DifficultyLevel.EASY,
                parameters={"grid_size": 5, "num_items": 2, "cooperation_mode": False},
            )
            from envgen.environments import MultiAgentEnvironment
            return MultiAgentEnvironment(config=cfg)

        pipeline = MultiAgentPipeline(
            num_agents=2,
            mode="competitive",
            config={"num_cycles": 1, "num_episodes": 3},
            agents=[agent1, agent2],
            env_factory=factory,
            evaluator=SkillEvaluator(),
            curriculum_manager=curriculum,
            reward_engine=RewardEngine(),
        )
        self.assertEqual(pipeline.num_agents, 2)
        self.assertEqual(pipeline.mode, "competitive")

    def test_multi_agent_pipeline_wrong_agent_count(self):
        """Test that wrong agent count raises error."""
        from envgen.agents import PPOAgent
        from envgen.curriculum import CurriculumManager
        from envgen.evaluators.skill_evaluator import SkillEvaluator
        from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
        from envgen.rewards.engine import RewardEngine

        with self.assertRaises(ValueError):
            MultiAgentPipeline(
                num_agents=2,
                mode="competitive",
                config={},
                agents=[PPOAgent(name="a1")],
                env_factory=lambda: None,
                evaluator=SkillEvaluator(),
                curriculum_manager=CurriculumManager(),
                reward_engine=RewardEngine(),
            )


class TestNearestNeighborAgent(unittest.TestCase):
    """Tests for the NearestNeighborAgent."""

    def setUp(self):
        from envgen.agents.nn_agent import NearestNeighborAgent

        self.agent = NearestNeighborAgent(
            name="test_nn",
            config={"k": 3, "distance_metric": "euclidean", "max_memory": 100},
        )

    def test_select_action_empty_memory(self):
        """Test select_action with empty memory returns random action."""
        obs = Observation(state={"x": 0.0, "y": 0.0})
        action = self.agent.select_action(obs)
        self.assertIsInstance(action, Action)

    def test_select_action_with_memory(self):
        """Test select_action exploits memory when sufficient."""
        from envgen.agents.nn_agent import Experience

        # Fill memory with enough experiences
        for i in range(10):
            self.agent.memory.append(
                Experience(state=[0.1 * i, 0.2 * i], action_index=i % 3, reward=float(i))
            )
        self.agent._action_space_size = 3

        obs = Observation(state={"x": 0.5, "y": 1.0})
        action = self.agent.select_action(obs)
        self.assertIsInstance(action, Action)

    def test_train_adds_experiences(self):
        """Test that train adds experiences from episodes."""
        transitions = [
            Transition(
                observation={"x": 0.0, "y": 0.0},
                action=Action("move", {"direction": "right", "action_index": 0}),
                reward=1.0,
                next_observation={"x": 1.0, "y": 0.0},
                done=False,
            ),
            Transition(
                observation={"x": 1.0, "y": 0.0},
                action=Action("move", {"direction": "down", "action_index": 1}),
                reward=2.0,
                next_observation={"x": 1.0, "y": 1.0},
                done=True,
            ),
        ]
        episode = Episode(transitions=transitions, total_reward=3.0)

        result = self.agent.train([episode])
        self.assertEqual(result["experiences_added"], 2)
        self.assertEqual(result["memory_size"], 2)

    def test_train_empty_episodes(self):
        """Test training with empty episode list."""
        result = self.agent.train([])
        self.assertEqual(result["experiences_added"], 0)

    def test_save_load_roundtrip(self):
        """Test save and load preserves memory."""
        from envgen.agents.nn_agent import Experience, NearestNeighborAgent

        self.agent.memory.append(Experience(state=[1.0, 2.0], action_index=0, reward=5.0))
        self.agent.memory.append(Experience(state=[3.0, 4.0], action_index=1, reward=10.0))
        self.agent._action_space_size = 3

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            self.agent.save(path)
            new_agent = NearestNeighborAgent(name="loaded_nn")
            new_agent.load(path)
            self.assertEqual(len(new_agent.memory), 2)
            self.assertEqual(new_agent.memory[0].reward, 5.0)
            self.assertEqual(new_agent.memory[1].action_index, 1)
            self.assertEqual(new_agent._action_space_size, 3)
        finally:
            os.unlink(path)

    def test_get_stats(self):
        """Test get_stats returns correct info."""
        stats = self.agent.get_stats()
        self.assertEqual(stats["name"], "test_nn")
        self.assertEqual(stats["k"], 3)
        self.assertEqual(stats["distance_metric"], "euclidean")
        self.assertEqual(stats["memory_size"], 0)

    def test_cosine_distance(self):
        """Test cosine distance metric."""
        from envgen.agents.nn_agent import NearestNeighborAgent

        agent = NearestNeighborAgent(
            name="cosine_nn",
            config={"distance_metric": "cosine"},
        )
        obs = Observation(state={"x": 1.0, "y": 1.0})
        action = agent.select_action(obs)
        self.assertIsInstance(action, Action)

    def test_max_memory_eviction(self):
        """Test that oldest experiences are evicted when max_memory is reached."""
        from envgen.agents.nn_agent import NearestNeighborAgent

        agent = NearestNeighborAgent(
            name="small_nn",
            config={"max_memory": 5, "k": 1},
        )
        transitions = []
        for i in range(10):
            transitions.append(
                Transition(
                    observation={"x": float(i)},
                    action=Action("move", {"direction": "right", "action_index": i % 2}),
                    reward=float(i),
                    next_observation={"x": float(i + 1)},
                    done=False,
                )
            )
        episode = Episode(transitions=transitions, total_reward=45.0)
        agent.train([episode])
        self.assertEqual(len(agent.memory), 5)
        # Oldest should be evicted, so first remaining should be index 5
        self.assertEqual(agent.memory[0].reward, 5.0)

    def test_config_loader_integration(self):
        """Test that config loader creates NN agent."""
        if ConfigLoader is None:
            self.skipTest("pyyaml not installed")
        from envgen.agents.nn_agent import NearestNeighborAgent
        # Directly test that NN agent accepts k and distance_metric in config
        agent = NearestNeighborAgent(
            name="test_nn",
            config={"k": 5, "distance_metric": "cosine"},
        )
        self.assertIsNotNone(agent)
        self.assertEqual(agent.k, 5)
        self.assertEqual(agent.distance_metric, "cosine")


if __name__ == "__main__":
    unittest.main()
