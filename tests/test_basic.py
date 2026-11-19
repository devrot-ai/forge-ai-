"""Basic tests for the EnvGen-LLM framework."""

import sys
import unittest
from pathlib import Path

# isort: off
sys.path.insert(0, str(Path(__file__).parent.parent))
# isort: on

from envgen.agents.ppo_agent import PPOAgent
from envgen.curriculum.manager import CurriculumManager
from envgen.environments.base import EnvironmentConfig
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.llm.generator import EnvironmentGenerator, MockLLMClient
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import (
    Action,
    DifficultyLevel,
    Episode,
    EvaluationResult,
    Observation,
    SkillScore,
    Transition,
)


class TestTypes(unittest.TestCase):
    """Test core data types."""

    def test_observation(self):
        obs = Observation(state={"x": 1, "y": 2})
        self.assertEqual(obs.state["x"], 1)
        self.assertEqual(obs.state["y"], 2)
        d = obs.to_dict()
        self.assertIn("state", d)

    def test_action(self):
        action = Action(action_type="move", parameters={"direction": "up"})
        self.assertEqual(action.action_type, "move")
        self.assertEqual(action.parameters["direction"], "up")

    def test_episode(self):
        ep = Episode()
        obs1 = Observation(state={"pos": [0, 0]})
        obs2 = Observation(state={"pos": [0, 1]})
        action = Action(action_type="move")
        t = Transition(observation=obs1, action=action, reward=1.0, next_observation=obs2, done=False)
        ep.add_transition(t)
        self.assertEqual(ep.length, 1)
        self.assertEqual(ep.total_reward, 1.0)

    def test_evaluation_result(self):
        result = EvaluationResult(
            agent_id="test",
            skill_scores=[
                SkillScore("navigation", 0.8, 8.0),
                SkillScore("planning", 0.3, 3.0),
                SkillScore("reasoning", 0.6, 6.0),
            ],
            overall_score=0.57,
        )
        self.assertEqual(result.get_skill_score("navigation"), 0.8)
        self.assertIsNone(result.get_skill_score("unknown"))
        weak = result.get_weak_skills(threshold=0.5)
        self.assertEqual(len(weak), 1)
        self.assertEqual(weak[0].skill_name, "planning")

    def test_environment_config(self):
        cfg = EnvironmentConfig(
            name="test_env",
            description="A test environment",
            difficulty=DifficultyLevel.MEDIUM,
            parameters={"grid_size": 5},
            target_skills=["navigation"],
        )
        d = cfg.to_dict()
        self.assertEqual(d["name"], "test_env")
        self.assertEqual(d["difficulty"], "medium")

        restored = EnvironmentConfig.from_dict(d)
        self.assertEqual(restored.name, "test_env")
        self.assertEqual(restored.difficulty, DifficultyLevel.MEDIUM)


class TestPuzzleEnvironment(unittest.TestCase):
    """Test the puzzle environment."""

    def setUp(self):
        self.cfg = EnvironmentConfig(
            name="test_puzzle",
            description="Test puzzle",
            difficulty=DifficultyLevel.EASY,
            parameters={
                "grid_size": 5,
                "num_items": 2,
                "num_obstacles": 1,
                "complexity": 0.2,
                "noise_level": 0.0,
            },
            target_skills=["navigation", "planning"],
            max_steps=50,
            seed=42,
        )
        self.env = PuzzleEnvironment(self.cfg)

    def test_reset(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("position", obs.state)
        self.assertFalse(self.env.done)

    def test_step(self):
        self.env.reset()
        self.env.get_observation()
        actions = self.env.get_available_actions()
        self.assertTrue(len(actions) > 0)

        action = Action(action_type="move", parameters={"direction": "right"})
        next_obs, reward, done, info = self.env.step(action)
        self.assertIsInstance(next_obs, Observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_render(self):
        self.env.reset()
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("@", rendered)  # Agent marker

    def test_collect_item(self):
        self.env.reset()
        # Move to an item position and collect
        for _ in range(10):
            if self.env.is_terminal():
                break
            actions = self.env.get_available_actions()
            for a in actions:
                if a.action_type == "collect":
                    obs, reward, done, info = self.env.step(a)
                    if info.get("collected"):
                        self.assertIn("reasoning", info.get("skills_demonstrated", []))
                        return
            # Random move
            action = self.env.random_action()
            self.env.step(action)

        # If we didn't collect, that's OK — depends on random layout


class TestPPOAgent(unittest.TestCase):
    """Test the PPO agent."""

    def setUp(self):
        self.agent = PPOAgent(
            name="test_agent",
            config={"learning_rate": 0.1, "initial_epsilon": 0.5},
        )

    def test_select_action(self):
        obs = Observation(state={"position": [0, 0], "collected_items": [], "items_remaining": 3, "grid_size": 5})
        actions = [
            Action(action_type="move", parameters={"direction": "up"}),
            Action(action_type="move", parameters={"direction": "right"}),
        ]
        action = self.agent.select_action(obs, actions)
        self.assertIn(action, actions)

    def test_train(self):
        # Create a simple episode
        obs1 = Observation(state={"position": [0, 0], "collected_items": [], "items_remaining": 3, "grid_size": 5})
        obs2 = Observation(state={"position": [0, 1], "collected_items": [], "items_remaining": 3, "grid_size": 5})
        action = Action(action_type="move", parameters={"direction": "right"})
        t = Transition(observation=obs1, action=action, reward=0.5, next_observation=obs2, done=False)
        ep = Episode()
        ep.add_transition(t)
        ep.environment_config = {"parameters": {"grid_size": 5, "num_items": 3}}

        metrics = self.agent.train([ep])
        self.assertIn("loss", metrics)
        self.assertIn("num_updates", metrics)
        self.assertGreater(metrics["num_updates"], 0)

    def test_save_load(self):
        import os
        import tempfile

        # Train a bit
        obs = Observation(state={"position": [0, 0], "collected_items": [], "items_remaining": 3, "grid_size": 5})
        action = Action(action_type="move", parameters={"direction": "right"})
        t = Transition(observation=obs, action=action, reward=0.5, next_observation=obs, done=False)
        ep = Episode()
        ep.add_transition(t)
        self.agent.train([ep])

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            self.agent.save(path)
            new_agent = PPOAgent(name="loaded_agent")
            new_agent.load(path)
            self.assertEqual(len(new_agent.value_function), len(self.agent.value_function))
        finally:
            os.unlink(path)


class TestSkillEvaluator(unittest.TestCase):
    """Test the skill evaluator."""

    def setUp(self):
        self.evaluator = SkillEvaluator()

    def test_skill_names(self):
        names = self.evaluator.get_skill_names()
        self.assertIn("navigation", names)
        self.assertIn("planning", names)
        self.assertIn("reasoning", names)
        self.assertIn("memory", names)
        self.assertIn("efficiency", names)

    def test_empty_episodes(self):
        result = self.evaluator.evaluate([], "test_agent")
        self.assertEqual(result.overall_score, 0.0)
        self.assertEqual(len(result.skill_scores), 0)

    def test_evaluate_episodes(self):
        # Create episodes with some skill demonstrations
        episodes = []
        for _ in range(5):
            obs1 = Observation(state={"position": [0, 0], "collected_items": [], "items_remaining": 3, "grid_size": 5})
            obs2 = Observation(
                state={"position": [0, 1], "collected_items": ["key"], "items_remaining": 2, "grid_size": 5}
            )
            action = Action(action_type="move", parameters={"direction": "right"})
            t = Transition(
                observation=obs1,
                action=action,
                reward=1.0,
                next_observation=obs2,
                done=False,
                info={"moved": True, "skills_demonstrated": ["navigation", "planning"]},
            )
            ep = Episode()
            ep.add_transition(t)
            ep.environment_config = {"parameters": {"grid_size": 5, "num_items": 3}}
            episodes.append(ep)

        result = self.evaluator.evaluate(episodes, "test_agent")
        self.assertGreater(len(result.skill_scores), 0)
        self.assertGreater(result.overall_score, 0)
        self.assertIn("mean_reward", result.episode_stats)


class TestCurriculumManager(unittest.TestCase):
    """Test the curriculum manager."""

    def setUp(self):
        self.curriculum = CurriculumManager()
        self.curriculum.add_stage("easy", DifficultyLevel.EASY, skill_targets=["navigation"])
        self.curriculum.add_stage("medium", DifficultyLevel.MEDIUM, skill_targets=["navigation", "planning"])

    def test_initial_stage(self):
        self.assertEqual(self.curriculum.current_stage, 0)
        stage = self.curriculum.get_current_stage()
        self.assertEqual(stage.name, "easy")

    def test_update(self):
        result = EvaluationResult(
            agent_id="test",
            skill_scores=[
                SkillScore("navigation", 0.8, 8.0),
                SkillScore("planning", 0.7, 7.0),
            ],
            overall_score=0.75,
        )
        record = self.curriculum.update(result)
        self.assertIn("action", record)
        self.assertIn("stage", record)

    def test_progress_report(self):
        result = EvaluationResult(
            agent_id="test",
            skill_scores=[SkillScore("navigation", 0.5, 5.0)],
            overall_score=0.5,
        )
        self.curriculum.update(result)
        report = self.curriculum.get_progress_report()
        self.assertIn("current_stage", report)
        self.assertIn("skill_progress", report)


class TestRewardEngine(unittest.TestCase):
    """Test the reward engine."""

    def test_add_component(self):
        engine = RewardEngine()
        engine.add_component(
            "test",
            1.0,
            lambda t: 1.0 if t.info.get("success") else 0.0,
        )
        self.assertEqual(len(engine.components), 1)

    def test_compute(self):
        engine = RewardEngine()
        engine.add_component(
            "success",
            1.0,
            RewardEngine.sparse_reward("success", 1.0, 0.0),
        )
        engine.add_component(
            "step",
            0.5,
            RewardEngine.step_penalty(-0.01),
        )

        obs = Observation(state={"step": 0})
        t = Transition(
            observation=obs,
            action=Action(action_type="test"),
            reward=0.0,
            next_observation=obs,
            done=False,
            info={"success": True},
        )
        reward = engine.compute(t)
        # success=1.0*1.0 + step=0.5*(-0.01) = 0.995
        self.assertAlmostEqual(reward, 0.995, places=3)

    def test_get_weights(self):
        engine = RewardEngine()
        engine.add_component("a", 1.0, lambda t: 0.0)
        engine.add_component("b", 2.0, lambda t: 0.0)
        weights = engine.get_weights()
        self.assertEqual(weights["a"], 1.0)
        self.assertEqual(weights["b"], 2.0)


class TestMockLLM(unittest.TestCase):
    """Test the mock LLM client."""

    def test_generate(self):
        client = MockLLMClient()
        response = client.generate("Generate environments for navigation and planning")
        self.assertIsInstance(response, str)

    def test_generate_json(self):
        client = MockLLMClient()
        response = client.generate_json("Generate environments for navigation and planning tasks")
        self.assertIn("environments", response)
        self.assertIsInstance(response["environments"], list)

    def test_call_count(self):
        client = MockLLMClient()
        self.assertEqual(client.call_count, 0)
        client.generate("test")
        self.assertEqual(client.call_count, 1)


class TestEnvironmentGenerator(unittest.TestCase):
    """Test the environment generator."""

    def setUp(self):
        self.client = MockLLMClient()
        self.generator = EnvironmentGenerator(self.client)

    def test_generate_initial(self):
        configs = self.generator.generate_initial(
            task_description="Navigate and collect items",
            skills=["navigation", "planning"],
            difficulty=DifficultyLevel.EASY,
            num_environments=3,
        )
        self.assertTrue(len(configs) > 0)
        self.assertIsInstance(configs[0], EnvironmentConfig)

    def test_adapt_environments(self):
        # Generate initial
        configs = self.generator.generate_initial(
            task_description="Navigate and collect items",
            skills=["navigation", "planning"],
            difficulty=DifficultyLevel.EASY,
        )

        # Create evaluation result
        result = EvaluationResult(
            agent_id="test",
            skill_scores=[
                SkillScore("navigation", 0.8, 8.0),
                SkillScore("planning", 0.2, 2.0),
            ],
            overall_score=0.5,
        )

        adapted = self.generator.adapt_environments(
            evaluation_result=result,
            current_configs=configs,
            task_description="Navigate and collect items",
        )
        self.assertTrue(len(adapted) > 0)


if __name__ == "__main__":
    unittest.main()
