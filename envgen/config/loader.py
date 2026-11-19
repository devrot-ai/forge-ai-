"""Configuration loading from YAML files."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and parse YAML configuration files.

    Reads YAML config files and constructs pipeline components
    (agent, environment, LLM, curriculum, reward) from the config sections.
    """

    def __init__(self):
        try:
            global yaml
            import yaml
        except ImportError:
            raise ImportError(
                "The 'pyyaml' package is required for ConfigLoader. " "Install it with: pip install pyyaml"
            )

    def load(self, path: str) -> Dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            path: Path to the YAML file.

        Returns:
            Dictionary with configuration sections.
        """
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return config or {}

    def get_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task configuration.

        Returns:
            Dict with 'description' and 'skills'.
        """
        task = config.get("task", {})
        return {
            "description": task.get("description", "Default task"),
            "skills": task.get("skills", ["navigation", "planning", "reasoning", "memory", "efficiency"]),
        }

    def get_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training configuration.

        Returns:
            Dict with training parameters.
        """
        training = config.get("training", {})
        return {
            "max_cycles": training.get("max_cycles", 10),
            "episodes_per_cycle": training.get("episodes_per_cycle", 50),
            "eval_episodes": training.get("eval_episodes", 20),
            "initial_difficulty": training.get("initial_difficulty", "easy"),
            "num_initial_envs": training.get("num_initial_envs", 5),
            "save_interval": training.get("save_interval", 5),
            "log_dir": training.get("log_dir", ""),
            "use_mlflow": training.get("use_mlflow", False),
            "mlflow_tracking_uri": training.get("mlflow_tracking_uri", ""),
        }

    def get_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent configuration.

        Returns:
            Dict with agent type and parameters.
        """
        agent = config.get("agent", {})
        return {
            "type": agent.get("type", "ppo"),
            "learning_rate": agent.get("learning_rate", 0.1),
            "discount_factor": agent.get("discount_factor", 0.95),
            "initial_epsilon": agent.get("initial_epsilon", 0.3),
            "epsilon_min": agent.get("epsilon_min", 0.05),
            "epsilon_decay": agent.get("epsilon_decay", 0.95),
            "max_experience": agent.get("max_experience", 100),
        }

    def get_curriculum_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract curriculum configuration.

        Returns:
            Dict with curriculum parameters.
        """
        curriculum = config.get("curriculum", {})
        return {
            "advance_threshold": curriculum.get("advance_threshold", 0.7),
            "regression_threshold": curriculum.get("regression_threshold", 0.4),
            "min_improvement": curriculum.get("min_improvement", 0.05),
        }

    def get_reward_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reward configuration.

        Returns:
            Dict with reward components.
        """
        reward = config.get("reward", {})
        return {
            "clip_range": reward.get("clip_range", [-10.0, 10.0]),
            "components": reward.get("components", []),
        }

    def get_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environment configuration.

        Returns:
            Dict with environment type and default parameters.
        """
        env = config.get("environment", {})
        return {
            "type": env.get("type", "puzzle"),
            "default_grid_size": env.get("default_grid_size", 5),
            "default_num_items": env.get("default_num_items", 3),
            "default_num_obstacles": env.get("default_num_obstacles", 2),
        }

    def get_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract LLM configuration.

        Returns:
            Dict with LLM client type and parameters.
        """
        llm = config.get("llm", {})
        return {
            "client": llm.get("client", "mock"),
            "model": llm.get("model", "gpt-4o-mini"),
        }

    def create_agent(self, config: Dict[str, Any]):
        """Create an agent instance from config.

        Args:
            config: Full YAML config dict.

        Returns:
            An Agent instance.
        """
        agent_cfg = self.get_agent_config(config)
        agent_type = agent_cfg["type"]

        if agent_type == "ppo":
            from envgen.agents.ppo_agent import PPOAgent

            return PPOAgent(
                name="ppo_agent",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "ppo_torch":
            from envgen.agents.ppo_torch import PPOAgentTorch

            return PPOAgentTorch(
                name="ppo_torch",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "grpo":
            from envgen.agents.grpo_agent import GRPOAgent

            return GRPOAgent(
                name="grpo_agent",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "grpo_torch":
            from envgen.agents.grpo_torch import GRPOAgentTorch

            return GRPOAgentTorch(
                name="grpo_torch",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type in ("sb3_ppo", "sb3_sac", "sb3_a2c", "sb3_dqn"):
            from envgen.agents.sb3_agent import SB3Agent

            algo = agent_type.split("_")[-1]  # Extract 'ppo', 'sac', 'a2c', 'dqn'
            return SB3Agent(
                algorithm=algo,
                name=agent_type,
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "rnn":
            from envgen.agents.rnn_agent import RNNAgent

            return RNNAgent(
                name="rnn_agent",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "transformer":
            from envgen.agents.transformer_agent import TransformerAgent

            return TransformerAgent(
                name="transformer_agent",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        elif agent_type == "nn":
            from envgen.agents.nn_agent import NearestNeighborAgent

            return NearestNeighborAgent(
                name="nn_agent",
                config={k: v for k, v in agent_cfg.items() if k != "type"},
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def create_llm_client(self, config: Dict[str, Any]):
        """Create an LLM client from config.

        Args:
            config: Full YAML config dict.

        Returns:
            An LLMClient instance.
        """
        llm_cfg = self.get_llm_config(config)
        client_type = llm_cfg["client"]

        if client_type == "mock":
            from envgen.llm.generator import MockLLMClient

            return MockLLMClient()
        elif client_type == "openai":
            from envgen.llm.openai_client import OpenAILLMClient

            return OpenAILLMClient(model=llm_cfg.get("model", "gpt-4o-mini"))
        elif client_type == "ollama":
            from envgen.llm.ollama_client import OllamaLLMClient

            return OllamaLLMClient(llm_cfg)
        elif client_type == "anthropic":
            from envgen.llm.anthropic_client import AnthropicLLMClient

            return AnthropicLLMClient(llm_cfg)
        else:
            raise ValueError(f"Unknown LLM client type: {client_type}")

    def create_environment_factory(self, config: Dict[str, Any]):
        """Create an environment factory function from config.

        Args:
            config: Full YAML config dict.

        Returns:
            A callable that creates environments from EnvironmentConfig.
        """
        env_cfg = self.get_environment_config(config)
        env_type = env_cfg["type"]

        if env_type == "puzzle":
            from envgen.environments.puzzle_env import PuzzleEnvironment

            def factory(cfg):
                return PuzzleEnvironment(cfg)

            return factory
        elif env_type == "maze":
            from envgen.environments.maze_env import MazeEnvironment

            def factory(cfg):
                return MazeEnvironment(cfg)

            return factory
        elif env_type == "resource":
            from envgen.environments.resource_env import ResourceEnvironment

            def factory(cfg):
                return ResourceEnvironment(cfg)

            return factory
        elif env_type == "text":
            from envgen.environments.text_env import TextTaskEnvironment

            def factory(cfg):
                return TextTaskEnvironment(cfg)

            return factory
        elif env_type == "cartpole":
            from envgen.environments.cartpole_env import CartPoleEnvironment

            def factory(cfg):
                return CartPoleEnvironment(cfg)

            return factory
        elif env_type == "multi_agent":
            from envgen.environments.multi_agent_env import MultiAgentEnvironment

            def factory(cfg):
                return MultiAgentEnvironment(cfg)

            return factory
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
