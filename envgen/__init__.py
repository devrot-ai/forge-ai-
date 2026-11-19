"""
EnvGen-LLM: Self-Improving AI Agents via LLM-Generated Training Environments.
"""

__version__ = "1.0.0"

from envgen.agents.base import Agent
from envgen.curriculum.manager import CurriculumManager
from envgen.environments.base import Environment, EnvironmentConfig
from envgen.evaluators.base import Evaluator
from envgen.pipeline.trainer import TrainingPipeline
from envgen.rewards.engine import RewardEngine

__all__ = [
    "TrainingPipeline",
    "Environment",
    "EnvironmentConfig",
    "Agent",
    "Evaluator",
    "CurriculumManager",
    "RewardEngine",
]
