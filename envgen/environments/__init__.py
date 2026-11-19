"""Environment definitions and generators."""

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.environments.cartpole_env import CartPoleEnvironment
from envgen.environments.maze_env import MazeEnvironment
from envgen.environments.multi_agent_env import MultiAgentEnvironment
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.environments.resource_env import ResourceEnvironment
from envgen.environments.text_env import TextTaskEnvironment

try:
    from envgen.environments.gymnasium_wrapper import GymnasiumEnvironment
except ImportError:
    # Gymnasium wrapper is optional; requires 'gymnasium' and 'numpy' packages
    GymnasiumEnvironment = None  # type: ignore

try:
    from envgen.environments.parallel_env import ParallelEnvironment
except (ImportError, OSError):
    # Parallel environment is optional; may fail in some environments
    ParallelEnvironment = None  # type: ignore


__all__ = [
    "Environment",
    "EnvironmentConfig",
    "PuzzleEnvironment",
    "MazeEnvironment",
    "ResourceEnvironment",
    "TextTaskEnvironment",
    "CartPoleEnvironment",
    "MultiAgentEnvironment",
    "GymnasiumEnvironment",
    "ParallelEnvironment",
]
