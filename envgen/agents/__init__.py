"""Agent implementations (RL and LLM-based)."""

from envgen.agents.base import Agent
from envgen.agents.grpo_agent import GRPOAgent
from envgen.agents.ppo_agent import PPOAgent

try:
    from envgen.agents.ppo_torch import PPOAgentTorch
except ImportError:
    # PyTorch agent is optional; requires 'torch' and 'numpy' packages
    PPOAgentTorch = None  # type: ignore

try:
    from envgen.agents.grpo_torch import GRPOAgentTorch
except ImportError:
    # PyTorch GRPO agent is optional; requires 'torch' and 'numpy' packages
    GRPOAgentTorch = None  # type: ignore

try:
    from envgen.agents.sb3_agent import SB3Agent
except ImportError:
    # SB3 agent is optional; requires 'stable-baselines3', 'gymnasium', and 'numpy'
    SB3Agent = None  # type: ignore

try:
    from envgen.agents.rnn_agent import RNNAgent
except ImportError:
    # RNN agent is optional; requires 'torch' and 'numpy' packages
    RNNAgent = None  # type: ignore

try:
    from envgen.agents.transformer_agent import TransformerAgent
except ImportError:
    # Transformer agent is optional; requires 'torch' and 'numpy' packages
    TransformerAgent = None  # type: ignore

from envgen.agents.replay_buffer import ReplayBuffer

try:
    from envgen.agents.nn_agent import NearestNeighborAgent
except ImportError:
    # NN agent is optional; requires 'numpy' package
    NearestNeighborAgent = None  # type: ignore

__all__ = [
    "Agent",
    "PPOAgent",
    "PPOAgentTorch",
    "GRPOAgent",
    "GRPOAgentTorch",
    "SB3Agent",
    "RNNAgent",
    "TransformerAgent",
    "ReplayBuffer",
    "NearestNeighborAgent",
]
