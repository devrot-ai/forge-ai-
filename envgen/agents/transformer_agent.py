"""Transformer-based agent with self-attention for long-range temporal reasoning.

This agent implements a Transformer encoder with multi-head self-attention
over observation sequences. Unlike feedforward or RNN agents, the Transformer
can attend to any position in the observation history, enabling long-range
dependency detection and complex temporal reasoning.

Requires: pip install torch numpy
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation

logger = logging.getLogger(__name__)


class TransformerAgent(Agent):
    """Transformer-based agent with self-attention for temporal reasoning.

    Uses a Transformer encoder with multi-head self-attention over a sliding
    window of observations. This enables the agent to:
    - Detect long-range temporal patterns
    - Attend to relevant past observations regardless of distance
    - Learn complex sequential dependencies

    The policy network architecture:
    - Observation sequence → positional encoding → Transformer encoder → MLP → action logits
    - Value network: Transformer output → MLP → scalar value

    Maintains a sliding window of observations (configurable max_seq_length).
    """

    def __init__(self, name: str = "transformer", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # torch is imported at module level; ImportError will propagate if missing

        # PPO hyperparameters
        self.learning_rate: float = self.config.get("learning_rate", 1e-3)
        self.discount_factor: float = self.config.get("discount_factor", 0.99)
        self.gae_lambda: float = self.config.get("gae_lambda", 0.95)
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)
        self.ent_coef: float = self.config.get("ent_coef", 0.01)
        self.vf_coef: float = self.config.get("vf_coef", 0.5)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.ppo_epochs: int = self.config.get("ppo_epochs", 3)

        # Transformer architecture
        self.hidden_size: int = self.config.get("hidden_size", 64)
        self.num_heads: int = self.config.get("num_heads", 4)
        self.num_layers: int = self.config.get("num_layers", 2)
        self.max_seq_length: int = self.config.get("max_seq_length", 32)
        self.ffn_hidden_size: int = self.config.get("ffn_hidden_size", 128)

        # Exploration
        self.epsilon: float = self.config.get("initial_epsilon", 0.3)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.98)

        # Observation/action dimensions (set after first observation)
        self.obs_dim: Optional[int] = None
        self.action_dim: Optional[int] = None

        # Networks
        self.positional_encoding = None
        self.transformer_encoder = None
        self.policy_head = None
        self.value_head = None
        self.optimizer = None

        # Observation history (sliding window)
        self.obs_history: List[np.ndarray] = []

        # Training buffer
        self.buffer: List[Dict[str, Any]] = []
        self.max_buffer_size: int = self.config.get("max_buffer_size", 2048)

        # Stats
        self.update_count: int = 0
        self.total_reward_history: List[float] = []

    def _flatten_observation(self, observation: Observation) -> np.ndarray:
        """Flatten observation dict to numpy array for PyTorch input."""
        # Unwrap Observation objects
        if hasattr(observation, 'state'):
            observation = observation.state
        if isinstance(observation, dict):
            values = []
            for key in sorted(observation.keys()):
                val = observation[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, np.ndarray):
                    values.extend(val.flatten().tolist())
                elif isinstance(val, (list, tuple)):
                    values.extend([float(v) for v in val])
                elif isinstance(val, str):
                    values.append(hash(val) % 1000 / 1000.0)
                elif isinstance(val, bool):
                    values.append(1.0 if val else 0.0)
                else:
                    values.append(0.0)
            return np.array(values, dtype=np.float32)
        elif isinstance(observation, (list, tuple)):
            return np.array([float(v) for v in observation], dtype=np.float32)
        elif isinstance(observation, np.ndarray):
            return observation.flatten().astype(np.float32)
        else:
            return np.array([float(observation)], dtype=np.float32)

    def _init_networks(self, obs_dim: int, action_dim: int) -> None:
        """Initialize neural networks once dimensions are known."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Positional encoding for sequence positions
        self.positional_encoding = PositionalEncoding(obs_dim, self.max_seq_length)

        # Ensure num_heads divides obs_dim
        actual_heads = self.num_heads
        while obs_dim % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=obs_dim,
            nhead=actual_heads,
            dim_feedforward=self.ffn_hidden_size,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # Policy head: Transformer output → action logits
        self.policy_head = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, action_dim),
        )

        # Value head: Transformer output → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        # Optimizer
        all_params = (
            list(self.transformer_encoder.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

    def select_action(self, observation: Observation, episode_id: Optional[int] = None) -> Action:
        """Select an action using the Transformer policy network.

        Maintains a sliding window of observations for self-attention.
        Resets history when a new episode starts.
        """
        import torch

        # Flatten and add to history
        obs_array = self._flatten_observation(observation)
        self.obs_history.append(obs_array)

        # Trim history to max sequence length
        if len(self.obs_history) > self.max_seq_length:
            self.obs_history = self.obs_history[-self.max_seq_length:]

        # Initialize networks on first observation
        if self.obs_dim is None:
            inferred_action_dim = getattr(self, "_inferred_action_dim", 6)
            self._init_networks(len(obs_array), inferred_action_dim)

        # Convert observation history to tensor with sequence dimension
        seq_length = len(self.obs_history)
        obs_seq = torch.FloatTensor(self.obs_history).unsqueeze(0)  # (1, seq_len, obs_dim)

        # Add positional encoding
        pos_enc = self.positional_encoding(obs_seq)

        # Forward pass through Transformer encoder
        transformer_out = self.transformer_encoder(pos_enc)

        # Use the last output token for action selection
        last_output = transformer_out[:, -1, :]  # (1, obs_dim)

        # Get action logits from policy head
        action_logits = self.policy_head(last_output)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            import torch.nn.functional as F
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return Action(action_type=str(action_idx), parameters={"action_idx": action_idx})

    def reset_episode(self) -> None:
        """Reset observation history for a new episode."""
        self.obs_history = []

    def store_transition(self, obs: Observation, action: Action, reward: float,
                         next_obs: Observation, done: bool) -> None:
        """Store a transition in the training buffer."""
        transition = {
            "obs": self._flatten_observation(obs),
            "action": action.parameters.get("action_idx", 0) if action.parameters else 0,
            "reward": reward,
            "next_obs": self._flatten_observation(next_obs),
            "done": done,
        }
        self.buffer.append(transition)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]

    def train(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Train the Transformer agent using PPO-style updates.

        Uses Generalized Advantage Estimation (GAE) for advantage computation
        and clipped surrogate objective for stable updates.
        """
        import torch
        import torch.nn.functional as F

        if not episodes or self.policy_head is None:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        # Collect all transitions from episodes
        all_transitions = []
        for episode in episodes:
            for transition in episode.transitions:
                all_transitions.append({
                    "obs": self._flatten_observation(transition.observation),
                    "action": transition.action.parameters.get("action_idx", 0) if transition.action.parameters else 0,
                    "reward": transition.reward,
                    "done": transition.done,
                })

        if not all_transitions:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        # Compute returns and advantages using GAE
        rewards = [t["reward"] for t in all_transitions]
        dones = [t["done"] for t in all_transitions]
        observations = [t["obs"] for t in all_transitions]
        actions = [t["action"] for t in all_transitions]

        # Compute GAE advantages
        advantages = self._compute_gae_advantages(rewards, dones)

        # Compute returns
        returns = []
        for j in range(len(rewards)):
            ret = sum(rewards[j + i] * (self.discount_factor ** i) for i in range(len(rewards) - j))
            returns.append(ret)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.LongTensor(actions)
        advantage_tensor = torch.FloatTensor(advantages)
        return_tensor = torch.FloatTensor(returns)

        total_loss = 0.0

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Process each observation as a single-step sequence
            # (for training, we use individual observations)
            obs_seq = obs_tensor.unsqueeze(1)  # (batch, 1, obs_dim)

            # Add positional encoding
            pos_enc = self.positional_encoding(obs_seq)

            # Forward pass through Transformer
            transformer_out = self.transformer_encoder(pos_enc)
            last_output = transformer_out[:, -1, :]  # (batch, obs_dim)

            # Get action logits
            action_logits = self.policy_head(last_output)
            action_probs = F.log_softmax(action_logits, dim=-1)

            # Policy loss (clipped surrogate)
            old_log_prob = action_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_prob = dist.log_prob(action_tensor)
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantage_tensor, clipped_ratio * advantage_tensor).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            policy_loss = policy_loss - self.ent_coef * entropy

            # Value loss
            values = self.value_head(last_output).squeeze(1)
            value_loss = F.mse_loss(values, return_tensor)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.transformer_encoder.parameters())
                + list(self.policy_head.parameters())
                + list(self.value_head.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()

        self.update_count += 1

        # Decay exploration more after training
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)

        return {
            "loss": total_loss / self.ppo_epochs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def _compute_gae_advantages(self, rewards: List[float], dones: List[int]) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        gae = 0.0
        advantages = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.discount_factor * (1 - dones[i]) * 0.0 - 0.0
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        # Normalize advantages
        if len(advantages) > 1:
            mean = sum(advantages) / len(advantages)
            std = (sum((a - mean) ** 2 for a in advantages) / len(advantages)) ** 0.5
            if std > 1e-8:
                advantages = [(a - mean) / std for a in advantages]

        return advantages

    def save(self, path: str) -> None:
        """Save agent state including Transformer weights."""
        import torch

        state = {
            "config": self.config,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "ffn_hidden_size": self.ffn_hidden_size,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "total_reward_history": self.total_reward_history[-1000:],
        }

        if self.transformer_encoder is not None:
            state["transformer_encoder_state_dict"] = self.transformer_encoder.state_dict()
            state["policy_head_state_dict"] = self.policy_head.state_dict()
            state["value_head_state_dict"] = self.value_head.state_dict()
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        save_path = os.path.join(path, f"{self.name}_transformer.pt") if os.path.isdir(path) else path
        torch.save(state, save_path)
        logger.info(f"Saved Transformer agent to {save_path}")

    def load(self, path: str) -> None:
        """Load agent state from checkpoint."""
        import torch

        load_path = path if os.path.isfile(path) else os.path.join(path, f"{self.name}_transformer.pt")
        state = torch.load(load_path)

        self.hidden_size = state.get("hidden_size", self.hidden_size)
        self.num_heads = state.get("num_heads", self.num_heads)
        self.num_layers = state.get("num_layers", self.num_layers)
        self.max_seq_length = state.get("max_seq_length", self.max_seq_length)
        self.ffn_hidden_size = state.get("ffn_hidden_size", self.ffn_hidden_size)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.update_count = state.get("update_count", 0)
        self.total_reward_history = state.get("total_reward_history", [])

        if "transformer_encoder_state_dict" in state:
            obs_dim = state.get("obs_dim", self.obs_dim or 64)
            action_dim = state.get("action_dim", self.action_dim or 6)
            self._init_networks(obs_dim, action_dim)
            self.transformer_encoder.load_state_dict(state["transformer_encoder_state_dict"])
            self.policy_head.load_state_dict(state["policy_head_state_dict"])
            self.value_head.load_state_dict(state["value_head_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Loaded Transformer agent from {load_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            "name": self.name,
            "type": "transformer",
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "buffer_size": len(self.buffer),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "history_length": len(self.obs_history),
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer sequences."""

    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]
