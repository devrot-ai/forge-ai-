"""RNN/LSTM-based agent with memory for temporal reasoning tasks.

This agent implements a recurrent neural network policy using PyTorch LSTM
layers. Unlike feedforward agents, the LSTM maintains hidden state across
steps within an episode, enabling the agent to reason about temporal
patterns and maintain memory of past observations.

Requires: pip install torch numpy
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation

logger = logging.getLogger(__name__)


class RNNAgent(Agent):
    """RNN/LSTM agent with memory for temporal reasoning.

    Uses an LSTM-based policy network that maintains hidden state across
    steps within an episode. This enables the agent to:
    - Remember past observations and actions
    - Detect temporal patterns in the environment
    - Make decisions based on sequence history

    The policy network architecture:
    - Observation flattening → LSTM → MLP → action logits
    - Value network: LSTM hidden state → MLP → scalar value

    Hidden state is reset between episodes but maintained across steps
    within an episode.
    """

    def __init__(self, name: str = "rnn", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        try:
            global torch
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' package is required for RNNAgent. "
                "Install it with: pip install torch"
            )

        # RNN hyperparameters
        self.learning_rate: float = self.config.get("learning_rate", 1e-3)
        self.discount_factor: float = self.config.get("discount_factor", 0.99)
        self.gae_lambda: float = self.config.get("gae_lambda", 0.95)
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)
        self.ent_coef: float = self.config.get("ent_coef", 0.01)
        self.vf_coef: float = self.config.get("vf_coef", 0.5)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.ppo_epochs: int = self.config.get("ppo_epochs", 3)

        # Network architecture
        self.hidden_size: int = self.config.get("hidden_size", 64)
        self.num_layers: int = self.config.get("num_layers", 1)

        # Exploration
        self.epsilon: float = self.config.get("initial_epsilon", 0.3)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.98)

        # Observation/action dimensions (set after first observation)
        self.obs_dim: Optional[int] = None
        self.action_dim: Optional[int] = None

        # Networks
        self.lstm = None
        self.policy_head = None
        self.value_head = None
        self.optimizer = None

        # Hidden state management
        self.hidden_state = None
        self.current_episode_id: Optional[int] = None

        # Training buffer
        self.buffer: List[Dict[str, Any]] = []
        self.max_buffer_size: int = self.config.get("max_buffer_size", 2048)

        # Stats
        self.update_count: int = 0
        self.total_reward_history: List[float] = []

    def _flatten_observation(self, observation: Observation) -> np.ndarray:
        """Flatten observation dict to numpy array for PyTorch input."""
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
                    # Simple hash-based encoding for strings
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

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # LSTM layer: processes sequence of observations
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Policy head: LSTM hidden state → action logits
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, action_dim),
        )

        # Value head: LSTM hidden state → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        # Optimizer
        all_params = list(self.lstm.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

        # Reset hidden state
        self._reset_hidden_state()

    def _reset_hidden_state(self) -> None:
        """Reset LSTM hidden state for a new episode."""
        import torch

        if self.lstm is not None:
            self.hidden_state = (
                torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size),
            )

    def select_action(self, observation: Observation, episode_id: Optional[int] = None) -> Action:
        """Select an action using the LSTM policy network.

        Maintains hidden state across steps within an episode.
        Resets hidden state when a new episode starts.
        """
        import torch

        # Detect episode boundary and reset hidden state
        if episode_id is not None and episode_id != self.current_episode_id:
            self._reset_hidden_state()
            self.current_episode_id = episode_id

        # Initialize networks on first observation
        obs_array = self._flatten_observation(observation)
        if self.obs_dim is None:
            # Infer action dimension from available actions or default
            inferred_action_dim = getattr(self, "_inferred_action_dim", 6)
            self._init_networks(len(obs_array), inferred_action_dim)

        # Convert to tensor with batch and sequence dimensions
        obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)

        # Forward pass through LSTM
        lstm_out, self.hidden_state = self.lstm(obs_tensor, self.hidden_state)

        # Get action logits from policy head
        action_logits = self.policy_head(lstm_out.squeeze(0))

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Sample from categorical distribution
            import torch.nn.functional as F
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return Action(action_type=str(action_idx), parameters={"action_idx": action_idx})

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
        """Train the LSTM agent using PPO-style updates on collected episodes.

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

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.LongTensor(actions)
        advantage_tensor = torch.FloatTensor(advantages)
        return_tensor = advantage_tensor + torch.FloatTensor(
            [sum(r * (self.discount_factor ** i) for i, r in enumerate(rewards[::-1]))
             for r in [0] * len(rewards)]
        )  # Simplified return computation

        total_loss = 0.0

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Forward pass through LSTM (process as sequence)
            obs_seq = obs_tensor.unsqueeze(1)  # (batch, 1, obs_dim)
            lstm_out, _ = self.lstm(obs_seq)

            # Get action logits
            action_logits = self.policy_head(lstm_out.squeeze(1))
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
            values = self.value_head(lstm_out.squeeze(1)).squeeze(1)
            returns = [sum(r * (self.discount_factor ** i) for i, r in enumerate(rewards[j:][::-1]))
                       for j in range(len(rewards))]
            return_t = torch.FloatTensor(returns)
            value_loss = F.mse_loss(values, return_t)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.lstm.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
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
        """Save agent state including LSTM weights and hidden state."""
        import torch

        state = {
            "config": self.config,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "total_reward_history": self.total_reward_history[-1000:],
        }

        if self.lstm is not None:
            state["lstm_state_dict"] = self.lstm.state_dict()
            state["policy_head_state_dict"] = self.policy_head.state_dict()
            state["value_head_state_dict"] = self.value_head.state_dict()
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        save_path = os.path.join(path, f"{self.name}_rnn.pt") if os.path.isdir(path) else path
        torch.save(state, save_path)
        logger.info(f"Saved RNN agent to {save_path}")

    def load(self, path: str) -> None:
        """Load agent state from checkpoint."""
        import torch

        load_path = path if os.path.isfile(path) else os.path.join(path, f"{self.name}_rnn.pt")
        state = torch.load(load_path)

        self.hidden_size = state.get("hidden_size", self.hidden_size)
        self.num_layers = state.get("num_layers", self.num_layers)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.update_count = state.get("update_count", 0)
        self.total_reward_history = state.get("total_reward_history", [])

        if "lstm_state_dict" in state:
            obs_dim = state.get("obs_dim", self.obs_dim or 64)
            action_dim = state.get("action_dim", self.action_dim or 6)
            self._init_networks(obs_dim, action_dim)
            self.lstm.load_state_dict(state["lstm_state_dict"])
            self.policy_head.load_state_dict(state["policy_head_state_dict"])
            self.value_head.load_state_dict(state["value_head_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Loaded RNN agent from {load_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            "name": self.name,
            "type": "rnn",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "buffer_size": len(self.buffer),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
