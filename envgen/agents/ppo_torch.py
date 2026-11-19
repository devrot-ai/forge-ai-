"""PyTorch-based PPO agent with neural network policy.

This agent implements Proximal Policy Optimization using PyTorch neural
networks for both the policy (action selection) and value function
(state value estimation). It supports the EnvGen-LLM agent interface
and can be dropped into the training pipeline.

Requires: pip install torch numpy
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation

logger = logging.getLogger(__name__)


class PPOAgentTorch(Agent):
    """PPO agent with PyTorch neural networks.

    Uses two MLP networks:
    - Policy network: maps flattened observations to action probabilities
    - Value network: maps flattened observations to state values

    Training uses the clipped surrogate objective from PPO with
    Generalized Advantage Estimation (GAE) for advantage computation.
    """

    def __init__(self, name: str = "ppo_torch", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        try:
            global torch
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' package is required for PPOAgentTorch. " "Install it with: pip install torch"
            )

        # PPO hyperparameters
        self.learning_rate: float = self.config.get("learning_rate", 3e-4)
        self.discount_factor: float = self.config.get("discount_factor", 0.99)
        self.gae_lambda: float = self.config.get("gae_lambda", 0.95)
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)
        self.ent_coef: float = self.config.get("ent_coef", 0.01)
        self.vf_coef: float = self.config.get("vf_coef", 0.5)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.ppo_epochs: int = self.config.get("ppo_epochs", 4)

        # Exploration (for action selection during interaction)
        self.epsilon: float = self.config.get("initial_epsilon", 0.3)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.98)

        # Network architecture
        self.hidden_size: int = self.config.get("hidden_size", 128)
        self.num_layers: int = self.config.get("num_layers", 2)

        # Automatic Mixed Precision
        self.use_amp: bool = self.config.get("use_amp", False)
        self.scaler: Optional[Any] = None
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else None)
            except (AttributeError, Exception):
                # Fallback for older PyTorch versions
                self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # Observation/action dimensions (set after first observation)
        self.obs_dim: Optional[int] = None
        self.action_dim: Optional[int] = None

        # Networks
        self.policy_net = None
        self.value_net = None
        self.optimizer = None

        # Training buffer
        self.buffer: List[Dict[str, Any]] = []
        self.max_buffer_size: int = self.config.get("max_buffer_size", 2048)

        # Stats
        self.update_count: int = 0
        self.total_reward_history: List[float] = []

    def _init_networks(self, obs_dim: int, action_dim: int) -> None:
        """Initialize neural networks once dimensions are known."""
        import torch
        import torch.nn as nn

        # Build policy network: obs -> hidden layers -> action logits
        policy_layers = [nn.Linear(obs_dim, self.hidden_size)]
        for _ in range(self.num_layers - 1):
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        policy_layers.append(nn.Linear(self.hidden_size, action_dim))
        self.policy_net = nn.Sequential(*policy_layers)

        # Build value network: obs -> hidden layers -> value
        value_layers = [nn.Linear(obs_dim, self.hidden_size)]
        for _ in range(self.num_layers - 1):
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        value_layers.append(nn.Linear(self.hidden_size, 1))
        self.value_net = nn.Sequential(*value_layers)

        # Optimizer
        all_params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

        logger.info(
            f"PPOAgentTorch networks initialized: obs_dim={obs_dim}, "
            f"action_dim={action_dim}, hidden={self.hidden_size}"
        )

    def _flatten_observation(self, observation: Observation) -> np.ndarray:
        """Flatten an observation dict into a fixed-size numpy array.

        Extracts numeric features from the observation state:
        - Agent position (x, y)
        - Goal position (x, y)
        - Number of items remaining
        - Number of collected items
        - Grid size
        - Distance to goal
        - Visited cell count
        """
        state = observation.state

        # Extract features
        agent_pos = state.get("agent_pos", [0, 0])
        goal_pos = state.get("goal_pos", [0, 0])
        grid_size = state.get("grid_size", 5)

        # Normalize positions to [0, 1]
        max_pos = max(grid_size, 1)
        agent_x = agent_pos[0] / max_pos if len(agent_pos) > 0 else 0.0
        agent_y = agent_pos[1] / max_pos if len(agent_pos) > 1 else 0.0
        goal_x = goal_pos[0] / max_pos if len(goal_pos) > 0 else 0.0
        goal_y = goal_pos[1] / max_pos if len(goal_pos) > 1 else 0.0

        # Item features
        items_remaining = state.get("items_remaining", 0)
        items_collected = state.get("items_collected", 0)
        max_items = max(state.get("num_items", 3), 1)

        # Distance to goal (Manhattan)
        dist_to_goal = abs(agent_x - goal_x) + abs(agent_y - goal_y)

        # Visited cells
        visited = state.get("visited", 0)
        visited_ratio = visited / (grid_size * grid_size) if grid_size > 0 else 0.0

        # Step progress
        current_step = state.get("current_step", 0)
        max_steps = max(state.get("max_steps", 50), 1)
        step_ratio = current_step / max_steps

        features = np.array(
            [
                agent_x,
                agent_y,
                goal_x,
                goal_y,
                items_remaining / max_items,
                items_collected / max_items,
                dist_to_goal,
                visited_ratio,
                step_ratio,
                1.0 / max(grid_size, 1),  # Grid size (inverted for normalization)
            ],
            dtype=np.float32,
        )

        return features

    def _get_obs_dim(self) -> int:
        """Return the observation feature dimension."""
        return 10  # Fixed feature vector size

    def _encode_action(self, action: Action, available_actions: List[Action]) -> int:
        """Encode an action as an index into available actions."""
        action_type = action.action_type
        action_params = json.dumps(action.parameters, sort_keys=True) if action.parameters else ""

        for i, avail in enumerate(available_actions):
            avail_type = avail.action_type
            avail_params = json.dumps(avail.parameters, sort_keys=True) if avail.parameters else ""
            if avail_type == action_type and avail_params == action_params:
                return i
        return 0  # Default to first action if not found

    def _decode_action(self, action_idx: int, available_actions: List[Action]) -> Action:
        """Decode an action index back to an Action object."""
        idx = min(max(action_idx, 0), len(available_actions) - 1)
        return available_actions[idx]

    def select_action(self, observation: Observation, available_actions: List[Action]) -> Action:
        """Select an action using the policy network with epsilon-greedy exploration."""
        import torch

        # Initialize networks on first call
        if self.policy_net is None:
            self.obs_dim = self._get_obs_dim()
            self.action_dim = len(available_actions)
            self._init_networks(self.obs_dim, self.action_dim)

        # Handle variable action space: if action count changed, rebuild
        if self.action_dim != len(available_actions):
            self.action_dim = len(available_actions)
            self._init_networks(self.obs_dim, self.action_dim)

        # Flatten observation
        obs_vec = self._flatten_observation(observation)
        obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Get action probabilities from policy
        with torch.no_grad():
            logits = self.policy_net(obs_tensor)
            probs = torch.softmax(logits, dim=-1)

        # Sample from the distribution
        action_idx = torch.multinomial(probs, num_samples=1).item()
        return self._decode_action(action_idx, available_actions)

    def train(self, episodes: List[Episode]) -> Dict[str, float]:
        """Train the agent using PPO with GAE advantage estimation.

        Args:
            episodes: List of completed episodes.

        Returns:
            Training metrics including policy loss, value loss, and entropy.
        """
        import torch

        if not episodes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Collect transitions from episodes
        transitions = []
        for episode in episodes:
            for transition in episode.transitions:
                transitions.append(transition)

        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Initialize networks if not yet done
        if self.policy_net is None:
            self.obs_dim = self._get_obs_dim()
            default_actions = self._reconstruct_actions(transitions[0])
            self.action_dim = len(default_actions)
            self._init_networks(self.obs_dim, self.action_dim)

        # Store in buffer
        self.buffer.extend(
            [
                {
                    "obs": self._flatten_observation(t.observation),
                    "action": self._encode_action(t.action, self._reconstruct_actions(t)),
                    "reward": t.reward,
                    "done": t.done,
                }
                for t in transitions
            ]
        )

        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size :]

        # Train if we have enough data
        if len(self.buffer) < 32:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Convert buffer to tensors
        obs_array = np.array([b["obs"] for b in self.buffer], dtype=np.float32)
        action_array = np.array([b["action"] for b in self.buffer], dtype=np.int64)
        reward_array = np.array([b["reward"] for b in self.buffer], dtype=np.float32)
        done_array = np.array([b["done"] for b in self.buffer], dtype=np.float32)

        obs_tensor = torch.FloatTensor(obs_array)
        action_tensor = torch.LongTensor(action_array)
        reward_tensor = torch.FloatTensor(reward_array)
        done_tensor = torch.FloatTensor(done_array)

        # Compute GAE advantages
        with torch.no_grad():
            values = self.value_net(obs_tensor).squeeze(-1)

        # Bootstrap value for last state
        last_value = 0.0
        advantages = torch.zeros_like(reward_tensor)
        gae = 0.0

        for t in reversed(range(len(reward_tensor))):
            if t == len(reward_tensor) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = reward_tensor[t] + self.discount_factor * next_value * (1 - done_tensor[t]) - values[t]
            gae = delta + self.discount_factor * self.gae_lambda * (1 - done_tensor[t]) * gae
            advantages[t] = gae

        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.ppo_epochs):
            # Current policy probabilities (with AMP if enabled)
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else None, enabled=torch.cuda.is_available()):
                    logits = self.policy_net(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
            else:
                logits = self.policy_net(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)

            # Probability of taken actions
            action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            action_probs = probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

            # Old policy ratio
            with torch.no_grad():
                old_probs = torch.exp(action_log_probs.detach())

            ratio = torch.exp(action_log_probs - action_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (with AMP if enabled)
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else None, enabled=torch.cuda.is_available()):
                    current_values = self.value_net(obs_tensor).squeeze(-1)
            else:
                current_values = self.value_net(obs_tensor).squeeze(-1)
            returns = advantages + values.detach()
            value_loss = torch.nn.functional.mse_loss(current_values, returns)

            # Entropy bonus for exploration
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            # Combined loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            # Optimize (with AMP scaler if enabled)
            self.optimizer.zero_grad()
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update stats
        self.update_count += 1
        for episode in episodes:
            self.total_reward_history.append(episode.total_reward)
            self.total_episodes_trained += 1
            self.total_steps_trained += episode.length

        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
            "num_updates": self.update_count,
            "epsilon": self.epsilon,
        }

    def _reconstruct_actions(self, transition: Any) -> List[Action]:
        """Reconstruct a list of available actions for encoding.

        Since we don't have the exact available actions at training time,
        we use a default set of actions based on the environment type.
        """
        # Default action set for puzzle environment
        return [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("collect", {}),
            Action("check_goal", {}),
        ]

    def save(self, path: str) -> None:
        """Save the agent's state including neural network weights."""
        import torch

        save_data = {
            "name": self.name,
            "config": self.config,
            "agent_id": self.agent_id,
            "total_episodes_trained": self.total_episodes_trained,
            "total_steps_trained": self.total_steps_trained,
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "total_reward_history": self.total_reward_history[-100:],  # Last 100
        }

        # Save network state dicts
        if self.policy_net is not None:
            save_data["policy_state_dict"] = self.policy_net.state_dict()
        if self.value_net is not None:
            save_data["value_state_dict"] = self.value_net.state_dict()

        # Save to path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Save network weights separately (for torch.load compatibility)
        weights_path = path + ".pt"
        torch.save(save_data, weights_path)

        # Save metadata as JSON
        meta_path = path + ".json"
        meta_save = {k: v for k, v in save_data.items() if k not in ("policy_state_dict", "value_state_dict")}
        with open(meta_path, "w") as f:
            json.dump(meta_save, f, indent=2, default=str)

        logger.info(f"PPOAgentTorch saved to {path}")

    def load(self, path: str) -> None:
        """Load the agent's state including neural network weights."""
        import torch

        weights_path = path + ".pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        save_data = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Restore config
        self.name = save_data.get("name", self.name)
        self.config = save_data.get("config", self.config)
        self.total_episodes_trained = save_data.get("total_episodes_trained", 0)
        self.total_steps_trained = save_data.get("total_steps_trained", 0)
        self.update_count = save_data.get("update_count", 0)
        self.epsilon = save_data.get("epsilon", self.epsilon)
        self.obs_dim = save_data.get("obs_dim")
        self.action_dim = save_data.get("action_dim")
        self.total_reward_history = save_data.get("total_reward_history", [])

        # Restore networks
        if self.obs_dim and self.action_dim:
            self._init_networks(self.obs_dim, self.action_dim)
            if "policy_state_dict" in save_data:
                self.policy_net.load_state_dict(save_data["policy_state_dict"])
            if "value_state_dict" in save_data:
                self.value_net.load_state_dict(save_data["value_state_dict"])

        logger.info(f"PPOAgentTorch loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "update_count": self.update_count,
                "epsilon": self.epsilon,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "buffer_size": len(self.buffer),
            }
        )
        if self.total_reward_history:
            recent = self.total_reward_history[-10:]
            stats["avg_recent_reward"] = float(np.mean(recent))
        return stats
