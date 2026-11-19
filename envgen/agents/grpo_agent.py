"""GRPO agent — Group Relative Policy Optimization.

This agent implements GRPO, an alternative to PPO that generates multiple
rollouts per state (a group), computes relative advantages within each group,
and updates the policy based on these relative rewards. This approach can
provide more stable learning signals by comparing actions within the same
context rather than against absolute baselines.

Reference-inspired design: Group-based relative policy optimization where
multiple trajectories from the same state are compared to compute advantages.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation

logger = logging.getLogger(__name__)


class GRPOAgent(Agent):
    """Group Relative Policy Optimization agent.

    For each state, the agent generates multiple rollouts (a group),
    computes relative advantages within the group, and updates the policy
    to favor actions that performed better than their group peers.

    This is a table-based implementation that can be swapped into the
    training pipeline. For neural network support, see PPOAgentTorch.
    """

    def __init__(self, name: str = "grpo_agent", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        # GRPO parameters
        self.group_size: int = self.config.get("group_size", 4)
        self.learning_rate: float = self.config.get("learning_rate", 0.1)
        self.discount_factor: float = self.config.get("discount_factor", 0.95)
        self.temperature: float = self.config.get("temperature", 1.0)
        self.advantage_normalize: bool = self.config.get("advantage_normalize", True)
        self.clip_epsilon: float = self.config.get("clip_epsilon", 0.2)

        # Exploration
        self.epsilon: float = self.config.get("initial_epsilon", 0.3)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.95)

        # Policy: state_hash -> {action_key -> logit}
        self.policy_logits: Dict[str, Dict[str, float]] = {}

        # Value function: state_hash -> value
        self.value_function: Dict[str, float] = {}

        # Experience buffer
        self.experience: List[Episode] = []
        self.max_experience: int = self.config.get("max_experience", 100)

        # Stats
        self.update_count: int = 0
        self.total_reward_history: List[float] = []

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create a hashable key from the state."""
        # Simplified hashing for grid-based states
        key_parts = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, (list, tuple)):
                key_parts.append(f"{k}:{tuple(v)}")
            elif isinstance(v, dict):
                key_parts.append(f"{k}:{json.dumps(v, sort_keys=True)}")
            else:
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts[:8])  # Limit to first 8 keys

    def _hash_action(self, action: Action) -> str:
        """Create a hashable key from the action."""
        params_str = json.dumps(action.parameters, sort_keys=True) if action.parameters else ""
        return f"{action.action_type}:{params_str}"

    def select_action(self, observation: Observation, available_actions: List[Action]) -> Action:
        """Select an action using softmax policy with temperature scaling."""
        state_key = self._hash_state(observation.state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Get logits for available actions
        logits = self.policy_logits.get(state_key, {})
        if not logits:
            return random.choice(available_actions)

        # Compute softmax with temperature
        action_scores = []
        for action in available_actions:
            action_key = self._hash_action(action)
            score = logits.get(action_key, 0.0) / max(self.temperature, 0.01)
            action_scores.append((action, score))

        # Softmax
        max_score = max(s for _, s in action_scores)
        exp_scores = [math.exp(s - max_score) for _, s in action_scores]
        total = sum(exp_scores)

        if total == 0:
            return random.choice(available_actions)

        probs = [e / total for e in exp_scores]

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        for (action, _), prob in zip(action_scores, probs):
            cumulative += prob
            if r <= cumulative:
                return action

        return action_scores[-1][0]

    def train(self, episodes: List[Episode]) -> Dict[str, float]:
        """Train using GRPO: group-based relative advantage estimation.

        For each unique state in the episodes, groups of transitions are
        formed. Within each group, advantages are computed relative to the
        group mean, and the policy is updated to favor higher-advantage actions.

        Args:
            episodes: List of completed episodes.

        Returns:
            Training metrics.
        """
        if not episodes:
            return {"loss": 0.0, "num_updates": 0, "avg_advantage": 0.0}

        # Store episodes
        self.experience.extend(episodes)
        if len(self.experience) > self.max_experience:
            self.experience = self.experience[-self.max_experience :]

        # Collect all transitions grouped by state
        state_groups: Dict[str, List[tuple]] = {}
        for episode in episodes:
            for transition in episode.transitions:
                state_key = self._hash_state(transition.observation.state)
                action_key = self._hash_action(transition.action)
                next_state_key = self._hash_state(transition.next_observation.state)

                # Compute TD target
                next_value = self.value_function.get(next_state_key, 0.0)
                td_target = transition.reward + self.discount_factor * next_value * (1 - transition.done)

                if state_key not in state_groups:
                    state_groups[state_key] = []
                state_groups[state_key].append((action_key, td_target, transition.reward, transition.done))

        # GRPO update: for each state, form groups and compute relative advantages
        total_loss = 0.0
        total_advantage = 0.0
        num_updates = 0

        for state_key, transitions in state_groups.items():
            # Form groups of size self.group_size
            for i in range(0, len(transitions), self.group_size):
                group = transitions[i : i + self.group_size]
                if not group:
                    continue

                # Compute group mean return
                group_returns = [td for _, td, _, _ in group]
                group_mean = sum(group_returns) / len(group_returns)
                group_std = (sum((r - group_mean) ** 2 for r in group_returns) / len(group_returns)) ** 0.5

                # Compute relative advantages
                for action_key, td_target, reward, done in group:
                    advantage = td_target - group_mean
                    if self.advantage_normalize and group_std > 1e-8:
                        advantage /= group_std

                    # Update policy logits
                    if state_key not in self.policy_logits:
                        self.policy_logits[state_key] = {}

                    current_logit = self.policy_logits[state_key].get(action_key, 0.0)

                    # GRPO update: increase logits for positive advantage actions
                    # Use clipped update similar to PPO
                    old_logit = current_logit
                    new_logit = current_logit + self.learning_rate * advantage

                    # Clip the update magnitude
                    logit_change = new_logit - old_logit
                    max_change = self.clip_epsilon * max(abs(old_logit), 1.0)
                    logit_change = max(-max_change, min(max_change, logit_change))
                    new_logit = old_logit + logit_change

                    self.policy_logits[state_key][action_key] = new_logit

                    # Update value function
                    if state_key not in self.value_function:
                        self.value_function[state_key] = 0.0
                    self.value_function[state_key] = (
                        self.value_function[state_key] * (1 - self.learning_rate) + td_target * self.learning_rate
                    )

                    total_loss += logit_change**2
                    total_advantage += advantage
                    num_updates += 1

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update stats
        self.update_count += 1
        for episode in episodes:
            self.total_reward_history.append(episode.total_reward)
            self.total_episodes_trained += 1
            self.total_steps_trained += episode.length

        avg_loss = total_loss / max(num_updates, 1)
        avg_adv = total_advantage / max(num_updates, 1)

        return {
            "loss": avg_loss,
            "num_updates": num_updates,
            "avg_advantage": avg_adv,
            "epsilon": self.epsilon,
            "group_size": self.group_size,
        }

    def save(self, path: str) -> None:
        """Save the agent's state to disk."""
        save_data = {
            "name": self.name,
            "config": self.config,
            "agent_id": self.agent_id,
            "total_episodes_trained": self.total_episodes_trained,
            "total_steps_trained": self.total_steps_trained,
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "policy_logits": self.policy_logits,
            "value_function": self.value_function,
            "total_reward_history": self.total_reward_history[-100:],
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path + ".json", "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"GRPOAgent saved to {path}")

    def load(self, path: str) -> None:
        """Load the agent's state from disk."""
        load_path = path + ".json"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        with open(load_path) as f:
            save_data = json.load(f)

        self.name = save_data.get("name", self.name)
        self.config = save_data.get("config", self.config)
        self.total_episodes_trained = save_data.get("total_episodes_trained", 0)
        self.total_steps_trained = save_data.get("total_steps_trained", 0)
        self.update_count = save_data.get("update_count", 0)
        self.epsilon = save_data.get("epsilon", self.epsilon)
        self.policy_logits = save_data.get("policy_logits", {})
        self.value_function = save_data.get("value_function", {})
        self.total_reward_history = save_data.get("total_reward_history", [])

        logger.info(f"GRPOAgent loaded from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "update_count": self.update_count,
                "epsilon": self.epsilon,
                "group_size": self.group_size,
                "num_states": len(self.policy_logits),
                "buffer_size": len(self.experience),
            }
        )
        if self.total_reward_history:
            recent = self.total_reward_history[-10:]
            stats["avg_recent_reward"] = sum(recent) / len(recent)
        return stats
