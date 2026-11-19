"""PPO-style agent with value-based learning for demonstration.

This agent implements a simplified PPO-like learning algorithm that:
1. Uses a value function to estimate state values
2. Learns action preferences through experience
3. Improves its policy over training cycles

This is a demonstration implementation — for production use, integrate
with a proper RL library like Stable Baselines3 or Ray RLlib.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

from envgen.agents.base import Agent
from envgen.utils.types import Action, Episode, Observation


class PPOAgent(Agent):
    """Simplified PPO-style agent for demonstration.

    This agent learns through experience by:
    1. Maintaining a value function for states
    2. Tracking action-value estimates
    3. Using exploration-exploitation balance (epsilon-greedy with decay)
    4. Updating policies based on episode returns

    The agent demonstrates measurable improvement over training cycles,
    making it suitable for the self-improving pipeline.
    """

    def __init__(self, name: str = "ppo_agent", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

        # Learning parameters
        self.learning_rate: float = self.config.get("learning_rate", 0.1)
        self.discount_factor: float = self.config.get("discount_factor", 0.95)
        self.epsilon: float = self.config.get("initial_epsilon", 0.3)
        self.epsilon_min: float = self.config.get("epsilon_min", 0.05)
        self.epsilon_decay: float = self.config.get("epsilon_decay", 0.95)

        # Value function: state_hash -> value
        self.value_function: Dict[str, float] = {}

        # Action preferences: state_hash -> {action_key -> preference}
        self.action_preferences: Dict[str, Dict[str, float]] = {}

        # Experience buffer
        self.experience: List[Episode] = []
        self.max_experience: int = self.config.get("max_experience", 100)

        # Learning stats
        self.update_count: int = 0
        self.total_reward_history: List[float] = []

    def select_action(self, observation: Observation, available_actions: List[Action]) -> Action:
        """Select an action using epsilon-greedy policy."""
        state_key = self._hash_state(observation.state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Exploitation: choose action with highest preference
        prefs = self.action_preferences.get(state_key, {})
        if not prefs:
            return random.choice(available_actions)

        # Score available actions
        best_action = None
        best_score = float("-inf")

        for action in available_actions:
            action_key = self._hash_action(action)
            score = prefs.get(action_key, 0.0)

            # Add value function bonus
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return random.choice(available_actions)

        return best_action

    def train(self, episodes: List[Episode]) -> Dict[str, float]:
        """Train the agent on a batch of episodes.

        Uses temporal difference learning to update value function
        and action preferences.

        Args:
            episodes: List of completed episodes.

        Returns:
            Training metrics.
        """
        if not episodes:
            return {"loss": 0.0, "num_updates": 0}

        # Store episodes in experience buffer
        self.experience.extend(episodes)
        if len(self.experience) > self.max_experience:
            # Keep most recent and highest-reward episodes
            self.experience.sort(key=lambda e: e.total_reward, reverse=True)
            self.experience = self.experience[: self.max_experience]

        num_updates = 0
        total_loss = 0.0

        # Train on each episode
        for episode in episodes:
            if not episode.transitions:
                continue

            # Compute returns using Monte Carlo
            returns = self._compute_returns(episode)

            # Update value function and action preferences
            for i, transition in enumerate(episode.transitions):
                state_key = self._hash_state(transition.observation.state)
                action_key = self._hash_action(transition.action)

                # TD target
                target = returns[i]

                # Current value estimate
                current_value = self.value_function.get(state_key, 0.0)

                # TD error
                td_error = target - current_value

                # Update value function
                new_value = current_value + self.learning_rate * td_error
                self.value_function[state_key] = new_value

                # Update action preference
                if state_key not in self.action_preferences:
                    self.action_preferences[state_key] = {}

                pref = self.action_preferences[state_key].get(action_key, 0.0)
                # Boost preference for actions with positive TD error
                pref_update = self.learning_rate * td_error * 0.5
                self.action_preferences[state_key][action_key] = pref + pref_update

                num_updates += 1
                total_loss += td_error**2

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Record stats
        avg_reward = sum(e.total_reward for e in episodes) / len(episodes)
        self.total_reward_history.append(avg_reward)
        self.update_count += num_updates

        metrics = {
            "loss": total_loss / max(1, num_updates),
            "num_updates": num_updates,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "num_states": len(self.value_function),
            "reward_trend": self._compute_reward_trend(),
        }

        return metrics

    def _compute_returns(self, episode: Episode) -> List[float]:
        """Compute discounted returns for each step in an episode."""
        returns = []
        discounted_return = 0.0

        # Backward pass
        for i in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[i]
            discounted_return = transition.reward + self.discount_factor * discounted_return
            returns.insert(0, discounted_return)

        return returns

    def _compute_reward_trend(self) -> float:
        """Compute the trend in recent rewards."""
        if len(self.total_reward_history) < 2:
            return 0.0

        recent = self.total_reward_history[-5:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create a hashable key from state."""
        # Simplified state representation
        key_parts = [
            str(state.get("position", [0, 0])),
            str(state.get("collected_items", [])),
            str(state.get("items_remaining", 0)),
            str(state.get("grid_size", 5)),
        ]
        return "|".join(key_parts)

    def _hash_action(self, action: Action) -> str:
        """Create a hashable key from action."""
        return f"{action.action_type}:{json.dumps(action.parameters, sort_keys=True)}"

    def save(self, path: str) -> None:
        """Save agent state to file."""
        data = {
            "name": self.name,
            "agent_id": self.agent_id,
            "config": self.config,
            "epsilon": self.epsilon,
            "value_function": self.value_function,
            "action_preferences": self.action_preferences,
            "update_count": self.update_count,
            "total_reward_history": self.total_reward_history,
            "total_episodes_trained": self.total_episodes_trained,
            "total_steps_trained": self.total_steps_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load agent state from file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.epsilon = data.get("epsilon", self.epsilon)
        self.value_function = data.get("value_function", {})
        self.action_preferences = data.get("action_preferences", {})
        self.update_count = data.get("update_count", 0)
        self.total_reward_history = data.get("total_reward_history", [])
        self.total_episodes_trained = data.get("total_episodes_trained", 0)
        self.total_steps_trained = data.get("total_steps_trained", 0)

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        base_stats = super().get_stats()
        base_stats.update(
            {
                "epsilon": self.epsilon,
                "num_states_learned": len(self.value_function),
                "update_count": self.update_count,
                "reward_trend": self._compute_reward_trend(),
            }
        )
        return base_stats
