# Creating Custom Environments

This guide shows you how to create your own environment types and integrate them with the EnvGen-LLM framework.

## Understanding the Environment Interface

All environments in EnvGen-LLM extend the `Environment` base class. The key methods you need to implement are:

- `reset()` — Reset the environment to its initial state
- `step(action)` — Execute an action and return the result
- `get_observation()` — Return the current observation
- `get_available_actions()` — Return the list of available actions
- `render()` — Visualize the current state (optional)

## Step 1: Create Your Environment Class

Create a new file `envgen/environments/my_env.py`:

```python
"""Custom environment example: Number Guessing Game."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from envgen.environments.base import Environment
from envgen.utils.types import (
    Action,
    EnvironmentConfig,
    Observation,
    Transition,
)


class NumberGuessingEnvironment(Environment):
    """A number guessing game environment.

    The agent tries to guess a secret number within a range.
    The environment provides hints (higher/lower) after each guess.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__(config)
        self.config = config or EnvironmentConfig()

        # Game parameters
        self.min_value = self.config.parameters.get("min_value", 1)
        self.max_value = self.config.parameters.get("max_value", 100)
        self.max_steps = self.config.parameters.get("max_steps", 10)

        # Action space: guess lower, guess higher, guess current midpoint
        self.action_space = 3  # 0=lower, 1=higher, 2=submit midpoint

        # Game state
        self.secret_number: int = 0
        self.current_guess: int = 0
        self.steps_taken: int = 0
        self.game_over: bool = False

    def reset(self) -> Observation:
        """Reset the game with a new secret number."""
        self.secret_number = random.randint(self.min_value, self.max_value)
        self.current_guess = (self.min_value + self.max_value) // 2
        self.steps_taken = 0
        self.game_over = False
        return self.get_observation()

    def step(self, action: Action) -> Transition:
        """Execute a guess action."""
        if self.game_over:
            return self._make_terminal_transition()

        old_obs = self.get_observation()

        # Execute action
        if action.action == "lower":
            self.max_value = self.current_guess - 1
        elif action.action == "higher":
            self.min_value = self.current_guess + 1
        elif action.action == "submit":
            # Submit current guess
            if self.current_guess == self.secret_number:
                action.reward = 10.0
                self.game_over = True
            else:
                action.reward = -1.0
                self.game_over = True

        self.current_guess = (self.min_value + self.max_value) // 2
        self.steps_taken += 1

        # Check for max steps
        if self.steps_taken >= self.max_steps and not self.game_over:
            self.game_over = True
            action.reward = -5.0

        # Step penalty
        action.reward = action.reward - 0.1

        new_obs = self.get_observation()
        return Transition(
            observation=old_obs,
            action=action,
            next_observation=new_obs,
            reward=action.reward,
            done=self.game_over,
        )

    def get_observation(self) -> Observation:
        """Return the current game state."""
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "current_guess": self.current_guess,
            "steps_taken": self.steps_taken,
            "range_size": self.max_value - self.min_value + 1,
        }

    def get_available_actions(self) -> List[Action]:
        """Return available actions."""
        if self.game_over:
            return [Action(action="reset")]

        actions = [
            Action(action="lower"),
            Action(action="higher"),
            Action(action="submit"),
        ]
        return actions

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current game state."""
        output = f"Secret range: [{self.min_value}, {self.max_value}]\n"
        output += f"Current guess: {self.current_guess}\n"
        output += f"Steps taken: {self.steps_taken}/{self.max_steps}\n"

        if self.game_over:
            output += f"Game over! Secret was: {self.secret_number}\n"

        if mode == "human":
            print(output)
        return output
```

## Step 2: Register Your Environment

Update `envgen/environments/__init__.py`:

```python
from envgen.environments.my_env import NumberGuessingEnvironment

__all__ = [
    # ... existing exports ...
    "NumberGuessingEnvironment",
]
```

## Step 3: Add to Config Loader

Update `envgen/config/loader.py`:

```python
def create_env_factory(env_cfg: Dict[str, Any]):
    env_type = env_cfg.get("type", "puzzle")

    if env_type == "my_env":
        from envgen.environments.my_env import NumberGuessingEnvironment
        return lambda: NumberGuessingEnvironment()
    # ... existing cases ...
```

## Step 4: Add to CLI

Update `envgen/cli/train.py`:

```python
parser.add_argument(
    "--env-type",
    choices=["puzzle", "maze", "resource", "text", "cartpole", "multi_agent", "my_env"],
    default=None,
)
```

## Step 5: Test Your Environment

```python
from envgen.environments.my_env import NumberGuessingEnvironment
from envgen.utils.types import Action, EnvironmentConfig

# Create environment
config = EnvironmentConfig(
    name="number_guessing",
    difficulty="easy",
    parameters={"min_value": 1, "max_value": 50, "max_steps": 8},
)
env = NumberGuessingEnvironment(config)

# Reset and play
obs = env.reset()
print(f"Initial observation: {obs}")

actions = env.get_available_actions()
print(f"Available actions: {[a.action for a in actions]}")

# Take a step
transition = env.step(Action(action="lower"))
print(f"Reward: {transition.reward}, Done: {transition.done}")

# Render
env.render()
```

## Design Principles

When creating custom environments, follow these principles:

### 1. Use EnvironmentConfig for Parameters

Make your environment configurable through `EnvironmentConfig.parameters`:

```python
def __init__(self, config: Optional[EnvironmentConfig] = None):
    self.config = config or EnvironmentConfig()
    self.grid_size = self.config.parameters.get("grid_size", 10)
    self.num_items = self.config.parameters.get("num_items", 5)
```

### 2. Support Difficulty Levels

Scale your environment based on difficulty:

```python
def _setup_difficulty(self):
    difficulty = self.config.difficulty
    if difficulty == "easy":
        self.grid_size = 8
    elif difficulty == "medium":
        self.grid_size = 12
    elif difficulty == "hard":
        self.grid_size = 16
    else:  # expert
        self.grid_size = 20
```

### 3. Provide Meaningful Observations

Observations should contain all information needed for decision-making:

```python
def get_observation(self) -> Observation:
    return {
        "position": self.agent_position,
        "items": self.visible_items,
        "distance_to_goal": self._calc_distance(),
        "steps_remaining": self.max_steps - self.steps_taken,
    }
```

### 4. Design Clear Reward Signals

Rewards should guide the agent toward the goal:

```python
# Positive rewards for progress
if item_collected:
    reward = 5.0

# Negative rewards for inefficiency
reward -= 0.1  # Step penalty

# Large reward for success
if goal_reached:
    reward = 10.0
```

## Next Steps

- **[LLM Integration Guide](../guides/llm_integration.md)** — Generate environments with LLMs
- **[API Reference](../api/environments.md)** — Complete environment API documentation
- **[Experiment Tracking Guide](../guides/experiment_tracking.md)** — Track your custom environment experiments
