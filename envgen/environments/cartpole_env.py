"""CartPole continuous control environment.

This environment implements the classic CartPole balancing problem where the agent
must apply force to a cart to keep a pole balanced upright. It demonstrates
continuous control tasks in the framework.

Physics:
- Cart moves horizontally on a track
- Pole is attached to cart via frictionless hinge
- Goal: keep pole upright as long as possible

Skills trained:
- reasoning: Understanding physics dynamics
- planning: Anticipating pole movement
- efficiency: Maintaining balance with minimal force
- memory: Learning from past states
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, DifficultyLevel, Observation


class CartPoleEnvironment(Environment):
    """CartPole balancing environment with configurable physics.

    The agent controls a cart that can move left/right to balance a pole.
    Configuration parameters control:
    - Gravity: Force pulling the pole down
    - Pole length/mass: Affects dynamics
    - Cart mass: Affects responsiveness
    - Force magnitude: How much force can be applied
    - Angle threshold: When the pole is considered "fallen"

    Actions:
    - move left: Apply force to the left
    - move right: Apply force to the right
    - stay: Apply no force

    Observation:
    - cart_position: Horizontal position of cart
    - cart_velocity: Horizontal velocity of cart
    - pole_angle: Angle of pole from vertical (radians)
    - pole_angular_velocity: Angular velocity of pole
    """

    # Action definitions
    ACTION_LEFT = "left"
    ACTION_RIGHT = "right"
    ACTION_STAY = "stay"

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        # Parse config parameters
        params = config.parameters

        # Physics parameters (scaled by difficulty)
        difficulty = config.difficulty
        if difficulty == DifficultyLevel.EASY:
            self.gravity: float = 8.0
            self.pole_length: float = 0.2
            self.pole_mass: float = 0.05
            self.cart_mass: float = 1.0
            self.force_magnitude: float = 5.0
            self.angle_threshold: float = math.radians(30)
            self.cart_bound: float = 4.0
            self.dt: float = 0.02
        elif difficulty == DifficultyLevel.MEDIUM:
            self.gravity = 9.81
            self.pole_length = 0.3
            self.pole_mass = 0.1
            self.cart_mass = 0.8
            self.force_magnitude = 8.0
            self.angle_threshold = math.radians(20)
            self.cart_bound = 3.5
            self.dt = 0.02
        elif difficulty == DifficultyLevel.HARD:
            self.gravity = 9.81
            self.pole_length = 0.5
            self.pole_mass = 0.15
            self.cart_mass = 0.5
            self.force_magnitude = 10.0
            self.angle_threshold = math.radians(15)
            self.cart_bound = 3.0
            self.dt = 0.02
        else:  # EXPERT
            self.gravity = 12.0
            self.pole_length = 0.6
            self.pole_mass = 0.2
            self.cart_mass = 0.3
            self.force_magnitude = 12.0
            self.angle_threshold = math.radians(10)
            self.cart_bound = 2.5
            self.dt = 0.02

        # Override with explicit params if provided
        self.gravity = params.get("gravity", self.gravity)
        self.pole_length = params.get("pole_length", self.pole_length)
        self.pole_mass = params.get("pole_mass", self.pole_mass)
        self.cart_mass = params.get("cart_mass", self.cart_mass)
        self.force_magnitude = params.get("force_magnitude", self.force_magnitude)
        self.angle_threshold = params.get("angle_threshold", self.angle_threshold)
        self.cart_bound = params.get("cart_bound", self.cart_bound)
        self.dt = params.get("dt", self.dt)

        # Total mass
        self.total_mass = self.cart_mass + self.pole_mass

        # State variables
        self.cart_position: float = 0.0
        self.cart_velocity: float = 0.0
        self.pole_angle: float = 0.0
        self.pole_angular_velocity: float = 0.0

        # Episode tracking
        self.steps_survived: int = 0
        self._available_actions: List[Action] = [
            Action("move", {"direction": self.ACTION_LEFT}),
            Action("move", {"direction": self.ACTION_RIGHT}),
            Action("move", {"direction": self.ACTION_STAY}),
            Action("noop", {}),
        ]

    def reset(self) -> Observation:
        """Reset the environment with small random initial conditions."""
        self.current_step = 0
        self._done = False
        self.steps_survived = 0

        # Small random initial conditions
        self.cart_position = self._rng.uniform(-0.05, 0.05)
        self.cart_velocity = self._rng.uniform(-0.05, 0.05)
        self.pole_angle = self._rng.uniform(-0.05, 0.05)
        self.pole_angular_velocity = self._rng.uniform(-0.05, 0.05)

        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action and update physics."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills_this_step: List[str] = []

        # Determine force
        force = 0.0
        action_type = action.action_type
        params = action.parameters or {}

        if action_type == "move":
            direction = params.get("direction", self.ACTION_STAY)
            if direction == self.ACTION_LEFT:
                force = -self.force_magnitude
            elif direction == self.ACTION_RIGHT:
                force = self.force_magnitude
            else:
                force = 0.0
        elif action_type == "noop":
            force = 0.0

        # Update physics using Euler integration
        self._update_physics(force)
        self.steps_survived += 1

        # Calculate reward
        # Alive reward
        reward += 1.0

        # Angle penalty (encourage keeping pole upright)
        angle_penalty = abs(self.pole_angle) / self.angle_threshold
        reward -= angle_penalty * 0.1

        # Velocity penalty (encourage smooth movement)
        vel_penalty = abs(self.cart_velocity) * 0.01
        reward -= vel_penalty

        # Position penalty (encourage staying near center)
        pos_penalty = abs(self.cart_position) / self.cart_bound
        reward -= pos_penalty * 0.05

        # Check termination conditions
        angle_fallen = abs(self.pole_angle) >= self.angle_threshold
        cart_out_of_bounds = abs(self.cart_position) >= self.cart_bound

        if angle_fallen:
            reward -= 10.0
            info["termination_reason"] = "pole_fallen"
            self._done = True
        elif cart_out_of_bounds:
            reward -= 5.0
            info["termination_reason"] = "cart_out_of_bounds"
            self._done = True

        if self.current_step >= self.max_steps:
            self._done = True
            info["termination_reason"] = "max_steps"

        # Efficiency bonus for surviving longer
        if self.steps_survived > self.max_steps * 0.8:
            reward += 5.0
            skills_this_step.append("efficiency")

        skills_this_step.extend(["reasoning", "planning"])
        info["skills_demonstrated"] = skills_this_step
        info["steps_survived"] = self.steps_survived

        self.current_step += 1

        return self.get_observation(), reward, self._done, info

    def get_observation(self) -> Observation:
        """Return the current observation."""
        state = {
            "cart_position": self.cart_position,
            "cart_velocity": self.cart_velocity,
            "pole_angle": self.pole_angle,
            "pole_angular_velocity": self.pole_angular_velocity,
            "steps_survived": self.steps_survived,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

        return Observation(state=state)

    def get_available_actions(self) -> List[Action]:
        """Return the list of valid actions."""
        return self._available_actions.copy()

    def render(self) -> str:
        """Return a text representation of the current state."""
        # Simple ASCII rendering
        width = 40
        center = width // 2
        cart_pos = int(self.cart_position / self.cart_bound * center) + center
        cart_pos = max(2, min(width - 3, cart_pos))

        # Pole position at top
        pole_top = int(math.sin(self.pole_angle) * 10)

        lines = []
        # Pole
        pole_row = " " * (cart_pos + pole_top - 2) + "/" if pole_top >= 0 else " " * (cart_pos + pole_top) + "/"
        lines.append(pole_row)
        # Cart
        cart_row = " " * cart_pos + "[===]"
        lines.append(cart_row)
        # Track
        lines.append("-" * width)

        info = f"Angle: {self.pole_angle:.2f} rad | Steps: {self.steps_survived}"
        lines.append(info)

        return "\n".join(lines)

    def _update_physics(self, force: float) -> None:
        """Update the physics state using Euler integration.

        Based on the classic CartPole dynamics from Barto, Sutton, & Anderson (1983).
        """
        cos_theta = math.cos(self.pole_angle)
        sin_theta = math.sin(self.pole_angle)

        # Intermediate calculation for acceleration
        temp = (
            force + self.pole_mass * self.pole_length * (self.pole_angular_velocity**2) * sin_theta
        ) / self.total_mass

        # Angular acceleration
        pole_angle_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * cos_theta**2 / self.total_mass)
        )

        # Cart acceleration
        cart_acc = temp - self.pole_mass * self.pole_length * pole_angle_acc * cos_theta / self.total_mass

        # Update state using Euler integration
        self.cart_velocity += cart_acc * self.dt
        self.cart_position += self.cart_velocity * self.dt
        self.pole_angular_velocity += pole_angle_acc * self.dt
        self.pole_angle += self.pole_angular_velocity * self.dt
