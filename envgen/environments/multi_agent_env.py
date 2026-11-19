"""Multi-agent competitive environment.

This environment implements a multi-agent scenario where two agents operate on a
shared grid, competing or cooperating to collect items. It demonstrates the
framework's ability to handle multi-agent interactions.

Game mechanics:
- Two agents on a shared grid
- Items scattered across the grid
- First agent to collect target count wins (competitive mode)
- In cooperative mode, both agents share rewards

Skills trained:
- navigation: Moving efficiently on the grid
- planning: Strategizing item collection order
- reasoning: Adapting to opponent behavior
- efficiency: Collecting items in fewer steps
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, DifficultyLevel, Observation


class MultiAgentEnvironment(Environment):
    """Multi-agent competitive/cooperative environment on a shared grid.

    Two agents operate on a shared grid, competing or cooperating to collect items.
    Configuration parameters control:
    - Grid size: Width and height of the grid
    - Number of items: Items scattered on the grid
    - Target count: Items needed to win (competitive mode)
    - Cooperation mode: Competitive vs cooperative rewards
    - Visibility range: How far each agent can see

    Actions:
    - move up/down/left/right: Move in cardinal directions
    - collect: Collect item at current position
    - noop: Do nothing

    Observation:
    - own_position: Agent's current position
    - opponent_position: Last known opponent position (may be hidden)
    - visible_items: Items within visibility range
    - own_inventory: Items collected by this agent
    - opponent_inventory: Items collected by opponent (may be hidden)
    """

    # Action definitions
    ACTION_UP = "up"
    ACTION_DOWN = "down"
    ACTION_LEFT = "left"
    ACTION_RIGHT = "right"
    ACTION_COLLECT = "collect"
    ACTION_NOOP = "noop"

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        # Parse config parameters
        params = config.parameters

        # Grid configuration
        difficulty = config.difficulty
        if difficulty == DifficultyLevel.EASY:
            self.grid_width: int = 8
            self.grid_height: int = 8
            self.num_items: int = 5
            self.target_count: int = 3
            self.visibility_range: int = 4
        elif difficulty == DifficultyLevel.MEDIUM:
            self.grid_width = 12
            self.grid_height = 12
            self.num_items = 8
            self.target_count = 5
            self.visibility_range = 3
        elif difficulty == DifficultyLevel.HARD:
            self.grid_width = 16
            self.grid_height = 16
            self.num_items = 12
            self.target_count = 7
            self.visibility_range = 2
        else:  # EXPERT
            self.grid_width = 20
            self.grid_height = 20
            self.num_items = 15
            self.target_count = 10
            self.visibility_range = 2

        # Override with explicit params
        self.grid_width = params.get("grid_width", self.grid_width)
        self.grid_height = params.get("grid_height", self.grid_height)
        self.num_items = params.get("num_items", self.num_items)
        self.target_count = params.get("target_count", self.target_count)
        self.visibility_range = params.get("visibility_range", self.visibility_range)
        self.cooperation_mode: bool = params.get("cooperation_mode", False)

        # State variables
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.opponent_pos: Tuple[int, int] = (0, 0)
        self.items: List[Tuple[int, int]] = []
        self.agent_inventory: int = 0
        self.opponent_inventory: int = 0
        self.opponent_last_known: Optional[Tuple[int, int]] = None
        self.winner: Optional[str] = None

        # Available actions
        self._available_actions: List[Action] = [
            Action("move", {"direction": self.ACTION_UP}),
            Action("move", {"direction": self.ACTION_DOWN}),
            Action("move", {"direction": self.ACTION_LEFT}),
            Action("move", {"direction": self.ACTION_RIGHT}),
            Action("collect", {}),
            Action("noop", {}),
        ]

    def reset(self) -> Observation:
        """Reset the environment with new random positions."""
        self.current_step = 0
        self._done = False
        self.agent_inventory = 0
        self.opponent_inventory = 0
        self.winner = None

        # Place agent at random position
        self.agent_pos = (
            self._rng.randint(0, self.grid_width - 1),
            self._rng.randint(0, self.grid_height - 1),
        )

        # Place opponent at random position (not on agent)
        while True:
            self.opponent_pos = (
                self._rng.randint(0, self.grid_width - 1),
                self._rng.randint(0, self.grid_height - 1),
            )
            if self.opponent_pos != self.agent_pos:
                break

        self.opponent_last_known = self.opponent_pos

        # Scatter items
        self.items = []
        for _ in range(self.num_items):
            while True:
                pos = (
                    self._rng.randint(0, self.grid_width - 1),
                    self._rng.randint(0, self.grid_height - 1),
                )
                if pos not in self.items and pos != self.agent_pos and pos != self.opponent_pos:
                    self.items.append(pos)
                    break

        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action and update the environment."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills_this_step: List[str] = []

        action_type = action.action_type
        params = action.parameters or {}

        # Handle primary agent action
        if action_type == "move":
            reward, info, skills_this_step = self._handle_move(params.get("direction", ""))
        elif action_type == "collect":
            reward, info, skills_this_step = self._handle_collect()
        elif action_type == "noop":
            pass

        # Step penalty for efficiency
        reward -= 0.05

        # Simulate opponent action (simple random policy)
        self._simulate_opponent()

        # Check win conditions
        if self.agent_inventory >= self.target_count:
            self.winner = "agent"
            reward += 10.0
            self._done = True
            info["result"] = "win"
        elif self.opponent_inventory >= self.target_count:
            self.winner = "opponent"
            reward -= 5.0
            self._done = True
            info["result"] = "lose"

        # Check if all items collected
        if not self.items and self.agent_inventory + self.opponent_inventory >= self.num_items:
            if self.cooperation_mode:
                reward += 5.0
            elif self.agent_inventory > self.opponent_inventory:
                reward += 5.0
                self.winner = "agent"
                info["result"] = "win"
            elif self.agent_inventory < self.opponent_inventory:
                reward -= 3.0
                self.winner = "opponent"
                info["result"] = "lose"
            self._done = True

        if self.current_step >= self.max_steps:
            self._done = True
            if self.winner is None:
                if self.agent_inventory > self.opponent_inventory:
                    self.winner = "agent"
                    info["result"] = "win"
                elif self.agent_inventory < self.opponent_inventory:
                    self.winner = "opponent"
                    info["result"] = "lose"
                else:
                    info["result"] = "draw"

        skills_this_step.extend(["navigation", "planning"])
        info["skills_demonstrated"] = skills_this_step
        info["agent_inventory"] = self.agent_inventory
        info["opponent_inventory"] = self.opponent_inventory

        self.current_step += 1

        return self.get_observation(), reward, self._done, info

    def get_observation(self) -> Observation:
        """Return the current observation."""
        # Determine visible items (within visibility range)
        visible_items = [
            (x, y)
            for x, y in self.items
            if abs(x - self.agent_pos[0]) + abs(y - self.agent_pos[1]) <= self.visibility_range
        ]

        # Determine if opponent is visible
        opponent_visible = (
            abs(self.opponent_pos[0] - self.agent_pos[0]) + abs(self.opponent_pos[1] - self.agent_pos[1])
        ) <= self.visibility_range

        # Check if there's an item at current position
        item_at_position = self.agent_pos in self.items

        state = {
            "own_position": list(self.agent_pos),
            "opponent_position": list(self.opponent_pos) if opponent_visible else None,
            "visible_items": visible_items,
            "item_at_position": item_at_position,
            "own_inventory": self.agent_inventory,
            "opponent_inventory": self.opponent_inventory if opponent_visible else None,
            "target_count": self.target_count,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "remaining_items": len(self.items),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

        return Observation(state=state)

    def get_available_actions(self) -> List[Action]:
        """Return the list of valid actions."""
        return self._available_actions.copy()

    def render(self) -> str:
        """Return a text representation of the grid."""
        grid = [["." for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        # Mark items
        for x, y in self.items:
            if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                grid[y][x] = "*"

        # Mark opponent
        ox, oy = self.opponent_pos
        if 0 <= oy < self.grid_height and 0 <= ox < self.grid_width:
            grid[oy][ox] = "O"

        # Mark agent
        ax, ay = self.agent_pos
        if 0 <= ay < self.grid_height and 0 <= ax < self.grid_width:
            grid[ay][ax] = "@"

        lines = []
        for row in grid:
            lines.append(" ".join(row))

        info = (
            f"Agent: {self.agent_inventory}/{self.target_count} | "
            f"Opponent: {self.opponent_inventory}/{self.target_count}"
        )
        lines.append(info)

        return "\n".join(lines)

    def _handle_move(self, direction: str) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle a move action."""
        reward = 0.0
        info = {}
        skills = []

        dx, dy = 0, 0
        if direction == self.ACTION_UP:
            dy = -1
        elif direction == self.ACTION_DOWN:
            dy = 1
        elif direction == self.ACTION_LEFT:
            dx = -1
        elif direction == self.ACTION_RIGHT:
            dx = 1
        else:
            return -0.1, {"error": "invalid direction"}, []

        new_x = max(0, min(self.grid_width - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_height - 1, self.agent_pos[1] + dy))
        self.agent_pos = (new_x, new_y)

        # Collision with opponent
        if self.agent_pos == self.opponent_pos:
            reward -= 0.5
            info["collision"] = True

        skills.append("navigation")
        return reward, info, skills

    def _handle_collect(self) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle a collect action."""
        reward = 0.0
        info = {}
        skills = []

        if self.agent_pos in self.items:
            self.items.remove(self.agent_pos)
            self.agent_inventory += 1
            reward += 2.0
            info["collected"] = True
            skills.extend(["reasoning", "efficiency"])
        else:
            reward -= 0.2
            info["collected"] = False

        return reward, info, skills

    def _simulate_opponent(self) -> None:
        """Simulate opponent behavior using a simple heuristic policy."""
        # Find nearest item to opponent
        if not self.items:
            return

        nearest = None
        nearest_dist = float("inf")
        for item in self.items:
            dist = abs(item[0] - self.opponent_pos[0]) + abs(item[1] - self.opponent_pos[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = item

        if nearest:
            # Move toward nearest item
            dx = 0
            dy = 0
            if nearest[0] > self.opponent_pos[0]:
                dx = 1
            elif nearest[0] < self.opponent_pos[0]:
                dx = -1
            elif nearest[1] > self.opponent_pos[1]:
                dy = 1
            elif nearest[1] < self.opponent_pos[1]:
                dy = -1
            else:
                # At the item, collect it
                self.items.remove(nearest)
                self.opponent_inventory += 1
                return

            new_x = max(0, min(self.grid_width - 1, self.opponent_pos[0] + dx))
            new_y = max(0, min(self.grid_height - 1, self.opponent_pos[1] + dy))
            self.opponent_pos = (new_x, new_y)

            # Check if opponent collected an item
            if self.opponent_pos in self.items:
                self.items.remove(self.opponent_pos)
                self.opponent_inventory += 1
