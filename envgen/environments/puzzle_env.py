"""Puzzle environment — a configurable grid-based puzzle for demonstration.

This environment implements a grid-world puzzle where the agent must navigate
to collect items and reach a goal. The difficulty and parameters are controlled
by the EnvironmentConfig, making it suitable for LLM-generated variations.

Skills trained:
- navigation: Moving efficiently through the grid
- planning: Finding optimal paths to collect items
- reasoning: Understanding item dependencies and goal conditions
- memory: Remembering item locations and visited cells
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, Observation


class PuzzleEnvironment(Environment):
    """Grid-based puzzle environment with configurable parameters.

    The agent navigates a grid to collect items and reach a goal.
    Configuration parameters control:
    - Grid size
    - Number and types of items
    - Obstacles
    - Item dependencies (some items require others)
    - Goal conditions
    """

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        # Parse config parameters
        params = config.parameters
        self.grid_size: int = params.get("grid_size", 5)
        self.num_items: int = params.get("num_items", 3)
        self.num_obstacles: int = params.get("num_obstacles", 2)
        self.complexity: float = params.get("complexity", 0.3)
        self.noise_level: float = params.get("noise_level", 0.1)
        self.skill_focus: str = params.get("skill_focus", "general_task")
        self.scaffolding: bool = params.get("scaffolding", False)
        self.hint_frequency: float = params.get("hint_frequency", 0.0)

        # Grid state
        self.grid: List[List[str]] = []
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.items: Dict[Tuple[int, int], str] = {}
        self.obstacles: set[Tuple[int, int]] = set()
        self.collected_items: List[str] = []
        self.visited: set[Tuple[int, int]] = set()

        # Skill tracking
        self.skills_demonstrated: set[str] = set()

    def reset(self) -> Observation:
        """Reset the environment to a new random state."""
        self.current_step = 0
        self._done = False
        self.collected_items = []
        self.visited = set()
        self.skills_demonstrated = set()

        # Generate grid
        self._generate_grid()

        # Place agent at start
        self.agent_pos = (0, 0)
        self.visited.add(self.agent_pos)

        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action in the environment."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills_this_step: List[str] = []

        action_type = action.action_type
        params = action.parameters

        if action_type == "move":
            reward, info, skills_this_step = self._handle_move(params.get("direction", "stay"))
        elif action_type == "collect":
            reward, info, skills_this_step = self._handle_collect()
        elif action_type == "check_goal":
            reward, info, skills_this_step = self._handle_check_goal()
        elif action_type == "noop":
            pass
        else:
            reward = -0.1  # Unknown action penalty

        # Apply noise
        if self._rng.random() < self.noise_level:
            reward *= 0.5  # Noisy reward

        # Track skills
        self.skills_demonstrated.update(skills_this_step)
        info["skills_demonstrated"] = list(skills_this_step)

        # Step penalty for efficiency
        reward -= 0.01

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self._done = True

        obs = self.get_observation()
        return obs, reward * self.config.reward_scale, self._done, info

    def get_observation(self) -> Observation:
        """Return the current observation."""
        # Build local view around agent
        view_size = 3
        local_view = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                ny, nx = self.agent_pos[0] + dy, self.agent_pos[1] + dx
                cell = self._get_cell(ny, nx)
                row.append(cell)
            local_view.append(row)

        state = {
            "position": list(self.agent_pos),
            "local_view": local_view,
            "collected_items": self.collected_items.copy(),
            "items_remaining": len(self.items) - len(self.collected_items),
            "step": self.current_step,
            "max_steps": self.max_steps,
            "grid_size": self.grid_size,
            "skill_focus": self.skill_focus,
        }

        # Add hints if scaffolding is enabled
        if self.scaffolding and self._rng.random() < self.hint_frequency:
            hint = self._generate_hint()
            if hint:
                state["hint"] = hint

        info = {
            "visited_count": len(self.visited),
            "total_cells": self.grid_size * self.grid_size,
            "progress": len(self.collected_items) / max(1, self.num_items),
        }

        return Observation(state=state, info=info)

    def get_available_actions(self) -> List[Action]:
        """Return available actions."""
        actions = [
            Action(action_type="move", parameters={"direction": "up"}),
            Action(action_type="move", parameters={"direction": "down"}),
            Action(action_type="move", parameters={"direction": "left"}),
            Action(action_type="move", parameters={"direction": "right"}),
            Action(action_type="move", parameters={"direction": "stay"}),
            Action(action_type="noop"),
        ]

        # Can collect if standing on an item
        if self.agent_pos in self.items:
            actions.append(Action(action_type="collect"))

        # Can check goal
        actions.append(Action(action_type="check_goal"))

        return actions

    def _generate_grid(self) -> None:
        """Generate the grid with items and obstacles."""
        # Initialize empty grid
        self.grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.items = {}
        self.obstacles = set()

        # Place goal
        self.goal_pos = (
            self._rng.randint(1, self.grid_size - 1),
            self._rng.randint(1, self.grid_size - 1),
        )
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = "G"

        # Place items
        item_types = ["key", "gem", "coin", "token", "artifact"]
        placed = 0
        attempts = 0
        while placed < self.num_items and attempts < 100:
            y = self._rng.randint(0, self.grid_size - 1)
            x = self._rng.randint(0, self.grid_size - 1)
            pos = (y, x)
            if pos != (0, 0) and pos != self.goal_pos and pos not in self.items and pos not in self.obstacles:
                item_type = item_types[placed % len(item_types)]
                self.items[pos] = item_type
                self.grid[y][x] = item_type[0].upper()
                placed += 1
            attempts += 1

        # Place obstacles
        placed_obs = 0
        attempts = 0
        while placed_obs < self.num_obstacles and attempts < 100:
            y = self._rng.randint(0, self.grid_size - 1)
            x = self._rng.randint(0, self.grid_size - 1)
            pos = (y, x)
            if pos != (0, 0) and pos != self.goal_pos and pos not in self.items and pos not in self.obstacles:
                self.obstacles.add(pos)
                self.grid[y][x] = "#"
                placed_obs += 1
            attempts += 1

    def _get_cell(self, y: int, x: int) -> str:
        """Get the cell content at position (y, x)."""
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
            if (y, x) == self.agent_pos:
                return "@"
            return self.grid[y][x]
        return " "

    def _handle_move(self, direction: str) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle a move action."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills: List[str] = []

        dy, dx = 0, 0
        if direction == "up":
            dy = -1
        elif direction == "down":
            dy = 1
        elif direction == "left":
            dx = -1
        elif direction == "right":
            dx = 1
        # "stay" means no movement

        ny, nx = self.agent_pos[0] + dy, self.agent_pos[1] + dx

        # Check bounds and obstacles
        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
            if (ny, nx) not in self.obstacles:
                self.agent_pos = (ny, nx)
                self.visited.add(self.agent_pos)

                # Reward for visiting new cells (exploration)
                if len(self.visited) > info.get("prev_visited", 0):
                    reward += 0.1
                    skills.append("navigation")

                # Reward for moving toward goal (planning)
                old_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
                # Already updated agent_pos, so compute new distance
                new_dist = abs(ny - self.goal_pos[0]) + abs(nx - self.goal_pos[1])
                if new_dist < old_dist:
                    reward += 0.05
                    skills.append("planning")

                info["moved"] = True
                info["position"] = list(self.agent_pos)
            else:
                reward -= 0.1  # Hit obstacle
                info["moved"] = False
                info["hit_obstacle"] = True
        else:
            reward -= 0.05  # Out of bounds
            info["moved"] = False
            info["out_of_bounds"] = True

        info["prev_visited"] = len(self.visited)
        return reward, info, skills

    def _handle_collect(self) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle a collect action."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills: List[str] = []

        if self.agent_pos in self.items:
            item = self.items.pop(self.agent_pos)
            self.collected_items.append(item)
            reward = 1.0
            info["collected"] = item
            info["success"] = True
            skills.append("reasoning")
            skills.append("memory")

            # Update grid display
            y, x = self.agent_pos
            self.grid[y][x] = "."
        else:
            reward = -0.1
            info["collected"] = None
            info["error"] = "No item at current position"

        return reward, info, skills

    def _handle_check_goal(self) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle checking if goal conditions are met."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills: List[str] = []

        # Goal: collect all items and be at goal position
        all_collected = len(self.collected_items) >= self.num_items
        at_goal = self.agent_pos == self.goal_pos

        if all_collected and at_goal:
            reward = 5.0
            self._done = True
            info["success"] = True
            info["goal_reached"] = True
            skills.append("planning")
            skills.append("reasoning")
        elif all_collected:
            reward = 1.0
            info["success"] = False
            info["all_items_collected"] = True
            info["need_to_reach_goal"] = True
        elif at_goal:
            reward = 0.5
            info["success"] = False
            info["at_goal"] = True
            info["items_remaining"] = self.num_items - len(self.collected_items)
        else:
            reward = -0.1
            info["success"] = False
            info["items_collected"] = len(self.collected_items)
            info["items_needed"] = self.num_items

        return reward, info, skills

    def _generate_hint(self) -> Optional[str]:
        """Generate a hint for the agent (scaffolding)."""
        if not self.items:
            return "All items collected! Head to the goal (G)."

        # Find nearest uncollected item
        nearest = None
        nearest_dist = float("inf")
        for pos in self.items:
            dist = abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = pos

        if nearest:
            dy = nearest[0] - self.agent_pos[0]
            dx = nearest[1] - self.agent_pos[1]
            directions = []
            if dy < 0:
                directions.append("up")
            elif dy > 0:
                directions.append("down")
            if dx < 0:
                directions.append("left")
            elif dx > 0:
                directions.append("right")

            if directions:
                return f"Nearest {self.items[nearest]} is {' and '.join(directions)} ({nearest_dist} steps away)."

        return None

    def render(self) -> str:
        """Return a text representation of the grid."""
        lines = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                if (y, x) == self.agent_pos:
                    row.append("@")
                elif (y, x) == self.goal_pos:
                    row.append("G")
                elif (y, x) in self.items:
                    row.append(self.items[(y, x)][0].upper())
                elif (y, x) in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
