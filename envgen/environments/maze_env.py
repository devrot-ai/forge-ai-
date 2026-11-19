"""Maze environment — procedural maze navigation task.

The agent navigates through a procedurally generated maze to find the exit.
Difficulty is controlled by maze size, complexity, and dead-end density.

Skills trained:
- navigation: Moving through the maze
- planning: Finding the path to the exit
- reasoning: Understanding maze structure
- memory: Remembering explored paths
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, Observation


class MazeEnvironment(Environment):
    """Procedural maze navigation environment.

    The agent starts at the entrance and must find the exit by navigating
    through corridors. The maze is generated using a recursive backtracking
    algorithm with configurable complexity.
    """

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        params = config.parameters
        self.maze_width: int = params.get("maze_width", 7)
        self.maze_height: int = params.get("maze_height", 7)
        self.complexity: float = params.get("complexity", 0.5)
        self.dead_end_density: float = params.get("dead_end_density", 0.2)
        self.noise_level: float = params.get("noise_level", 0.05)
        self.skill_focus: str = params.get("skill_focus", "navigation")

        # Ensure odd dimensions for maze generation
        if self.maze_width % 2 == 0:
            self.maze_width += 1
        if self.maze_height % 2 == 0:
            self.maze_height += 1

        # Maze state
        self.maze: List[List[int]] = []  # 1 = wall, 0 = path
        self.start: Tuple[int, int] = (1, 1)
        self.exit: Tuple[int, int] = (self.maze_width - 2, self.maze_height - 2)
        self.agent_pos: Tuple[int, int] = self.start
        self.visited: Set[Tuple[int, int]] = set()
        self.steps_taken: int = 0
        self.optimal_path_length: int = 0

        # Skills
        self.skills_demonstrated: set = set()

    def _generate_maze(self) -> None:
        """Generate a maze using recursive backtracking."""
        # Initialize all walls
        self.maze = [[1] * self.maze_width for _ in range(self.maze_height)]

        # Recursive backtracking
        stack: List[Tuple[int, int]] = []
        start = (1, 1)
        self.maze[start[0]][start[1]] = 0
        stack.append(start)

        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            current = stack[-1]
            neighbors = []

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 < nx < self.maze_width - 1 and 0 < ny < self.maze_height - 1:
                    if self.maze[nx][ny] == 1:
                        neighbors.append((nx, ny))

            if neighbors:
                # Choose a random unvisited neighbor
                next_cell = random.choice(neighbors)
                # Remove wall between current and next
                wall_x = (current[0] + next_cell[0]) // 2
                wall_y = (current[1] + next_cell[1]) // 2
                self.maze[wall_x][wall_y] = 0
                self.maze[next_cell[0]][next_cell[1]] = 0
                stack.append(next_cell)
            else:
                stack.pop()

        # Add some extra passages based on complexity (more complexity = fewer extra passages)
        num_extra = int(self.maze_width * self.maze_height * (1 - self.complexity) * 0.1)
        for _ in range(num_extra):
            x = random.randint(1, self.maze_width - 2)
            y = random.randint(1, self.maze_height - 2)
            if self.maze[x][y] == 1:
                # Check if removing this wall connects two open cells
                open_neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height:
                        if self.maze[nx][ny] == 0:
                            open_neighbors += 1
                if open_neighbors >= 2:
                    self.maze[x][y] = 0

        # Set exit position (ensure it's on a path)
        self.exit = (self.maze_width - 2, self.maze_height - 2)
        self.maze[self.exit[0]][self.exit[1]] = 0

        # Compute optimal path length using BFS
        self.optimal_path_length = self._bfs_distance(self.start, self.exit)

    def _bfs_distance(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Compute shortest path distance using BFS."""
        if start == end:
            return 0

        visited = {start}
        queue = [(start, 0)]

        while queue:
            (x, y), dist = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == end:
                    return dist + 1
                if (
                    0 <= nx < self.maze_width
                    and 0 <= ny < self.maze_height
                    and self.maze[nx][ny] == 0
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

        return self.maze_width * self.maze_height  # Fallback

    def reset(self) -> Observation:
        """Reset the maze environment."""
        self.current_step = 0
        self._done = False
        self.steps_taken = 0
        self.visited = {self.start}
        self.skills_demonstrated = set()
        self.agent_pos = self.start

        # Generate new maze
        self._generate_maze()

        return self.get_observation()

    def get_observation(self) -> Observation:
        """Get the current observation."""
        # Local view: show nearby cells
        view_size = 3
        local_view = []
        for dy in range(-view_size // 2, view_size // 2 + 1):
            row = []
            for dx in range(-view_size // 2, view_size // 2 + 1):
                nx = self.agent_pos[0] + dx
                ny = self.agent_pos[1] + dy
                if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height:
                    cell = self.maze[nx][ny]
                    if (nx, ny) == self.exit:
                        cell = 2  # Exit marker
                    row.append(cell)
                else:
                    row.append(1)  # Wall out of bounds
            local_view.append(row)

        state = {
            "agent_pos": list(self.agent_pos),
            "exit_pos": list(self.exit),
            "maze_width": self.maze_width,
            "maze_height": self.maze_height,
            "local_view": local_view,
            "visited": len(self.visited),
            "steps_taken": self.steps_taken,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "distance_to_exit": abs(self.agent_pos[0] - self.exit[0]) + abs(self.agent_pos[1] - self.exit[1]),
        }

        return Observation(state=state, info={"maze": self.maze})

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action in the maze."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills = []

        action_type = action.action_type
        params = action.parameters

        if action_type == "move":
            direction = params.get("direction", "stay")
            reward, skills = self._handle_move(direction)
        elif action_type == "noop":
            pass
        else:
            reward = -0.1

        # Step penalty
        reward -= 0.01
        self.steps_taken += 1

        # Check if reached exit
        if self.agent_pos == self.exit:
            reward += 5.0
            self._done = True
            skills.append("navigation")
            skills.append("planning")

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self._done = True

        # Apply noise
        if self._rng.random() < self.noise_level:
            reward *= 0.5

        self.skills_demonstrated.update(skills)
        info["skills_demonstrated"] = list(skills)

        obs = self.get_observation()
        return obs, reward, self._done, info

    def _handle_move(self, direction: str) -> tuple[float, List[str]]:
        """Handle a move action."""
        reward = 0.0
        skills = []

        dx, dy = 0, 0
        if direction == "up":
            dx, dy = -1, 0
        elif direction == "down":
            dx, dy = 1, 0
        elif direction == "left":
            dx, dy = 0, -1
        elif direction == "right":
            dx, dy = 0, 1

        if dx != 0 or dy != 0:
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy

            if 0 <= new_x < self.maze_width and 0 <= new_y < self.maze_height and self.maze[new_x][new_y] == 0:
                self.agent_pos = (new_x, new_y)
                self.visited.add(self.agent_pos)
                skills.append("navigation")

                # Bonus for visiting new cells
                reward += 0.1

                # Penalty for backtracking to heavily visited areas
                if len(self.visited) > (self.maze_width * self.maze_height * 0.5):
                    reward += 0.05
            else:
                # Hit a wall
                reward -= 0.1
        else:
            # Stay
            reward -= 0.05

        return reward, skills

    def get_available_actions(self) -> List[Action]:
        """Return available actions."""
        return [
            Action("move", {"direction": "up"}),
            Action("move", {"direction": "down"}),
            Action("move", {"direction": "left"}),
            Action("move", {"direction": "right"}),
            Action("noop", {}),
        ]

    def render(self) -> str:
        """Render the maze as a string."""
        lines = []
        for y in range(self.maze_height):
            row = ""
            for x in range(self.maze_width):
                if (x, y) == self.agent_pos:
                    row += "A"
                elif (x, y) == self.exit:
                    row += "E"
                elif self.maze[x][y] == 1:
                    row += "#"
                elif (x, y) in self.visited:
                    row += "."
                else:
                    row += " "
            lines.append(row)
        return "\n".join(lines)

    def is_terminal(self) -> bool:
        """Check if the environment is in a terminal state."""
        return self._done
