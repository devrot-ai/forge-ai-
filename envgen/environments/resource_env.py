"""Resource management environment — collect and allocate resources.

The agent manages limited resources over time, making decisions about
collection, allocation, and consumption. It must balance short-term needs
with long-term goals under scarcity constraints.

Skills trained:
- planning: Allocating resources optimally
- reasoning: Understanding trade-offs
- memory: Tracking resource levels over time
- efficiency: Maximizing output per resource unit
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, Observation


class ResourceEnvironment(Environment):
    """Resource management environment with configurable constraints.

    The agent has a set of resources (energy, materials, time) and must
    complete tasks that consume resources. It can collect more resources,
    but collection is limited and stochastic. The goal is to maximize
    completed tasks within the time budget.
    """

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        params = config.parameters
        self.num_resource_types: int = params.get("num_resource_types", 3)
        self.initial_resources: float = params.get("initial_resources", 10.0)
        self.max_resources: float = params.get("max_resources", 20.0)
        self.collection_rate: float = params.get("collection_rate", 1.5)
        self.scarcity: float = params.get("scarcity", 0.3)
        self.num_tasks: int = params.get("num_tasks", 5)
        self.noise_level: float = params.get("noise_level", 0.1)
        self.skill_focus: str = params.get("skill_focus", "planning")

        # Resource names
        self.resource_names = [f"resource_{i}" for i in range(self.num_resource_types)]

        # State
        self.resources: Dict[str, float] = {}
        self.tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.total_collected: float = 0.0
        self.total_consumed: float = 0.0

        # Skills
        self.skills_demonstrated: set = set()

    def _generate_tasks(self) -> None:
        """Generate tasks with varying resource requirements."""
        self.tasks = []
        for i in range(self.num_tasks):
            # Each task requires some amount of each resource
            requirements = {}
            total_cost = 0.0
            for rname in self.resource_names:
                cost = random.uniform(0.5, 2.0) * (1 + i * 0.2)
                requirements[rname] = cost
                total_cost += cost

            # Reward scales with task difficulty
            reward = total_cost * random.uniform(1.5, 3.0)

            self.tasks.append(
                {
                    "id": f"task_{i}",
                    "requirements": requirements,
                    "reward": reward,
                    "completed": False,
                }
            )

        # Sort by reward (descending) to make ordering meaningful
        self.tasks.sort(key=lambda t: t["reward"], reverse=True)

    def reset(self) -> Observation:
        """Reset the resource environment."""
        self.current_step = 0
        self._done = False
        self.completed_tasks = []
        self.failed_tasks = []
        self.total_collected = 0.0
        self.total_consumed = 0.0
        self.skills_demonstrated = set()

        # Initialize resources
        for rname in self.resource_names:
            self.resources[rname] = self.initial_resources

        # Generate new tasks
        self._generate_tasks()

        return self.get_observation()

    def get_observation(self) -> Observation:
        """Get the current observation."""
        # Resource levels (normalized)
        resource_levels = {rname: self.resources[rname] / self.max_resources for rname in self.resource_names}

        # Task info
        pending_tasks = [t for t in self.tasks if not t["completed"]]
        task_info = []
        for t in pending_tasks[:5]:  # Show up to 5 pending tasks
            task_info.append(
                {
                    "id": t["id"],
                    "reward": t["reward"],
                    "can_afford": all(self.resources[r] >= req for r, req in t["requirements"].items()),
                }
            )

        state = {
            "resources": dict(self.resources),
            "resource_levels": resource_levels,
            "num_resources": self.num_resource_types,
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(pending_tasks),
            "total_collected": self.total_collected,
            "total_consumed": self.total_consumed,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "task_info": task_info,
            # Compatibility fields for feature extraction
            "agent_pos": [0, 0],
            "goal_pos": [0, 0],
            "grid_size": 1,
            "items_remaining": len(pending_tasks),
            "items_collected": len(self.completed_tasks),
            "num_items": self.num_tasks,
            "visited": len(self.completed_tasks),
        }

        return Observation(state=state, info={"tasks": self.tasks})

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action in the resource environment."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills = []

        action_type = action.action_type
        params = action.parameters

        if action_type == "collect":
            reward, skills = self._handle_collect()
        elif action_type == "execute_task":
            task_idx = params.get("task_index", 0)
            reward, skills = self._handle_execute_task(task_idx)
        elif action_type == "wait":
            # Wait consumes a step but no resources
            reward -= 0.01
        else:
            reward = -0.1

        # Step penalty
        reward -= 0.01

        self.current_step += 1

        # Check termination
        all_done = all(t["completed"] for t in self.tasks)
        if all_done or self.current_step >= self.max_steps:
            self._done = True

        # Bonus for completing all tasks
        if all_done:
            reward += 10.0
            skills.append("planning")
            skills.append("efficiency")

        # Apply noise
        if self._rng.random() < self.noise_level:
            reward *= 0.8

        self.skills_demonstrated.update(skills)
        info["skills_demonstrated"] = list(skills)

        obs = self.get_observation()
        return obs, reward, self._done, info

    def _handle_collect(self) -> tuple[float, List[str]]:
        """Collect resources (stochastic based on scarcity)."""
        skills = ["reasoning"]
        reward = 0.0

        for rname in self.resource_names:
            # Collection is stochastic
            if random.random() > self.scarcity:
                amount = self.collection_rate * random.uniform(0.5, 1.5)
                self.resources[rname] = min(self.resources[rname] + amount, self.max_resources)
                self.total_collected += amount
                reward += 0.05
            else:
                # Scarcity: failed to collect
                reward -= 0.02

        return reward, skills

    def _handle_execute_task(self, task_idx: int) -> tuple[float, List[str]]:
        """Attempt to execute a task."""
        skills = ["planning"]
        reward = 0.0

        pending = [t for t in self.tasks if not t["completed"]]
        if not pending:
            reward -= 0.5
            return reward, skills

        # Clamp index
        idx = min(max(task_idx, 0), len(pending) - 1)
        task = pending[idx]

        # Check if we can afford the task
        can_afford = all(self.resources[r] >= req for r, req in task["requirements"].items())

        if can_afford:
            # Consume resources
            for r, req in task["requirements"].items():
                self.resources[r] -= req
                self.total_consumed += req

            # Mark task as completed
            task["completed"] = True
            self.completed_tasks.append(task["id"])

            reward += task["reward"]
            skills.append("efficiency")
        else:
            # Failed to execute: not enough resources
            reward -= 1.0
            self.failed_tasks.append(task["id"])
            skills.append("reasoning")

        return reward, skills

    def get_available_actions(self) -> List[Action]:
        """Return available actions."""
        pending = [t for t in self.tasks if not t["completed"]]
        actions = [
            Action("collect", {}),
            Action("wait", {}),
        ]

        # Add execute actions for pending tasks
        for i, task in enumerate(pending[:5]):
            actions.append(Action("execute_task", {"task_index": i}))

        return actions

    def render(self) -> str:
        """Render the resource state."""
        lines = ["Resource Management Environment"]
        lines.append("-" * 40)

        # Resources
        lines.append("Resources:")
        for rname in self.resource_names:
            level = self.resources[rname]
            filled = int(level / self.max_resources * 20)
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  {rname}: [{bar}] {level:.1f}/{self.max_resources:.1f}")

        # Tasks
        lines.append("Tasks:")
        for task in self.tasks:
            status = "[X]" if task["completed"] else "[ ]"
            lines.append(f"  {status} {task['id']}: reward={task['reward']:.1f}")

        lines.append(f"Completed: {len(self.completed_tasks)}/{self.num_tasks}")
        lines.append(f"Step: {self.current_step}/{self.max_steps}")

        return "\n".join(lines)

    def is_terminal(self) -> bool:
        """Check if the environment is in a terminal state."""
        return self._done
