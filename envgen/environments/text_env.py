"""Text-based reasoning environment for demonstration.

This environment implements text-based reasoning puzzles where the agent must
solve problems by selecting the correct answer from multiple choices. Task types
include arithmetic, sequence completion, logical deduction, and pattern matching.

Skills trained:
- reasoning: Solving logical and mathematical problems
- planning: Choosing the right approach for each problem type
- memory: Remembering patterns and rules
- efficiency: Solving problems in fewer steps
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, DifficultyLevel, Observation


class TextTaskEnvironment(Environment):
    """Text-based reasoning environment with configurable puzzle types.

    The agent solves text-based puzzles by selecting answers from multiple choices.
    Configuration parameters control:
    - Task type: arithmetic, sequence, logic, pattern, or mixed
    - Difficulty: EASY (simple) to EXPERT (multi-step reasoning)
    - Number of choices: 2-5 answer options
    - Hint availability: whether hints can be requested

    Actions:
    - select_answer: Choose answer by index (0-based)
    - request_hint: Ask for a hint (with penalty)
    - submit: Submit current selection as final answer
    - noop: Do nothing
    """

    # Action definitions
    ACTION_SELECT = "select_answer"
    ACTION_HINT = "request_hint"
    ACTION_SUBMIT = "submit"
    ACTION_NOOP = "noop"

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        # Parse config parameters
        params = config.parameters
        self.task_type: str = params.get("task_type", "mixed")
        self.num_choices: int = params.get("num_choices", 4)
        self.allow_hints: bool = params.get("allow_hints", True)
        self.hint_penalty: float = params.get("hint_penalty", 0.5)
        self.complexity: float = params.get("complexity", 0.3)

        # Current puzzle state
        self.current_question: str = ""
        self.current_answer: str = ""
        self.current_choices: List[str] = []
        self.selected_answer_idx: Optional[int] = None
        self.hints_used: int = 0
        self.correct: bool = False
        self.submitted: bool = False

        # Available actions list
        self._available_actions: List[Action] = []

    def reset(self) -> Observation:
        """Reset the environment with a new puzzle."""
        self.current_step = 0
        self._done = False
        self.selected_answer_idx = None
        self.hints_used = 0
        self.correct = False
        self.submitted = False

        # Generate a new puzzle
        self._generate_puzzle()
        self._update_available_actions()

        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action in the environment."""
        reward = 0.0
        info: Dict[str, Any] = {}
        skills_this_step: List[str] = []

        action_type = action.action_type
        params = action.parameters or {}

        if action_type == self.ACTION_SELECT:
            reward, info, skills_this_step = self._handle_select(params.get("index", 0))
        elif action_type == self.ACTION_HINT:
            reward, info, skills_this_step = self._handle_hint()
        elif action_type == self.ACTION_SUBMIT:
            reward, info, skills_this_step = self._handle_submit()
        elif action_type == self.ACTION_NOOP:
            pass
        else:
            reward = -0.1  # Unknown action penalty

        # Step penalty for efficiency
        reward -= 0.05

        self.current_step += 1
        info["skills_demonstrated"] = skills_this_step
        info["hints_used"] = self.hints_used

        if self.submitted or self.current_step >= self.max_steps:
            self._done = True

        return self.get_observation(), reward, self._done, info

    def get_observation(self) -> Observation:
        """Return the current observation."""
        state = {
            "question": self.current_question,
            "choices": self.current_choices,
            "selected_index": self.selected_answer_idx,
            "hints_used": self.hints_used,
            "submitted": self.submitted,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "task_type": self.task_type,
        }

        return Observation(
            state=state,
        )

    def get_available_actions(self) -> List[Action]:
        """Return the list of valid actions from the current state."""
        return self._available_actions.copy()

    def render(self) -> str:
        """Return a text representation of the current puzzle."""
        lines = [
            f"Question: {self.current_question}",
            "Choices:",
        ]
        for i, choice in enumerate(self.current_choices):
            marker = " <-- SELECTED" if i == self.selected_answer_idx else ""
            lines.append(f"  [{i}] {choice}{marker}")
        if self.submitted:
            lines.append(f"Answer: {self.current_answer}")
            lines.append(f"Correct: {self.correct}")
        return "\n".join(lines)

    def _update_available_actions(self) -> None:
        """Update the list of available actions based on current state."""
        actions = []

        # Add select_answer actions for each choice
        for i in range(len(self.current_choices)):
            actions.append(Action(self.ACTION_SELECT, {"index": i}))

        # Add hint action if allowed and not yet submitted
        if self.allow_hints and not self.submitted:
            actions.append(Action(self.ACTION_HINT, {}))

        # Add submit action if an answer is selected
        if self.selected_answer_idx is not None and not self.submitted:
            actions.append(Action(self.ACTION_SUBMIT, {}))

        # Always allow noop
        actions.append(Action(self.ACTION_NOOP, {}))

        self._available_actions = actions

    def _generate_puzzle(self) -> None:
        """Generate a new puzzle based on config."""
        # Determine task type
        if self.task_type == "mixed":
            task_type = self._rng.choice(["arithmetic", "sequence", "logic", "pattern"])
        else:
            task_type = self.task_type

        # Determine difficulty-based complexity
        difficulty = self.config.difficulty
        if difficulty == DifficultyLevel.EASY:
            complexity = 0.25
        elif difficulty == DifficultyLevel.MEDIUM:
            complexity = 0.5
        elif difficulty == DifficultyLevel.HARD:
            complexity = 0.75
        else:  # EXPERT
            complexity = 1.0

        # Generate puzzle based on type
        if task_type == "arithmetic":
            self._generate_arithmetic(complexity)
        elif task_type == "sequence":
            self._generate_sequence(complexity)
        elif task_type == "logic":
            self._generate_logic(complexity)
        elif task_type == "pattern":
            self._generate_pattern(complexity)
        else:
            self._generate_arithmetic(complexity)

        # Ensure we have the right number of choices
        self._adjust_choices()

    def _generate_arithmetic(self, complexity: float) -> None:
        """Generate an arithmetic puzzle."""
        if complexity < 0.4:
            # Simple addition/subtraction
            a = self._rng.randint(1, 20)
            b = self._rng.randint(1, 20)
            if self._rng.random() < 0.5:
                self.current_question = f"What is {a} + {b}?"
                self.current_answer = str(a + b)
            else:
                a, b = max(a, b), min(a, b)
                self.current_question = f"What is {a} - {b}?"
                self.current_answer = str(a - b)
        elif complexity < 0.7:
            # Multiplication or two-step
            a = self._rng.randint(2, 12)
            b = self._rng.randint(2, 12)
            if self._rng.random() < 0.5:
                self.current_question = f"What is {a} × {b}?"
                self.current_answer = str(a * b)
            else:
                c = self._rng.randint(1, 10)
                self.current_question = f"What is {a} + {b} × {c}?"
                self.current_answer = str(a + b * c)
        else:
            # Multi-step arithmetic
            a = self._rng.randint(10, 50)
            b = self._rng.randint(2, 10)
            c = self._rng.randint(1, 20)
            ops = self._rng.choice(
                [
                    (f"What is ({a} + {b}) × {c}?", (a + b) * c),
                    (f"What is {a} × {b} - {c}?", a * b - c),
                    (f"What is ({a} - {b}) + {c} × 2?", (a - b) + c * 2),
                ]
            )
            self.current_question = ops[0]
            self.current_answer = str(ops[1])

        self.current_choices = self._generate_numeric_choices(int(self.current_answer))

    def _generate_sequence(self, complexity: float) -> None:
        """Generate a sequence completion puzzle."""
        if complexity < 0.4:
            # Simple arithmetic sequence
            start = self._rng.randint(1, 10)
            step = self._rng.randint(2, 5)
            seq = [start + i * step for i in range(5)]
            answer = start + 5 * step
            self.current_question = f"What comes next? {', '.join(map(str, seq))}, ?"
            self.current_answer = str(answer)
        elif complexity < 0.7:
            # Geometric or squared sequence
            if self._rng.random() < 0.5:
                start = self._rng.randint(1, 3)
                ratio = self._rng.randint(2, 3)
                seq = [start * (ratio**i) for i in range(5)]
                answer = start * (ratio**5)
            else:
                seq = [(i + 1) ** 2 for i in range(5)]
                answer = 36
            self.current_question = f"What comes next? {', '.join(map(str, seq))}, ?"
            self.current_answer = str(answer)
        else:
            # Fibonacci-like or complex pattern
            a, b = self._rng.randint(1, 5), self._rng.randint(1, 5)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            answer = seq[-1] + seq[-2]
            self.current_question = f"What comes next? {', '.join(map(str, seq))}, ?"
            self.current_answer = str(answer)

        self.current_choices = self._generate_numeric_choices(int(self.current_answer))

    def _generate_logic(self, complexity: float) -> None:
        """Generate a logical deduction puzzle."""
        if complexity < 0.4:
            # Simple if-then
            self.current_question = "If all cats are mammals and Fluffy is a cat, then Fluffy is a:"
            self.current_answer = "mammal"
            self.current_choices = ["mammal", "reptile", "bird", "fish"]
        elif complexity < 0.7:
            # Comparative logic
            self.current_question = "If A > B and B > C, which is true?"
            self.current_answer = "A > C"
            self.current_choices = ["A > C", "C > A", "A = C", "B > A"]
        else:
            # Multi-step logic
            self.current_question = "If P implies Q, Q implies R, and P is true, then:"
            self.current_answer = "R is true"
            self.current_choices = ["R is true", "R is false", "Q is false", "Cannot determine"]

    def _generate_pattern(self, complexity: float) -> None:
        """Generate a pattern recognition puzzle."""
        if complexity < 0.4:
            # Letter pattern
            self.current_question = "What comes next? A, C, E, G, ?"
            self.current_answer = "I"
            self.current_choices = ["I", "H", "J", "K"]
        elif complexity < 0.7:
            # Number-letter pattern
            self.current_question = "What comes next? AB, BC, CD, DE, ?"
            self.current_answer = "EF"
            self.current_choices = ["EF", "FG", "DF", "FE"]
        else:
            # Complex pattern
            self.current_question = "What comes next? 1A, 2B, 4D, 8H, ?"
            self.current_answer = "16L"
            self.current_choices = ["16L", "16J", "12L", "10L"]

    def _generate_numeric_choices(self, correct: int) -> List[str]:
        """Generate multiple-choice answers with one correct answer."""
        choices = {str(correct)}
        attempts = 0
        while len(choices) < self.num_choices and attempts < 100:
            # Generate a plausible wrong answer
            offset = self._rng.randint(1, max(abs(correct) // 2, 1))
            if self._rng.random() < 0.5:
                wrong = correct + offset
            else:
                wrong = correct - offset
            choices.add(str(wrong))
            attempts += 1

        # Fill remaining if needed
        while len(choices) < self.num_choices:
            choices.add(str(correct + len(choices) * 3))

        # Shuffle and return
        result = list(choices)[: self.num_choices]
        self._rng.shuffle(result)
        return result

    def _adjust_choices(self) -> None:
        """Ensure the number of choices matches config."""
        if len(self.current_choices) < self.num_choices:
            # Add filler choices
            while len(self.current_choices) < self.num_choices:
                self.current_choices.append(f"Option {len(self.current_choices)}")
        elif len(self.current_choices) > self.num_choices:
            self.current_choices = self.current_choices[: self.num_choices]

    def _handle_select(self, index: int) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle selecting an answer."""
        info = {}
        skills = ["reasoning"]

        if 0 <= index < len(self.current_choices):
            self.selected_answer_idx = index
            info["selected_choice"] = self.current_choices[index]
            reward = 0.0
        else:
            info["error"] = "Invalid answer index"
            reward = -0.2

        self._update_available_actions()
        return reward, info, skills

    def _handle_hint(self) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle requesting a hint."""
        if not self.allow_hints:
            return -0.5, {"error": "Hints not allowed"}, []

        self.hints_used += 1
        hint = self._generate_hint()
        skills = ["planning"]
        return -self.hint_penalty, {"hint": hint, "hints_used": self.hints_used}, skills

    def _handle_submit(self) -> tuple[float, Dict[str, Any], List[str]]:
        """Handle submitting the selected answer."""
        info = {}
        skills = ["reasoning", "efficiency"]

        if self.selected_answer_idx is None:
            return -0.5, {"error": "No answer selected"}, []

        self.submitted = True
        selected_choice = self.current_choices[self.selected_answer_idx]

        # Check correctness
        if str(selected_choice) == str(self.current_answer):
            self.correct = True
            # Reward based on efficiency
            steps_used = self.current_step
            max_reward = 10.0
            efficiency_bonus = max(0, 1 - steps_used / self.max_steps) * 5.0
            hint_penalty = self.hints_used * self.hint_penalty
            reward = max_reward + efficiency_bonus - hint_penalty
            info["result"] = "correct"
        else:
            self.correct = False
            reward = -1.0
            info["result"] = "incorrect"
            info["correct_answer"] = self.current_answer

        info["selected"] = selected_choice
        info["correct"] = self.current_answer
        self._update_available_actions()
        return reward, info, skills

    def _generate_hint(self) -> str:
        """Generate a hint for the current puzzle."""
        if "arithmetic" in self.current_question.lower() or "what is" in self.current_question.lower():
            return "Try breaking the problem into smaller steps."
        elif "comes next" in self.current_question.lower():
            return "Look at the difference between consecutive terms."
        elif "if" in self.current_question.lower() and (
            "then" in self.current_question.lower() or "true" in self.current_question.lower()
        ):
            return "Follow the logical chain step by step."
        else:
            return "Look for the underlying pattern or rule."
