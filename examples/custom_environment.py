"""Custom environment tutorial.

This example demonstrates how to create a new environment type from scratch.
It walks through creating a simple "NumberGuessingEnvironment" where the agent
must guess a hidden number by receiving higher/lower feedback.

This tutorial covers:
1. **Extending the Environment base class** — implementing the required methods
2. **Defining observations and actions** — designing the state and action space
3. **Implementing step logic** — environment dynamics and rewards
4. **Supporting EnvironmentConfig** — making the environment LLM-generatable
5. **Registering with the framework** — using the new environment in the pipeline

Run with:
    python examples/custom_environment.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, DifficultyLevel, Observation

# =============================================================================
# Step 1: Create the custom environment class
# =============================================================================


class NumberGuessingEnvironment(Environment):
    """A number guessing game environment.

    The agent must guess a hidden number within a range. After each guess,
    the environment provides feedback: the guess was too high, too low, or correct.

    This is a simple environment that demonstrates the core patterns for
    creating custom environments in EnvGen-LLM.
    """

    def __init__(self, config: EnvironmentConfig):
        """Initialize the number guessing environment.

        Args:
            config: EnvironmentConfig with parameters:
                - min_value: Minimum number in the range (default: 1)
                - max_value: Maximum number in the range (default: 100)
                - max_steps: Maximum number of guesses (default: 10)
        """
        super().__init__(config)
        self.min_value = self.config.parameters.get("min_value", 1)
        self.max_value = self.config.parameters.get("max_value", 100)

        # Internal state
        self.target_number: int = 0
        self.guess_history: List[int] = []
        self.current_low = self.min_value
        self.current_high = self.max_value

    def reset(self) -> Observation:
        """Reset the environment to a fresh state.

        Returns:
            The initial observation.
        """
        self.current_step = 0
        self._done = False
        self.target_number = self._rng.randint(self.min_value, self.max_value)
        self.guess_history = []
        self.current_low = self.min_value
        self.current_high = self.max_value

        return self.get_observation()

    def get_observation(self) -> Observation:
        """Get the current observation.

        The observation includes the current range, step count, and
        the last feedback (higher/lower/correct).

        Returns:
            Observation with state dictionary.
        """
        state = {
            "current_low": self.current_low,
            "current_high": self.current_high,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "range_size": self.current_high - self.current_low + 1,
            "num_guesses_made": len(self.guess_history),
            "last_feedback": self._get_last_feedback(),
        }

        return Observation(state=state)

    def get_available_actions(self) -> List[Action]:
        """Get the available actions.

        Actions represent different guessing strategies:
        - "guess_mid": Guess the midpoint of the current range
        - "guess_low": Guess the lower quarter
        - "guess_high": Guess the upper quarter
        - "guess_low_bound": Guess near the lower bound
        - "guess_high_bound": Guess near the upper bound

        Returns:
            List of available Action objects.
        """
        return [
            Action(action_type="guess_mid"),
            Action(action_type="guess_low"),
            Action(action_type="guess_high"),
            Action(action_type="guess_low_bound"),
            Action(action_type="guess_high_bound"),
        ]

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute an action and return the result.

        Args:
            action: The guessing strategy Action to use.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            obs = self.get_observation()
            return obs, 0.0, True, {"error": "Episode already done"}

        # Compute the actual guess based on the action
        guess = self._compute_guess(action.action_type)
        self.guess_history.append(guess)

        # Compute reward
        reward = self._compute_reward(guess)

        # Update the search range based on feedback
        if guess < self.target_number:
            self.current_low = max(self.current_low, guess + 1)
        elif guess > self.target_number:
            self.current_high = min(self.current_high, guess - 1)

        # Ensure range is valid
        if self.current_low > self.current_high:
            self.current_low, self.current_high = self.current_high, self.current_low

        self.current_step += 1
        if guess == self.target_number or self.current_step >= self.max_steps:
            self._done = True

        info = {
            "guess": guess,
            "target": self.target_number,
            "feedback": self._get_last_feedback(),
        }

        obs = self.get_observation()
        return obs, reward * self.config.reward_scale, self._done, info

    def render(self) -> str:
        """Render the current state of the environment.

        Returns:
            Text representation of the environment state.
        """
        obs = self.get_observation()
        state = obs.state

        lines = [
            "=" * 53,
            "  Number Guessing Environment",
            "-" * 53,
            f"  Range: [{self.current_low:3d}, {self.current_high:3d}]",
            f"  Range size: {state['range_size']:2d}",
            f"  Step: {self.current_step}/{self.max_steps}",
            f"  Guesses made: {len(self.guess_history):2d}",
            f"  Last feedback: {state['last_feedback']}",
        ]

        if self.guess_history:
            last_guess = self.guess_history[-1]
            lines.append(f"  Last guess: {last_guess:3d}")

        if self._done:
            lines.append(f"  Target was: {self.target_number:3d}")
            if self.guess_history and self.guess_history[-1] == self.target_number:
                lines.append("  [Correct!]")
            else:
                lines.append("  [Out of guesses!]")

        lines.append("=" * 53)

        return "\n".join(lines)

    def _compute_guess(self, action_type: str) -> int:
        """Convert an action type to an actual number guess.

        Args:
            action_type: The guessing strategy action type string.

        Returns:
            The integer guess.
        """
        low = self.current_low
        high = self.current_high
        mid = (low + high) // 2
        quarter = (high - low) // 4

        if action_type == "guess_mid":
            return mid
        elif action_type == "guess_low":
            return low + quarter
        elif action_type == "guess_high":
            return high - quarter
        elif action_type == "guess_low_bound":
            return low + 1 if low < high else low
        elif action_type == "guess_high_bound":
            return high - 1 if high > low else high
        else:
            return mid  # Default to midpoint

    def _compute_reward(self, guess: int) -> float:
        """Compute the reward for a guess.

        Args:
            guess: The number guessed.

        Returns:
            Reward value.
        """
        if guess == self.target_number:
            # Big reward for correct guess, scaled by remaining steps
            remaining_ratio = 1.0 - (self.current_step / self.max_steps)
            return 10.0 * max(0.1, remaining_ratio)
        elif self.current_step >= self.max_steps:
            return -5.0  # Penalty for running out of steps

        # Small reward for narrowing the range
        range_reduction = (self.max_value - self.min_value) / max(1, self.current_high - self.current_low + 1)
        return 0.5 * min(1.0, range_reduction / 10.0)

    def _get_last_feedback(self) -> str:
        """Get feedback on the last guess.

        Returns:
            String feedback: 'higher', 'lower', 'correct', or 'none'.
        """
        if not self.guess_history:
            return "none"

        last_guess = self.guess_history[-1]
        if last_guess == self.target_number:
            return "correct"
        elif last_guess < self.target_number:
            return "higher"
        else:
            return "lower"


# =============================================================================
# Step 2: Demonstrate the custom environment
# =============================================================================


def demonstrate_basic_usage():
    """Demonstrate basic usage of the custom environment."""
    print("=" * 70)
    print("Step 1: Basic Environment Usage")
    print("=" * 70)
    print()

    # Create the environment with proper EnvironmentConfig
    config = EnvironmentConfig(
        name="number_guessing_demo",
        description="Guess a number between 1 and 100",
        difficulty=DifficultyLevel.EASY,
        parameters={"min_value": 1, "max_value": 100},
        max_steps=10,
    )
    env = NumberGuessingEnvironment(config)
    obs = env.reset()

    print("Initial observation:")
    print(f"  Range: [{obs.state['current_low']}, {obs.state['current_high']}]")
    print(f"  Step: {obs.state['step']}/{obs.state['max_steps']}")
    print()

    # Get available actions
    actions = env.get_available_actions()
    print(f"Available actions: {[a.action_type for a in actions]}")
    print()

    # Take a few steps
    print("Taking steps:")
    for i in range(5):
        action = actions[0]  # Always guess midpoint (binary search!)
        next_obs, reward, done, info = env.step(action)

        print(f"  Step {i + 1}: action={action.action_type}, " f"reward={reward:.3f}, " f"done={done}")
        print(f"    New range: [{next_obs.state['current_low']}, " f"{next_obs.state['current_high']}]")

        obs = next_obs
        if done:
            break

    # Render the environment
    print()
    print(env.render())


def demonstrate_configured_usage():
    """Demonstrate creating the environment from an EnvironmentConfig."""
    print()
    print("=" * 70)
    print("Step 2: Environment from Config (LLM-Generatable)")
    print("=" * 70)
    print()

    # Simulate an LLM-generated config
    config = EnvironmentConfig(
        name="EasyNumberGuessing",
        description="Guess a number between 1 and 10 in 5 tries.",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "min_value": 1,
            "max_value": 10,
        },
        target_skills=["reasoning", "planning"],
        max_steps=5,
    )

    print(f"Config name: {config.name}")
    print(f"Difficulty: {config.difficulty.value}")
    print(f"Parameters: {config.parameters}")
    print(f"Target skills: {config.target_skills}")
    print()

    # Create environment from config
    env = NumberGuessingEnvironment(config)

    print("Running 3 episodes:")
    for ep in range(3):
        env.reset()
        total_reward = 0.0
        steps = 0

        while not env.done:
            actions = env.get_available_actions()
            action = actions[0]  # Binary search strategy
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        print(
            f"  Episode {ep + 1}: {steps} steps, reward={total_reward:.3f}, "
            f"found={env.guess_history and env.guess_history[-1] == env.target_number}"
        )


def demonstrate_difficulty_scaling():
    """Demonstrate how difficulty scales the environment."""
    print()
    print("=" * 70)
    print("Step 3: Difficulty Scaling")
    print("=" * 70)
    print()

    difficulty_configs = {
        DifficultyLevel.EASY: {
            "min_value": 1,
            "max_value": 10,
            "max_steps": 5,
        },
        DifficultyLevel.MEDIUM: {
            "min_value": 1,
            "max_value": 100,
            "max_steps": 7,
        },
        DifficultyLevel.HARD: {
            "min_value": 1,
            "max_value": 1000,
            "max_steps": 10,
        },
        DifficultyLevel.EXPERT: {
            "min_value": 1,
            "max_value": 10000,
            "max_steps": 13,
        },
    }

    print(f"{'Difficulty':<10} {'Range':<15} {'Max Steps':<12} {'Binary Search OK?':<20}")
    print("-" * 57)

    for difficulty, params in difficulty_configs.items():
        # Can binary search solve it?
        import math

        range_size = params["max_value"] - params["min_value"] + 1
        min_steps_needed = math.ceil(math.log2(range_size))
        can_solve = params["max_steps"] >= min_steps_needed

        print(
            f"{difficulty.value:<10} [{params['min_value']}, {params['max_value']}]"
            f"{params['max_steps']:<12} {'Yes' if can_solve else 'No (too hard!)':<20}"
        )

    print()
    print("Binary search needs ceil(log2(N)) steps to find a number in range [1, N].")
    print("At EXPERT difficulty, the range is too large for the allowed steps,")
    print("forcing the agent to use heuristics and learn from feedback patterns.")


def demonstrate_in_pipeline():
    """Demonstrate using the custom environment in the training pipeline."""
    print()
    print("=" * 70)
    print("Step 4: Using in Training Pipeline")
    print("=" * 70)
    print()

    from envgen.agents.ppo_agent import PPOAgent
    from envgen.evaluators.skill_evaluator import SkillEvaluator
    from envgen.utils.types import Episode, Transition

    # Create environment with proper config
    config = EnvironmentConfig(
        name="number_guessing_pipeline",
        description="Number guessing for pipeline demo",
        difficulty=DifficultyLevel.EASY,
        parameters={"min_value": 1, "max_value": 100},
        max_steps=10,
    )
    env = NumberGuessingEnvironment(config)

    # Create agent
    agent = PPOAgent(
        name="number_guess_agent",
        config={"learning_rate": 0.1, "initial_epsilon": 1.0, "epsilon_decay": 0.95},
    )

    # Train for a few episodes
    episodes = []
    for ep_idx in range(20):
        obs = env.reset()
        transitions = []

        while not env.done:
            available = env.get_available_actions()
            action = agent.select_action(obs, available)

            next_obs, reward, done, info = env.step(action)
            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            transitions.append(transition)
            obs = next_obs

        episode = Episode(
            transitions=transitions,
            total_reward=sum(t.reward for t in transitions),
        )
        episodes.append(episode)
        agent.train([episode])

    # Evaluate
    evaluator = SkillEvaluator()
    eval_result = evaluator.evaluate(episodes, agent_id="number_guess_agent")

    mean_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
    success_count = sum(1 for ep in episodes if ep.transitions and ep.transitions[-1].reward > 0)

    print(f"Trained {len(episodes)} episodes")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Success rate: {success_count}/{len(episodes)} ({success_count/len(episodes)*100:.1f}%)")
    print(f"Overall skill score: {eval_result.overall_score:.3f}")
    print()
    print("Skill scores:")
    for ss in eval_result.skill_scores:
        print(f"  {ss.skill_name}: {ss.score:.3f}")


def main():
    """Run the custom environment tutorial."""
    print()
    print("=" * 72)
    print(" Custom Environment Tutorial".center(72))
    print("=" * 72)
    print()
    print("This tutorial shows how to create a new environment type from scratch.")
    print("We'll build a NumberGuessingEnvironment step by step.")
    print()

    demonstrate_basic_usage()
    demonstrate_configured_usage()
    demonstrate_difficulty_scaling()
    demonstrate_in_pipeline()

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("To create a custom environment:")
    print("  1. Extend the Environment base class")
    print("  2. Implement reset(), get_observation(), get_available_actions(), step()")
    print("  3. Optionally implement render() for visualization")
    print("  4. Support EnvironmentConfig for LLM-generated parameters")
    print("  5. Use it in the TrainingPipeline like any other environment")
    print()
    print("The NumberGuessingEnvironment above demonstrates all these patterns.")
    print("You can follow the same structure to create your own environments.")
    print()


if __name__ == "__main__":
    main()
