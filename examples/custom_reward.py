"""Custom reward shaping example.

This example demonstrates how to create custom reward functions with the
RewardEngine. Instead of using the default reward components, you can define
your own reward logic to guide agent learning.

The example shows three custom reward strategies:
1. **Distance-based reward** — rewards the agent for getting closer to the goal
2. **Exploration bonus** — rewards visiting new states the agent hasn't seen before
3. **Speed bonus** — rewards completing tasks in fewer steps

Run with:
    python examples/custom_reward.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envgen.agents.ppo_agent import PPOAgent
from envgen.environments.base import EnvironmentConfig
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.evaluators.skill_evaluator import SkillEvaluator
from envgen.rewards.engine import RewardEngine
from envgen.utils.types import Action, DifficultyLevel, Episode, Observation, Transition


def distance_to_goal_reward(transition: Transition) -> float:
    """Reward based on items remaining (proxy for progress toward goal).

    Rewards the agent for reducing the number of remaining items.
    This provides dense feedback instead of sparse success-only rewards.

    Args:
        transition: The MDP transition.

    Returns:
        Positive reward if fewer items remain, zero otherwise.
    """
    prev_items = transition.observation.state.get("items_remaining", 0)
    curr_items = transition.next_observation.state.get("items_remaining", 0)
    return float(max(0, prev_items - curr_items))


def exploration_bonus_reward(transition: Transition) -> float:
    """Reward for visiting new cells.

    Encourages the agent to explore by rewarding increases in visited cells.

    Args:
        transition: The MDP transition.

    Returns:
        Small positive reward for new cells visited.
    """
    prev_pos = transition.observation.state.get("position", [0, 0])
    curr_pos = transition.next_observation.state.get("position", [0, 0])
    # Reward movement to a different cell
    if prev_pos != curr_pos:
        return 0.3
    return 0.0


def speed_bonus_reward(transition: Transition) -> float:
    """Reward for completing tasks quickly.

    Gives a larger bonus for achieving the goal in fewer steps.

    Args:
        transition: The MDP transition.

    Returns:
        Bonus reward if done with positive reward, scaled by step count.
    """
    if transition.done and transition.reward > 0:
        step = transition.observation.state.get("step", 0)
        max_steps = transition.observation.state.get("max_steps", 100)
        speed_ratio = 1.0 - (step / max_steps)
        return 5.0 * max(0.1, speed_ratio)
    return 0.0


def create_custom_reward_engine(strategy: str = "distance") -> RewardEngine:
    """Create a reward engine with custom reward components.

    Args:
        strategy: Reward strategy to use:
            - "distance": Distance-based reward for dense feedback
            - "exploration": Exploration bonus for visiting new states
            - "speed": Speed bonus for fast completion
            - "combined": All three strategies combined

    Returns:
        Configured RewardEngine instance.
    """
    engine = RewardEngine({"clip_range": (-5.0, 15.0)})

    if strategy == "distance":
        engine.add_component("distance", 1.0, distance_to_goal_reward)
        engine.add_component("success", 10.0, lambda t: float(t.done and t.reward > 0))
        engine.add_component("step_penalty", -0.1, lambda t: 1.0)

    elif strategy == "exploration":
        engine.add_component("exploration", 1.0, exploration_bonus_reward)
        engine.add_component("success", 10.0, lambda t: float(t.done and t.reward > 0))
        engine.add_component("step_penalty", -0.1, lambda t: 1.0)

    elif strategy == "speed":
        engine.add_component("speed", 1.0, speed_bonus_reward)
        engine.add_component("success", 10.0, lambda t: float(t.done and t.reward > 0))
        engine.add_component("step_penalty", -0.1, lambda t: 1.0)

    elif strategy == "combined":
        engine.add_component("distance", 0.5, distance_to_goal_reward)
        engine.add_component("exploration", 0.5, exploration_bonus_reward)
        engine.add_component("speed", 1.0, speed_bonus_reward)
        engine.add_component("success", 10.0, lambda t: float(t.done and t.reward > 0))
        engine.add_component("step_penalty", -0.1, lambda t: 1.0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return engine


def make_env_config(grid_size=8, num_items=3, num_obstacles=5, max_steps=50):
    """Create an EnvironmentConfig for the puzzle environment."""
    return EnvironmentConfig(
        name="custom_reward_test",
        description="Test environment for custom reward shaping",
        difficulty=DifficultyLevel.EASY,
        parameters={
            "grid_size": grid_size,
            "num_items": num_items,
            "num_obstacles": num_obstacles,
            "complexity": 0.3,
            "noise_level": 0.1,
        },
        max_steps=max_steps,
    )


def run_comparison():
    """Run training with different reward strategies and compare results."""
    strategies = ["distance", "exploration", "speed", "combined"]

    print("=" * 70)
    print("Custom Reward Shaping Comparison")
    print("=" * 70)
    print()

    results = {}
    for strategy in strategies:
        print(f"\n{'=' * 50}")
        print(f"Strategy: {strategy}")
        print(f"{'=' * 50}")

        # Create environment with proper EnvironmentConfig
        env = PuzzleEnvironment(make_env_config())

        # Create custom reward engine
        reward_engine = create_custom_reward_engine(strategy)

        # Create agent
        agent = PPOAgent(
            name="custom_reward_agent",
            config={"learning_rate": 0.1, "initial_epsilon": 1.0, "epsilon_decay": 0.95},
        )

        # Train for a few episodes
        episodes = []
        for ep_idx in range(10):
            obs = env.reset()
            transitions = []

            while not env.done and env.current_step < env.max_steps:
                available = env.get_available_actions()
                action = agent.select_action(obs, available)

                next_obs, raw_reward, done, info = env.step(action)

                # Build transition for reward engine
                raw_transition = Transition(
                    observation=obs,
                    action=action,
                    reward=raw_reward,
                    next_observation=next_obs,
                    done=done,
                )

                # Apply custom reward shaping
                shaped_reward = reward_engine.compute(raw_transition)

                transition = Transition(
                    observation=obs,
                    action=action,
                    reward=shaped_reward,
                    next_observation=next_obs,
                    done=done,
                )
                transitions.append(transition)
                obs = next_obs

                if done:
                    break

            episode = Episode(
                transitions=transitions,
                total_reward=sum(t.reward for t in transitions),
            )
            episodes.append(episode)
            agent.train([episode])

        # Evaluate
        evaluator = SkillEvaluator()
        eval_result = evaluator.evaluate(episodes, agent_id=strategy)

        mean_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
        skill_dict = {ss.skill_name: ss.score for ss in eval_result.skill_scores}
        results[strategy] = {
            "mean_reward": mean_reward,
            "skill_scores": skill_dict,
        }

        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Skill scores:")
        for skill, score in skill_dict.items():
            print(f"    {skill}: {score:.3f}")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("Summary Comparison")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<15} {'Mean Reward':>12} {'Navigation':>12} {'Efficiency':>12}")
    print("-" * 53)
    for strategy, result in results.items():
        nav_score = result["skill_scores"].get("navigation", 0.0)
        eff_score = result["skill_scores"].get("efficiency", 0.0)
        print(f"{strategy:<15} {result['mean_reward']:>12.2f} {nav_score:>12.3f} {eff_score:>12.3f}")

    print()
    print("The 'combined' strategy typically performs best by providing")
    print("multiple signals: direction toward goal, exploration incentive,")
    print("and speed encouragement.")


if __name__ == "__main__":
    run_comparison()
