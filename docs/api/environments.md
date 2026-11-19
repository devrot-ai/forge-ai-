# Environments API

Reference documentation for all environment implementations.

## Environment Base Class

::: envgen.environments.base.Environment
    options:
      show_source: true
      show_root_heading: true

## PuzzleEnvironment

Grid-world navigation with items and goals.

::: envgen.environments.puzzle_env.PuzzleEnvironment
    options:
      show_source: true

## MazeEnvironment

Procedural maze navigation.

::: envgen.environments.maze_env.MazeEnvironment
    options:
      show_source: true

## ResourceEnvironment

Resource management under constraints.

::: envgen.environments.resource_env.ResourceEnvironment
    options:
      show_source: true

## TextTaskEnvironment

Text-based reasoning puzzles.

::: envgen.environments.text_env.TextTaskEnvironment
    options:
      show_source: true

## CartPoleEnvironment

Classic CartPole balancing.

::: envgen.environments.cartpole_env.CartPoleEnvironment
    options:
      show_source: true

## MultiAgentEnvironment

Multi-agent competitive/cooperative grid.

::: envgen.environments.multi_agent_env.MultiAgentEnvironment
    options:
      show_source: true

## GymnasiumEnvironment

Gymnasium compatibility wrapper.

::: envgen.environments.gymnasium_wrapper.GymnasiumEnvironment
    options:
      show_source: true

## ParallelEnvironment

Parallel environment execution with multiprocessing.

::: envgen.environments.parallel_env.ParallelEnvironment
    options:
      show_source: true

## EnvironmentRegistry

Save and load environment configurations.

::: envgen.environments.registry.EnvironmentRegistry
    options:
      show_source: true
