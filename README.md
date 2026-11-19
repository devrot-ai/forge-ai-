# EnvGen-LLM

This project teaches AI agents how to learn by having another AI create their training challenges.

Imagine you want to train a robot to navigate mazes. Normally you would spend days designing those mazes, figuring out how hard they should be, and slowly making them tougher as the robot improves. That is boring and slow.

So instead, we asked a large language model to do it.

You tell the LLM what you want in plain English. It builds a training environment. An agent practices inside it. We score how well the agent did. Then the LLM reads those scores and builds a better, harder environment focused on what the agent struggled with. The whole thing repeats, and the agent gets better every round.

It is like having a personal coach who watches you train, sees where you are weak, and designs the next workout just for you.

## How it works

The whole system runs in a loop with four steps.

**Step one: The LLM creates a challenge.** You say something like "make a maze that tests navigation and memory." The LLM turns that into a real training environment with walls, items, and goals.

**Step two: The agent practices.** The agent explores the environment, tries different actions, gets rewards for good moves and penalties for bad ones, and slowly learns what works.

**Step three: We score the agent.** After training, an evaluator checks five skills: navigation, planning, reasoning, memory, and efficiency. Each gets a score from zero to one.

**Step four: The LLM adapts.** If memory scored low, the LLM creates a bigger maze with hidden shortcuts. If navigation was fine but planning was weak, it adds more obstacles that require thinking ahead.

Then we go back to step one. The agent keeps getting better.

```
LLM creates challenge
       |
       v
Agent practices
       |
       v
We score the agent
       |
       v
LLM sees what was hard and creates a better challenge
       |
       +----> (repeat)
```

## Getting started

Install the project:

```bash
pip install -e .
```

That is enough to run everything with a built-in mock LLM. No API keys needed.

If you want neural network agents or real LLM support, install the extras:

```bash
pip install -e ".[torch, llm]"
```

Run the demo:

```bash
python examples/demo_pipeline.py
```

You will see output like this:

```
Cycle 1/5
  Environment: Simple Grid Puzzle (Easy)
  Training 20 episodes...
  Evaluation results:
    Navigation:  0.62
    Planning:    0.48
    Reasoning:   0.55
    Memory:      0.38
    Efficiency:  0.60

Cycle 2/5
  Environment: Medium Grid Puzzle (Medium)
  Training 20 episodes...
  Evaluation results:
    Navigation:  0.71
    Planning:    0.55
    ...
```

The scores go up because the LLM keeps targeting the agent's weak spots.

To use a real LLM instead of the mock one, just add a flag:

```bash
python examples/demo_pipeline.py --llm openai
python examples/demo_pipeline.py --llm ollama
python examples/demo_pipeline.py --llm anthropic
```

## The big picture

Here are all the moving parts and how they connect:

The LLM is the coach. It creates training challenges based on your description and the agent's performance.

The environment is the gym. It is the actual challenge the agent practices in, like a maze or a puzzle grid.

The agent is the student. It learns by trying actions and getting rewards or penalties.

The evaluator is the teacher. It scores the agent on five different skills after each training round.

The reward engine shapes what the agent learns by giving points for good behavior and penalties for bad behavior.

The curriculum manager tracks difficulty and decides when to make things harder or easier.

```
+------------------+
|    LLM           |  Creates challenges
|   (Coach)        |
+--------+---------+
         |
         v
+--------+---------+
|   Environment    |  The actual challenge
|     (Gym)        |
+--------+---------+
         |
         v
+--------+---------+
|     Agent        |  Learns by trying things
|    (Student)     |
+--------+---------+
         |
         v
+--------+---------+     +--------------+
|   Evaluator      |---->|  Reward      |
|    (Teacher)     |     |   Engine     |
+--------+---------+     +--------------+
         |
         v
+--------+---------+
|   Curriculum   |  Decides next difficulty
|    (Tracker)   |
+--------+---------+
         |
         +----> Back to LLM with feedback
```

## Environments

Environments are the training grounds. Think of them as different types of gyms for your agent.

Puzzle: A grid where the agent moves around, collects items, and reaches a goal. It teaches basic navigation and planning.

Maze: A procedurally generated maze the agent must escape. It teaches spatial reasoning and memory.

Resource: The agent collects and manages limited resources to complete tasks. It teaches decision making under constraints.

Text: The agent answers math problems, logic puzzles, and pattern questions. It teaches reasoning and problem solving.

CartPole: A classic physics problem where the agent balances a pole on a moving cart. It teaches timing and continuous control.

Multi-Agent: Two agents on the same grid, either competing or cooperating. It teaches strategy and interaction.

Here is what a puzzle environment looks like:

```
+---+---+---+---+---+
|   | # |   | # |   |    # = wall
+---+---+---+---+---+    @ = agent
| @ |   | # |   | * |    * = item to collect
+---+---+---+---+---+    G = goal
|   | # |   | # |   |
+---+---+---+---+---+
| # |   | * |   | G |
+---+---+---+---+---+

The agent needs to navigate around walls, collect items, and reach the goal.
```

## Agent types

Different agents learn in different ways. Here is what each one does in simple terms.

PPO (Simple): Keeps a table of what worked before. Fast to run, no GPU needed. Good for quick experiments.

PPO-Torch: A neural network that recognizes patterns. This is real deep learning. Use it when you want the agent to generalize beyond what it has seen.

GRPO: Learns from groups of similar situations. Good when you want to compare multiple strategies at once.

GRPO-Torch: Neural network version of GRPO. Combines both ideas.

RNN/LSTM: Remembers what happened earlier. Has actual memory. Use it for tasks where the past matters, like remembering where items were.

Transformer: Looks at the entire history of what happened all at once. Best for complex tasks with long sequences of events.

Nearest-Neighbor: Asks "what worked in similar situations before?" A simple baseline for comparison.

SB3: Uses proven algorithms from the Stable-Baselines3 library. PPO, SAC, A2C, and DQN are all available.

A simple way to think about it:

PPO is like studying flashcards. You remember what worked last time.

RNN is like remembering a story from start to finish. The order matters.

Transformer is like reading the whole book, then answering questions about any part of it.

Nearest-Neighbor is like saying "last time I saw something like this, I did X and it worked."

## LLM backends

The LLM is the coach that designs training scenarios. You have four options.

Mock: Built-in simulator. No real AI, no API keys. Free and great for testing.

OpenAI: Uses GPT models via the OpenAI API. You pay per use but get powerful environment generation.

Ollama: Runs models on your own computer. Free if you have a good GPU. No data leaves your machine.

Anthropic: Uses Claude models via the Anthropic API. Another paid option with strong reasoning.

## Things you will want to do

Train an agent:

```bash
python -m envgen train --config configs/default.yaml
```

You can control every detail:

```bash
python -m envgen train --config configs/default.yaml \
    --agent ppo_torch \
    --env-type maze \
    --cycles 20 \
    --episodes 50
```

Compare different agents on the same task:

```bash
python examples/agent_comparison.py
```

See which environment is hardest for your agent:

```bash
python examples/multi_env_benchmark.py
```

Test a saved agent without retraining:

```bash
python -m envgen evaluate --checkpoint my_agent.json --episodes 50
```

Watch training progress in your browser:

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs
python -m envgen dashboard --log-dir ./logs
```

Then open http://localhost:8080 to see charts of how your agent improved.

Find the best settings automatically:

```bash
python -m envgen hyperparam-search --agent ppo_torch --episodes 10
```

Optimize reward weights:

```bash
python -m envgen optimize-rewards --agent ppo_torch --episodes 10
```

## Understanding the scores

When training runs, you will see five skill scores. Here is what each one means.

Navigation: Can the agent get from point A to point B without hitting walls?

Planning: Does the agent think ahead, or does it just react to what it sees right now?

Reasoning: Can the agent figure out rules and patterns in the environment?

Memory: Does the agent remember what it saw earlier in the episode?

Efficiency: Does the agent take the shortest path, or does it wander around wasting moves?

Scores range from zero to one. A score of zero point nine means the agent is really good at that skill. A score of zero point three means there is room to improve.

## Configuration

Everything is controlled through a YAML file. The default one is at configs/default.yaml:

```yaml
task:
  description: "Navigate a grid, collect items, reach the goal"
  skills: [navigation, planning, reasoning, memory, efficiency]

training:
  max_cycles: 10
  episodes_per_cycle: 50

agent:
  type: ppo_torch
  config:
    learning_rate: 0.001
    hidden_size: 128

environment:
  type: puzzle
  initial_difficulty: easy

llm:
  type: mock
```

Change the agent type, environment type, LLM backend, or any hyperparameter here. The CLI flags override these values when you run a command.

## Docker

You can run everything inside a container:

```bash
docker compose run --rm train
docker compose up dashboard
```

## Project structure

```
envgen-llm/
envgen/                  Main library
  agents/                AI agents
  environments/          Training scenarios
  llm/                   LLM integrations
  pipeline/              Training loop
  rewards/               Scoring system
  evaluators/            Skill measurement
  curriculum/            Difficulty progression
  benchmark/             Comparison tools
  dashboard/             Web UI
  cli/                   Command-line tools
  config/                YAML loader
  utils/                 Helpers
examples/                Demo scripts
tests/                   Test suite
notebooks/               Jupyter tutorials
docs/                    Documentation site
configs/                 Configuration files
benchmarks/              Performance tests
```

## Learn more

The notebooks are the best way to learn. Open them in Jupyter and run each cell:

notebooks/getting_started.ipynb walks you through the basics step by step.

notebooks/agent_comparison.ipynb compares all agent types side by side.

notebooks/environment_exploration.ipynb lets you try every environment.

notebooks/llm_integration.ipynb shows how to connect real LLM models.

For other resources:

TROUBLESHOOTING.md covers common problems and how to fix them.

CHANGELOG.md lists what changed in each version.

CONTRIBUTING.md explains how to help improve the project.

MIGRATION.md documents how to upgrade between versions.

## License

MIT. Do what you want with this code. See LICENSE for the full text.

## Need help

Check TROUBLESHOOTING.md first. If your issue is not there, open a GitHub issue and we will help.
