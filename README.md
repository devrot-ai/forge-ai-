# EnvGen-LLM: Self-Improving AI Agents

[![CI](https://github.com/user/envgen-llm/workflows/CI/badge.svg)](https://github.com/user/envgen-llm/actions)
[![Codecov](https://codecov.io/gh/user/envgen-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/user/envgen-llm)
[![PyPI](https://img.shields.io/pypi/v/envgen-llm.svg)](https://pypi.org/project/envgen-llm/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A framework that generates problems, creates test environments, trains agents to solve them,
evaluates their performance, and generates progressively harder tasks.

## Overview

EnvGen-LLM implements a **self-improving training pipeline** inspired by:
- **EnvGen** (Zala et al., COLM 2024): LLM-generated environments for RL agents
- **Microsoft Orchard-Agentic**: Agentic reinforcement learning
- **Meta HyperAgents**: Automated environment generation
- **OpenAI agent training**: GRPO and reward engineering

## Features

- **Real LLM Integration**: Connect to OpenAI (GPT-4o-mini), Anthropic (Claude), or local Ollama (llama3.2) for genuine environment generation, or use the built-in mock LLM for offline development
- **PyTorch PPO Agent**: Neural network-based PPO with GAE advantage estimation and clipped surrogate objective
- **GRPO Agent**: Group Relative Policy Optimization for alternative training strategies
- **PyTorch GRPO Agent**: Neural network-based GRPO with group-relative advantage estimation
- **Multiple Environments**: Grid puzzle navigation, procedural maze generation, resource management, text-based reasoning, CartPole balancing, and multi-agent competitive scenarios
- **Gymnasium Compatibility**: Wrap any environment for use with Stable-Baselines3, Ray RLlib, and other Gymnasium-compatible libraries
- **YAML Configuration**: Configure the entire pipeline via YAML files with CLI overrides
- **Persistent Checkpoints**: Save and resume training sessions with full state preservation
- **Experiment Logging**: Structured JSON-lines logging of training metrics for tracking and comparison
- **Benchmark Suite**: Standardized agent evaluation across all environments with reproducible seeds
- **Environment Registry**: Save, load, and manage generated environment configurations with metadata
- **Curriculum Learning**: Progressive difficulty management with automatic stage advancement
- **Web Dashboard**: Flask-based dashboard with Chart.js for visualizing training progress
- **MLflow Integration**: Optional MLflow tracking for experiment management
- **Weights & Biases Integration**: Optional W&B tracking for experiment management
- **Parallel Environments**: Run multiple environment instances in parallel for faster training
- **Early Stopping**: Automatically stop training when performance plateaus
- **Hyperparameter Search**: Automated grid search over agent hyperparameters with ranked results
- **Experience Replay Buffer**: Store and sample past transitions for off-policy learning
- **On-Policy Buffer**: Trajectory storage with mini-batch sampling for PPO/GRPO training
- **Multi-Agent Training**: Train multiple agents simultaneously in competitive or cooperative modes
- **Curriculum Logging**: Track difficulty progression over training cycles in experiment logs
- **Automatic Mixed Precision**: Faster GPU training with FP16 for PyTorch agents
- **RNN/LSTM Agent**: Recurrent neural network agent with memory for temporal reasoning tasks
- **Transformer Agent**: Self-attention based agent with multi-head attention over observation sequences for long-range temporal reasoning
- **Automatic Reward Design**: Intrinsic motivation via curiosity-based and count-based rewards for exploration
- **Transfer Learning**: Fine-tune pre-trained agents on new environments with layer freezing
- **Population-Based Training**: Evolve agent population with hyperparameter mutation and config sharing
- **Reward Function Optimizer**: Automated grid search over reward weights for optimal training signals
- **Multi-environment Curriculum**: Rotate through multiple environments during training for generalist agents
- **LLM-based Curriculum Manager**: Uses LLM to analyze agent performance and decide curriculum progression with reasoning
- **Reward Function Visualizer**: Analyze experiment logs to see reward breakdown and trends over training cycles
- **Nearest-Neighbor Agent**: Non-parametric k-NN baseline agent with experience memory for simple comparison against learned agents
- **Multi-environment Benchmark**: Cross-environment comparison script for training and evaluating agents across all environments
- **Multi-agent Training Example**: Complete demo showing competitive and cooperative multi-agent training
- **Python Packaging**: Installable via `pip install -e .` with optional dependency groups
- **Docker Support**: Multi-stage Dockerfile with CPU/GPU variants and docker-compose
- **CI/CD**: GitHub Actions workflow testing on Python 3.9–3.12

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Training Pipeline                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Env Generator│───▶│   Agent      │───▶│ Evaluator│  │
│  │  (LLM)       │    │  (RL/LLM)    │    │          │  │
│  └──────┬───────┘    └──────────────┘    │          │  │
│         │                                 └────┬─────┘  │
│         │                                      │        │
│         │                    ┌─────────────────┘        │
│         │                    │                          │
│  ┌──────▼───────┐    ┌──────▼───────┐                  │
│  │  Curriculum  │◀───│   Reward     │                  │
│  │  Manager     │    │   Engine     │                  │
│  └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install optional dependencies
pip install openai torch numpy gymnasium pyyaml

# Run the demo pipeline (mock LLM)
python examples/demo_pipeline.py

# Run with OpenAI (requires OPENAI_API_KEY)
python examples/demo_pipeline.py --llm openai --model gpt-4o-mini

# Run with local Ollama (requires Ollama running)
python examples/demo_pipeline.py --llm ollama --model llama3.2

# Train with YAML config
python -m envgen train --config configs/default.yaml --cycles 10 --episodes 50

# Train with checkpoint save/resume
python -m envgen train --config configs/default.yaml --checkpoint-save-dir ./checkpoints
```

## Command-Line Interface

```bash
# Basic usage
python -m envgen train --config configs/default.yaml

# Common options
python -m envgen train --config configs/default.yaml \
    --cycles 20 \
    --episodes 50 \
    --llm openai \
    --agent ppo_torch \
    --env-type maze

# Train with PyTorch GRPO agent
python -m envgen train --config configs/default.yaml \
    --agent grpo_torch \
    --cycles 20 \
    --episodes 50

# Resume from checkpoint
python -m envgen train --config configs/default.yaml --resume --checkpoint-load-dir ./checkpoints

# Train with local Ollama
python -m envgen train --config configs/default.yaml --llm ollama

# Train with CartPole environment
python -m envgen train --config configs/default.yaml --env-type cartpole

# Train with multi-agent environment
python -m envgen train --config configs/default.yaml --env-type multi_agent

# Train with W&B tracking
python -m envgen train --config configs/default.yaml --log-dir ./logs --wandb

# Train with AMP (mixed precision) for faster GPU training
python -m envgen train --config configs/default.yaml --agent ppo_torch

# Evaluate a trained agent from checkpoint
python -m envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json --episodes 50

# Train with experiment logging
python -m envgen train --config configs/default.yaml --log-dir ./logs

# Run benchmark evaluation
python -m envgen benchmark --agent ppo_torch --episodes 10

# Run benchmark on specific environments
python -m envgen benchmark --agent ppo_torch --episodes 10 --env-types puzzle,maze

# Train with Anthropic Claude (requires ANTHROPIC_API_KEY)
python -m envgen train --config configs/default.yaml --llm anthropic

# Train with MLflow tracking
python -m envgen train --config configs/default.yaml --log-dir ./logs --mlflow

# Launch web dashboard
python -m envgen dashboard --log-dir ./logs --port 8080

# Train with early stopping
python -m envgen train --config configs/default.yaml --early-stopping --patience 3

# Run hyperparameter search
python -m envgen hyperparam-search --agent ppo --episodes 10

# Custom hyperparameter grid
python -m envgen hyperparam-search --agent ppo_torch --param learning_rate=0.001,0.01 --param hidden_size=64,128

# Multi-agent training (competitive mode)
python -m envgen train --config configs/default.yaml --multi-agent --num-agents 2 --agent-mode competitive

# Multi-agent training (cooperative mode)
python -m envgen train --config configs/default.yaml --multi-agent --num-agents 3 --agent-mode cooperative

# Train with RNN/LSTM agent (memory-based)
python -m envgen train --config configs/default.yaml --agent rnn --cycles 10 --episodes 20

# Optimize reward function weights
python -m envgen optimize-rewards --agent ppo --episodes 10

# Reward optimization with detailed grid
python -m envgen optimize-rewards --agent ppo_torch --env-type maze --grid detailed --episodes 20
```

## Environments

| Environment | Description | Actions |
|---|---|---|
| **PuzzleEnvironment** | Grid-world navigation with items and goals | move, collect, check_goal |
| **MazeEnvironment** | Procedurally generated maze navigation | move (up/down/left/right) |
| **ResourceEnvironment** | Resource collection and task execution | collect, execute_task, allocate |
| **TextTaskEnvironment** | Text-based reasoning puzzles (arithmetic, sequences, logic, patterns) | select_answer, request_hint, submit |
| **CartPoleEnvironment** | Classic CartPole balancing with configurable physics | move (left/right/stay) |
| **MultiAgentEnvironment** | Two agents competing/cooperating on a shared grid | move (up/down/left/right), collect |

## Agents

| Agent | Description | Backend |
|---|---|---|
| **PPOAgent** | Simplified PPO with table-based Q-values | Standard library |
| **PPOAgentTorch** | Neural network PPO with GAE and clipped objective | PyTorch |
| **GRPOAgent** | Group Relative Policy Optimization | Standard library |
| **GRPOAgentTorch** | Neural network GRPO with group-relative advantages | PyTorch |
| **RNNAgent** | LSTM-based agent with memory for temporal reasoning | PyTorch |
| **TransformerAgent** | Self-attention agent with multi-head attention over observation sequences | PyTorch |
| **NearestNeighborAgent** | Non-parametric k-NN baseline with experience memory and distance-based retrieval | Standard library |
| **SB3Agent** | Stable-Baselines3 wrapper (PPO, SAC, A2C, DQN) | Stable-Baselines3 |
| **ReplayBuffer** | Experience replay with uniform and prioritized sampling | Standard library |

## Utilities

| Utility | Description |
|---|---|
| **ParallelEnvironment** | Run multiple environments in parallel worker processes for faster training |
| **EarlyStoppingMonitor** | Auto-stop training when performance plateaus (configurable patience) |
| **HyperparameterSearch** | Automated grid search over agent hyperparameters with ranked results |
| **ReplayBuffer** | Store and sample past transitions for off-policy learning |
| **OnPolicyBuffer** | Trajectory storage with mini-batch sampling and GAE return computation |
| **MultiAgentPipeline** | Train multiple agents simultaneously in competitive or cooperative modes |
| **RewardOptimizer** | Automated grid search over reward function weights with Pareto front analysis |
| **MultiEnvCurriculum** | Rotate through multiple environments during training for generalist agents |
| **AutoRewardDesigner** | Intrinsic motivation via curiosity-based and count-based rewards |
| **TransferLearningPipeline** | Fine-tune pre-trained agents on new environments with layer freezing |
| **PopulationBasedTrainer** | Evolve agent population with hyperparameter mutation and config sharing |

## LLM Backends

| Backend | Description |
|---|---|
| **MockLLMClient** | Built-in mock for offline development and testing |
| **OpenAILLMClient** | Real LLM via OpenAI API with JSON mode support |
| **OllamaLLMClient** | Local LLM via Ollama HTTP API (no external packages required) |
| **AnthropicLLMClient** | Real LLM via Anthropic Claude API with JSON mode support |

## Gymnasium Integration

```python
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.environments.gymnasium_wrapper import GymnasiumEnvironment
from envgen.environments.base import EnvironmentConfig
from envgen.utils.types import DifficultyLevel

# Create and wrap an environment
config = EnvironmentConfig(
    name='puzzle',
    description='Grid puzzle',
    difficulty=DifficultyLevel.EASY,
    parameters={'grid_size': 5, 'num_items': 3}
)
env = GymnasiumEnvironment(PuzzleEnvironment(config))

# Use with any Gymnasium-compatible library
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Project Structure

```
envgen-llm/
├── envgen/
│   ├── environments/     # Environment definitions & generators
│   │   ├── puzzle_env.py     # Grid puzzle navigation
│   │   ├── maze_env.py       # Procedural maze navigation
│   │   ├── resource_env.py   # Resource management tasks
│   │   ├── text_env.py       # Text-based reasoning puzzles
│   │   ├── cartpole_env.py   # CartPole balancing physics
│   │   ├── multi_agent_env.py  # Multi-agent competitive/cooperative
│   │   ├── gymnasium_wrapper.py  # Gymnasium compatibility
│   │   ├── parallel_env.py   # Parallel environment execution
│   │   └── registry.py       # Environment save/load registry
│   ├── agents/           # RL agent implementations
│   │   ├── ppo_agent.py      # Table-based PPO
│   │   ├── ppo_torch.py      # PyTorch neural network PPO
│   │   ├── grpo_agent.py     # Group Relative Policy Optimization
│   │   ├── grpo_torch.py     # PyTorch neural network GRPO
│   │   ├── sb3_agent.py      # Stable-Baselines3 wrapper
│   │   ├── rnn_agent.py          # RNN/LSTM agent with memory
│   │   ├── transformer_agent.py  # Transformer agent with self-attention
│   │   ├── nn_agent.py           # Nearest-neighbor baseline agent
│   │   └── replay_buffer.py      # Experience replay buffer
│   ├── evaluators/       # Performance evaluation & skill metrics
│   ├── curriculum/       # Progressive difficulty management
│   │   ├── manager.py          # Rule-based curriculum manager
│   │   ├── llm_curriculum.py   # LLM-based curriculum manager
│   │   └── multi_env_curriculum.py  # Multi-environment curriculum rotation
│   ├── rewards/          # Reward engineering & shaping
│   │   ├── engine.py         # Reward computation engine
│   │   └── auto_reward.py    # Automatic reward design (intrinsic motivation)
│   ├── pipeline/         # Main training loop & orchestration
│   │   ├── trainer.py            # Main training pipeline
│   │   ├── transfer_trainer.py   # Transfer learning pipeline
│   │   └── multi_agent_trainer.py  # Multi-agent training pipeline
│   ├── llm/              # LLM integration for env generation
│   │   ├── generator.py      # Environment generator
│   │   ├── openai_client.py  # OpenAI API client
│   │   ├── ollama_client.py  # Ollama HTTP API client (no external deps)
│   │   └── anthropic_client.py  # Anthropic Claude API client
│   ├── config/           # YAML config loading
│   ├── cli/              # Command-line interface
│   │   ├── train.py            # Training CLI
│   │   ├── benchmark.py        # Benchmark CLI
│   │   ├── evaluate.py         # Offline evaluation CLI
│   │   ├── dashboard.py        # Dashboard CLI
│   │   ├── hyperparam_search.py  # Hyperparameter search CLI
│   │   └── optimize_rewards.py   # Reward optimization CLI
│   ├── dashboard/        # Web dashboard for visualization
│   │   ├── server.py         # Flask web server
│   │   └── templates/        # HTML templates with Chart.js
│   ├── benchmark/        # Standardized benchmark suite
│   │   ├── runner.py           # Benchmark runner
│   │   ├── hyperparam_search.py  # Hyperparameter grid search
│   │   ├── reward_optimizer.py   # Reward weight optimization
│   │   └── pbt.py                # Population-based training
│   └── utils/            # Shared utilities
│       ├── logger.py             # Experiment logging (JSON lines)
│       ├── mlflow_logger.py      # MLflow integration
│       ├── wandb_logger.py       # Weights & Biases integration
│       ├── early_stopping.py     # Early stopping monitor
│       ├── reward_visualizer.py  # Reward function visualization
│       └── on_policy_buffer.py   # On-policy experience buffer
├── configs/              # Configuration files
├── examples/             # Demo scripts
│   ├── demo_pipeline.py          # Full pipeline demo
│   ├── multi_env_comparison.py   # Compare environments
│   ├── agent_comparison.py       # Compare agents
│   ├── custom_reward.py          # Custom reward shaping
│   ├── llm_env_generation.py     # LLM-generated environments
│   ├── custom_environment.py     # Custom environment tutorial
│   ├── multi_agent_training.py   # Multi-agent competitive/cooperative demo
│   └── multi_env_benchmark.py    # Cross-environment benchmark comparison
├── tests/                # Unit and integration tests (281+ tests)
├── notebooks/            # Jupyter notebooks
│   ├── getting_started.ipynb      # Interactive tutorial
│   ├── agent_comparison.ipynb     # Compare different agents
│   ├── environment_exploration.ipynb  # Explore all environments
│   └── llm_integration.ipynb      # LLM-generated environments
├── benchmarks/           # Performance benchmarks
│   ├── env_speed.py          # Environment speed benchmarks
│   └── agent_speed.py        # Agent training speed benchmarks
├── pyproject.toml        # Python packaging
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Container orchestration
└── .github/workflows/    # CI/CD (GitHub Actions)
```

## Configuration

Configure the pipeline via `configs/default.yaml`:

```yaml
task:
  description: "Navigate a grid world, collect items, and reach the goal"
  skills: [navigation, planning, reasoning, memory, efficiency]

training:
  max_cycles: 10
  episodes_per_cycle: 50
  eval_episodes: 15

agent:
  type: ppo_torch
  config:
    learning_rate: 0.001
    gamma: 0.99
    hidden_size: 128

environment:
  type: puzzle
  initial_difficulty: easy

llm:
  type: mock  # or "openai", "ollama", "anthropic"
```

## Web Dashboard

Visualize training progress with a web-based dashboard:

```bash
# After training with --log-dir, launch the dashboard
python -m envgen dashboard --log-dir ./logs --port 8080
```

The dashboard displays:
- Reward over cycles line chart
- Skill scores over cycles
- Final skill scores comparison table
- Multi-run comparison support

## MLflow Integration

Optional MLflow tracking for experiment management:

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs --mlflow
```

View experiments in MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns
```

## Weights & Biases Integration

Optional W&B tracking as an alternative to MLflow:

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs --wandb --wandb-project my-project
```

## Offline Evaluation

Evaluate a trained agent from a checkpoint without retraining:

```bash
# Evaluate on default environment
python -m envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json --episodes 50

# Evaluate on specific environments and difficulties
python -m envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json \
    --env-types puzzle,maze,cartpole \
    --difficulties easy,medium,hard \
    --episodes 20 \
    --output results.json
```

## Automatic Mixed Precision

Enable FP16 mixed precision for faster GPU training with PyTorch agents:

```python
# Via config
agent:
  type: ppo_torch
  config:
    use_amp: true
    learning_rate: 0.001
```

AMP automatically uses FP16 on GPU and falls back to FP32 on CPU.

## Python Packaging

Install as a proper Python package:

```bash
# Base installation (zero dependencies)
pip install -e .

# With optional dependencies
pip install -e ".[llm,torch,gymnasium]"

# All optional dependencies
pip install -e ".[all]"
```

After installation, the `envgen` command is available:
```bash
envgen train --config configs/default.yaml
envgen benchmark --agent ppo_torch --episodes 10
envgen dashboard --log-dir ./logs
envgen evaluate --checkpoint checkpoints/checkpoint_cycle_5.json
envgen hyperparam-search --agent ppo --episodes 10
```

## Docker Support

```bash
# Build and train
docker compose run --rm train

# Launch dashboard
docker compose up dashboard

# Launch MLflow tracking
docker compose up mlflow

# Development shell
docker compose run --rm dev
```

## Experiment Logging

Track training metrics across cycles with structured JSON-lines logging:

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs
```

Each run creates a timestamped directory with:
- `metrics.jsonl` — Per-cycle metrics (rewards, skill scores, difficulty, timing)
- `config.json` — Pipeline configuration used
- `summary.json` — Aggregate stats after training completes

## Benchmark Suite

Evaluate any agent across all environments with reproducible seeds:

```bash
# Benchmark PPO agent across all environments
python -m envgen benchmark --agent ppo_torch --episodes 10

# Benchmark specific environments
python -m envgen benchmark --agent grpo_torch --episodes 10 --env-types puzzle,maze

# Benchmark specific difficulties
python -m envgen benchmark --agent ppo_torch --episodes 10 --difficulties easy,medium
```

## Environment Registry

Save and load generated environment configurations for reproducibility:

```python
from envgen.environments.registry import EnvironmentRegistry
from envgen.environments.base import EnvironmentConfig
from envgen.utils.types import DifficultyLevel

registry = EnvironmentRegistry(registry_dir="./env_registry")

# Save a config
config = EnvironmentConfig(
    name="custom_puzzle",
    description="Custom puzzle environment",
    difficulty=DifficultyLevel.MEDIUM,
    parameters={"grid_size": 8, "num_items": 5},
)
env_id = registry.save(config, cycle=1, prompt="Generate a puzzle...")

# Load it back
loaded = registry.load(env_id)

# List all configs
configs = registry.list()
```

## Parallel Environments

Run multiple environment instances in parallel for faster training:

```python
from envgen.environments.parallel_env import ParallelEnvironment
from envgen.environments.puzzle_env import PuzzleEnvironment
from envgen.environments.base import EnvironmentConfig
from envgen.utils.types import DifficultyLevel

config = EnvironmentConfig(
    name="parallel_test",
    description="Parallel training",
    difficulty=DifficultyLevel.EASY,
    parameters={"grid_size": 5, "num_items": 3},
)

env = ParallelEnvironment(
    env_factory=lambda: PuzzleEnvironment(config),
    num_workers=4,
)

obs_list = env.reset()
action_list = [agent.select_action(obs) for obs in obs_list]
next_obs, rewards, dones, infos = env.step(action_list)
```

## Early Stopping

Automatically stop training when performance plateaus:

```bash
# Enable early stopping with default patience (3 cycles)
python -m envgen train --config configs/default.yaml --early-stopping

# Custom patience and metric
python -m envgen train --config configs/default.yaml \
    --early-stopping --patience 5 --early-stopping-metric mean_reward
```

## Hyperparameter Search

Automated grid search over agent hyperparameters:

```bash
# Search with default parameter grid
python -m envgen hyperparam-search --agent ppo --episodes 10

# Custom parameter grid
python -m envgen hyperparam-search --agent ppo_torch \
    --param learning_rate=0.001,0.01,0.1 \
    --param hidden_size=64,128,256 \
    --episodes 5

# Search on specific environment
python -m envgen hyperparam-search --agent grpo \
    --env-type maze --episodes 10
```

## Experience Replay Buffer

Store and sample past transitions for off-policy learning:

```python
from envgen.agents.replay_buffer import ReplayBuffer
from envgen.utils.types import Transition, Observation, Action

buffer = ReplayBuffer(capacity=10000, alpha=0.6)

# Add transitions
buffer.add(Transition(
    observation=Observation(state={"x": 0}),
    action=Action("move"),
    reward=1.0,
    next_observation=Observation(state={"x": 1}),
    done=False,
))

# Sample a batch (prioritized)
transitions, indices, weights = buffer.sample(batch_size=64, prioritized=True)

# Update priorities after training
for idx, td_error in zip(indices, td_errors):
    buffer.update_priority(idx, abs(td_error) + 1e-5)
```

## Research Areas

- Agentic reinforcement learning
- GRPO (Group Relative Policy Optimization)
- Automated environment generation
- Reward engineering

## Development

### Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Hooks include **black** (formatting), **isort** (import sorting), and **flake8** (linting).

### Performance Benchmarks

Measure environment and agent training speed:

```bash
# Benchmark agent training speed
python benchmarks/agent_speed.py

# Benchmark environment episode collection speed
python benchmarks/env_speed.py

# Benchmark specific agents
python -c "from benchmarks.agent_speed import benchmark_agent_speed; benchmark_agent_speed(['ppo', 'ppo_torch'], num_episodes=20)"
```

### Jupyter Notebooks

Interactive tutorials for learning the framework:

```bash
# Start Jupyter
jupyter notebook

# Available notebooks:
# - notebooks/getting_started.ipynb       - Step-by-step introduction
# - notebooks/agent_comparison.ipynb       - Compare PPO, GRPO, RNN, Transformer agents
# - notebooks/environment_exploration.ipynb - Explore all 6 environment types
# - notebooks/llm_integration.ipynb        - LLM-generated environments with different backends
```

### Publishing to PyPI

Releases are automated via GitHub Actions. To publish:

```bash
git tag v0.3.0
git push origin v0.3.0
```

The CI workflow builds the package, checks it with `twine`, and uploads to PyPI.

## Examples

```bash
# Full pipeline demo with mock LLM
python examples/demo_pipeline.py --cycles 5 --episodes 20

# Compare same agent across environments
python examples/multi_env_comparison.py

# Compare different agents on same environment
python examples/agent_comparison.py

# Custom reward shaping
python examples/custom_reward.py

# LLM-generated environments
python examples/llm_env_generation.py

# Custom environment tutorial
python examples/custom_environment.py

# Multi-environment benchmark comparison
python examples/multi_env_benchmark.py --agent ppo --episodes 10 --cycles 2
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Migration Guide

See [MIGRATION.md](MIGRATION.md) for breaking changes and upgrade paths between versions.
