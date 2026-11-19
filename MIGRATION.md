# Migration Guide

This guide documents breaking changes and upgrade paths between EnvGen-LLM versions.

## Table of Contents

- [v0.1.0 → v0.2.0](#v010--v020)
- [v0.2.0 → v0.3.0](#v020--v030)
- [v0.3.0 → v0.4.0](#v030--v040)
- [v0.4.0 → v0.5.0](#v040--v050)
- [v0.5.0 → v0.6.0](#v050--v060)
- [v0.6.0 → v0.7.0](#v060--v070)

---

## v0.1.0 → v0.2.0

### Overview

v0.2.0 added the core infrastructure: real LLM integration, PyTorch agents, Gymnasium compatibility, YAML config, CLI, and checkpoints.

### New Features

- **OpenAI LLM Client**: `OpenAILLMClient` for real LLM-based environment generation
- **PyTorch PPO Agent**: `PPOAgentTorch` with neural network policy/value networks
- **Gymnasium Wrapper**: `GymnasiumEnvironment` for Gymnasium ecosystem compatibility
- **New Environments**: `MazeEnvironment` and `ResourceEnvironment`
- **GRPO Agent**: `GRPOAgent` implementing Group Relative Policy Optimization
- **YAML Config + CLI**: `python -m envgen train --config configs/default.yaml`
- **Persistent Checkpoints**: Save/load full pipeline state

### Breaking Changes

- **None** — All changes are additive.

### New Dependencies (Optional)

- `openai>=1.0` — For OpenAI LLM client
- `torch>=2.0` — For PyTorch agents
- `gymnasium>=0.29` — For Gymnasium compatibility
- `pyyaml>=6.0` — For YAML config loading
- `numpy>=1.24` — For array operations

### Migration Steps

1. Install optional dependencies as needed: `pip install torch numpy pyyaml`
2. Update imports if using new agents: `from envgen.agents import PPOAgentTorch`
3. Use CLI instead of direct Python scripts: `python -m envgen train --config configs/default.yaml`

---

## v0.2.0 → v0.3.0

### Overview

v0.3.0 added expanded test suite, PyTorch GRPO agent, and Stable-Baselines3 integration.

### New Features

- **PyTorch GRPO**: `GRPOAgentTorch` with neural network GRPO
- **Stable-Baselines3**: `SB3Agent` wrapping PPO, SAC, A2C, DQN algorithms
- **Expanded Tests**: 58+ tests covering all new components

### Breaking Changes

- **None** — All changes are additive.

### New Dependencies (Optional)

- `stable-baselines3>=2.0` — For SB3 agent integration

### Migration Steps

1. Install SB3 if needed: `pip install stable-baselines3 gymnasium`
2. Use SB3 agents via CLI: `python -m envgen train --config configs/default.yaml --agent sb3_ppo`

---

## v0.3.0 → v0.4.0

### Overview

v0.4.0 added TextTaskEnvironment, RNN/LSTM agent, reward optimizer, and multi-environment curriculum.

### New Features

- **TextTaskEnvironment**: Text-based reasoning puzzles (arithmetic, sequences, logic, patterns)
- **RNN/LSTM Agent**: `RNNAgent` with memory-based decision making
- **Reward Optimizer**: Automated grid search over reward weights
- **Multi-environment Curriculum**: Rotate through multiple environments during training

### Breaking Changes

- **None** — All changes are additive.

### Migration Steps

1. Use text environment: `python -m envgen train --config configs/default.yaml --env-type text`
2. Use RNN agent: `python -m envgen train --config configs/default.yaml --agent rnn`
3. Optimize rewards: `python -m envgen optimize-rewards --agent ppo_torch --episodes 10`

---

## v0.4.0 → v0.5.0

### Overview

v0.5.0 added Transformer agent, Automatic Reward Design, Transfer Learning, and Population-Based Training.

### New Features

- **Transformer Agent**: `TransformerAgent` with self-attention over observation sequences
- **Automatic Reward Design**: `AutoRewardDesigner` with curiosity-based intrinsic rewards
- **Transfer Learning**: `TransferLearningPipeline` for fine-tuning on new environments
- **Population-Based Training**: `PopulationBasedTrainer` for automated hyperparameter evolution

### Breaking Changes

- **None** — All changes are additive.

### Migration Steps

1. Use Transformer agent: `python -m envgen train --config configs/default.yaml --agent transformer`
2. Enable ARD in your agent config
3. Use PBT: `python -m envgen train --config configs/default.yaml --pbt`

---

## v0.5.0 → v0.6.0

### Overview

v0.6.0 added LLM-based curriculum, reward visualizer, multi-agent training example, Ollama client, CartPole environment, and Multi-agent environment.

### New Features

- **LLM Curriculum**: `LLMCurriculumManager` uses LLM for curriculum decisions
- **Reward Visualizer**: `RewardVisualizer` for experiment log analysis
- **Multi-agent Training**: `MultiAgentPipeline` for competitive/cooperative training
- **Ollama Client**: `OllamaLLMClient` for local LLM support (no external deps)
- **CartPole Environment**: Classic CartPole balancing task
- **Multi-agent Environment**: Two agents on shared grid

### Breaking Changes

- **None** — All changes are additive.

### Migration Steps

1. Use Ollama: `python -m envgen train --config configs/default.yaml --llm ollama`
2. Use CartPole: `python -m envgen train --config configs/default.yaml --env-type cartpole`
3. Multi-agent training: `python -m envgen train --config configs/default.yaml --multi-agent --num-agents 2`

---

## v0.6.0 → v0.7.0

### Overview

v0.7.0 adds Nearest-Neighbor agent as a simple baseline, multi-environment benchmark script, and this migration guide.

### New Features

- **Nearest-Neighbor Agent**: `NearestNeighborAgent` — non-parametric k-NN baseline
- **Multi-env Benchmark**: `examples/multi_env_benchmark.py` for cross-environment comparison
- **Migration Guide**: This document

### Breaking Changes

- **None** — All changes are additive.

### New API

```python
from envgen.agents import NearestNeighborAgent

agent = NearestNeighborAgent(
    name="nn_baseline",
    config={
        "k": 5,
        "distance_metric": "euclidean",
        "max_memory": 10000,
        "exploration_rate": 0.3,
    }
)
```

### CLI Usage

```bash
# Train with NN agent
python -m envgen train --config configs/default.yaml --agent nn

# Multi-environment benchmark
python examples/multi_env_benchmark.py --agent nn --episodes 10 --cycles 2
```

### Migration Steps

1. Use NN agent as a baseline for comparison: `--agent nn`
2. Run multi-env benchmark: `python examples/multi_env_benchmark.py`

---

## General Notes

### Optional Dependencies

All optional dependencies use graceful fallbacks. The core framework works with zero external dependencies. Install only what you need:

```bash
# LLM integration
pip install openai anthropic

# PyTorch agents
pip install torch numpy

# Gymnasium + SB3
pip install gymnasium stable-baselines3

# YAML config
pip install pyyaml

# Experiment tracking
pip install mlflow wandb

# Dashboard
pip install flask

# Development
pip install pytest black isort ruff bandit pre-commit
```

### Backward Compatibility

All versions maintain backward compatibility. Old code continues to work with new versions. New features are additive only.

### Version Checking

```python
import envgen
print(envgen.__version__)  # e.g., "0.7.0"
```
