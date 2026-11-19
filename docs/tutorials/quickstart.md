# Quickstart Tutorial

Get up and running with EnvGen-LLM in under 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/envgen-llm/envgen-llm.git
cd envgen-llm

# Install in development mode (no external dependencies required)
pip install -e .
```

### Optional Dependencies

Install additional features as needed:

```bash
# PyTorch agents (PPO-Torch, GRPO-Torch, RNN, Transformer)
pip install -e ".[torch]"

# LLM integration (OpenAI, Anthropic)
pip install -e ".[llm]"

# Gymnasium compatibility & Stable-Baselines3
pip install -e ".[gymnasium]"

# Experiment tracking
pip install -e ".[mlflow]"

# Web dashboard
pip install -e ".[dashboard]"

# All optional features
pip install -e ".[all]"
```

## Run Your First Training

The simplest way to start is running the demo script:

```bash
python examples/demo_pipeline.py
```

This runs a complete training cycle with:

- **Mock LLM** — Generates environments without API keys
- **PPO Agent** — Simple policy gradient agent
- **Puzzle Environment** — Grid-world navigation task
- **3 Training Cycles** — With curriculum progression

### Expected Output

```
=== EnvGen-LLM Training Pipeline ===
Cycle 1/3
  Environment: puzzle_env (difficulty: EASY)
  Training 10 episodes...
  Evaluation:
    navigation: 0.45
    planning: 0.32
    reasoning: 0.28
    memory: 0.35
    efficiency: 0.40
  ...
```

## Customize Your Training

### Choose a Different Agent

```bash
# PyTorch PPO with neural networks
python examples/demo_pipeline.py --agent ppo_torch

# GRPO with group-relative policy optimization
python examples/demo_pipeline.py --agent grpo_torch

# RNN/LSTM agent with memory
python examples/demo_pipeline.py --agent rnn

# Transformer agent with self-attention
python examples/demo_pipeline.py --agent transformer
```

### Choose a Different Environment

```bash
# Maze navigation
python examples/demo_pipeline.py --env-type maze

# Resource management
python examples/demo_pipeline.py --env-type resource

# Text-based reasoning
python examples/demo_pipeline.py --env-type text

# CartPole balancing
python examples/demo_pipeline.py --env-type cartpole

# Multi-agent competition
python examples/demo_pipeline.py --env-type multi_agent
```

### Adjust Training Parameters

```bash
# More training cycles
python examples/demo_pipeline.py --cycles 10

# More episodes per cycle
python examples/demo_pipeline.py --episodes 50

# Combine options
python examples/demo_pipeline.py --agent ppo_torch --env-type maze --cycles 5 --episodes 20
```

## Using the CLI

For more advanced configuration, use the YAML-based CLI:

```bash
# Train with default config
python -m envgen train --config configs/default.yaml

# Override parameters
python -m envgen train --config configs/default.yaml --cycles 10 --episodes 50

# Use a specific agent
python -m envgen train --config configs/default.yaml --agent ppo_torch
```

## Next Steps

- **[Training Your First Agent](first_agent.md)** — Detailed guide on training and evaluating agents
- **[Creating Custom Environments](custom_env.md)** — Build your own environment types
- **[LLM Integration Guide](../guides/llm_integration.md)** — Connect real LLMs for environment generation
- **[API Reference](../api/agents.md)** — Complete documentation for all components
