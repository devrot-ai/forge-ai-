# EnvGen-LLM: Self-Improving AI Agent Training

**EnvGen-LLM** is a research framework that uses Large Language Models (LLMs) to automatically generate, adapt, and optimize training environments for reinforcement learning agents.

## Key Features

- **6 Environment Types**: Puzzle, Maze, Resource, Text, CartPole, Multi-Agent
- **9+ Agent Types**: PPO, PPO-Torch, GRPO, GRPO-Torch, SB3, RNN/LSTM, Transformer
- **4 LLM Backends**: Mock, OpenAI, Ollama, Anthropic
- **Advanced Training**: Curriculum learning, early stopping, hyperparameter search, population-based training
- **Experiment Tracking**: JSON logging, MLflow, Weights & Biases
- **Visualization**: Web dashboard with real-time metrics
- **Full Tooling**: Docker, CI/CD, Python packaging, CLI

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/envgen-llm/envgen-llm.git
cd envgen-llm

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[torch,llm,gymnasium]"
```

### Run the Demo

```bash
# Run with mock LLM (no API keys needed)
python examples/demo_pipeline.py --cycles 3 --episodes 10

# Run with a specific agent
python examples/demo_pipeline.py --agent ppo_torch --cycles 3 --episodes 10

# Run with a specific environment
python examples/demo_pipeline.py --env-type maze --cycles 3 --episodes 10
```

### Using the CLI

```bash
# Train with YAML config
python -m envgen train --config configs/default.yaml

# Benchmark an agent
python -m envgen benchmark --agent ppo_torch --episodes 50

# Evaluate a checkpoint
python -m envgen evaluate --checkpoint checkpoints/agent.pt --episodes 20

# View experiment dashboard
python -m envgen dashboard --log-dir ./logs
```

## Project Structure

```
envgen-llm/
├── envgen/              # Main package
│   ├── agents/          # Agent implementations
│   ├── environments/    # Environment implementations
│   ├── llm/             # LLM clients
│   ├── pipeline/        # Training pipelines
│   ├── rewards/         # Reward engineering
│   ├── curriculum/      # Curriculum management
│   ├── evaluators/      # Performance evaluation
│   ├── benchmark/       # Benchmarking tools
│   ├── dashboard/       # Web dashboard
│   ├── config/          # YAML config loader
│   ├── cli/             # Command-line interface
│   └── utils/           # Utilities
├── examples/            # Example scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Test suite
├── configs/             # YAML configurations
├── docs/                # Documentation source
└── notebooks/           # Interactive tutorials
```

## Next Steps

- **[Quickstart Tutorial](tutorials/quickstart.md)** — Step-by-step introduction
- **[Training Your First Agent](tutorials/first_agent.md)** — Detailed training guide
- **[Creating Custom Environments](tutorials/custom_env.md)** — Build your own environments
- **[API Reference](api/agents.md)** — Complete API documentation

## Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/envgen-llm/envgen-llm/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/envgen-llm/envgen-llm/discussions)
- **License**: MIT License
