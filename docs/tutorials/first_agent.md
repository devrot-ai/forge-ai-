# Training Your First Agent

This guide walks you through training an agent from scratch, understanding the training loop, and evaluating performance.

## Understanding the Training Loop

EnvGen-LLM implements a self-improving training loop:

1. **Environment Generation** — An LLM creates a training environment with specific parameters
2. **Agent Training** — The agent trains in the environment for a set number of episodes
3. **Evaluation** — The agent's skills are measured across 5 dimensions
4. **Adaptation** — Based on weak skills, the LLM generates new targeted environments
5. **Curriculum Progression** — Difficulty increases as the agent improves

## Step 1: Choose Your Agent

Each agent type has different strengths:

| Agent | Best For | Dependencies |
|-------|----------|-------------|
| `ppo` | Quick prototyping, no dependencies | None |
| `ppo_torch` | Neural network training | PyTorch |
| `grpo_torch` | Group-relative optimization | PyTorch |
| `rnn` | Tasks requiring memory | PyTorch |
| `transformer` | Long-range temporal reasoning | PyTorch |
| `sb3_ppo` | Production-grade PPO | Stable-Baselines3 |

For this guide, we'll use `ppo_torch`:

```bash
pip install -e ".[torch]"
```

## Step 2: Choose Your Environment

| Environment | Task Type |
|-------------|-----------|
| `puzzle` | Grid navigation with items |
| `maze` | Maze navigation |
| `resource` | Resource management |
| `text` | Text-based reasoning |
| `cartpole` | Balancing control |
| `multi_agent` | Multi-agent competition |

For this guide, we'll use `maze`:

```bash
python examples/demo_pipeline.py --agent ppo_torch --env-type maze --cycles 5 --episodes 20
```

## Step 3: Monitor Training

During training, you'll see output like:

```
=== EnvGen-LLM Training Pipeline ===

Cycle 1/5
  Environment: maze_env (difficulty: EASY)
  Training 20 episodes...
  Total reward: 45.2
  Success rate: 0.35

  Evaluation:
    navigation: 0.42
    planning: 0.38
    reasoning: 0.30
    memory: 0.35
    efficiency: 0.40

  Curriculum: Advancing to MEDIUM difficulty

Cycle 2/5
  Environment: maze_env (difficulty: MEDIUM)
  ...
```

### Understanding Skill Scores

- **Navigation** (0-1): How well the agent moves toward goals
- **Planning** (0-1): How well the agent plans sequences of actions
- **Reasoning** (0-1): How well the agent solves logical problems
- **Memory** (0-1): How well the agent remembers past states
- **Efficiency** (0-1): How efficiently the agent uses steps

## Step 4: Save and Load Checkpoints

Training automatically saves checkpoints:

```bash
# Training with checkpoint saving
python -m envgen train --config configs/default.yaml --checkpoint-save-dir ./checkpoints
```

Resume from a checkpoint:

```bash
python -m envgen train --config configs/default.yaml --checkpoint-load-path ./checkpoints/latest.pt
```

## Step 5: Evaluate Your Agent

After training, evaluate your agent:

```bash
# Evaluate from checkpoint
python -m envgen evaluate --checkpoint ./checkpoints/latest.pt --episodes 50

# Evaluate across all environments
python -m envgen benchmark --agent ppo_torch --episodes 50
```

## Step 6: Experiment with Hyperparameters

Use the hyperparameter search to find optimal settings:

```bash
python -m envgen hyperparam-search --agent ppo_torch --episodes 20
```

Or manually adjust in your config:

```yaml
# configs/custom.yaml
agent:
  type: ppo_torch
  learning_rate: 0.0005
  hidden_size: 256
  gamma: 0.995
  epsilon: 0.1
```

## Advanced: Using Experiment Logging

Track your experiments with JSON logging:

```bash
python -m envgen train --config configs/default.yaml --log-dir ./logs
```

View results in the web dashboard:

```bash
python -m envgen dashboard --log-dir ./logs
```

Or use MLflow for advanced tracking:

```bash
pip install -e ".[mlflow]"
python -m envgen train --config configs/default.yaml --mlflow
```

## Tips for Better Training

1. **Start simple**: Begin with `ppo` agent and `puzzle` environment to understand the framework
2. **Increase gradually**: Start with 3 cycles and 10 episodes, then increase
3. **Monitor skills**: Pay attention to which skills are weak and adjust accordingly
4. **Use curriculum**: Let the curriculum manager handle difficulty progression
5. **Compare agents**: Use the benchmark to compare different agents on the same task

## Next Steps

- **[Creating Custom Environments](custom_env.md)** — Build your own environment types
- **[LLM Integration Guide](../guides/llm_integration.md)** — Connect real LLMs
- **[Experiment Tracking Guide](../guides/experiment_tracking.md)** — Track and compare experiments
- **[API Reference](../api/agents.md)** — Complete agent documentation
