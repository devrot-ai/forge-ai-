# Multi-Agent Training Guide

Train multiple agents simultaneously in shared environments.

## Overview

EnvGen-LLM supports training multiple agents in the same environment with two modes:

| Mode | Description |
|------|-------------|
| **Competitive** | Agents compete against each other (zero-sum) |
| **Cooperative** | Agents work together toward a shared goal |

## Basic Usage

### Via CLI

```bash
# Train 2 agents competitively
python -m envgen train --config configs/default.yaml \
    --multi-agent --num-agents 2 --agent-mode competitive

# Train 3 agents cooperatively
python -m envgen train --config configs/default.yaml \
    --multi-agent --num-agents 3 --agent-mode cooperative
```

### Via Demo Script

```bash
python examples/demo_pipeline.py --env-type multi_agent --cycles 5 --episodes 20
```

## How It Works

### Competitive Mode

In competitive mode, agents compete for resources:

- Each agent receives inverse rewards (one's gain is another's loss)
- First agent to collect target items wins
- Opponent uses heuristic AI or a second trained agent

### Cooperative Mode

In cooperative mode, agents work together:

- All agents share the same reward signal
- Collective goal: collect all items efficiently
- Communication through shared observations

## Configuration

### YAML Configuration

```yaml
# configs/multi_agent.yaml
training:
  multi_agent: true
  num_agents: 2
  agent_mode: competitive  # or cooperative

agent:
  type: ppo_torch
  learning_rate: 0.001

environment:
  type: multi_agent
  parameters:
    grid_size: 12
    num_items: 10
    cooperation_mode: false
    visibility_range: 5
```

### Programmatic Configuration

```python
from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline
from envgen.config.loader import ConfigLoader

config = ConfigLoader.load("configs/multi_agent.yaml")
pipeline = MultiAgentPipeline(config)
pipeline.train()
```

## Environment Settings

### Grid Size

Larger grids provide more space for strategic play:

```yaml
parameters:
  grid_size: 16  # 8-24 recommended
```

### Number of Items

More items extend the game duration:

```yaml
parameters:
  num_items: 15  # 5-30 recommended
```

### Visibility Range

Controls how much of the grid each agent can see:

```yaml
parameters:
  visibility_range: 3  # 1-10, 0 = full visibility
```

## Training Tips

### Start Simple

1. Begin with 2 agents on a small grid (8x8)
2. Use cooperative mode first to verify setup
3. Gradually increase complexity

### Monitor Individual Performance

```python
# Each agent's stats are logged separately
# Check logs for agent_0, agent_1, etc.
```

### Balance Difficulty

- Competitive: Equal starting positions
- Cooperative: Distributed items for efficient collection

## Advanced: Custom Opponent Policy

Replace the default heuristic opponent:

```python
from envgen.pipeline.multi_agent_trainer import MultiAgentPipeline

pipeline = MultiAgentPipeline(config)

# Set custom opponent policy
def custom_opponent_policy(observation):
    # Your custom logic here
    return action

pipeline.opponent_policy = custom_opponent_policy
pipeline.train()
```

## Evaluation

### Benchmark Multi-Agent

```bash
python -m envgen benchmark --agent ppo_torch --env-type multi_agent --episodes 50
```

### Compare Modes

```bash
# Competitive
python -m envgen train --config configs/default.yaml \
    --multi-agent --agent-mode competitive --log-dir ./logs/competitive

# Cooperative
python -m envgen train --config configs/default.yaml \
    --multi-agent --agent-mode cooperative --log-dir ./logs/cooperative

# Compare in dashboard
python -m envgen dashboard --log-dir ./logs
```

## Use Cases

### Research

- Study emergent behavior in multi-agent systems
- Compare competitive vs cooperative learning
- Analyze communication patterns

### Applications

- Game AI development
- Robotics coordination
- Resource allocation optimization

## Next Steps

- **[Experiment Tracking Guide](experiment_tracking.md)** — Track multi-agent experiments
- **[API Reference](../api/pipeline.md)** — MultiAgentPipeline API
- **[Creating Custom Environments](../tutorials/custom_env.md)** — Build multi-agent environments
