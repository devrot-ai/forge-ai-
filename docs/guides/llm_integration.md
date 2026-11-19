# LLM Integration Guide

Connect real LLMs to EnvGen-LLM for dynamic environment generation and adaptation.

## Overview

EnvGen-LLM supports four LLM backends:

| Backend | Description | Dependencies |
|---------|-------------|-------------|
| `mock` | Mock LLM for testing (no API keys) | None |
| `openai` | OpenAI GPT models | `openai` package |
| `ollama` | Local LLMs via Ollama | None (Ollama server required) |
| `anthropic` | Anthropic Claude models | `anthropic` package |

## Using OpenAI

### Installation

```bash
pip install -e ".[llm]"
```

### Configuration

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

```bash
# Via CLI
python -m envgen train --config configs/default.yaml --llm openai

# Via demo script
python examples/demo_pipeline.py --llm openai
```

### YAML Configuration

```yaml
# configs/openai_config.yaml
llm:
  type: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 1000
```

## Using Ollama (Local LLMs)

### Prerequisites

1. Install Ollama: <https://ollama.ai>
2. Pull a model:

```bash
ollama pull llama3.2
```

3. Start the Ollama server (starts automatically on most systems)

### Usage

```bash
# Via CLI (uses default model llama3.2)
python -m envgen train --config configs/default.yaml --llm ollama

# Specify a different model
python -m envgen train --config configs/default.yaml --llm ollama --llm-model mistral
```

### YAML Configuration

```yaml
llm:
  type: ollama
  model: llama3.2
  temperature: 0.7
  base_url: http://localhost:11434
```

## Using Anthropic

### Installation

```bash
pip install -e ".[llm]"
```

### Configuration

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Usage

```bash
python -m envgen train --config configs/default.yaml --llm anthropic
```

### YAML Configuration

```yaml
llm:
  type: anthropic
  model: claude-3-haiku-20240307
  temperature: 0.7
  max_tokens: 1000
```

## How LLMs Generate Environments

The LLM generates `EnvironmentConfig` objects based on:

1. **Task Description** — What the agent should learn
2. **Current Skill Scores** — Agent's strengths and weaknesses
3. **Curriculum Stage** — Current difficulty level
4. **Previous Environments** — History of generated environments

### Example LLM Prompt

```
Generate a training environment for an RL agent.

Task: Navigation and item collection
Current skills:
  navigation: 0.45
  planning: 0.32
  reasoning: 0.28
  memory: 0.35
  efficiency: 0.40

Weak skills: planning, reasoning
Difficulty: MEDIUM

Generate an EnvironmentConfig JSON with parameters that target
the agent's weak skills.
```

### Example LLM Response

```json
{
  "name": "planning_maze",
  "difficulty": "medium",
  "parameters": {
    "grid_size": 12,
    "num_items": 5,
    "num_obstacles": 8,
    "require_planning": true
  },
  "target_skills": ["planning", "reasoning"]
}
```

## Customizing LLM Prompts

You can customize the system prompt in `envgen/llm/generator.py`:

```python
from envgen.llm.generator import EnvironmentGenerator
from envgen.llm.openai_client import OpenAILLMClient

client = OpenAILLMClient(model="gpt-4", temperature=0.7)
generator = EnvironmentGenerator(client)

# Custom system prompt
generator.system_prompt = """
You are an expert RL environment designer. Generate challenging
environments that target the agent's weak skills while maintaining
a progressive difficulty curve.
"""
```

## Cost Considerations

| Backend | Cost |
|---------|------|
| Mock | Free |
| Ollama | Free (local) |
| OpenAI | Per-token pricing |
| Anthropic | Per-token pricing |

### Reducing Costs

1. **Use Ollama** for local, free environment generation
2. **Limit cycles** — Fewer cycles = fewer LLM calls
3. **Use smaller models** — GPT-3.5 or Claude Haiku are cheaper
4. **Cache environments** — Use the environment registry to reuse generated configs

## Troubleshooting

### "Connection refused" (Ollama)

Make sure Ollama is running:

```bash
ollama serve
```

### "Invalid API key" (OpenAI/Anthropic)

Check your environment variables:

```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### "Model not found"

Check available models:

```bash
# OpenAI
# See OpenAI documentation for available models

# Ollama
ollama list

# Anthropic
# See Anthropic documentation for available models
```

## Next Steps

- **[Experiment Tracking Guide](experiment_tracking.md)** — Track LLM-generated experiments
- **[API Reference](../api/llm.md)** — Complete LLM client documentation
- **[Creating Custom Environments](../tutorials/custom_env.md)** — Manual environment creation
