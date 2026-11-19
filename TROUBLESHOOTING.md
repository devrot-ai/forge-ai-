# Troubleshooting Guide

Common issues users encounter when working with EnvGen-LLM and how to resolve them.

## Missing Optional Dependencies

### PyTorch not found

**Error**: `ImportError: torch is not installed`

**Solution**: Install PyTorch for neural network agents (PPOAgentTorch, GRPOAgentTorch):

```bash
pip install torch numpy
# Or install all optional dependencies
pip install -e ".[torch]"
```

### OpenAI client not found

**Error**: `ImportError: openai is not installed`

**Solution**: Install the OpenAI SDK:

```bash
pip install openai
# Or
pip install -e ".[llm]"
```

### Anthropic client not found

**Error**: `ImportError: anthropic is not installed`

**Solution**: Install the Anthropic SDK:

```bash
pip install anthropic
# Or
pip install -e ".[llm]"
```

### Gymnasium not found

**Error**: `ImportError: gymnasium is not installed`

**Solution**: Install Gymnasium for environment compatibility:

```bash
pip install gymnasium
# Or
pip install -e ".[gymnasium]"
```

### Stable-Baselines3 not found

**Error**: `ImportError: stable_baselines3 is not installed`

**Solution**: Install SB3 and its dependencies:

```bash
pip install stable-baselines3 gymnasium
# Or
pip install -e ".[gymnasium]"
```

### Flask not found (Dashboard)

**Error**: `ImportError: flask is not installed`

**Solution**: Install Flask for the web dashboard:

```bash
pip install flask
# Or
pip install -e ".[dashboard]"
```

### MLflow not found

**Error**: `ImportError: mlflow is not installed`

**Solution**: Install MLflow for experiment tracking:

```bash
pip install mlflow
# Or
pip install -e ".[mlflow]"
```

## Ollama Connection Errors

### Connection refused

**Error**: `Connection refused: Ollama is not running`

**Solution**: Start the Ollama service first:

```bash
# Install Ollama if not already installed
# https://ollama.ai/

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2
```

### Wrong port

**Error**: `Connection refused on port 11434`

**Solution**: Check if Ollama is running on a different port:

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If running on a different port, specify it
python -m envgen train --llm ollama --llm-config base_url=http://localhost:11435
```

## API Key Errors

### OpenAI API key missing

**Error**: `OpenAI error: no API key provided`

**Solution**: Set the `OPENAI_API_KEY` environment variable:

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Or pass via config
python -m envgen train --llm openai --llm-config api_key=sk-...
```

### Anthropic API key missing

**Error**: `Anthropic error: no API key provided`

**Solution**: Set the `ANTHROPIC_API_KEY` environment variable:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### Invalid API key

**Error**: `401 Unauthorized: Invalid API key`

**Solution**: Verify your API key is correct and has not expired. Check your account dashboard at:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

## ParallelEnvironment on Windows

### Multiprocessing spawn/pickle error

**Error**: `PicklingError: Cannot pickle` or hang during parallel environment startup

**Cause**: Windows uses `spawn` multiprocessing context by default, which requires pickling of the environment factory. Some environments with complex state cannot be pickled.

**Solution**: Use single environment mode on Windows, or ensure the environment factory is a top-level function:

```python
# This works — factory is a top-level function
def env_factory():
    return PuzzleEnvironment(config)

# This may fail — lambda cannot be pickled
parallel_env = ParallelEnvironment(env_factory=lambda: PuzzleEnvironment(config))
```

Alternatively, run with a single worker:

```bash
python -m envgen train --workers 1
```

## Docker GPU Setup

### NVIDIA container toolkit not installed

**Error**: `CUDA error: no kernel image is available` or GPU not detected in container

**Solution**: Install the NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Running with GPU

```bash
# Build and run with GPU support
docker build --target gpu -t envgen-llm .
docker run --gpus all -v $(pwd)/logs:/app/logs envgen-llm train --agent ppo_torch
```

## Checkpoint Load Failures

### Corrupted checkpoint file

**Error**: `JSONDecodeError` or `FileNotFoundError` when loading checkpoint

**Solution**: Verify the checkpoint file exists and is valid JSON:

```bash
# Check if file exists
ls checkpoints/checkpoint_cycle_5.json

# Validate JSON
python -c "import json; json.load(open('checkpoints/checkpoint_cycle_5.json'))"
```

### Version mismatch

**Error**: `KeyError` or `AttributeError` when loading checkpoint from a different version

**Solution**: Ensure you're using the same version of EnvGen-LLM that created the checkpoint. Check the version in the checkpoint file:

```python
import json
with open("checkpoints/checkpoint_cycle_5.json") as f:
    data = json.load(f)
    print(data.get("version", "unknown"))
```

### PyTorch checkpoint not found

**Error**: `FileNotFoundError: agent_model.pt`

**Solution**: Make sure the `.pt` file is in the same directory as the JSON checkpoint:

```bash
ls checkpoints/
# Should contain both checkpoint_cycle_N.json and agent_model.pt
```

## SB3 Import Errors

### Missing Gymnasium dependency

**Error**: `ImportError: Gymnasium >=0.29 is required`

**Solution**: Install Gymnasium before SB3:

```bash
pip install "gymnasium>=0.29" "stable-baselines3>=2.0"
```

### Algorithm not found

**Error**: `NameError: name 'SAC' is not defined`

**Solution**: Make sure you're using the correct agent type string:

```bash
# Valid SB3 agent types
--agent sb3_ppo
--agent sb3_sac
--agent sb3_a2c
--agent sb3_dqn
```

## General Tips

### Core framework with zero dependencies

The core framework works without any optional dependencies. If you just want to run the basic demo:

```bash
# No extra packages needed
python examples/demo_pipeline.py --cycles 3 --episodes 10
```

### Installing all optional dependencies

To get everything at once:

```bash
pip install -e ".[all]"
```

### Checking installed version

```bash
python -c "import envgen; print(envgen.__version__)"
```

### Running tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run tests without parallel (avoids Windows multiprocessing issues)
python -m pytest tests/ -v -k "not parallel"
```
