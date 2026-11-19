# Docker Usage Guide

Run EnvGen-LLM in Docker containers for reproducible, isolated training environments.

## Prerequisites

- Docker Desktop installed
- Docker Compose (included with Docker Desktop)

## Quick Start

### Build and Run Training

```bash
# Build the image
docker build -t envgen-llm .

# Run training
docker run --rm -v $(pwd)/logs:/app/logs envgen-llm \
    python -m envgen train --config configs/default.yaml --log-dir /app/logs
```

### Using Docker Compose

```bash
# Start training service
docker-compose up train

# Start with dashboard
docker-compose up train dashboard
```

## Available Images

The Dockerfile supports multiple build targets:

| Target | Description | Build Command |
|--------|-------------|---------------|
| `base` | Core framework only | `docker build --target base -t envgen-llm:base .` |
| `dev` | Development with all tools | `docker build --target dev -t envgen-llm:dev .` |
| `full` | All optional dependencies | `docker build --target full -t envgen-llm:full .` |
| `gpu` | GPU support with CUDA | `docker build --target gpu -t envgen-llm:gpu .` |

## GPU Training

### Build GPU Image

```bash
docker build --target gpu -t envgen-llm:gpu .
```

### Run with GPU

```bash
docker run --rm --gpus all -v $(pwd)/logs:/app/logs envgen-llm:gpu \
    python -m envgen train --config configs/default.yaml --agent ppo_torch
```

## Volume Mounts

Mount volumes to persist data:

```bash
docker run --rm \
    -v $(pwd)/configs:/app/configs \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/checkpoints:/app/checkpoints \
    envgen-llm \
    python -m envgen train \
        --config /app/configs/default.yaml \
        --log-dir /app/logs \
        --checkpoint-save-dir /app/checkpoints
```

## Docker Compose Services

### Training

```yaml
# docker-compose.yml
services:
  train:
    build: .
    volumes:
      - ./configs:/app/configs
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    command: >
      python -m envgen train
      --config /app/configs/default.yaml
      --log-dir /app/logs
      --checkpoint-save-dir /app/checkpoints
```

### Dashboard

```yaml
  dashboard:
    build: --target full .
    ports:
      - "8080:8080"
    volumes:
      - ./logs:/app/logs
    command: python -m envgen dashboard --log-dir /app/logs
```

### MLflow

```yaml
  mlflow:
    build: --target full .
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    command: mlflow ui --backend-store-uri /app/mlruns
```

## Development Workflow

### Build Dev Image

```bash
docker build --target dev -t envgen-llm:dev .
```

### Run Interactive Shell

```bash
docker run --rm -it -v $(pwd)/..:/workspace envgen-llm:dev bash
```

### Run Tests

```bash
docker run --rm envgen-llm:dev python -m pytest tests/ -v
```

## Environment Variables

Pass API keys and configuration:

```bash
docker run --rm \
    -e OPENAI_API_KEY="your-key" \
    -e ANTHROPIC_API_KEY="your-key" \
    envgen-llm \
    python -m envgen train --config configs/default.yaml --llm openai
```

## Troubleshooting

### "No space left on device"

Clean up Docker images:

```bash
docker system prune -a
```

### GPU not detected

Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.key] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Port already in use

Change the port mapping:

```bash
docker run --rm -p 8081:8080 envgen-llm \
    python -m envgen dashboard --log-dir /app/logs --port 8080
```

## Next Steps

- **[Experiment Tracking Guide](experiment_tracking.md)** — Track experiments in Docker
- **[Multi-Agent Training Guide](multi_agent.md)** — Multi-agent in containers
- **[API Reference](../api/pipeline.md)** — Pipeline API documentation
