# EnvGen-LLM Dockerfile
# Multi-stage build for training and serving

# Base image
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install base package
RUN pip install --no-cache-dir -e .

# Development image with all optional dependencies
FROM base AS dev
RUN pip install --no-cache-dir -e ".[all]"

# Full image with all optional dependencies
FROM base AS full
RUN pip install --no-cache-dir -e ".[llm,torch,gymnasium,mlflow,dashboard]"

# CPU-only image (smaller, no GPU)
FROM base AS cpu
RUN pip install --no-cache-dir -e ".[torch,gymnasium,llm]"

# GPU image with CUDA support
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[torch,gymnasium,llm]"
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu121

# Default to full image
FROM full

# Create logs directory
RUN mkdir -p /app/logs /app/checkpoints

# Expose dashboard port
EXPOSE 8080

# Default command
ENTRYPOINT ["envgen"]
CMD ["--help"]
