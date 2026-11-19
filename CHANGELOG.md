# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-19

### 🎉 Production Release

EnvGen-LLM reaches v1.0.0 — a feature-complete, production-ready research framework for self-improving AI agent training via LLM-generated environments.

### Summary

- **281+ passing tests** across all modules
- **6 environments**: puzzle, maze, resource, text, cartpole, multi-agent
- **10+ agent types**: PPO, PPO-Torch, GRPO, GRPO-Torch, SB3 (PPO/SAC/A2C/DQN), RNN/LSTM, Transformer, Nearest-Neighbor
- **4 LLM backends**: Mock, OpenAI, Ollama, Anthropic
- **Full tooling**: dashboard, MLflow, W&B, CI/CD, Docker, packaging, benchmarks
- **Advanced training**: PBT, transfer learning, ARD, multi-agent, parallel envs, early stopping, hyperparameter search, replay buffer, on-policy buffer, AMP, curriculum logging, reward optimizer, reward visualizer, LLM curriculum, multi-env curriculum
- **Comprehensive documentation**: README, CHANGELOG, TROUBLESHOOTING, CONTRIBUTING, MIGRATION, 4 Jupyter notebooks, MkDocs site

### Added

#### Documentation
- **Jupyter Notebooks** — 3 new interactive notebooks: `agent_comparison.ipynb` (compare all agent types), `environment_exploration.ipynb` (explore all 6 environments), `llm_integration.ipynb` (demonstrate LLM backends)
- **Smoke Tests** — `tests/test_smoke.py` with 31 tests validating all imports, agent/environment creation, pipeline components, config loader, and version export
- **README Final Polish** — Updated test count, project structure diagram, all notebooks listed, all CLI commands documented

### Changed

- Version bumped from 0.7.0 to 1.0.0
- Project declared feature-complete and production-ready

## [0.7.0] - 2026-11-19

### Added

#### Agents
- **Nearest-Neighbor Agent** — Non-parametric k-NN baseline agent (`NearestNeighborAgent`) that stores (state, action, reward) tuples and selects actions by finding the k most similar past states. Supports euclidean and cosine distance metrics, configurable memory size, and exploration rate. Selectable via `--agent nn`.

#### Tools
- **Multi-environment Benchmark** — `examples/multi_env_benchmark.py` trains a single agent across all 6 environments with fixed seeds and produces formatted comparison tables with skill scores, rewards, and timing. Supports JSON output.

#### Documentation
- **Migration Guide** — `MIGRATION.md` documents breaking changes and upgrade paths from v0.1.0 through v0.7.0.

#### Developer Experience
- Test suite expanded from 274 to 284 tests
- 10 new tests for NearestNeighborAgent covering select_action, train, save/load, cosine distance, memory eviction, and config loader integration

### Changed

- Version bumped from 0.6.0 to 0.7.0

## [0.6.0] - 2026-11-19

### Added

#### Curriculum
- **LLM-based Curriculum Manager** — Uses an LLM to analyze agent performance and make curriculum decisions (advance/stay/regress) with reasoning. Falls back to rule-based CurriculumManager when LLM is unavailable. Tracks skill trends over time for informed decisions.

#### Tools
- **Reward Function Visualizer** — Reads experiment logs (JSON-lines) and extracts reward components over time. Shows reward breakdown (success, progress, step penalty, shaping) with trend analysis and multi-run comparison.

#### Examples
- **Multi-agent Training Example** — Complete demo (`examples/multi_agent_training.py`) showing competitive and cooperative multi-agent training with formatted comparison results.

#### Developer Experience
- Test suite expanded from 254 to 272 tests
- 20 new tests: 7 for LLM curriculum manager, 8 for reward visualizer, 3 for multi-agent training example
- All new features properly exported and documented

### Changed

- Version bumped from 0.5.0 to 0.6.0

## [0.5.0] - 2026-11-18

### Added

#### Agents
- **Transformer Agent** — Self-attention based agent with Transformer encoder for long-range temporal reasoning. Processes observation sequences with positional encoding and multi-head attention, enabling detection of complex sequential dependencies.

#### Training Pipeline
- **Automatic Reward Design (ARD)** — Intrinsic motivation via curiosity-based (forward model prediction error) and count-based (state visitation) rewards. Encourages exploration without manual reward engineering.
- **Transfer Learning Pipeline** — Fine-tune pre-trained agents on new environments with configurable learning rate, layer freezing, and partial weight transfer.
- **Population-Based Training (PBT)** — Evolve a population of agents with different hyperparameters. Periodically mutates worst performers and shares best configurations for automated hyperparameter optimization.

#### Developer Experience
- Test suite expanded from 231 to 259 tests
- New agent type: `--agent transformer`
- All new features fully documented in README

### Changed

- Version bumped from 0.4.0 to 0.5.0

## [0.4.0] - 2026-11-18

### Added

#### Agents
- **RNN/LSTM Agent** — Recurrent neural network agent with LSTM-based policy for temporal reasoning. Maintains hidden state across steps within episodes, enabling memory-based decision making.

#### Training Pipeline
- **Reward Function Optimizer** — Automated grid search over reward weights (success_weight, progress_weight, step_penalty, shaping_weight) with Pareto front analysis. Accessible via `python -m envgen optimize-rewards`.
- **Multi-environment Curriculum** — `MultiEnvCurriculumManager` rotates through multiple environments during training with round-robin, skill-based, and random strategies for generalist agent training.

#### Developer Experience
- Test suite expanded from 219 to 231 tests
- New CLI command: `python -m envgen optimize-rewards`
- All new features fully documented in README with CLI examples

### Changed

- Version bumped from 0.3.0 to 0.4.0

## [0.3.0] - 2025-11-18

### Added

#### Training Pipeline
- **On-Policy Buffer** — Trajectory storage with mini-batch sampling and GAE return computation for PPO/GRPO training
- **Multi-Agent Training** — Train multiple agents simultaneously in competitive or cooperative modes via `--multi-agent` flag
- **Curriculum Stage Logging** — Track difficulty progression over training cycles in experiment logs
- **Automatic Mixed Precision** — FP16 mixed precision training for PyTorch agents (PPOAgentTorch, GRPOAgentTorch) with `use_amp` config option
- **Weights & Biases Integration** — Optional W&B tracking for experiment management via `--wandb` flag
- **Offline Evaluation CLI** — Evaluate trained agents from checkpoints without retraining via `python -m envgen evaluate`

#### Developer Experience
- Test suite expanded from 185 to 211 tests
- All new features fully documented in README with CLI examples

### Changed

- Version bumped from 0.2.0 to 0.3.0

## [0.2.0] - 2024-01-01

### Added

#### LLM Integration
- **OpenAI LLM Client** — Real LLM integration via OpenAI's chat completions API with JSON mode for structured environment config generation
- **Ollama LLM Client** — Local LLM support via Ollama HTTP API using only Python stdlib (urllib), no external dependencies
- **Anthropic LLM Client** — Anthropic Claude integration via Messages API with JSON mode support

#### Agents
- **PyTorch PPO Agent** — Neural network-based PPO with MLP policy/value networks, GAE advantage estimation, and clipped surrogate objective
- **GRPO Agent** — Group Relative Policy Optimization with table-based Q-value lookup and group-relative advantage estimation
- **GRPO PyTorch Agent** — Neural network GRPO combining MLP policy/value networks with group-relative advantage estimation
- **Stable-Baselines3 Integration** — Wrapper supporting PPO, SAC, A2C, and DQN algorithms via Gymnasium compatibility
- **Experience Replay Buffer** — Uniform and prioritized sampling for off-policy learning

#### Environments
- **Maze Environment** — Procedural maze navigation with configurable size, complexity, and dead-end density
- **Resource Environment** — Resource collection and allocation under constraints
- **Text Task Environment** — Text-based reasoning puzzles (arithmetic, sequences, logic, patterns) with multiple-choice answers
- **CartPole Environment** — Classic CartPole balancing with configurable physics (gravity, pole length/mass, force magnitude)
- **Multi-agent Environment** — Two agents on a shared grid competing or cooperating to collect items with partial observability
- **Parallel Environment** — Multiprocessing-based wrapper for parallel environment execution
- **Gymnasium Compatibility** — Wrapper making any EnvGen environment compatible with the Gymnasium RL ecosystem
- **Environment Registry** — Save/load environment configs with metadata for reproducibility

#### Training Pipeline
- **YAML Config Support** — Full YAML configuration via `pyyaml` for all pipeline components
- **CLI** — Command-line interface via `python -m envgen train --config <path>` with flags for agent, environment, LLM type, etc.
- **Persistent Checkpoints** — Save/load/resume full pipeline state including agent weights, curriculum, and evaluation history
- **Experiment Logging** — JSON-lines metrics logging with timestamped run directories
- **Benchmark Suite** — Standardized evaluation across all environments and difficulty levels via `python -m envgen benchmark`
- **Hyperparameter Search** — Automated grid search over agent hyperparameters via `python -m envgen hyperparam-search`
- **Early Stopping** — Auto-stop training when performance plateaus, configurable patience

#### Developer Experience
- **Python Packaging** — `pyproject.toml` with optional dependency groups (llm, torch, gymnasium, mlflow, dashboard, dev)
- **Web Dashboard** — Flask-based dashboard with Chart.js for visualizing training metrics and comparing runs
- **MLflow Integration** — Optional MLflow tracking for experiment management
- **Docker Support** — Multi-stage Dockerfile with CPU/GPU variants and docker-compose
- **CI/CD** — GitHub Actions testing on Python 3.9–3.12 with flake8, isort, and coverage
- **Pre-commit Hooks** — Black, isort, and flake8 for consistent code quality
- **PyPI Publishing** — Automated releases on git tags via GitHub Actions
- **Jupyter Notebook** — Interactive getting-started tutorial
- **Example Scripts** — Multi-environment comparison, agent comparison, custom reward, custom environment, LLM env generation
- **Performance Benchmarks** — Timing-based benchmarks for environment and agent speed
- **Type Hints** — Mypy configuration for type checking
- **Code Coverage** — Coverage reporting with pytest-cov
- **MIT License** — Open-source licensing
- **CONTRIBUTING.md** — Contribution guidelines
- **TROUBLESHOOTING.md** — Common errors and solutions

### Changed

- Version bumped from 0.1.0 to 0.2.0
- All code formatted with Black (line length 120)
- Import sorting with isort (black profile)
- Expanded test suite from 26 to 185+ tests

### Fixed

- Graceful ImportError handling for all optional dependencies (torch, openai, anthropic, gymnasium, flask, mlflow)
- Core framework works with zero external dependencies

## [0.1.0] - 2024-01-01

### Added

- Core framework for self-improving AI agent training
- **Puzzle Environment** — Grid-world navigation with items, obstacles, and goals
- **PPO Agent** — Simplified PPO with value functions and epsilon-greedy exploration
- **Skill Evaluator** — Per-skill evaluation across navigation, planning, reasoning, memory, and efficiency
- **Curriculum Manager** — Progressive difficulty stages with advancement/regression
- **Reward Engine** — Composable reward components with configurable weights
- **Mock LLM Client** — Mock LLM for testing environment generation
- **Training Pipeline** — Orchestrator coordinating all components in the self-improving loop
- **Basic Tests** — 26 unit tests covering core components
- **Demo Pipeline** — End-to-end demo with mock LLM
- **README** — Project documentation with architecture overview
