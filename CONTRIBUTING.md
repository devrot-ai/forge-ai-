# Contributing to EnvGen-LLM

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/envgen-llm/envgen-llm.git
   cd envgen-llm
   ```

2. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[all]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- **Formatting**: [Black](https://black.readthedocs.io/) with line length 120
- **Import sorting**: [isort](https://pycqa.github.io/isort/) with black profile
- **Linting**: [flake8](https://flake8.pycqa.org/)

Run formatters before committing:
```bash
black envgen/ tests/ examples/
isort envgen/ tests/ examples/
flake8 envgen/ tests/ examples/
```

Or use pre-commit hooks which run automatically on `git commit`.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=envgen --cov-report=term-missing

# Run specific test file
pytest tests/test_basic.py -v
```

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure code passes linting: `flake8 envgen/`
5. Submit a pull request with a clear description of changes

## Project Structure

```
envgen-llm/
├── envgen/              # Main package
│   ├── agents/          # RL agent implementations
│   ├── environments/    # Environment implementations
│   ├── evaluators/      # Skill evaluation
│   ├── llm/             # LLM client implementations
│   ├── pipeline/        # Training pipeline
│   ├── curriculum/      # Curriculum management
│   ├── rewards/         # Reward engineering
│   ├── config/          # YAML config loading
│   ├── cli/             # CLI commands
│   ├── benchmark/       # Benchmarking tools
│   ├── dashboard/       # Web dashboard
│   └── utils/           # Utilities
├── tests/               # Test suite
├── examples/            # Example scripts
├── benchmarks/          # Performance benchmarks
├── configs/             # YAML configuration files
└── notebooks/           # Jupyter notebooks
```
