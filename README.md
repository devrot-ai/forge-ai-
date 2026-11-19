# 🤖 EnvGen-LLM

**Teach AI agents to learn, by having AI create their training challenges.**

EnvGen-LLM is a framework where a large language model (like GPT or Claude) automatically creates training scenarios for smaller AI agents. The agents practice, get tested, and then the LLM creates *harder* challenges based on where they struggled. It's like having an AI coach that designs personalized workouts.

---

## 🎯 What Problem Does This Solve?

Training AI agents is hard. You need to:
1. **Design good training scenarios** (time-consuming)
2. **Know what skills to test** (requires expertise)
3. **Make things progressively harder** (trial and error)

EnvGen-LLM automates all of this. Instead of manually crafting training environments, you describe what you want in plain English, and an LLM builds it, tests your agent, and keeps improving the training based on results.

---

## 🔄 How It Works (The Simple Version)

```
   Step 1: LLM Creates Challenge
   ┌─────────────────────────┐
   │ "Make a maze that tests  │
   │  navigation and memory"  │
   └───────────┬─────────────┘
               ▼
   Step 2: Agent Practices
   ┌─────────────────────────┐
   │ Agent explores the maze │
   │ learns from mistakes    │
   └───────────┬─────────────┘
               ▼
   Step 3: Score the Agent
   ┌─────────────────────────┐
   │ Navigation: 0.7/1.0     │
   │ Memory:     0.4/1.0  ←  │
   │ Planning:   0.6/1.0     │
   └───────────┬─────────────┘
               ▼
   Step 4: LLM Makes It Harder
   ┌─────────────────────────┐
   │ "Memory is weak. Create  │
   │  a bigger maze with     │
   │  hidden shortcuts"      │
   └───────────┬─────────────┘
               ▼
         Repeat... Agent Gets Better!
```

**In technical terms:** This is called *self-improving reinforcement learning*. The LLM generates environments, the agent trains using reinforcement learning (RL), an evaluator scores 5 different skills, and the cycle repeats with increasingly difficult challenges.

---

## 🚀 Getting Started (5 Minutes)

### Install it

```bash
# Basic installation (works right away)
pip install -e .

# With extras for neural networks and real LLMs
pip install -e ".[torch,llm]"
```

### Run a demo

```bash
# This runs everything with a built-in mock LLM (no API keys needed)
python examples/demo_pipeline.py
```

You'll see output like this:

```
Cycle 1/5 ─────────────────────────────────
  Environment: "Simple Grid Puzzle" (Easy)
  Training 20 episodes...
  Evaluation results:
    Navigation:  ████████░░  0.62
    Planning:    ██████░░░░  0.48
    Reasoning:   ███████░░░  0.55
    Memory:      █████░░░░░  0.38
    Efficiency:  ████████░░  0.60

Cycle 2/5 ─────────────────────────────────
  Environment: "Medium Grid Puzzle" (Medium) ← Got harder!
  Training 20 episodes...
  Evaluation results:
    Navigation:  █████████░  0.71  ← Improved!
    Planning:    ███████░░░  0.55
    ...
```

### Use a real LLM (optional)

```bash
# With OpenAI (set OPENAI_API_KEY first)
python examples/demo_pipeline.py --llm openai

# With local Ollama (run Ollama first)
python examples/demo_pipeline.py --llm ollama

# With Anthropic Claude (set ANTHROPIC_API_KEY first)
python examples/demo_pipeline.py --llm anthropic
```

---

## 🏗️ The Big Picture

Here's how all the pieces fit together:

```
┌──────────────────────────────────────────────────────────────┐
│                     THE TRAINING LOOP                        │
│                                                              │
│   ┌─────────────┐                                           │
│   │   LLM       │  "Create a training challenge"             │
│   │  (Coach)    │                                           │
│   └──────┬──────┘                                           │
│          │ Environment Config                                │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ Environment │  The actual challenge (maze, puzzle, etc)  │
│   │  (Gym)      │                                           │
│   └──────┬──────┘                                           │
│          │ Agent practices here                              │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │    Agent    │  Learns by trying things & getting rewards │
│   │  (Student)  │                                           │
│   └──────┬──────┘                                           │
│          │ Episode data                                      │
│          ▼                                                   │
│   ┌─────────────┐     ┌─────────────┐                       │
│   │  Evaluator  │────▶│   Reward    │  Shapes what the      │
│   │ (Teacher)   │     │   Engine    │  agent learns to do   │
│   └──────┬──────┘     └─────────────┘                       │
│          │ Skill scores                                      │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ Curriculum  │  "Should we make it harder/easier?"       │
│   │  (Tracker)  │                                           │
│   └──────┬──────┘                                           │
│          │ Feedback for next round                           │
│          └──────────────┬───────────────────────────────────┘
│                         ▼                                    │
│                 Back to LLM with                            │
│                 "Agent struggled at X, focus on that"       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎮 What Can Agents Learn? (Environments)

Think of environments as different types of training gyms:

| Environment | What It Is | What It Teaches |
|---|---|---|
| **🧩 Puzzle** | A grid where the agent navigates, picks up items, and reaches a goal | Basic navigation and planning |
| **🏰 Maze** | A procedurally generated maze the agent must escape | Spatial reasoning and memory |
| **📦 Resource** | Collect and manage limited resources to complete tasks | Decision-making under constraints |
| **📝 Text** | Answer math problems, logic puzzles, and pattern questions | Reasoning and problem-solving |
| **🎪 CartPole** | Balance a pole on a moving cart (classic physics problem) | Timing and continuous control |
| **👥 Multi-Agent** | Two agents on the same grid, competing or cooperating | Strategy and interaction |

**Example — what a Puzzle environment looks like:**

```
  ┌───┬───┬───┬───┬───┐
  │   │ # │   │ # │   │   # = wall
  ├───┼───┼───┼───┼───┤   @ = agent (you)
  │ @ │   │ # │   │ * │   * = item to collect
  ├───┼───┼───┼───┼───┤   G = goal (finish here)
  │   │ # │   │ # │   │
  ├───┼───┼───┼───┼───┤
  │ # │   │ * │   │ G │
  └───┴───┴───┴───┴───┘

  The agent needs to: navigate around walls, collect items (*), and reach the goal (G)
```

---

## 🧠 Who Can Be the Agent? (Agent Types)

Different agents use different learning strategies. Think of them as different study methods:

| Agent | How It Learns | When to Use |
|---|---|---|
| **PPO** (Simple) | Keeps a table of what worked before | Quick experiments, no GPU needed |
| **PPO-Torch** | Neural network that recognizes patterns | When you want real deep learning |
| **GRPO** | Learns from groups of similar situations | When you want to compare multiple strategies |
| **GRPO-Torch** | Neural network version of GRPO | Best of both worlds |
| **RNN/LSTM** | Remembers what happened earlier (has memory) | Tasks where past matters |
| **Transformer** | Looks at the whole history at once | Complex tasks with long sequences |
| **Nearest-Neighbor** | "What worked in similar situations?" | Simple baseline for comparison |
| **SB3** | Uses battle-tested algorithms from Stable-Baselines3 | When you want proven methods |

**Simple analogy:**
- **PPO** = studying flashcards
- **RNN** = remembering a story from start to finish
- **Transformer** = reading the whole book, then answering questions
- **Nearest-Neighbor** = "Last time I saw this, I did X and it worked"

---

## 💬 Who Creates the Challenges? (LLM Backends)

The "coach" that designs training scenarios can be:

| Backend | What It Is | Cost |
|---|---|---|
| **Mock** | Built-in simulator (no real AI) | Free, for testing |
| **OpenAI** | GPT-4o-mini via API | Pay per use |
| **Ollama** | Runs on your computer | Free, needs good GPU |
| **Anthropic** | Claude via API | Pay per use |

---

## 🛠️ Common Things You'll Want to Do

### Train an agent

```bash
# Basic training
python -m envgen train --config configs/default.yaml

# With more control
python -m envgen train --config configs/default.yaml \
    --agent ppo_torch \        # Use neural network agent
    --env-type maze \          # Train on mazes
    --cycles 20 \              # 20 rounds of training
    --episodes 50              # 50 practice runs per round
```

### Compare agents

```bash
# See which agent performs best
python examples/agent_comparison.py
```

### Compare environments

```bash
# See which environment is hardest for your agent
python examples/multi_env_comparison.py
```

### Evaluate a saved agent

```bash
# Test a trained agent without retraining
python -m envgen evaluate --checkpoint my_agent.json --episodes 50
```

### Watch training progress in a browser

```bash
# Train with logging
python -m envgen train --config configs/default.yaml --log-dir ./logs

# Open the dashboard
python -m envgen dashboard --log-dir ./logs
# Then visit http://localhost:8080
```

### Find the best settings automatically

```bash
# Auto-search for the best learning rate, network size, etc.
python -m envgen hyperparam-search --agent ppo_torch --episodes 10
```

---

## 📊 Understanding the Output

When training runs, you'll see skill scores like this:

```
Skill Evaluation (Cycle 5/10):
  Navigation:  ██████████  0.92  ← Agent moves well
  Planning:    ████████░░  0.74  ← Agent plans decently
  Reasoning:   ███████░░░  0.65  ← Agent solves problems okay
  Memory:      ██████░░░░  0.58  ← Agent forgets sometimes
  Efficiency:  █████████░  0.85  ← Agent doesn't waste moves
```

**What these mean:**
- **Navigation** — Can the agent get from A to B without hitting walls?
- **Planning** — Does the agent think ahead, or just react?
- **Reasoning** — Can the agent figure out rules and patterns?
- **Memory** — Does the agent remember what it saw earlier?
- **Efficiency** — Does the agent take the shortest path, or wander around?

---

## 🔧 Configuration

Everything is controlled through a simple YAML file (`configs/default.yaml`):

```yaml
# What should the agent learn?
task:
  description: "Navigate a grid, collect items, reach the goal"
  skills: [navigation, planning, reasoning, memory, efficiency]

# How long to train?
training:
  max_cycles: 10           # Number of training rounds
  episodes_per_cycle: 50   # Practice runs per round

# Which agent to use?
agent:
  type: ppo_torch
  config:
    learning_rate: 0.001   # How fast it learns
    hidden_size: 128       # Size of the neural network

# What kind of challenges?
environment:
  type: puzzle
  initial_difficulty: easy

# Which LLM creates challenges?
llm:
  type: mock               # Change to "openai" for real AI
```

---

## 🐳 Run in Docker

```bash
# Train inside a container
docker compose run --rm train

# View the dashboard
docker compose up dashboard
```

---

## 📁 Project Layout

```
envgen-llm/
├── envgen/                  # Main library code
│   ├── agents/              # Different AI agents (PPO, GRPO, etc.)
│   ├── environments/        # Training scenarios (maze, puzzle, etc.)
│   ├── llm/                 # LLM integrations (OpenAI, Ollama, etc.)
│   ├── pipeline/            # The main training loop
│   ├── rewards/             # How agents get scored
│   ├── evaluators/          # Skill measurement
│   ├── curriculum/          # Difficulty progression
│   ├── benchmark/           # Testing and comparison tools
│   ├── dashboard/           # Web UI for viewing results
│   ├── cli/                 # Command-line tools
│   ├── config/              # YAML configuration loader
│   └── utils/               # Helper tools (logging, etc.)
├── examples/                # Ready-to-run demos
├── tests/                   # Test suite (316 tests)
├── notebooks/               # Interactive Jupyter tutorials
├── docs/                    # Full documentation site
├── configs/                 # Configuration files
├── benchmarks/              # Speed and performance tests
└── Dockerfile               # Container setup
```

---

## 📖 Learn More

- **[Getting Started Notebook](notebooks/getting_started.ipynb)** — Interactive tutorial
- **[Agent Comparison](notebooks/agent_comparison.ipynb)** — See which agent works best
- **[Environment Explorer](notebooks/environment_exploration.ipynb)** — Try all environments
- **[LLM Integration](notebooks/llm_integration.ipynb)** — Connect real AI models
- **[Full Documentation](https://your-docs-url)** — API reference and guides
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** — Common problems and fixes
- **[Changelog](CHANGELOG.md)** — What's new in each version
- **[Contributing](CONTRIBUTING.md)** — How to help improve the project

---

## 📜 License

MIT — do what you want with it. See [LICENSE](LICENSE) for details.

---

## 💡 Need Help?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Open an issue on GitHub for bugs or questions
- The project is actively maintained and welcomes contributions

Built with ❤️ for the AI research community.