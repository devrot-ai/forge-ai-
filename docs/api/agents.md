# Agents API

Reference documentation for all agent implementations.

## Agent Base Class

::: envgen.agents.base.Agent
    options:
      show_source: true
      show_root_heading: true

## PPOAgent

Simple table-based PPO agent.

::: envgen.agents.ppo_agent.PPOAgent
    options:
      show_source: true

## PPOAgentTorch

Neural network PPO agent with PyTorch.

::: envgen.agents.ppo_torch.PPOAgentTorch
    options:
      show_source: true

## GRPOAgent

Group Relative Policy Optimization agent.

::: envgen.agents.grpo_agent.GRPOAgent
    options:
      show_source: true

## GRPOAgentTorch

Neural network GRPO agent with PyTorch.

::: envgen.agents.grpo_torch.GRPOAgentTorch
    options:
      show_source: true

## RNNAgent

LSTM-based agent with memory.

::: envgen.agents.rnn_agent.RNNAgent
    options:
      show_source: true

## TransformerAgent

Self-attention based agent with Transformer encoder.

::: envgen.agents.transformer_agent.TransformerAgent
    options:
      show_source: true

## SB3Agent

Stable-Baselines3 wrapper agent.

::: envgen.agents.sb3_agent.SB3Agent
    options:
      show_source: true

## ReplayBuffer

Experience replay buffer with uniform and prioritized sampling.

::: envgen.agents.replay_buffer.ReplayBuffer
    options:
      show_source: true

## OnPolicyBuffer

On-policy trajectory buffer for PPO/GRPO.

::: envgen.agents.on_policy_buffer.OnPolicyBuffer
    options:
      show_source: true
