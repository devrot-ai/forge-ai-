# LLM Clients API

Reference documentation for LLM integration.

## LLMClient Base Class

::: envgen.llm.base.LLMClient
    options:
      show_source: true
      show_root_heading: true

## MockLLMClient

Mock LLM for testing without API keys.

::: envgen.llm.mock_client.MockLLMClient
    options:
      show_source: true

## OpenAILLMClient

OpenAI GPT integration.

::: envgen.llm.openai_client.OpenAILLMClient
    options:
      show_source: true

## OllamaLLMClient

Local LLM via Ollama HTTP API.

::: envgen.llm.ollama_client.OllamaLLMClient
    options:
      show_source: true

## AnthropicLLMClient

Anthropic Claude integration.

::: envgen.llm.anthropic_client.AnthropicLLMClient
    options:
      show_source: true

## EnvironmentGenerator

Uses LLM to generate and adapt environment configurations.

::: envgen.llm.generator.EnvironmentGenerator
    options:
      show_source: true
