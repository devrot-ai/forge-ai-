"""LLM integration for environment generation."""

from envgen.llm.generator import EnvironmentGenerator, LLMClient, MockLLMClient

try:
    from envgen.llm.openai_client import OpenAILLMClient
except ImportError:
    # OpenAI client is optional; requires 'openai' package
    OpenAILLMClient = None  # type: ignore

try:
    from envgen.llm.ollama_client import OllamaLLMClient
except ImportError:
    # Ollama client is optional; uses only stdlib but may fail in some envs
    OllamaLLMClient = None  # type: ignore

try:
    from envgen.llm.anthropic_client import AnthropicLLMClient
except ImportError:
    # Anthropic client is optional; requires 'anthropic' package
    AnthropicLLMClient = None  # type: ignore

__all__ = [
    "LLMClient",
    "MockLLMClient",
    "OpenAILLMClient",
    "OllamaLLMClient",
    "AnthropicLLMClient",
    "EnvironmentGenerator",
]
