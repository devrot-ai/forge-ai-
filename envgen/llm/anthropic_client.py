"""Anthropic LLM client for Claude integration.

This client connects to Anthropic's API to generate text and structured JSON
responses. It requires the 'anthropic' Python package.

Configuration:
- model: Model name (default: 'claude-3-haiku-20240307')
- temperature: Sampling temperature (default: 0.7)
- max_tokens: Maximum tokens in response (default: 4096)
- api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)

Example:
    client = AnthropicLLMClient({"model": "claude-3-haiku-20240307"})
    response = client.generate("Generate a puzzle environment config")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class AnthropicLLMClient:
    """LLM client that connects to Anthropic's API.

    Uses the Anthropic Messages API to generate text and JSON responses.
    Requires the 'anthropic' Python package and a valid API key.

    Configuration:
    - model: Model name (default: 'claude-3-haiku-20240307')
    - temperature: Sampling temperature (default: 0.7)
    - max_tokens: Maximum tokens in response (default: 4096)
    - api_key: Anthropic API key (optional; falls back to env var)

    Example:
        client = AnthropicLLMClient({"model": "claude-3-haiku-20240307"})
        response = client.generate("Generate a puzzle environment config")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Anthropic client.

        Args:
            config: Configuration dictionary with optional keys:
                - model: Model name (default: 'claude-3-haiku-20240307')
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 4096)
                - api_key: Anthropic API key
        """
        cfg = config or {}
        self.model: str = cfg.get("model", "claude-3-haiku-20240307")
        self.temperature: float = cfg.get("temperature", 0.7)
        self.max_tokens: int = cfg.get("max_tokens", 4096)
        self.api_key: str = cfg.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))

        # Lazy import
        self._client = None
        self._call_count: int = 0

    def _get_client(self):
        """Get or create the Anthropic client instance."""
        if self._client is None:
            try:
                import anthropic

                api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    raise RuntimeError(
                        "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                        "or pass 'api_key' in config."
                    )
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError as e:
                raise RuntimeError("Anthropic package not installed. Install it with: pip install anthropic") from e
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the Anthropic Messages API.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt for context

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API call fails or package is not installed
        """
        client = self._get_client()
        self._call_count += 1

        messages = [{"role": "user", "content": prompt}]
        system = system_prompt if system_prompt else None

        try:
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON using the Anthropic Messages API.

        Adds a system prompt instructing the model to return valid JSON,
        then parses the response.

        Args:
            prompt: User prompt text
            system_prompt: Optional additional system prompt

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            RuntimeError: If API call fails or response is not valid JSON
        """
        json_instruction = (
            "Return ONLY valid JSON. Do not include any explanation, "
            "markdown formatting, or code blocks. Just the raw JSON object."
        )

        if system_prompt:
            full_system = f"{system_prompt}\n\n{json_instruction}"
        else:
            full_system = json_instruction

        text = self.generate(prompt, system_prompt=full_system)

        # Try to extract JSON from the response
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown code block wrappers
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON from Anthropic response. " f"Response: {text[:200]}... Error: {e}"
            ) from e

    @property
    def call_count(self) -> int:
        """Return the number of API calls made."""
        return self._call_count
