"""Ollama LLM client for local LLM integration.

This client connects to a local Ollama instance via its HTTP API to generate
text and structured JSON responses. It requires no external Python packages —
only the Python standard library (urllib.request).

Ollama must be running locally (default: http://localhost:11434) with the
specified model downloaded (e.g., `ollama pull llama3.2`).

Skills trained:
- reasoning: LLM-generated environments target specific reasoning skills
- planning: LLM adapts difficulty based on agent performance
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OllamaLLMClient:
    """LLM client that connects to a local Ollama instance.

    Uses the Ollama HTTP API (/api/chat endpoint) to generate text and JSON
    responses. No external dependencies required — uses only Python stdlib.

    Configuration:
    - model: Model name (default: 'llama3.2')
    - temperature: Sampling temperature (default: 0.7)
    - base_url: Ollama API base URL (default: 'http://localhost:11434')
    - timeout: Request timeout in seconds (default: 120)

    Example:
        client = OllamaLLMClient({"model": "llama3.2", "temperature": 0.7})
        response = client.generate("Generate a puzzle environment config")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ollama client.

        Args:
            config: Configuration dictionary with optional keys:
                - model: Model name (default: 'llama3.2')
                - temperature: Sampling temperature (default: 0.7)
                - base_url: Ollama API base URL
                - timeout: Request timeout in seconds
        """
        cfg = config or {}
        self.model: str = cfg.get("model", "llama3.2")
        self.temperature: float = cfg.get("temperature", 0.7)
        self.base_url: str = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.timeout: int = cfg.get("timeout", 120)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the Ollama chat API.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt for context

        Returns:
            Generated text response

        Raises:
            RuntimeError: If Ollama is not running or request fails
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running and model '{self.model}' is loaded. "
                f"Error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Ollama: {e}") from e

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON using the Ollama chat API.

        Adds a system prompt instructing the model to return valid JSON,
        then parses the response.

        Args:
            prompt: User prompt text
            system_prompt: Optional additional system prompt

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            RuntimeError: If Ollama is not running or response is not valid JSON
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
                f"Failed to parse JSON from Ollama response. " f"Response: {text[:200]}... Error: {e}"
            ) from e

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama API is reachable, False otherwise
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False
