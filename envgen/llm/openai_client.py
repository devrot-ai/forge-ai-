"""OpenAI LLM client for environment generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from envgen.llm.generator import LLMClient

logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client using the OpenAI Python SDK.

    Connects to OpenAI's chat completions API to generate environment
    configurations and adapt them based on evaluation feedback.
    Supports JSON mode for structured output.
    """

    # Default model and settings
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model name to use (default: gpt-4o-mini).
            temperature: Sampling temperature (0.0-2.0). Higher = more creative.
            max_tokens: Maximum tokens in the response.
            organization: Optional OpenAI organization ID.
            base_url: Optional base URL for custom endpoints (e.g., proxies).

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAILLMClient. " "Install it with: pip install openai"
            )

        # Resolve API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. Set the OPENAI_API_KEY environment variable " "or pass api_key explicitly."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        # Initialize the OpenAI client
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if organization:
            kwargs["organization"] = organization
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)
        self._call_count = 0
        logger.info(f"OpenAILLMClient initialized with model={model}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response from OpenAI.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            The LLM's response text.

        Raises:
            Exception: If the API call fails.
        """
        self._call_count += 1
        logger.info(f"OpenAI call #{self._call_count}: {prompt[:80]}...")

        messages = self._build_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.info(f"OpenAI response #{self._call_count}: {len(text)} chars")
            return text.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a JSON response from OpenAI using JSON mode.

        Uses OpenAI's response_format feature to encourage valid JSON output.
        Falls back to parsing the text response if JSON mode is not supported.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            Exception: If the API call fails or response is not valid JSON.
        """
        self._call_count += 1
        logger.info(f"OpenAI JSON call #{self._call_count}: {prompt[:80]}...")

        # Ensure system prompt encourages JSON output
        json_system_prompt = "Respond with valid JSON only. Do not include markdown formatting or explanations."
        if system_prompt:
            json_system_prompt = system_prompt + "\n\n" + json_system_prompt

        messages = self._build_messages(prompt, json_system_prompt)

        try:
            # Use JSON response format for models that support it
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or "{}"
            logger.info(f"OpenAI JSON response #{self._call_count}: {len(text)} chars")

            # Parse and return JSON
            result = json.loads(text.strip())
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse OpenAI response as JSON: {e}")
            logger.warning(f"Raw response: {text[:200]}")
            raise ValueError(f"OpenAI returned invalid JSON: {text[:200]}") from e
        except Exception as e:
            # JSON mode may not be supported by all models/endpoints
            logger.warning(f"JSON mode failed, falling back to text: {e}")
            try:
                text = self.generate(prompt, system_prompt)
                # Try to extract JSON from markdown code blocks
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except Exception as parse_err:
                logger.error(f"Failed to parse response as JSON: {parse_err}")
                raise

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        """Build the messages list for the chat API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            List of message dictionaries.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def call_count(self) -> int:
        """Return the number of API calls made."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the API call counter."""
        self._call_count = 0
