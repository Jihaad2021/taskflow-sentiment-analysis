"""Anthropic Claude LLM implementation."""

import time
from typing import Any, Dict, Optional

from anthropic import Anthropic

from src.llm.base import BaseLLM
from src.utils.exceptions import LLMError
from src.utils.logger import setup_logger


class ClaudeLLM(BaseLLM):
    """Claude LLM implementation using Anthropic API."""

    # Pricing per million tokens (as of Jan 2025)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # NEW
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):  # CHANGE DEFAULT
        """Initialize Claude LLM.

        Args:
            api_key: Anthropic API key
            model: Claude model name
        """
        super().__init__(api_key, model)
        self._client = None  # Lazy initialization
        self.logger = setup_logger("ClaudeLLM")

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response from Claude."""
        try:
            start_time = time.time()

            # Add JSON instruction if needed
            if response_format == "json":
                prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No other text."

            # Call API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract content
            content = response.content[0].text

            # Count tokens
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens)

            execution_time = time.time() - start_time

            self.logger.info(
                f"Generated response: {input_tokens} in + {output_tokens} out "
                f"= ${cost:.4f} in {execution_time:.2f}s"
            )

            return {
                "content": content,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                "cost": cost,
                "execution_time": execution_time,
            }

        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise LLMError(f"Failed to generate response: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for tokens."""
        pricing = self.PRICING.get(self.model, {"input": 3.00, "output": 15.00})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
