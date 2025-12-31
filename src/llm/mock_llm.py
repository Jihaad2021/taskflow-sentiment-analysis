"""Mock LLM for testing without API calls."""

from typing import Dict


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self):
        """Initialize mock LLM."""
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate mock response.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments (ignored)

        Returns:
            Mock response dictionary
        """
        self.call_count += 1
        self.last_prompt = prompt

        # Extract first column name from prompt
        # Format: "1. Column: 'column_name'"
        import re

        match = re.search(r"1\. Column: '([^']+)'", prompt)

        if match:
            column_name = match.group(1)
        else:
            column_name = "unknown"

        return {
            "content": f'{{"column": "{column_name}", "reasoning": "Mock LLM selected first candidate"}}',
            "tokens": {"input": 100, "output": 50},
            "cost": 0.001,
        }
