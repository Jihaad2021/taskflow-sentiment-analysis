"""Mock LLM for testing without API calls."""

import re
from typing import Any, Dict, Optional

from src.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM that returns predefined responses."""

    call_count: int
    last_prompt: Optional[str]

    def __init__(self, api_key: str = "mock_key", model: str = "mock-model"):
        """Initialize mock LLM."""
        super().__init__(api_key, model)
        self.call_count = 0
        self.last_prompt = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate mock response."""
        self.call_count += 1
        self.last_prompt = prompt

        # Generate appropriate mock response
        if response_format == "json":
            content = self._generate_json_response(prompt)
        else:
            content = self._generate_text_response(prompt)

        # Mock token counts
        input_tokens = len(prompt) // 4
        output_tokens = len(content) // 4

        return {
            "content": content,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            "cost": 0.001,  # Mock cost
            "execution_time": 0.1,
        }

    def _generate_json_response(self, prompt: str) -> str:
        """Generate mock JSON response."""
        # For column detection
        if "Column:" in prompt:
            match = re.search(r"1\. Column: '([^']+)'", prompt)
            column = match.group(1) if match else "unknown"
            return f'{{"column": "{column}", "reasoning": "Mock LLM selection"}}'

        # Default JSON
        return '{"result": "mock response"}'

    def _generate_text_response(self, prompt: str) -> str:
        """Generate mock text response."""

        # For report writing
        if "Write a complete, professional report" in prompt or "Report Outline" in prompt:
            return self._generate_mock_report()

        return "This is a mock LLM response for testing purposes."

    def _generate_mock_report(self) -> str:
        """Generate mock report in Markdown."""
        return """# Sentiment Analysis Report

## Executive Summary

This report analyzes customer feedback from 100 comments. The analysis reveals predominantly positive sentiment (60%) with key themes around product quality and customer service.

**Key Findings:**
- 60% positive sentiment
- Quality mentioned in 40 comments
- Strong customer satisfaction

## Sentiment Analysis

The sentiment distribution shows:
- **Positive**: 60 comments (60%)
- **Negative**: 30 comments (30%)
- **Neutral**: 10 comments (10%)

This indicates overall positive customer perception.

## Key Topics

### Quality
Quality was the most discussed topic with 40 mentions and positive sentiment (0.7).

### Service
Customer service received 30 mentions with moderately positive feedback (0.5).

## Recommendations

Based on the analysis:
1. Continue focus on quality maintenance
2. Improve customer service response times
3. Address negative feedback promptly

## Conclusion

Overall customer sentiment is positive with clear areas for improvement in service delivery."""

    def count_tokens(self, text: str) -> int:
        """Count tokens (mock: ~4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost (mock: always $0.001)."""
        return 0.001
