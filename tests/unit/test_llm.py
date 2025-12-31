"""Test LLM implementations."""

import os

import pytest

from src.llm.anthropic_llm import ClaudeLLM
from src.llm.mock_llm import MockLLM


@pytest.fixture
def mock_llm():
    """Create MockLLM instance."""
    return MockLLM()


def test_mock_llm_initialization(mock_llm):
    """Test MockLLM can be initialized."""
    assert mock_llm is not None
    assert mock_llm.model == "mock-model"
    assert mock_llm.call_count == 0


def test_mock_llm_generate(mock_llm):
    """Test MockLLM generate method."""
    result = mock_llm.generate("Test prompt")

    assert "content" in result
    assert "tokens" in result
    assert "cost" in result
    assert mock_llm.call_count == 1
    assert mock_llm.last_prompt == "Test prompt"


def test_mock_llm_json_response(mock_llm):
    """Test MockLLM JSON response format."""
    result = mock_llm.generate("Test", response_format="json")

    content = result["content"]
    assert "{" in content
    assert "}" in content
    # Should be valid JSON structure
    assert '"' in content


def test_mock_llm_token_counting(mock_llm):
    """Test token counting."""
    text = "This is a test prompt with some words"
    count = mock_llm.count_tokens(text)

    # Should approximate ~4 chars per token
    assert count > 0
    assert count == len(text) // 4


def test_mock_llm_cost_calculation(mock_llm):
    """Test cost calculation."""
    cost = mock_llm.calculate_cost(100, 50)

    assert cost > 0
    assert isinstance(cost, float)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_llm_real_api():
    """Test ClaudeLLM with real API (only if key present)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    llm = ClaudeLLM(api_key=api_key)

    result = llm.generate("Say hello in one word.", max_tokens=10)

    assert "content" in result
    assert len(result["content"]) > 0
    assert result["tokens"]["input"] > 0
    assert result["tokens"]["output"] > 0
    assert result["cost"] > 0


def test_claude_llm_initialization():
    """Test ClaudeLLM initialization."""
    llm = ClaudeLLM(api_key="fake_key")

    assert llm is not None
    assert llm.model == "claude-3-5-sonnet-20241022"


def test_claude_llm_token_counting():
    """Test Claude token counting."""
    llm = ClaudeLLM(api_key="fake_key")

    text = "This is a test prompt"
    count = llm.count_tokens(text)

    assert count > 0
    assert count == len(text) // 4


def test_claude_llm_cost_calculation():
    """Test Claude cost calculation."""
    llm = ClaudeLLM(api_key="fake_key")

    # 1000 input + 500 output tokens
    cost = llm.calculate_cost(1000, 500)

    # Should be: (1000/1M * 3.00) + (500/1M * 15.00)
    expected = (1000 / 1_000_000 * 3.00) + (500 / 1_000_000 * 15.00)
    assert abs(cost - expected) < 0.0001


def test_claude_llm_pricing_exists():
    """Test that pricing is defined for models."""
    llm = ClaudeLLM(api_key="fake_key")

    assert llm.model in llm.PRICING
    pricing = llm.PRICING[llm.model]
    assert "input" in pricing
    assert "output" in pricing
    assert pricing["input"] > 0
    assert pricing["output"] > 0


def test_mock_llm_column_detection_response(mock_llm):
    """Test MockLLM column detection response."""
    prompt = "1. Column: 'comment'\n2. Column: 'text'"

    result = mock_llm.generate(prompt, response_format="json")
    content = result["content"]

    # Should extract first column
    assert "comment" in content
    assert "column" in content.lower()
