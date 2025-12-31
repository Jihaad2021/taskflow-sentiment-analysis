"""Test ColumnDetectorAgent."""

import pandas as pd
import pytest

from src.agents.column_detector import ColumnDetectorAgent
from src.models.schemas import ColumnDetectorInput
from src.utils.exceptions import ValidationError


@pytest.fixture
def agent():
    """Create ColumnDetectorAgent instance."""
    return ColumnDetectorAgent()


@pytest.fixture
def sample_df_obvious():
    """DataFrame with obvious 'comment' column."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "comment": [
                "This product is amazing! Love it.",
                "Not bad, but could be better.",
                "Worst purchase ever. Very disappointed.",
            ],
            "rating": [5, 3, 1],
        }
    )


@pytest.fixture
def sample_df_ambiguous():
    """DataFrame with multiple text columns."""
    return pd.DataFrame(
        {
            "title": ["Great", "OK", "Bad"],
            "description": ["This is a great product", "It's okay I guess", "Terrible quality"],
            "review": [
                "I love this product so much! Highly recommend.",
                "Decent product for the price.",
                "Waste of money. Do not buy.",
            ],
        }
    )


@pytest.fixture
def sample_df_no_header():
    """DataFrame without obvious text column name."""
    return pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [
                "First comment here with some text",
                "Second comment also has content",
                "Third comment is also here",
            ],
            "col3": ["A", "B", "C"],
        }
    )


def test_agent_initialization(agent):
    """Test agent can be initialized."""
    assert agent is not None
    assert agent.config.name == "ColumnDetectorAgent"


def test_detect_obvious_column(agent, sample_df_obvious):
    """Test detection of obvious 'comment' column."""
    input_data = ColumnDetectorInput(dataframe=sample_df_obvious)
    output = agent.execute(input_data)

    assert output.column_name == "comment"
    assert output.confidence > 0.7
    assert output.method == "heuristic"
    assert "comment" in output.reasoning.lower()


def test_detect_with_user_hint(agent, sample_df_ambiguous):
    """Test detection with user hint."""
    input_data = ColumnDetectorInput(dataframe=sample_df_ambiguous, user_hint="review")
    output = agent.execute(input_data)

    assert output.column_name == "review"
    assert output.confidence == 1.0
    assert "user hint" in output.reasoning.lower()


def test_detect_longest_text_column(agent, sample_df_ambiguous):
    """Test detection picks longest text column."""
    input_data = ColumnDetectorInput(dataframe=sample_df_ambiguous)
    output = agent.execute(input_data)

    # 'review' column has longest text
    assert output.column_name == "review"
    assert output.confidence > 0.5


def test_detect_no_header(agent, sample_df_no_header):
    """Test detection without obvious column names."""
    input_data = ColumnDetectorInput(dataframe=sample_df_no_header)
    output = agent.execute(input_data)

    # Should detect 'col2' (longest text)
    assert output.column_name == "col2"
    assert len(output.candidates) > 0


def test_empty_dataframe(agent):
    """Test with empty DataFrame."""
    df = pd.DataFrame()
    input_data = ColumnDetectorInput(dataframe=df)

    with pytest.raises(ValidationError, match="Invalid DataFrame"):
        agent.execute(input_data)


def test_no_text_columns(agent):
    """Test with DataFrame containing only numeric columns."""
    df = pd.DataFrame({"id": [1, 2, 3], "count": [10, 20, 30], "score": [0.5, 0.7, 0.9]})
    input_data = ColumnDetectorInput(dataframe=df)

    with pytest.raises(ValidationError, match="No text columns found"):
        agent.execute(input_data)


def test_candidates_returned(agent, sample_df_ambiguous):
    """Test that top candidates are returned."""
    input_data = ColumnDetectorInput(dataframe=sample_df_ambiguous)
    output = agent.execute(input_data)

    assert len(output.candidates) <= 3
    assert len(output.candidates) > 0

    # Candidates should be sorted by confidence
    for i in range(len(output.candidates) - 1):
        assert output.candidates[i].confidence >= output.candidates[i + 1].confidence


def test_sample_values_truncated(agent):
    """Test that long sample values are truncated."""
    df = pd.DataFrame({"comment": ["a" * 200, "b" * 150, "Short text"]})
    input_data = ColumnDetectorInput(dataframe=df)
    output = agent.execute(input_data)

    for candidate in output.candidates:
        for sample in candidate.sample_values:
            assert len(sample) <= 100


def test_llm_fallback_triggered(agent):
    """Test that LLM fallback is triggered for low confidence."""
    # DataFrame with no obvious column names and short text
    df = pd.DataFrame({"a": ["hi", "ok", "no"], "b": [1, 2, 3], "c": ["yes", "maybe", "sure"]})

    input_data = ColumnDetectorInput(dataframe=df)
    output = agent.execute(input_data)

    # With short text, confidence should be low, triggering LLM
    # Mock LLM should return a result
    assert output.method == "llm"
    assert output.confidence >= 0.7  # LLM boosts confidence


def test_llm_fallback_with_mock():
    """Test LLM fallback with custom mock."""

    class CustomMockLLM:
        def generate(self, prompt, **kwargs):
            return {
                "content": '{"column": "feedback", "reasoning": "Contains user feedback"}',
                "tokens": {"input": 100, "output": 50},
                "cost": 0.001,
            }

    agent = ColumnDetectorAgent(llm=CustomMockLLM())

    df = pd.DataFrame({"feedback": ["short", "text", "here"], "id": [1, 2, 3]})

    input_data = ColumnDetectorInput(dataframe=df)
    output = agent.execute(input_data)

    assert output.column_name == "feedback"
    assert output.method == "llm"
    assert "LLM:" in output.reasoning


def test_llm_fallback_fails_gracefully():
    """Test that agent falls back to heuristic if LLM fails."""

    class FailingLLM:
        def generate(self, prompt, **kwargs):
            raise Exception("LLM API error")

    agent = ColumnDetectorAgent(llm=FailingLLM())

    df = pd.DataFrame({"text": ["a", "b", "c"], "id": [1, 2, 3]})  # Low confidence

    input_data = ColumnDetectorInput(dataframe=df)
    output = agent.execute(input_data)

    # Should still return result (heuristic fallback)
    assert output.column_name == "text"
    assert output.method == "heuristic"


def test_llm_invalid_json_response():
    """Test handling of invalid JSON from LLM."""

    class BadJSONLLM:
        def generate(self, prompt, **kwargs):
            return {
                "content": "This is not valid JSON",
                "tokens": {"input": 100, "output": 50},
                "cost": 0.001,
            }

    agent = ColumnDetectorAgent(llm=BadJSONLLM())

    df = pd.DataFrame(
        {
            "data": ["x", "y", "z"],
        }
    )

    input_data = ColumnDetectorInput(dataframe=df)
    output = agent.execute(input_data)

    # Should fall back to heuristic
    assert output.column_name == "data"
    assert output.method == "heuristic"
