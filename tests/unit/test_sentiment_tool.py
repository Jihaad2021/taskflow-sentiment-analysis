"""Test SentimentTool."""

import pytest

from src.tools.sentiment_tool import SentimentTool


@pytest.fixture
def sentiment_tool():
    """Create SentimentTool instance."""
    return SentimentTool(
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device="cpu"
    )


def test_tool_initialization(sentiment_tool):
    """Test tool can be initialized."""
    assert sentiment_tool is not None
    assert sentiment_tool.model is not None
    assert sentiment_tool.tokenizer is not None


def test_positive_sentiment(sentiment_tool):
    """Test positive sentiment detection."""
    text = "This product is amazing! I love it so much!"
    result = sentiment_tool.analyze(text)

    assert result["label"] == "positive"
    assert result["score"] > 0.5
    assert "scores" in result
    assert len(result["scores"]) == 3


def test_negative_sentiment(sentiment_tool):
    """Test negative sentiment detection."""
    text = "Terrible product. Waste of money. Very disappointed."
    result = sentiment_tool.analyze(text)

    assert result["label"] == "negative"
    assert result["score"] > 0.5


def test_neutral_sentiment(sentiment_tool):
    """Test neutral sentiment detection."""
    text = "The product arrived on time. It works as described."
    result = sentiment_tool.analyze(text)

    assert result["label"] in ["neutral", "positive"]  # Might be positive
    assert result["score"] > 0.3


def test_batch_analysis(sentiment_tool):
    """Test batch processing."""
    texts = ["Great product!", "Terrible experience.", "It's okay, nothing special."]

    results = sentiment_tool.analyze_batch(texts)

    assert len(results) == 3
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
