"""Test EmotionTool."""

import pytest

from src.tools.emotion_tool import EmotionTool


@pytest.fixture
def emotion_tool():
    """Create EmotionTool instance."""
    return EmotionTool(model_name="j-hartmann/emotion-english-distilroberta-base", device="cpu")


def test_tool_initialization(emotion_tool):
    """Test tool can be initialized."""
    assert emotion_tool is not None
    assert emotion_tool.model is not None
    assert emotion_tool.tokenizer is not None


def test_joy_emotion(emotion_tool):
    """Test joy emotion detection."""
    text = "I'm so happy and excited about this! Best day ever!"
    result = emotion_tool.analyze(text)

    assert result["emotion"] == "joy"
    assert result["confidence"] > 0.5
    assert "scores" in result
    assert len(result["scores"]) == 7


def test_anger_emotion(emotion_tool):
    """Test anger emotion detection."""
    text = "This is absolutely infuriating! I'm so angry right now!"
    result = emotion_tool.analyze(text)

    assert result["emotion"] == "anger"
    assert result["confidence"] > 0.3


def test_sadness_emotion(emotion_tool):
    """Test sadness emotion detection."""
    text = "I'm feeling really sad and disappointed about everything."
    result = emotion_tool.analyze(text)

    assert result["emotion"] in ["sadness", "neutral"]
    assert result["confidence"] > 0.3


def test_batch_analysis(emotion_tool):
    """Test batch processing."""
    texts = ["I'm so happy!", "This makes me angry.", "Feeling sad today."]

    results = emotion_tool.analyze_batch(texts)

    assert len(results) == 3
    assert all("emotion" in r for r in results)
    assert all("confidence" in r for r in results)
