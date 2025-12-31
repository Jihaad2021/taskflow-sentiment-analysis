"""Test TopicTool."""

import pytest

from src.tools.topic_tool import TopicTool


@pytest.fixture
def topic_tool():
    """Create TopicTool instance."""
    return TopicTool(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def test_tool_initialization(topic_tool):
    """Test tool can be initialized."""
    assert topic_tool is not None
    assert topic_tool.model is not None
    assert topic_tool.tokenizer is not None


def test_topic_extraction(topic_tool):
    """Test basic topic extraction."""
    text = "Machine learning models require training data and validation datasets."
    result = topic_tool.analyze(text)

    assert "topics" in result
    assert "relevance_scores" in result
    assert "primary_topic" in result
    assert len(result["topics"]) > 0


def test_product_review_topics(topic_tool):
    """Test topic extraction from product review."""
    text = "Great battery life and excellent camera quality with fast performance."
    result = topic_tool.analyze(text)

    topics = result["topics"]
    assert len(topics) > 0

    # Should extract product-related keywords
    all_topics = " ".join(topics).lower()
    assert any(word in all_topics for word in ["battery", "camera", "quality", "performance"])


def test_relevance_scores(topic_tool):
    """Test that relevance scores are provided."""
    text = "Customer service was helpful and responsive with quick solutions."
    result = topic_tool.analyze(text)

    topics = result["topics"]
    scores = result["relevance_scores"]

    # All topics should have scores
    for topic in topics:
        assert topic in scores
        assert 0 <= scores[topic] <= 1


def test_primary_topic(topic_tool):
    """Test primary topic selection."""
    text = "Software development requires programming skills and testing knowledge."
    result = topic_tool.analyze(text)

    assert result["primary_topic"] in result["topics"]
    # Primary should be first topic
    assert result["primary_topic"] == result["topics"][0]


def test_batch_analysis(topic_tool):
    """Test batch processing."""
    texts = [
        "Machine learning and data science applications.",
        "Mobile app development with Flutter.",
        "Cloud infrastructure and deployment.",
    ]

    results = topic_tool.analyze_batch(texts)

    assert len(results) == 3
    assert all("topics" in r for r in results)
    assert all("primary_topic" in r for r in results)
