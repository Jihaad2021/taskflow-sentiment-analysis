"""Test AnalysisOrchestratorAgent."""

import pytest

from src.agents.orchestrator import AnalysisOrchestratorAgent
from src.models.schemas import AnalysisOrchestratorInput


@pytest.fixture
def orchestrator():
    """Create AnalysisOrchestratorAgent instance."""
    return AnalysisOrchestratorAgent(device="cpu")


@pytest.fixture
def sample_comments():
    """Sample comments for testing."""
    return [
        "This product is amazing! I love it so much.",
        "Terrible quality, very disappointed with my purchase.",
        "The customer service at Apple was excellent and helpful.",
        "Fast delivery and good packaging, happy with the order.",
        "Not worth the money, waste of time and effort.",
    ]


def test_agent_initialization(orchestrator):
    """Test agent can be initialized."""
    assert orchestrator is not None
    assert orchestrator.sentiment_tool is not None
    assert orchestrator.emotion_tool is not None
    assert orchestrator.topic_tool is not None
    assert orchestrator.entity_tool is not None
    assert orchestrator.keyphrase_tool is not None


def test_analyze_single_comment(orchestrator):
    """Test analysis of single comment."""
    input_data = AnalysisOrchestratorInput(comments=["Great product, very satisfied!"])

    output = orchestrator.execute(input_data)

    assert output.total_comments == 1
    assert len(output.individual_results) == 1
    assert output.execution_time > 0


def test_analyze_multiple_comments(orchestrator, sample_comments):
    """Test analysis of multiple comments."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)

    output = orchestrator.execute(input_data)

    assert output.total_comments == 5
    assert len(output.individual_results) == 5

    # Check all comments analyzed
    for result in output.individual_results:
        assert result.sentiment is not None
        assert result.emotion is not None
        assert result.topics is not None
        assert result.entities is not None
        assert result.keyphrases is not None


def test_sentiment_distribution(orchestrator, sample_comments):
    """Test sentiment distribution aggregation."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)
    output = orchestrator.execute(input_data)

    sentiment_dist = output.sentiment_distribution

    # Should have sentiment counts
    assert isinstance(sentiment_dist, dict)
    assert len(sentiment_dist) > 0

    # Total should equal number of comments
    total = sum(sentiment_dist.values())
    assert total == 5


def test_emotion_distribution(orchestrator, sample_comments):
    """Test emotion distribution aggregation."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)
    output = orchestrator.execute(input_data)

    emotion_dist = output.emotion_distribution

    assert isinstance(emotion_dist, dict)
    assert len(emotion_dist) > 0

    # Total should equal number of comments
    total = sum(emotion_dist.values())
    assert total == 5


def test_top_topics(orchestrator, sample_comments):
    """Test topic extraction and ranking."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)
    output = orchestrator.execute(input_data)

    top_topics = output.top_topics

    assert isinstance(top_topics, list)
    assert len(top_topics) > 0

    # Check TopicSummary structure
    for topic in top_topics:
        assert hasattr(topic, "topic")
        assert hasattr(topic, "count")
        assert hasattr(topic, "avg_sentiment")
        assert topic.count > 0
        assert -1.0 <= topic.avg_sentiment <= 1.0


def test_entities_extraction(orchestrator):
    """Test entity extraction and aggregation."""
    comments = [
        "Apple released a new iPhone in California.",
        "Microsoft and Google are competing in AI.",
        "Tesla's factory in Texas is expanding.",
    ]

    input_data = AnalysisOrchestratorInput(comments=comments)
    output = orchestrator.execute(input_data)

    entities = output.entities

    assert isinstance(entities, list)
    # Should detect some entities
    assert len(entities) > 0

    # Check EntitySummary structure
    for entity in entities:
        assert hasattr(entity, "text")
        assert hasattr(entity, "type")
        assert hasattr(entity, "count")
        assert entity.count > 0


def test_keyphrases_aggregation(orchestrator, sample_comments):
    """Test keyphrase extraction and aggregation."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)
    output = orchestrator.execute(input_data)

    keyphrases = output.keyphrases

    assert isinstance(keyphrases, list)
    assert len(keyphrases) > 0


def test_individual_results_structure(orchestrator):
    """Test individual comment analysis structure."""
    input_data = AnalysisOrchestratorInput(comments=["Great product with excellent quality!"])
    output = orchestrator.execute(input_data)

    result = output.individual_results[0]

    # Check structure
    assert result.comment_id == "comment_0"
    assert result.text == "Great product with excellent quality!"
    assert result.sentiment.label in ["positive", "negative", "neutral"]
    assert 0 <= result.sentiment.score <= 1
    assert len(result.topics.topics) > 0
    assert result.execution_time > 0


def test_performance_tracking(orchestrator, sample_comments):
    """Test that execution time is tracked."""
    input_data = AnalysisOrchestratorInput(comments=sample_comments)
    output = orchestrator.execute(input_data)

    # Total execution time
    assert output.execution_time > 0

    # Individual execution times
    for result in output.individual_results:
        assert result.execution_time > 0


def test_empty_input_validation(orchestrator):
    """Test validation of empty comment list."""
    from pydantic import ValidationError

    # Should raise validation error
    with pytest.raises(ValidationError):
        AnalysisOrchestratorInput(comments=[])
