"""Test PrePromptEvaluatorAgent."""

import pytest

from src.agents.pre_evaluator import PrePromptEvaluatorAgent
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    CommentAnalysis,
    EmotionResult,
    EntityResult,
    KeyphraseResult,
    PrePromptEvaluatorInput,
    SentimentResult,
    TopicResult,
    TopicSummary,
)


@pytest.fixture
def evaluator():
    """Create PrePromptEvaluatorAgent instance."""
    return PrePromptEvaluatorAgent(min_coverage=0.8, min_confidence=0.6, min_quality_score=70.0)


@pytest.fixture
def good_results():
    """High-quality analysis results."""
    individual_results = []

    for i in range(10):
        result = CommentAnalysis(
            comment_id=f"comment_{i}",
            text=f"Sample comment {i}",
            sentiment=SentimentResult(
                label="positive" if i % 3 == 0 else "negative" if i % 3 == 1 else "neutral",
                score=0.85,
                scores={"positive": 0.85, "negative": 0.10, "neutral": 0.05},
            ),
            emotion=EmotionResult(
                emotion="joy",
                confidence=0.80,
                scores={
                    "joy": 0.80,
                    "anger": 0.10,
                    "sadness": 0.05,
                    "fear": 0.02,
                    "neutral": 0.02,
                    "surprise": 0.01,
                    "disgust": 0.0,
                },
            ),
            topics=TopicResult(
                topics=["quality", "service", "price"],
                relevance_scores={"quality": 0.9, "service": 0.7, "price": 0.6},
                primary_topic="quality",
            ),
            entities=EntityResult(entities=[]),
            keyphrases=KeyphraseResult(
                keyphrases=["great product", "excellent"],
                scores={"great product": 0.9, "excellent": 0.8},
            ),
            execution_time=0.5,
        )
        individual_results.append(result)

    return AnalysisOrchestratorOutput(
        total_comments=10,
        sentiment_distribution={"positive": 4, "negative": 3, "neutral": 3},
        emotion_distribution={"joy": 6, "anger": 2, "sadness": 2},
        top_topics=[
            TopicSummary(
                topic="quality",
                count=8,
                avg_sentiment=0.5,
                sample_comments=["Sample 1", "Sample 2"],
            ),
            TopicSummary(topic="service", count=6, avg_sentiment=0.3, sample_comments=["Sample 3"]),
        ],
        entities=[],
        keyphrases=["great", "excellent", "quality"],
        individual_results=individual_results,
        execution_time=5.0,
    )


@pytest.fixture
def poor_results():
    """Low-quality analysis results."""
    individual_results = []

    for i in range(10):
        result = CommentAnalysis(
            comment_id=f"comment_{i}",
            text=f"Sample {i}",
            sentiment=SentimentResult(
                label="positive",  # All same
                score=0.45,  # Low confidence
                scores={"positive": 0.45, "negative": 0.30, "neutral": 0.25},
            ),
            emotion=EmotionResult(
                emotion="neutral",
                confidence=0.40,  # Low confidence
                scores={
                    "joy": 0.10,
                    "anger": 0.10,
                    "sadness": 0.10,
                    "fear": 0.10,
                    "neutral": 0.40,
                    "surprise": 0.10,
                    "disgust": 0.10,
                },
            ),
            topics=TopicResult(
                topics=["general"], relevance_scores={"general": 0.5}, primary_topic="general"
            ),
            entities=EntityResult(entities=[]),
            keyphrases=KeyphraseResult(keyphrases=[], scores={}),
            execution_time=0.3,
        )
        individual_results.append(result)

    return AnalysisOrchestratorOutput(
        total_comments=10,
        sentiment_distribution={"positive": 10},  # All same
        emotion_distribution={"neutral": 10},  # All same
        top_topics=[TopicSummary(topic="general", count=10, avg_sentiment=0.0, sample_comments=[])],
        entities=[],
        keyphrases=[],
        individual_results=individual_results,
        execution_time=3.0,
    )


def test_agent_initialization(evaluator):
    """Test agent can be initialized."""
    assert evaluator is not None
    assert evaluator.min_coverage == 0.8
    assert evaluator.min_confidence == 0.6
    assert evaluator.min_quality_score == 70.0


def test_evaluate_good_results(evaluator, good_results):
    """Test evaluation of high-quality results."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    assert output.status in ["pass", "warning"]
    assert output.quality_score >= 60.0
    assert output.should_proceed is True
    assert len(output.checks) == 5


def test_evaluate_poor_results(evaluator, poor_results):
    """Test evaluation of low-quality results."""
    input_data = PrePromptEvaluatorInput(tool_results=poor_results)
    output = evaluator.execute(input_data)

    # Should warn or fail due to low confidence and uniformity
    assert output.status in ["warning", "fail"]
    assert output.quality_score < 80.0


def test_data_coverage_check(evaluator, good_results):
    """Test data coverage quality check."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    coverage_check = next(c for c in output.checks if c.name == "Data Coverage")

    assert coverage_check.passed is True
    assert coverage_check.score == 100.0  # 10/10 processed


def test_sentiment_quality_check(evaluator, good_results):
    """Test sentiment quality check."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    sentiment_check = next(c for c in output.checks if c.name == "Sentiment Quality")

    # Good results have diverse sentiment
    assert sentiment_check.passed is True
    assert sentiment_check.score >= 60.0


def test_topic_quality_check(evaluator, good_results):
    """Test topic quality check."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    topic_check = next(c for c in output.checks if c.name == "Topic Quality")

    assert topic_check.passed is True
    assert topic_check.score >= 60.0


def test_confidence_check(evaluator, good_results):
    """Test confidence check."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    confidence_check = next(c for c in output.checks if c.name == "Confidence Check")

    # Good results have high confidence
    assert confidence_check.passed is True
    assert confidence_check.score >= 60.0


def test_low_coverage_fails(evaluator):
    """Test that low data coverage fails."""
    # Only 5/10 processed
    results = AnalysisOrchestratorOutput(
        total_comments=10,
        sentiment_distribution={"positive": 5},
        emotion_distribution={"joy": 5},
        top_topics=[],
        entities=[],
        keyphrases=[],
        individual_results=[],  # Empty - low coverage
        execution_time=1.0,
    )

    input_data = PrePromptEvaluatorInput(tool_results=results)
    output = evaluator.execute(input_data)

    assert output.status == "fail"
    assert output.should_proceed is False


def test_recommendations_generated(evaluator, poor_results):
    """Test that recommendations are generated for poor results."""
    input_data = PrePromptEvaluatorInput(tool_results=poor_results)
    output = evaluator.execute(input_data)

    assert len(output.recommendations) > 0


def test_metadata_calculation(evaluator, good_results):
    """Test metadata calculation."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    metadata = output.metadata

    assert "avg_confidence" in metadata
    assert "data_coverage" in metadata
    assert "num_topics" in metadata
    assert metadata["data_coverage"] == 1.0  # 100%


def test_quality_score_calculation(evaluator, good_results):
    """Test overall quality score calculation."""
    input_data = PrePromptEvaluatorInput(tool_results=good_results)
    output = evaluator.execute(input_data)

    # Quality score should be weighted average
    assert 0 <= output.quality_score <= 100
    assert output.quality_score >= 70.0  # Good results should pass


def test_critical_check_failure(evaluator):
    """Test that critical check failure prevents proceeding."""
    # Very low confidence
    results = AnalysisOrchestratorOutput(
        total_comments=10,
        sentiment_distribution={"positive": 10},
        emotion_distribution={"neutral": 10},
        top_topics=[],
        entities=[],
        keyphrases=[],
        individual_results=[
            CommentAnalysis(
                comment_id="test",
                text="test",
                sentiment=SentimentResult(
                    label="positive",
                    score=0.3,  # Very low
                    scores={"positive": 0.3, "negative": 0.35, "neutral": 0.35},
                ),
                emotion=EmotionResult(
                    emotion="neutral",
                    confidence=0.3,  # Very low
                    scores={
                        "joy": 0.1,
                        "anger": 0.1,
                        "sadness": 0.1,
                        "fear": 0.1,
                        "neutral": 0.3,
                        "surprise": 0.15,
                        "disgust": 0.15,
                    },
                ),
                topics=TopicResult(topics=[], relevance_scores={}, primary_topic=""),
                entities=EntityResult(entities=[]),
                keyphrases=KeyphraseResult(keyphrases=[], scores={}),
                execution_time=0.1,
            )
        ]
        * 10,
        execution_time=1.0,
    )

    input_data = PrePromptEvaluatorInput(tool_results=results)
    output = evaluator.execute(input_data)

    # Should fail due to low confidence
    assert output.should_proceed is False
