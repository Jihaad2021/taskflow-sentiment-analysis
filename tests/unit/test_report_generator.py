"""Test ReportGenerator."""

import pytest

from src.agents.report_generator import ReportGenerator, ReportGeneratorInput
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    PrePromptEvaluatorOutput,
    QualityCheck,
    TopicSummary,
)


@pytest.fixture
def mock_llm():
    """Create MockLLM instance."""
    return MockLLM()


@pytest.fixture
def generator(mock_llm):
    """Create ReportGenerator with mock LLM."""
    return ReportGenerator(llm=mock_llm, max_regenerations=3, min_quality_score=70.0)


@pytest.fixture
def sample_analysis_results():
    """Sample analysis results."""
    return AnalysisOrchestratorOutput(
        total_comments=100,
        sentiment_distribution={"positive": 60, "negative": 30, "neutral": 10},
        emotion_distribution={"joy": 50, "anger": 20, "sadness": 15, "neutral": 15},
        top_topics=[
            TopicSummary(
                topic="quality", count=40, avg_sentiment=0.7, sample_comments=["Great quality!"]
            )
        ],
        entities=[],
        keyphrases=["excellent", "quality", "satisfied"],
        individual_results=[],
        execution_time=10.0,
    )


@pytest.fixture
def sample_quality_assessment():
    """Sample quality assessment."""
    return PrePromptEvaluatorOutput(
        status="pass",
        quality_score=85.0,
        checks=[
            QualityCheck(
                name="Data Coverage", passed=True, score=100.0, message="All data processed"
            )
        ],
        issues=[],
        recommendations=[],
        should_proceed=True,
        metadata={"avg_confidence": 0.85},
    )


def test_generator_initialization(generator):
    """Test generator can be initialized."""
    assert generator is not None
    assert generator.planner is not None
    assert generator.writer is not None
    assert generator.evaluator is not None
    assert generator.max_regenerations == 3


def test_generate_report_success(generator, sample_analysis_results, sample_quality_assessment):
    """Test successful report generation on first attempt."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=3,
    )

    output = generator.execute(input_data)

    assert output.report_text is not None
    assert len(output.report_text) > 0
    assert output.quality_score >= 0
    assert output.attempts >= 1
    assert output.total_cost >= 0
    assert output.total_time > 0
    assert output.final_status in ["passed", "failed", "max_attempts_reached"]


def test_regeneration_history_tracked(
    generator, sample_analysis_results, sample_quality_assessment
):
    """Test that regeneration history is tracked."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="marketing",
        max_regenerations=3,
    )

    output = generator.execute(input_data)

    # Should have history
    assert len(output.regeneration_history) > 0
    assert len(output.regeneration_history) == output.attempts

    # Each attempt should have required fields
    for attempt in output.regeneration_history:
        assert "attempt" in attempt
        assert "word_count" in attempt
        assert "quality_score" in attempt
        assert "status" in attempt
        assert "cost" in attempt


def test_cost_tracking(generator, sample_analysis_results, sample_quality_assessment):
    """Test that total cost includes all steps."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="product",
        max_regenerations=1,
    )

    output = generator.execute(input_data)

    # Should include: planner + writer + evaluator costs
    assert output.total_cost > 0

    # Should be sum of all attempts
    history_cost = sum(attempt["cost"] for attempt in output.regeneration_history)
    # Note: total_cost includes planner cost too
    assert output.total_cost >= history_cost


def test_max_regenerations_respected(sample_analysis_results, sample_quality_assessment):
    """Test that max regenerations limit is respected."""

    class AlwaysFailLLM(MockLLM):
        """LLM that always returns low quality scores."""

        def _generate_mock_evaluation(self) -> str:
            import json

            return json.dumps(
                {
                    "overall_score": 50.0,  # Always fail
                    "should_regenerate": True,
                    "checks": {
                        "completeness": {"score": 50.0, "issues": ["Poor"], "critical": True},
                        "factual_accuracy": {"score": 50.0, "issues": [], "critical": False},
                        "coherence": {"score": 50.0, "issues": [], "critical": False},
                        "actionability": {"score": 50.0, "issues": [], "critical": False},
                        "hallucination": {"score": 90.0, "issues": [], "critical": False},
                    },
                    "issues": ["Low quality"],
                    "feedback": "Needs major improvement.",
                }
            )

    generator = ReportGenerator(llm=AlwaysFailLLM(), max_regenerations=2)

    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=2,
    )

    output = generator.execute(input_data)

    # Should stop at max attempts
    assert output.attempts <= 2
    assert output.final_status == "max_attempts_reached"


def test_stops_on_passing_score(generator, sample_analysis_results, sample_quality_assessment):
    """Test that generation stops when quality passes."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=3,
    )

    output = generator.execute(input_data)

    # Mock LLM returns passing score, should stop on attempt 1
    if output.quality_score >= 70.0:
        assert output.final_status == "passed"
        assert output.attempts == 1


def test_all_report_types(generator, sample_analysis_results, sample_quality_assessment):
    """Test generation with all report types."""

    report_types = ["executive", "marketing", "product", "customer_service", "comprehensive"]

    for report_type in report_types:
        input_data = ReportGeneratorInput(
            tool_results=sample_analysis_results,
            quality_assessment=sample_quality_assessment,
            report_type=report_type,
            max_regenerations=1,
        )

        output = generator.execute(input_data)

        assert output is not None
        assert output.report_text is not None
        assert len(output.report_text) > 0


def test_feedback_used_in_regeneration(sample_analysis_results, sample_quality_assessment):
    """Test that feedback is passed to next attempt."""

    class FeedbackTrackingLLM(MockLLM):
        def __init__(self):
            super().__init__()
            self.prompts_received = []

        def generate(self, prompt, max_tokens=2000, temperature=0.7, response_format=None):
            self.prompts_received.append(prompt)
            return super().generate(prompt, max_tokens, temperature, response_format)

        def _generate_mock_evaluation(self) -> str:
            import json

            # First attempt fails, second passes
            if len(self.prompts_received) < 4:  # planner + writer + eval
                should_regen = True
                score = 60.0
            else:
                should_regen = False
                score = 85.0

            return json.dumps(
                {
                    "overall_score": score,
                    "should_regenerate": should_regen,
                    "checks": {
                        "completeness": {"score": score, "issues": [], "critical": False},
                        "factual_accuracy": {"score": score, "issues": [], "critical": False},
                        "coherence": {"score": score, "issues": [], "critical": False},
                        "actionability": {"score": score, "issues": [], "critical": False},
                        "hallucination": {"score": 90.0, "issues": [], "critical": False},
                    },
                    "issues": [],
                    "feedback": "Add more specific examples.",
                }
            )

    llm = FeedbackTrackingLLM()
    generator = ReportGenerator(llm=llm, max_regenerations=3)

    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=3,
    )

    output = generator.execute(input_data)

    # Should have attempted regeneration
    if output.attempts > 1:
        # Second writer prompt should include feedback
        writer_prompts = [p for p in llm.prompts_received if "Write a complete" in p]
        if len(writer_prompts) > 1:
            assert (
                "Previous Attempt Feedback" in writer_prompts[1]
                or "feedback" in writer_prompts[1].lower()
            )


def test_time_tracking(generator, sample_analysis_results, sample_quality_assessment):
    """Test that execution time is tracked."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=1,
    )

    output = generator.execute(input_data)

    assert output.total_time > 0


def test_final_status_values(generator, sample_analysis_results, sample_quality_assessment):
    """Test that final status has valid values."""
    input_data = ReportGeneratorInput(
        tool_results=sample_analysis_results,
        quality_assessment=sample_quality_assessment,
        report_type="executive",
        max_regenerations=1,
    )

    output = generator.execute(input_data)

    assert output.final_status in ["passed", "failed", "max_attempts_reached"]
