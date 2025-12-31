"""Test ReportEvaluatorAgent."""

import pytest

from src.agents.report_evaluator import ReportEvaluatorAgent
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    ReportEvaluatorInput,
    ReportOutline,
    ReportPlannerOutput,
    ReportSection,
)


@pytest.fixture
def mock_llm():
    """Create MockLLM instance."""
    return MockLLM()


@pytest.fixture
def evaluator(mock_llm):
    """Create ReportEvaluatorAgent with mock LLM."""
    return ReportEvaluatorAgent(llm=mock_llm, min_score=70.0)


@pytest.fixture
def good_report():
    """Sample good quality report."""
    return """# Customer Sentiment Analysis Report

## Executive Summary

Analysis of 100 customer comments reveals predominantly positive sentiment at 60%, with quality and service as key discussion topics.

**Key Metrics:**
- Positive: 60 comments (60%)
- Negative: 30 comments (30%)
- Neutral: 10 comments (10%)

## Sentiment Distribution

The sentiment analysis shows strong positive customer perception with 60% positive comments. Negative sentiment accounts for 30% of feedback, primarily related to service issues.

## Key Topics

### Quality (40 mentions)
Quality emerged as the top topic with positive sentiment (0.7 score).

### Service (30 mentions)
Customer service received moderate positive feedback (0.5 score).

## Recommendations

Based on the analysis:
1. Maintain current quality standards
2. Improve service response times
3. Address negative feedback proactively

## Conclusion

Overall customer sentiment is positive with clear improvement opportunities in service delivery."""


@pytest.fixture
def poor_report():
    """Sample poor quality report."""
    return """# Report

Some analysis was done.

There are customers who like the product and some who don't.

That's all."""


@pytest.fixture
def sample_data():
    """Sample analysis data."""
    return AnalysisOrchestratorOutput(
        total_comments=100,
        sentiment_distribution={"positive": 60, "negative": 30, "neutral": 10},
        emotion_distribution={"joy": 50, "anger": 20, "sadness": 15, "neutral": 15},
        top_topics=[],
        entities=[],
        keyphrases=[],
        individual_results=[],
        execution_time=10.0,
    )


@pytest.fixture
def sample_plan():
    """Sample report plan."""
    outline = ReportOutline(
        title="Customer Sentiment Analysis Report",
        sections=[
            ReportSection(
                name="Executive Summary",
                priority="high",
                key_points=["Overview"],
                data_sources=["sentiment"],
                estimated_length="2-3 paragraphs",
            )
        ],
        recommended_length="5-7 pages",
    )

    return ReportPlannerOutput(
        report_outline=outline,
        key_insights=["Positive sentiment"],
        recommended_focus=["Quality"],
        narrative_flow=["Intro", "Analysis"],
        estimated_sections=1,
        llm_cost=0.002,
    )


def test_agent_initialization(evaluator):
    """Test agent can be initialized."""
    assert evaluator is not None
    assert evaluator.llm is not None
    assert evaluator.min_score == 70.0


def test_evaluate_good_report(evaluator, good_report, sample_data, sample_plan):
    """Test evaluation of good quality report."""
    input_data = ReportEvaluatorInput(
        report_text=good_report, original_data=sample_data, report_plan=sample_plan
    )

    output = evaluator.execute(input_data)

    assert output.status == "pass"
    assert output.quality_score >= 70.0
    assert output.should_regenerate is False
    assert len(output.checks) == 5


def test_evaluate_poor_report(evaluator, poor_report, sample_data, sample_plan):
    """Test evaluation of poor quality report."""

    class FailingMockLLM(MockLLM):
        def _generate_mock_evaluation(self) -> str:
            import json

            return json.dumps(
                {
                    "overall_score": 40.0,
                    "should_regenerate": True,
                    "checks": {
                        "completeness": {
                            "score": 30.0,
                            "issues": ["Missing sections"],
                            "critical": True,
                        },
                        "factual_accuracy": {"score": 50.0, "issues": [], "critical": False},
                        "coherence": {"score": 40.0, "issues": ["Poor flow"], "critical": False},
                        "actionability": {
                            "score": 20.0,
                            "issues": ["No recommendations"],
                            "critical": True,
                        },
                        "hallucination": {"score": 80.0, "issues": [], "critical": False},
                    },
                    "issues": ["Missing critical sections", "No actionable recommendations"],
                    "feedback": "Report needs major improvements in completeness and actionability.",
                }
            )

    evaluator_fail = ReportEvaluatorAgent(llm=FailingMockLLM(), min_score=70.0)

    input_data = ReportEvaluatorInput(
        report_text=poor_report, original_data=sample_data, report_plan=sample_plan
    )

    output = evaluator_fail.execute(input_data)

    assert output.status == "fail"
    assert output.quality_score < 70.0
    assert output.should_regenerate is True


def test_evaluation_checks_structure(evaluator, good_report, sample_data, sample_plan):
    """Test that all evaluation checks are present."""
    input_data = ReportEvaluatorInput(
        report_text=good_report, original_data=sample_data, report_plan=sample_plan
    )

    output = evaluator.execute(input_data)

    # Should have all 5 checks
    assert "completeness" in output.checks
    assert "factual_accuracy" in output.checks
    assert "coherence" in output.checks
    assert "actionability" in output.checks
    assert "hallucination" in output.checks

    # Each check should have required fields
    for check in output.checks.values():
        assert hasattr(check, "score")
        assert hasattr(check, "issues")
        assert hasattr(check, "critical")
        assert 0 <= check.score <= 100


def test_quality_score_range(evaluator, good_report, sample_data, sample_plan):
    """Test that quality score is in valid range."""
    input_data = ReportEvaluatorInput(
        report_text=good_report, original_data=sample_data, report_plan=sample_plan
    )

    output = evaluator.execute(input_data)

    assert 0 <= output.quality_score <= 100


def test_feedback_provided_on_failure(evaluator, sample_data, sample_plan):
    """Test that feedback is provided when regeneration needed."""

    class FailMockLLM(MockLLM):
        def _generate_mock_evaluation(self) -> str:
            import json

            return json.dumps(
                {
                    "overall_score": 60.0,
                    "should_regenerate": True,
                    "checks": {
                        "completeness": {"score": 50.0, "issues": [], "critical": False},
                        "factual_accuracy": {"score": 70.0, "issues": [], "critical": False},
                        "coherence": {"score": 60.0, "issues": [], "critical": False},
                        "actionability": {"score": 60.0, "issues": [], "critical": False},
                        "hallucination": {"score": 90.0, "issues": [], "critical": False},
                    },
                    "issues": ["Low overall quality"],
                    "feedback": "Needs more detail and better structure.",
                }
            )

    eval_agent = ReportEvaluatorAgent(llm=FailMockLLM())

    input_data = ReportEvaluatorInput(
        report_text="Short report", original_data=sample_data, report_plan=sample_plan
    )

    output = eval_agent.execute(input_data)

    if output.should_regenerate:
        assert output.feedback is not None
        assert len(output.feedback) > 0


def test_cost_tracking(evaluator, good_report, sample_data, sample_plan):
    """Test that LLM cost is tracked."""
    input_data = ReportEvaluatorInput(
        report_text=good_report, original_data=sample_data, report_plan=sample_plan
    )

    output = evaluator.execute(input_data)

    assert output.llm_cost >= 0


def test_fallback_evaluation(good_report, sample_data, sample_plan):
    """Test fallback evaluation when LLM fails."""

    class BrokenLLM(MockLLM):
        def _generate_json_response(self, prompt: str) -> str:
            return "invalid json {"

    evaluator = ReportEvaluatorAgent(llm=BrokenLLM())

    input_data = ReportEvaluatorInput(
        report_text=good_report, original_data=sample_data, report_plan=sample_plan
    )

    # Should not crash, use fallback
    output = evaluator.execute(input_data)

    assert output is not None
    assert output.quality_score >= 0
    assert len(output.checks) == 5


def test_weights_sum_to_one(evaluator):
    """Test that evaluation weights sum to 1.0."""
    total_weight = sum(evaluator.WEIGHTS.values())
    assert abs(total_weight - 1.0) < 0.01


def test_critical_issues_trigger_regeneration(evaluator, sample_data, sample_plan):
    """Test that critical issues trigger regeneration."""

    class CriticalIssueLLM(MockLLM):
        def _generate_mock_evaluation(self) -> str:
            import json

            return json.dumps(
                {
                    "overall_score": 75.0,  # Above threshold
                    "should_regenerate": True,  # But critical issue
                    "checks": {
                        "completeness": {"score": 90.0, "issues": [], "critical": False},
                        "factual_accuracy": {
                            "score": 40.0,
                            "issues": ["Wrong data"],
                            "critical": True,
                        },
                        "coherence": {"score": 80.0, "issues": [], "critical": False},
                        "actionability": {"score": 75.0, "issues": [], "critical": False},
                        "hallucination": {"score": 90.0, "issues": [], "critical": False},
                    },
                    "issues": ["Critical factual error"],
                    "feedback": "Fix the incorrect data.",
                }
            )

    eval_agent = ReportEvaluatorAgent(llm=CriticalIssueLLM())

    input_data = ReportEvaluatorInput(
        report_text="Report with wrong data", original_data=sample_data, report_plan=sample_plan
    )

    output = eval_agent.execute(input_data)

    # Should regenerate despite good score
    assert output.should_regenerate is True
    assert output.status == "fail"
