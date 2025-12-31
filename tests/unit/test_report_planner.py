"""Test ReportPlannerAgent."""

import json

import pytest

from src.agents.report_planner import ReportPlannerAgent
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    PrePromptEvaluatorOutput,
    QualityCheck,
    ReportPlannerInput,
    TopicSummary,
)
from src.utils.exceptions import AgentExecutionError


class PlannerMockLLM(MockLLM):
    """Mock LLM that returns valid report plan JSON."""

    def _generate_json_response(self, prompt: str) -> str:
        """Generate mock report plan JSON."""

        # Extract report type from prompt
        report_type = "executive"
        if "MARKETING" in prompt:
            report_type = "marketing"
        elif "PRODUCT" in prompt:
            report_type = "product"

        return json.dumps(
            {
                "title": f"{report_type.title()} Sentiment Analysis Report",
                "sections": [
                    {
                        "name": "Executive Summary",
                        "priority": "high",
                        "key_points": ["Overview of findings", "Key metrics"],
                        "data_sources": ["sentiment", "emotion"],
                        "estimated_length": "2-3 paragraphs",
                    },
                    {
                        "name": "Sentiment Analysis",
                        "priority": "high",
                        "key_points": ["Distribution", "Trends"],
                        "data_sources": ["sentiment"],
                        "estimated_length": "3-4 paragraphs",
                    },
                    {
                        "name": "Key Topics",
                        "priority": "medium",
                        "key_points": ["Main themes"],
                        "data_sources": ["topics"],
                        "estimated_length": "2-3 paragraphs",
                    },
                ],
                "recommended_length": "5-7 pages",
                "key_insights": ["Positive sentiment dominates", "Quality is main topic"],
                "recommended_focus": ["Customer satisfaction", "Product quality"],
                "narrative_flow": ["Introduction", "Analysis", "Recommendations"],
            }
        )


@pytest.fixture
def mock_llm():
    """Create PlannerMockLLM instance."""
    return PlannerMockLLM()


@pytest.fixture
def planner(mock_llm):
    """Create ReportPlannerAgent with mock LLM."""
    return ReportPlannerAgent(llm=mock_llm)


@pytest.fixture
def sample_analysis_results():
    """Sample analysis results."""
    return AnalysisOrchestratorOutput(
        total_comments=100,
        sentiment_distribution={"positive": 60, "negative": 30, "neutral": 10},
        emotion_distribution={"joy": 50, "anger": 20, "sadness": 15, "neutral": 15},
        top_topics=[
            TopicSummary(
                topic="quality",
                count=40,
                avg_sentiment=0.7,
                sample_comments=["Great quality!", "Excellent product"],
            ),
            TopicSummary(
                topic="service", count=30, avg_sentiment=0.5, sample_comments=["Good service"]
            ),
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


def test_agent_initialization(planner):
    """Test agent can be initialized."""
    assert planner is not None
    assert planner.llm is not None
    assert len(planner.VALID_REPORT_TYPES) == 5


def test_create_executive_plan(planner, sample_analysis_results, sample_quality_assessment):
    """Test creating executive report plan."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="executive",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    assert output.report_outline is not None
    assert len(output.report_outline.sections) > 0
    assert output.estimated_sections > 0
    assert len(output.key_insights) > 0
    assert output.llm_cost >= 0


def test_create_marketing_plan(planner, sample_analysis_results, sample_quality_assessment):
    """Test creating marketing report plan."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="marketing",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    assert "Marketing" in output.report_outline.title or "marketing" in output.report_outline.title


def test_invalid_report_type(planner, sample_analysis_results, sample_quality_assessment):
    """Test error on invalid report type."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="invalid_type",
        quality_assessment=sample_quality_assessment,
    )

    with pytest.raises(AgentExecutionError, match="Invalid report type"):
        planner.execute(input_data)


def test_report_outline_structure(planner, sample_analysis_results, sample_quality_assessment):
    """Test report outline has correct structure."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="product",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)
    outline = output.report_outline

    # Check structure
    assert hasattr(outline, "title")
    assert hasattr(outline, "sections")
    assert hasattr(outline, "recommended_length")

    # Check sections
    assert len(outline.sections) > 0
    for section in outline.sections:
        assert hasattr(section, "name")
        assert hasattr(section, "priority")
        assert hasattr(section, "key_points")
        assert section.priority in ["high", "medium", "low"]


def test_key_insights_generated(planner, sample_analysis_results, sample_quality_assessment):
    """Test that key insights are generated."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="executive",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    assert len(output.key_insights) > 0
    assert all(isinstance(insight, str) for insight in output.key_insights)


def test_recommended_focus_generated(planner, sample_analysis_results, sample_quality_assessment):
    """Test that recommended focus areas are generated."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="marketing",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    assert len(output.recommended_focus) > 0


def test_narrative_flow_generated(planner, sample_analysis_results, sample_quality_assessment):
    """Test that narrative flow is generated."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="comprehensive",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    assert len(output.narrative_flow) > 0


def test_all_report_types(planner, sample_analysis_results, sample_quality_assessment):
    """Test all valid report types."""

    for report_type in planner.VALID_REPORT_TYPES:
        input_data = ReportPlannerInput(
            tool_results=sample_analysis_results,
            report_type=report_type,
            quality_assessment=sample_quality_assessment,
        )

        output = planner.execute(input_data)

        assert output is not None
        assert output.report_outline is not None
        assert output.estimated_sections > 0


def test_llm_cost_tracking(planner, sample_analysis_results, sample_quality_assessment):
    """Test that LLM cost is tracked."""
    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="executive",
        quality_assessment=sample_quality_assessment,
    )

    output = planner.execute(input_data)

    # Mock LLM returns 0.001
    assert output.llm_cost > 0


def test_default_plan_on_llm_failure(sample_analysis_results, sample_quality_assessment):
    """Test fallback to default plan if LLM fails."""

    class FailingLLM(MockLLM):
        def _generate_json_response(self, prompt: str) -> str:
            return "invalid json {"

    planner = ReportPlannerAgent(llm=FailingLLM())

    input_data = ReportPlannerInput(
        tool_results=sample_analysis_results,
        report_type="executive",
        quality_assessment=sample_quality_assessment,
    )

    # Should not crash, use default plan
    output = planner.execute(input_data)

    assert output is not None
    assert output.report_outline is not None
    assert len(output.report_outline.sections) > 0
