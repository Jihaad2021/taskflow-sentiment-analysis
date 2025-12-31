"""Test ReportWriterAgent."""

import pytest

from src.agents.report_writer import ReportWriterAgent
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    ReportOutline,
    ReportPlannerOutput,
    ReportSection,
    ReportWriterInput,
    TopicSummary,
)


@pytest.fixture
def mock_llm():
    """Create MockLLM instance."""
    return MockLLM()


@pytest.fixture
def writer(mock_llm):
    """Create ReportWriterAgent with mock LLM."""
    return ReportWriterAgent(llm=mock_llm)


@pytest.fixture
def sample_report_plan():
    """Sample report plan."""
    outline = ReportOutline(
        title="Customer Sentiment Analysis Report",
        sections=[
            ReportSection(
                name="Executive Summary",
                priority="high",
                key_points=["Overview", "Key metrics"],
                data_sources=["sentiment", "emotion"],
                estimated_length="2-3 paragraphs",
            ),
            ReportSection(
                name="Sentiment Analysis",
                priority="high",
                key_points=["Distribution", "Trends"],
                data_sources=["sentiment"],
                estimated_length="3-4 paragraphs",
            ),
            ReportSection(
                name="Recommendations",
                priority="high",
                key_points=["Action items"],
                data_sources=["all"],
                estimated_length="2-3 paragraphs",
            ),
        ],
        recommended_length="5-7 pages",
    )

    return ReportPlannerOutput(
        report_outline=outline,
        key_insights=["Positive sentiment dominates", "Quality is key topic"],
        recommended_focus=["Customer satisfaction", "Product quality"],
        narrative_flow=["Introduction", "Analysis", "Recommendations"],
        estimated_sections=3,
        llm_cost=0.002,
    )


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


def test_agent_initialization(writer):
    """Test agent can be initialized."""
    assert writer is not None
    assert writer.llm is not None


def test_generate_report(writer, sample_report_plan, sample_analysis_results):
    """Test report generation."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    assert output.report_text is not None
    assert len(output.report_text) > 0
    assert output.word_count > 0
    assert output.generation_time > 0
    assert output.llm_cost >= 0


def test_report_has_markdown_headers(writer, sample_report_plan, sample_analysis_results):
    """Test that report has proper Markdown headers."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    # Should have headers
    assert "#" in output.report_text


def test_sections_extracted(writer, sample_report_plan, sample_analysis_results):
    """Test that sections are extracted from report."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    # Should extract section names
    assert len(output.sections_generated) > 0
    assert all(isinstance(s, str) for s in output.sections_generated)


def test_word_count_calculated(writer, sample_report_plan, sample_analysis_results):
    """Test word count calculation."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    # Should have reasonable word count
    assert output.word_count > 50

    # Verify calculation
    actual_count = len(output.report_text.split())
    assert output.word_count == actual_count


def test_report_structure(writer, sample_report_plan, sample_analysis_results):
    """Test report has proper structure."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)
    text = output.report_text

    # Should start with title
    assert text.startswith("#")

    # Should have multiple sections
    assert text.count("##") >= 2


def test_regeneration_with_feedback(writer, sample_report_plan, sample_analysis_results):
    """Test regeneration with feedback."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan,
        tool_results=sample_analysis_results,
        regeneration_feedback="Add more specific examples and metrics.",
    )

    output = writer.execute(input_data)

    # Should still generate report with feedback
    assert output.report_text is not None
    assert len(output.report_text) > 0


def test_clean_report_text(writer):
    """Test report text cleaning."""
    # Test with markdown code blocks
    text_with_blocks = "```markdown\n# Title\nContent\n```"
    cleaned = writer._clean_report_text(text_with_blocks)

    assert "```" not in cleaned
    assert "# Title" in cleaned

    # Test with multiple blank lines
    text_with_blanks = "# Title\n\n\n\nContent"
    cleaned = writer._clean_report_text(text_with_blanks)

    # Should normalize to max 2 newlines
    assert "\n\n\n" not in cleaned


def test_extract_sections(writer):
    """Test section extraction."""
    report_text = """# Main Title

## Section 1

Content here.

## Section 2

More content.

### Subsection 2.1

Details."""

    sections = writer._extract_sections(report_text)

    # Should extract both ## and ### headers
    assert len(sections) >= 2
    assert "Section 1" in sections
    assert "Section 2" in sections


def test_extract_sections_removes_formatting(writer):
    """Test that section extraction removes markdown formatting."""
    report_text = """## **Bold Section**

## __Underlined Section__"""

    sections = writer._extract_sections(report_text)

    # Should remove ** and __
    assert "Bold Section" in sections
    assert "Underlined Section" in sections
    assert "**" not in sections[0]
    assert "__" not in sections[1]


def test_cost_tracking(writer, sample_report_plan, sample_analysis_results):
    """Test that LLM cost is tracked."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    # Mock LLM returns 0.001
    assert output.llm_cost > 0


def test_generation_time_tracked(writer, sample_report_plan, sample_analysis_results):
    """Test that generation time is tracked."""
    input_data = ReportWriterInput(
        report_plan=sample_report_plan, tool_results=sample_analysis_results
    )

    output = writer.execute(input_data)

    assert output.generation_time > 0
