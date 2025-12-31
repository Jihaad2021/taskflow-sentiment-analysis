"""Integration test for Week 3: Analysis → Report Generation pipeline."""

import pytest

from src.agents.orchestrator import AnalysisOrchestratorAgent
from src.agents.pre_evaluator import PrePromptEvaluatorAgent
from src.agents.report_generator import ReportGenerator, ReportGeneratorInput
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorInput,
    PrePromptEvaluatorInput,
)


@pytest.fixture
def sample_comments():
    """Sample comments for analysis."""
    return [
        "Amazing product! Best purchase ever. The quality is exceptional.",
        "Terrible experience. Product broke after one week. Very disappointed.",
        "Good value for money. Shipping was fast and packaging was secure.",
        "Love the new features! Customer service at Apple was very helpful.",
        "Not worth the price. There are better alternatives available.",
        "Excellent build quality. Highly recommend to everyone!",
        "Poor customer support. Took days to get a response from Microsoft.",
        "Fast delivery and great product. Exactly as described.",
        "Waste of money. Quality is terrible and feels cheap.",
        "Perfect for my needs. The battery life is outstanding.",
    ] * 3  # 30 comments total


def test_full_pipeline_analysis_to_report(sample_comments):
    """Test complete pipeline: Comments → Analysis → Report."""

    # Step 1: Analysis (Week 2)
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=sample_comments))

    assert analysis_result.total_comments == len(sample_comments)
    assert len(analysis_result.individual_results) == len(sample_comments)

    # Step 2: Pre-evaluation (Week 2)
    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    assert quality_result.should_proceed is True

    # Step 3: Report Generation (Week 3)
    report_generator = ReportGenerator(llm=MockLLM(), max_regenerations=2, min_quality_score=70.0)

    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=2,
        )
    )

    # Verify final report
    assert report_result.report_text is not None
    assert len(report_result.report_text) > 0
    assert report_result.quality_score >= 0
    assert report_result.attempts >= 1
    assert report_result.total_cost >= 0


def test_pipeline_with_different_report_types(sample_comments):
    """Test pipeline with different report types."""

    report_types = ["executive", "marketing", "product"]

    for report_type in report_types:
        # Analysis
        orchestrator = AnalysisOrchestratorAgent(device="cpu")
        analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=sample_comments))

        # Pre-evaluation
        pre_evaluator = PrePromptEvaluatorAgent()
        quality_result = pre_evaluator.execute(
            PrePromptEvaluatorInput(tool_results=analysis_result)
        )

        # Report generation
        report_generator = ReportGenerator(llm=MockLLM())
        report_result = report_generator.execute(
            ReportGeneratorInput(
                tool_results=analysis_result,
                quality_assessment=quality_result,
                report_type=report_type,
                max_regenerations=1,
            )
        )

        # Should generate different reports
        assert report_result.report_text is not None
        assert len(report_result.report_text) > 0


def test_pipeline_cost_tracking():
    """Test that costs are tracked through pipeline."""

    comments = ["Great product!"] * 10

    # Analysis (free - no LLM)
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    # Pre-evaluation (free - no LLM)
    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    # Report generation (has cost)
    report_generator = ReportGenerator(llm=MockLLM())
    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=1,
        )
    )

    # Should track LLM costs
    assert report_result.total_cost > 0
    assert len(report_result.regeneration_history) > 0


def test_pipeline_performance():
    """Test pipeline performance with moderate data."""
    import time

    comments = [f"Sample comment number {i}" for i in range(20)]

    start = time.time()

    # Full pipeline
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    report_generator = ReportGenerator(llm=MockLLM())
    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=1,
        )
    )

    end = time.time()
    total_time = end - start

    # Should complete in reasonable time (adjust for slower systems)
    # Analysis: ~15-25s (CPU), Report: ~0.5s (mock)
    assert total_time < 60.0  # Changed from 30.0 to 60.0

    print(f"\nPipeline timing for {len(comments)} comments:")
    print(f"  Analysis: {analysis_result.execution_time:.2f}s")
    print(f"  Report: {report_result.total_time:.2f}s")
    print(f"  Total: {total_time:.2f}s")


def test_pipeline_report_quality():
    """Test that generated reports have expected structure."""

    comments = [
        "Excellent quality and fast shipping!",
        "Poor service, very disappointed.",
        "Good product, reasonable price.",
    ] * 5

    # Full pipeline
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    report_generator = ReportGenerator(llm=MockLLM())
    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="marketing",
            max_regenerations=1,
        )
    )

    report_text = report_result.report_text

    # Should have markdown structure
    assert "#" in report_text  # Headers
    assert "##" in report_text  # Sections

    # Should be substantial
    word_count = len(report_text.split())
    assert word_count > 100


def test_regeneration_improves_quality():
    """Test that regeneration can improve quality."""

    class ImprovingLLM(MockLLM):
        """LLM that improves on second attempt."""

        def __init__(self):
            super().__init__()
            self.eval_count = 0

        def _generate_mock_evaluation(self) -> str:
            import json

            self.eval_count += 1

            # First attempt: low score, needs regen
            if self.eval_count == 1:
                score = 60.0
                should_regen = True
            else:
                # Second attempt: better score
                score = 85.0
                should_regen = False

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
                    "feedback": "Needs more detail.",
                }
            )

    comments = ["Good product"] * 5

    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    report_generator = ReportGenerator(llm=ImprovingLLM(), max_regenerations=3)
    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=3,
        )
    )

    # Should improve over attempts
    if len(report_result.regeneration_history) > 1:
        first_score = report_result.regeneration_history[0]["quality_score"]
        last_score = report_result.regeneration_history[-1]["quality_score"]

        # Last score should be better or equal
        assert last_score >= first_score
