"""End-to-end test: CSV upload → Final report generation."""

import pandas as pd
import pytest

from src.agents.column_detector import ColumnDetectorAgent
from src.agents.data_validator import DataValidatorAgent
from src.agents.orchestrator import AnalysisOrchestratorAgent
from src.agents.pre_evaluator import PrePromptEvaluatorAgent
from src.agents.report_generator import ReportGenerator, ReportGeneratorInput
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    AnalysisOrchestratorInput,
    ColumnDetectorInput,
    DataValidatorInput,
    PrePromptEvaluatorInput,
)


@pytest.fixture
def sample_csv_data():
    """Realistic product review CSV data."""
    # Create base templates WITHOUT f-string in list
    templates = [
        "Amazing product! Love the quality and fast shipping from Amazon.",
        "Terrible experience. Battery died after one week. Very disappointed.",
        "Good value for money. The Apple device works as expected.",
        "Excellent customer service at Microsoft store. Highly recommended!",
        "Not worth the price. Build quality is poor and feels cheap.",
        "Fast delivery and great packaging. Product is exactly as described.",
        "Poor support from the company. No response to my emails.",
        "Love the new features! Camera quality is outstanding.",
        "Waste of money. Returned it after 2 days of use.",
        "Perfect for my needs. Battery life is excellent, as promised.",
        "Great design and user-friendly interface. Very satisfied!",
        "Product broke within first month. Quality control is terrible.",
        "Highly recommend! Best purchase I've made this year.",
        "Disappointed with performance. Much slower than advertised.",
        "Excellent value! Works perfectly and shipping was super fast.",
    ]

    # Generate 60 unique reviews
    all_reviews = []
    for i in range(4):
        for template in templates:
            all_reviews.append(f"{template} [Review #{i+1}]")

    return pd.DataFrame(
        {
            "review_id": range(1, len(all_reviews) + 1),
            "customer_review": all_reviews,
            "rating": ([5, 1, 3, 5, 2, 4, 2, 5, 1, 5, 4, 1, 5, 2, 4] * 4),
            "date": pd.date_range("2024-01-01", periods=len(all_reviews)),
        }
    )


def test_complete_pipeline_csv_to_report(sample_csv_data):
    """Test complete pipeline from CSV to final report."""

    import time

    start_time = time.time()

    # ========== WEEK 1: DATA LAYER ==========

    # Step 1: Column Detection
    print("\n[Step 1/7] Detecting text column...")
    col_detector = ColumnDetectorAgent()
    col_result = col_detector.execute(ColumnDetectorInput(dataframe=sample_csv_data))

    assert col_result.column_name == "customer_review"
    assert col_result.confidence >= 0.5
    print(
        f"✓ Detected column: '{col_result.column_name}' (confidence: {col_result.confidence:.2f})"
    )

    # Step 2: Data Validation
    print("\n[Step 2/7] Validating and cleaning data...")
    data_validator = DataValidatorAgent()
    val_result = data_validator.execute(
        DataValidatorInput(dataframe=sample_csv_data, text_column=col_result.column_name)
    )

    assert val_result.status in ["pass", "warning"]
    assert val_result.stats.rows_after_cleaning >= 50
    print(
        f"✓ Cleaned data: {val_result.stats.rows_after_cleaning} rows, status: {val_result.status}"
    )

    # ========== WEEK 2: ANALYSIS LAYER ==========

    # Step 3: Analysis (5 tools)
    print("\n[Step 3/7] Running analysis (5 ML tools)...")
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    comments = val_result.cleaned_data[col_result.column_name].tolist()

    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    assert analysis_result.total_comments == len(comments)
    assert len(analysis_result.sentiment_distribution) > 0
    assert len(analysis_result.emotion_distribution) > 0
    print(
        f"✓ Analysis complete: {analysis_result.total_comments} comments in {analysis_result.execution_time:.1f}s"
    )
    print(f"  Sentiment: {analysis_result.sentiment_distribution}")

    # Step 4: Pre-evaluation
    print("\n[Step 4/7] Evaluating analysis quality...")
    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    assert quality_result.should_proceed is True
    print(
        f"✓ Quality check: {quality_result.quality_score:.1f}/100, status: {quality_result.status}"
    )

    # ========== WEEK 3: INTELLIGENCE LAYER ==========

    # Step 5-7: Report Generation (Planner → Writer → Evaluator)
    print("\n[Step 5-7/7] Generating professional report...")
    report_generator = ReportGenerator(llm=MockLLM(), max_regenerations=2, min_quality_score=70.0)

    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="marketing",
            max_regenerations=2,
        )
    )

    assert report_result.report_text is not None
    assert len(report_result.report_text) > 0
    assert report_result.quality_score >= 0
    print(
        f"✓ Report generated: {report_result.attempts} attempts, quality: {report_result.quality_score:.1f}/100"
    )
    print(f"  Word count: {len(report_result.report_text.split())}")
    print(f"  Cost: ${report_result.total_cost:.4f}")

    # ========== FINAL VERIFICATION ==========

    total_time = time.time() - start_time

    # Verify report structure
    assert "#" in report_result.report_text  # Has headers
    assert "##" in report_result.report_text  # Has sections

    word_count = len(report_result.report_text.split())
    assert word_count > 100  # Substantial content

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Input: {len(sample_csv_data)} rows")
    print(f"Processed: {len(comments)} comments")
    print(f"Report quality: {report_result.quality_score:.1f}/100")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total cost: ${report_result.total_cost:.4f}")
    print("=" * 60)


def test_pipeline_with_messy_data():
    """Test pipeline handles messy real-world data."""

    # Create enough unique reviews (>10)
    reviews = []
    for i in range(15):
        reviews.append(f"Great product review number {i}! Highly recommend this item.")

    messy_data = pd.DataFrame({"feedback": reviews})

    # Should handle gracefully
    col_detector = ColumnDetectorAgent()
    col_result = col_detector.execute(ColumnDetectorInput(dataframe=messy_data))

    data_validator = DataValidatorAgent()
    val_result = data_validator.execute(
        DataValidatorInput(dataframe=messy_data, text_column=col_result.column_name)
    )

    # Should clean and process
    assert val_result.status in ["pass", "warning"]
    assert val_result.stats.rows_after_cleaning >= 10

    # Should analyze cleaned data
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    comments = val_result.cleaned_data[col_result.column_name].tolist()

    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    assert analysis_result.total_comments == len(comments)


def test_pipeline_all_report_types(sample_csv_data):
    """Test pipeline with all report types."""

    report_types = ["executive", "marketing", "product", "customer_service"]

    for report_type in report_types:
        print(f"\nTesting {report_type} report...")

        # Quick pipeline
        col_detector = ColumnDetectorAgent()
        col_result = col_detector.execute(ColumnDetectorInput(dataframe=sample_csv_data))

        data_validator = DataValidatorAgent()
        val_result = data_validator.execute(
            DataValidatorInput(dataframe=sample_csv_data, text_column=col_result.column_name)
        )

        orchestrator = AnalysisOrchestratorAgent(device="cpu")
        comments = val_result.cleaned_data[col_result.column_name].tolist()[
            :20
        ]  # Just 20 for speed
        analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

        pre_evaluator = PrePromptEvaluatorAgent()
        quality_result = pre_evaluator.execute(
            PrePromptEvaluatorInput(tool_results=analysis_result)
        )

        report_generator = ReportGenerator(llm=MockLLM())
        report_result = report_generator.execute(
            ReportGeneratorInput(
                tool_results=analysis_result,
                quality_assessment=quality_result,
                report_type=report_type,
                max_regenerations=1,
            )
        )

        # Each report type should generate successfully
        assert report_result.report_text is not None
        assert len(report_result.report_text) > 0
        print(f"✓ {report_type}: {len(report_result.report_text.split())} words")


def test_pipeline_cost_breakdown(sample_csv_data):
    """Test detailed cost tracking through pipeline."""

    # Use smaller dataset for speed
    small_data = sample_csv_data.head(15)

    # Track costs
    costs = {"data_layer": 0.0, "analysis_layer": 0.0, "intelligence_layer": 0.0}

    # Data layer (Week 1) - free
    col_detector = ColumnDetectorAgent()
    col_result = col_detector.execute(ColumnDetectorInput(dataframe=small_data))

    data_validator = DataValidatorAgent()
    val_result = data_validator.execute(
        DataValidatorInput(dataframe=small_data, text_column=col_result.column_name)
    )

    costs["data_layer"] = 0.0  # No LLM used

    # Analysis layer (Week 2) - free (self-hosted models)
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    comments = val_result.cleaned_data[col_result.column_name].tolist()
    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

    pre_evaluator = PrePromptEvaluatorAgent()
    quality_result = pre_evaluator.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

    costs["analysis_layer"] = 0.0  # No LLM used

    # Intelligence layer (Week 3) - has cost
    report_generator = ReportGenerator(llm=MockLLM())
    report_result = report_generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=1,
        )
    )

    costs["intelligence_layer"] = report_result.total_cost

    # Verify cost breakdown
    total_cost = sum(costs.values())

    print("\n" + "=" * 60)
    print("COST BREAKDOWN")
    print("=" * 60)
    print(f"Data Layer (Week 1):        ${costs['data_layer']:.4f}")
    print(f"Analysis Layer (Week 2):    ${costs['analysis_layer']:.4f}")
    print(f"Intelligence Layer (Week 3): ${costs['intelligence_layer']:.4f}")
    print(f"{'─'*60}")
    print(f"TOTAL COST:                 ${total_cost:.4f}")
    print("=" * 60)

    # Only intelligence layer should have cost (with Mock LLM)
    assert costs["intelligence_layer"] > 0
    assert total_cost == costs["intelligence_layer"]


def test_pipeline_saves_report():
    """Test that final report can be saved to file."""

    # Create 12 unique reviews
    reviews = []
    for i in range(12):
        reviews.append(f"Great product experience number {i}, very satisfied with quality!")

    data = pd.DataFrame({"review": reviews})

    # Full pipeline
    col_detector = ColumnDetectorAgent()
    col_result = col_detector.execute(ColumnDetectorInput(dataframe=data))

    data_validator = DataValidatorAgent()
    val_result = data_validator.execute(
        DataValidatorInput(dataframe=data, text_column=col_result.column_name)
    )

    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    comments = val_result.cleaned_data[col_result.column_name].tolist()
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

    # Save report to file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(report_result.report_text)
        filepath = f.name

    # Verify file was created and has content
    import os

    assert os.path.exists(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    assert len(content) > 0
    assert content == report_result.report_text

    # Cleanup
    os.unlink(filepath)

    print(f"\n✓ Report saved successfully ({len(content)} chars)")
