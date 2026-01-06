"""Test real API integration before production."""
# ruff: noqa: E402

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

from src.agents.column_detector import ColumnDetectorAgent
from src.agents.data_validator import DataValidatorAgent
from src.agents.orchestrator import AnalysisOrchestratorAgent
from src.agents.pre_evaluator import PrePromptEvaluatorAgent
from src.agents.report_generator import ReportGenerator, ReportGeneratorInput
from src.llm.anthropic_llm import ClaudeLLM
from src.models.schemas import (
    AnalysisOrchestratorInput,
    ColumnDetectorInput,
    DataValidatorInput,
    PrePromptEvaluatorInput,
)


def test_real_api():
    """Test full pipeline with real Claude API."""

    # Load environment
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key.startswith("sk-ant-your"):
        print("❌ Please set ANTHROPIC_API_KEY in .env file")
        return

    print("=" * 60)
    print("TESTING REAL API INTEGRATION")
    print("=" * 60)

    # Create test data
    print("\n[1/6] Creating test data...")
    test_data = pd.DataFrame(
        {
            "review": [
                "Amazing product! Love the quality and fast delivery.",
                "Terrible experience. Product broke after one day.",
                "Good value for money. Works as expected.",
                "Excellent customer service. Highly recommended!",
                "Not worth the price. Very disappointed.",
                "Fast shipping and great packaging. Very satisfied!",
                "Poor quality. Returned it immediately.",
                "Love it! Best purchase I've made this year.",
                "Waste of money. Don't buy this product.",
                "Perfect! Exactly what I needed.",
            ]
            * 3  # 30 reviews
        }
    )
    print(f"✓ Created {len(test_data)} test reviews")

    # Week 1: Data Layer
    print("\n[2/6] Data Layer (Column Detection + Validation)...")
    col_detector = ColumnDetectorAgent()
    col_result = col_detector.execute(ColumnDetectorInput(dataframe=test_data))
    print(f"✓ Detected column: '{col_result.column_name}'")

    validator = DataValidatorAgent()
    val_result = validator.execute(
        DataValidatorInput(dataframe=test_data, text_column=col_result.column_name)
    )
    print(f"✓ Cleaned: {val_result.stats.rows_after_cleaning} rows")

    # Week 2: Analysis Layer
    print("\n[3/6] Analysis Layer (5 ML Tools)...")
    orchestrator = AnalysisOrchestratorAgent(device="cpu")
    comments = val_result.cleaned_data[col_result.column_name].tolist()

    analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))
    print(f"✓ Analyzed {analysis_result.total_comments} comments")
    print(f"  Sentiment: {analysis_result.sentiment_distribution}")

    # Week 2: Pre-Evaluation
    print("\n[4/6] Quality Check...")
    pre_eval = PrePromptEvaluatorAgent()
    quality_result = pre_eval.execute(PrePromptEvaluatorInput(tool_results=analysis_result))
    print(f"✓ Quality score: {quality_result.quality_score:.1f}/100")

    # Week 3: Report Generation (REAL API!)
    print("\n[5/6] Report Generation (REAL CLAUDE API)...")
    print("⚠️  This will use real API credits!")

    llm = ClaudeLLM(api_key=api_key)
    generator = ReportGenerator(llm=llm, max_regenerations=2)

    report_result = generator.execute(
        ReportGeneratorInput(
            tool_results=analysis_result,
            quality_assessment=quality_result,
            report_type="executive",
            max_regenerations=2,
        )
    )

    print("✓ Report generated!")
    print(f"  Quality: {report_result.quality_score:.1f}/100")
    print(f"  Word count: {len(report_result.report_text.split())}")
    print(f"  Attempts: {report_result.attempts}")
    print(f"  Cost: ${report_result.total_cost:.4f}")
    print(f"  Time: {report_result.total_time:.1f}s")

    # Save report
    print("\n[6/6] Saving report...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "test_real_api_report.md"
    with open(report_path, "w") as f:
        f.write(report_result.report_text)

    print(f"✓ Saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUCCESS! REAL API TEST COMPLETE")
    print("=" * 60)
    print(f"Total cost: ${report_result.total_cost:.4f}")
    print(f"Report quality: {report_result.quality_score:.1f}/100")
    print(f"Processing time: {report_result.total_time:.1f}s")
    print(f"\nReport saved to: {report_path}")
    print("=" * 60)

    # Show preview
    print("\nREPORT PREVIEW (first 500 chars):")
    print("-" * 60)
    print(report_result.report_text[:500])
    print("-" * 60)

    return report_result


if __name__ == "__main__":
    try:
        result = test_real_api()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
