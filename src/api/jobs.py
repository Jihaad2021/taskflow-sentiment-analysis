"""Background job processing for analysis and report generation."""

import os
from datetime import datetime

from dotenv import load_dotenv

from src.agents.column_detector import ColumnDetectorAgent
from src.agents.data_validator import DataValidatorAgent
from src.agents.orchestrator import AnalysisOrchestratorAgent
from src.agents.pre_evaluator import PrePromptEvaluatorAgent
from src.agents.report_generator import ReportGenerator, ReportGeneratorInput
from src.api.storage import storage
from src.llm.anthropic_llm import ClaudeLLM
from src.models.schemas import (
    AnalysisOrchestratorInput,
    ColumnDetectorInput,
    DataValidatorInput,
    PrePromptEvaluatorInput,
)
from src.utils.logger import setup_logger

# Load environment
load_dotenv()

logger = setup_logger("JobProcessor")


async def process_job(job_id: str):
    """Process analysis job in background.

    Args:
        job_id: Job ID to process
    """
    logger.info(f"Starting job {job_id}")

    try:
        # Get job
        job = storage.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Get upload
        upload = storage.get_upload(job["upload_id"])
        if not upload:
            logger.error(f"Upload {job['upload_id']} not found")
            storage.update_job(job_id, status="failed", error="Upload not found")
            return

        df = upload["dataframe"]

        # Update status
        storage.update_job(
            job_id, status="processing", progress=5.0, current_stage="Detecting text column..."
        )

        # Step 1: Column Detection (5%)
        logger.info(f"Job {job_id}: Detecting column")

        text_column = job["text_column"]
        if not text_column:
            # Auto-detect
            col_detector = ColumnDetectorAgent()
            col_result = col_detector.execute(ColumnDetectorInput(dataframe=df))
            text_column = col_result.column_name

        storage.update_job(job_id, progress=10.0, current_stage="Validating data...")

        # Step 2: Data Validation (10%)
        logger.info(f"Job {job_id}: Validating data")

        validator = DataValidatorAgent()
        val_result = validator.execute(DataValidatorInput(dataframe=df, text_column=text_column))

        if val_result.status == "fail":
            storage.update_job(
                job_id,
                status="failed",
                error="Data validation failed: " + ", ".join(val_result.issues),
            )
            return

        comments = val_result.cleaned_data[text_column].tolist()

        storage.update_job(
            job_id, progress=20.0, current_stage=f"Analyzing {len(comments)} comments..."
        )

        # Step 3: Analysis (20% -> 60%)
        logger.info(f"Job {job_id}: Running analysis on {len(comments)} comments")

        orchestrator = AnalysisOrchestratorAgent(device="cpu")
        analysis_result = orchestrator.execute(AnalysisOrchestratorInput(comments=comments))

        storage.update_job(job_id, progress=65.0, current_stage="Quality check...")

        # Step 4: Pre-evaluation (65%)
        logger.info(f"Job {job_id}: Pre-evaluation")

        pre_eval = PrePromptEvaluatorAgent()
        quality_result = pre_eval.execute(PrePromptEvaluatorInput(tool_results=analysis_result))

        if not quality_result.should_proceed:
            storage.update_job(
                job_id,
                status="failed",
                error="Analysis quality too low: " + ", ".join(quality_result.issues),
            )
            return

        storage.update_job(
            job_id, progress=70.0, current_stage="Generating report (this may take 30-60s)..."
        )

        # Step 5: Report Generation (70% -> 100%)
        logger.info(f"Job {job_id}: Generating report")

        # Initialize LLM
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("sk-ant-your"):
            logger.error("ANTHROPIC_API_KEY not configured")
            storage.update_job(job_id, status="failed", error="LLM API key not configured")
            return

        llm = ClaudeLLM(api_key=api_key)
        generator = ReportGenerator(llm=llm, max_regenerations=job.get("max_regenerations", 3))

        report_result = generator.execute(
            ReportGeneratorInput(
                tool_results=analysis_result,
                quality_assessment=quality_result,
                report_type=job["report_type"],
                max_regenerations=job.get("max_regenerations", 3),
            )
        )

        storage.update_job(job_id, progress=100.0, current_stage="Complete!")

        # Save result
        result = {
            "job_id": job_id,
            "report_text": report_result.report_text,
            "quality_score": report_result.quality_score,
            "word_count": len(report_result.report_text.split()),
            "cost": report_result.total_cost,
            "total_time": report_result.total_time,
            "created_at": datetime.utcnow(),
        }

        storage.update_job(job_id, status="completed", progress=100.0, result=result)

        logger.info(
            f"Job {job_id} completed: "
            f"quality={report_result.quality_score:.1f}, "
            f"cost=${report_result.total_cost:.4f}"
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        storage.update_job(job_id, status="failed", error=str(e))
