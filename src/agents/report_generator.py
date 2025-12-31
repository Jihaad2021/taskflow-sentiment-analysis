"""Report generator - orchestrates full report generation pipeline with regeneration."""

from typing import Optional

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.agents.report_evaluator import ReportEvaluatorAgent
from src.agents.report_planner import ReportPlannerAgent
from src.agents.report_writer import ReportWriterAgent
from src.llm.base import BaseLLM
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    PrePromptEvaluatorOutput,
    ReportEvaluatorInput,
    ReportPlannerInput,
    ReportWriterInput,
)
from src.utils.exceptions import AgentExecutionError


class ReportGeneratorInput(BaseModel):
    """Input for ReportGenerator."""

    tool_results: AnalysisOrchestratorOutput
    quality_assessment: PrePromptEvaluatorOutput
    report_type: str
    max_regenerations: int = 3


class ReportGeneratorOutput(BaseModel):
    """Output from ReportGenerator."""

    report_text: str
    quality_score: float
    attempts: int
    total_cost: float
    total_time: float
    final_status: str
    regeneration_history: list = []


class ReportGenerator(BaseAgent):
    """Generate report with quality checks and regeneration."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_regenerations: int = 3,
        min_quality_score: float = 70.0,
    ):
        """Initialize report generator.

        Args:
            llm: LLM instance for all agents
            max_regenerations: Maximum regeneration attempts
            min_quality_score: Minimum acceptable quality score
        """
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="ReportGenerator"))

        # Initialize the 3 agents
        self.planner = ReportPlannerAgent(llm=llm)
        self.writer = ReportWriterAgent(llm=llm)
        self.evaluator = ReportEvaluatorAgent(llm=llm, min_score=min_quality_score)

        self.max_regenerations = max_regenerations
        self.min_quality_score = min_quality_score

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Generate report with regeneration loop.

        Args:
            input_data: ReportGeneratorInput

        Returns:
            ReportGeneratorOutput with final report
        """
        if not isinstance(input_data, ReportGeneratorInput):
            raise AgentExecutionError("Invalid input type")

        import time

        start_time = time.time()

        total_cost = 0.0
        regeneration_history = []

        # Step 1: Plan report structure
        self.logger.info("Step 1: Planning report structure...")
        plan_result = self.planner.execute(
            ReportPlannerInput(
                tool_results=input_data.tool_results,
                report_type=input_data.report_type,
                quality_assessment=input_data.quality_assessment,
            )
        )
        total_cost += plan_result.llm_cost

        self.logger.info(
            f"Report plan created: {plan_result.estimated_sections} sections, "
            f"cost: ${plan_result.llm_cost:.4f}"
        )

        # Step 2: Generate report with regeneration loop
        report_text = None
        quality_score = 0.0
        final_status = "failed"
        feedback = None

        for attempt in range(1, input_data.max_regenerations + 1):
            self.logger.info(f"Step 2: Generating report (attempt {attempt})...")

            # Write report
            writer_result = self.writer.execute(
                ReportWriterInput(
                    report_plan=plan_result,
                    tool_results=input_data.tool_results,
                    regeneration_feedback=feedback,
                )
            )
            total_cost += writer_result.llm_cost

            self.logger.info(
                f"Report generated: {writer_result.word_count} words, "
                f"cost: ${writer_result.llm_cost:.4f}"
            )

            # Step 3: Evaluate report
            self.logger.info("Step 3: Evaluating report quality...")
            eval_result = self.evaluator.execute(
                ReportEvaluatorInput(
                    report_text=writer_result.report_text,
                    original_data=input_data.tool_results,
                    report_plan=plan_result,
                )
            )
            total_cost += eval_result.llm_cost

            self.logger.info(
                f"Evaluation complete: score={eval_result.quality_score:.1f}/100, "
                f"status={eval_result.status}"
            )

            # Store attempt info
            regeneration_history.append(
                {
                    "attempt": attempt,
                    "word_count": writer_result.word_count,
                    "quality_score": eval_result.quality_score,
                    "status": eval_result.status,
                    "issues": eval_result.issues,
                    "cost": writer_result.llm_cost + eval_result.llm_cost,
                }
            )

            # Check if we should stop
            report_text = writer_result.report_text
            quality_score = eval_result.quality_score

            if not eval_result.should_regenerate:
                self.logger.info(f"✅ Report passed quality checks on attempt {attempt}")
                final_status = "passed"
                break

            # Check if we can regenerate
            if attempt < input_data.max_regenerations:
                self.logger.warning(
                    f"⚠️  Report needs improvement (score: {quality_score:.1f}/100). "
                    f"Regenerating... ({attempt}/{input_data.max_regenerations})"
                )
                feedback = eval_result.feedback
            else:
                self.logger.warning(
                    f"❌ Maximum regenerations reached. "
                    f"Using best attempt (score: {quality_score:.1f}/100)"
                )
                final_status = "max_attempts_reached"

        total_time = time.time() - start_time

        # Create output
        output = ReportGeneratorOutput(
            report_text=report_text,
            quality_score=quality_score,
            attempts=len(regeneration_history),
            total_cost=total_cost,
            total_time=total_time,
            final_status=final_status,
            regeneration_history=regeneration_history,
        )

        self.log_execution(input_data, output)
        self.logger.info(
            f"Report generation complete: {output.attempts} attempts, "
            f"final score: {output.quality_score:.1f}/100, "
            f"cost: ${output.total_cost:.4f}, time: {output.total_time:.1f}s"
        )

        return output
