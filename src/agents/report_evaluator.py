"""Report evaluator agent - validates report quality using LLM."""

import json
from typing import Dict, Optional, Union

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.llm.base import BaseLLM
from src.llm.prompts.evaluator_prompt import create_evaluator_prompt
from src.models.schemas import (
    EvaluationCriterion,
    ReportEvaluatorInput,
    ReportEvaluatorOutput,
)
from src.utils.exceptions import AgentExecutionError


class ReportEvaluatorAgent(BaseAgent):
    """Evaluate report quality using LLM."""

    # Weights for overall score calculation
    WEIGHTS = {
        "completeness": 0.30,
        "factual_accuracy": 0.25,
        "coherence": 0.20,
        "actionability": 0.15,
        "hallucination": 0.10,
    }

    def __init__(self, llm: Optional[BaseLLM] = None, min_score: float = 70.0):
        """Initialize report evaluator.

        Args:
            llm: LLM instance (defaults to MockLLM for testing)
            min_score: Minimum quality score to pass (0-100)
        """
        from src.llm.mock_llm import MockLLM
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="ReportEvaluatorAgent"))

        self.llm: Union[BaseLLM, MockLLM]

        if llm is None:
            self.llm = MockLLM()
        else:
            self.llm = llm

        self.min_score = min_score

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Evaluate report quality.

        Args:
            input_data: ReportEvaluatorInput

        Returns:
            ReportEvaluatorOutput with evaluation results
        """
        if not isinstance(input_data, ReportEvaluatorInput):
            raise AgentExecutionError("Invalid input type")

        # Create evaluation prompt
        prompt = create_evaluator_prompt(
            input_data.report_text, input_data.original_data, input_data.report_plan
        )

        # Get LLM evaluation
        response = self.llm.generate(
            prompt,
            max_tokens=1500,
            temperature=0.3,  # Lower temp for more consistent evaluation
            response_format="json",
        )

        # Parse evaluation
        try:
            evaluation = self._parse_evaluation(response["content"])
        except Exception as e:
            self.logger.error(f"Failed to parse evaluation: {e}")
            # Fallback to basic evaluation
            evaluation = self._create_fallback_evaluation(input_data.report_text)

        # Create output
        output = ReportEvaluatorOutput(
            status=evaluation["status"],
            quality_score=evaluation["quality_score"],
            checks=evaluation["checks"],
            issues=evaluation["issues"],
            should_regenerate=evaluation["should_regenerate"],
            feedback=evaluation["feedback"],
            llm_cost=response.get("cost", 0.0),
        )

        self.log_execution(input_data, output)
        self.logger.info(
            f"Evaluated report: score={output.quality_score:.1f}/100, "
            f"status={output.status}, regenerate={output.should_regenerate}"
        )

        return output

    def _parse_evaluation(self, content: str) -> Dict:
        """Parse LLM evaluation response.

        Args:
            content: LLM JSON response

        Returns:
            Dictionary with evaluation data
        """
        # Clean content
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        # Parse JSON
        data = json.loads(content)

        # Create EvaluationCriterion objects
        checks = {}
        for criterion_name, criterion_data in data["checks"].items():
            checks[criterion_name] = EvaluationCriterion(**criterion_data)

        # Determine status
        status = "pass" if not data["should_regenerate"] else "fail"

        return {
            "status": status,
            "quality_score": data["overall_score"],
            "checks": checks,
            "issues": data.get("issues", []),
            "should_regenerate": data["should_regenerate"],
            "feedback": data.get("feedback", None),
        }

    def _create_fallback_evaluation(self, report_text: str) -> Dict:
        """Create fallback evaluation if LLM fails.

        Args:
            report_text: Report text

        Returns:
            Basic evaluation dictionary
        """
        self.logger.warning("Using fallback evaluation due to LLM failure")

        # Basic checks
        word_count = len(report_text.split())
        has_headers = "##" in report_text
        has_recommendations = "recommend" in report_text.lower()

        # Calculate simple score
        score = 50.0  # Base score
        if word_count > 200:
            score += 15.0
        if has_headers:
            score += 15.0
        if has_recommendations:
            score += 10.0

        # Create basic checks
        checks = {
            "completeness": EvaluationCriterion(
                score=score, issues=[] if word_count > 200 else ["Report too short"], critical=False
            ),
            "factual_accuracy": EvaluationCriterion(score=80.0, issues=[], critical=False),
            "coherence": EvaluationCriterion(score=70.0, issues=[], critical=False),
            "actionability": EvaluationCriterion(
                score=60.0 if has_recommendations else 40.0,
                issues=[] if has_recommendations else ["No recommendations found"],
                critical=False,
            ),
            "hallucination": EvaluationCriterion(score=90.0, issues=[], critical=False),
        }

        # Calculate weighted score
        weighted_score = sum(
            checks[name].score * self.WEIGHTS[name] for name in self.WEIGHTS.keys()
        )

        should_regenerate = weighted_score < self.min_score

        return {
            "status": "fail" if should_regenerate else "pass",
            "quality_score": weighted_score,
            "checks": checks,
            "issues": ["Fallback evaluation used - LLM unavailable"],
            "should_regenerate": should_regenerate,
            "feedback": "Report generated but quality could not be fully evaluated.",
        }
