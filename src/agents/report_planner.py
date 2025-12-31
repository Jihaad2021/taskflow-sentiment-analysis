"""Report planner agent - creates report structure using LLM."""

import json
from typing import Optional, Union

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.llm.base import BaseLLM
from src.llm.prompts.planner_prompt import create_planner_prompt
from src.models.schemas import (
    ReportOutline,
    ReportPlannerInput,
    ReportPlannerOutput,
    ReportSection,
)
from src.utils.exceptions import AgentExecutionError


class ReportPlannerAgent(BaseAgent):
    """Plan report structure using LLM."""

    VALID_REPORT_TYPES = ["executive", "marketing", "product", "customer_service", "comprehensive"]

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize report planner."""
        from src.llm.mock_llm import MockLLM
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="ReportPlannerAgent"))

        # Type as Union to satisfy mypy
        self.llm: Union[BaseLLM, MockLLM]

        if llm is None:
            self.llm = MockLLM()
        else:
            self.llm = llm

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Create report plan using LLM.

        Args:
            input_data: ReportPlannerInput

        Returns:
            ReportPlannerOutput with report outline
        """
        if not isinstance(input_data, ReportPlannerInput):
            raise AgentExecutionError("Invalid input type")

        # Validate report type
        if input_data.report_type not in self.VALID_REPORT_TYPES:
            raise AgentExecutionError(
                f"Invalid report type. Must be one of: {self.VALID_REPORT_TYPES}"
            )

        # Create prompt
        prompt = create_planner_prompt(
            input_data.tool_results, input_data.report_type, input_data.quality_assessment
        )

        # Generate with LLM
        response = self.llm.generate(
            prompt, max_tokens=2000, temperature=0.7, response_format="json"
        )

        # Parse response
        try:
            plan_data = self._parse_response(response["content"])
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Return default plan on failure
            plan_data = self._create_default_plan(input_data.report_type)

        # Create output
        output = ReportPlannerOutput(
            report_outline=plan_data["outline"],
            key_insights=plan_data["key_insights"],
            recommended_focus=plan_data["recommended_focus"],
            narrative_flow=plan_data["narrative_flow"],
            estimated_sections=len(plan_data["outline"].sections),
            llm_cost=response.get("cost", 0.0),
        )

        self.log_execution(input_data, output)
        self.logger.info(
            f"Created {input_data.report_type} report plan: "
            f"{output.estimated_sections} sections, cost: ${output.llm_cost:.4f}"
        )

        return output

    def _parse_response(self, content: str) -> dict:
        """Parse LLM JSON response.

        Args:
            content: LLM response content

        Returns:
            Dictionary with parsed plan
        """
        # Clean content
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        # Parse JSON
        data = json.loads(content)

        # Create ReportOutline
        sections = [ReportSection(**section) for section in data["sections"]]

        outline = ReportOutline(
            title=data["title"], sections=sections, recommended_length=data["recommended_length"]
        )

        return {
            "outline": outline,
            "key_insights": data.get("key_insights", []),
            "recommended_focus": data.get("recommended_focus", []),
            "narrative_flow": data.get("narrative_flow", []),
        }

    def _create_default_plan(self, report_type: str) -> dict:
        """Create default plan if LLM fails.

        Args:
            report_type: Type of report

        Returns:
            Default plan dictionary
        """
        self.logger.warning("Using default plan due to LLM failure")

        default_sections = [
            ReportSection(
                name="Executive Summary",
                priority="high",
                key_points=["Overview of findings", "Key metrics"],
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
                name="Key Topics",
                priority="medium",
                key_points=["Main themes", "Topic sentiment"],
                data_sources=["topics"],
                estimated_length="2-3 paragraphs",
            ),
            ReportSection(
                name="Recommendations",
                priority="high",
                key_points=["Action items", "Next steps"],
                data_sources=["all"],
                estimated_length="2-3 paragraphs",
            ),
        ]

        outline = ReportOutline(
            title=f"{report_type.title()} Sentiment Analysis Report",
            sections=default_sections,
            recommended_length="5-7 pages",
        )

        return {
            "outline": outline,
            "key_insights": ["Analysis completed", "Recommendations provided"],
            "recommended_focus": ["Sentiment trends", "Customer feedback"],
            "narrative_flow": ["Introduction", "Analysis", "Recommendations"],
        }
