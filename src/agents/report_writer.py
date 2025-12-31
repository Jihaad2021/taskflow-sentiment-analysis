"""Report writer agent - generates report text using LLM."""

import re
import time
from typing import Optional, Union

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.llm.base import BaseLLM
from src.llm.prompts.writer_prompt import create_writer_prompt
from src.models.schemas import (
    ReportWriterInput,
    ReportWriterOutput,
)
from src.utils.exceptions import AgentExecutionError


class ReportWriterAgent(BaseAgent):
    """Generate report text using LLM."""

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize report writer.

        Args:
            llm: LLM instance (defaults to MockLLM for testing)
        """
        from src.llm.mock_llm import MockLLM
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="ReportWriterAgent"))

        # Use provided LLM or default to Mock
        self.llm: Union[BaseLLM, MockLLM]

        if llm is None:
            self.llm = MockLLM()
        else:
            self.llm = llm

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Generate report text using LLM.

        Args:
            input_data: ReportWriterInput

        Returns:
            ReportWriterOutput with generated report
        """
        if not isinstance(input_data, ReportWriterInput):
            raise AgentExecutionError("Invalid input type")

        start_time = time.time()

        # Create prompt
        prompt = create_writer_prompt(
            input_data.report_plan, input_data.tool_results, input_data.regeneration_feedback
        )

        # Generate with LLM
        response = self.llm.generate(
            prompt,
            max_tokens=4000,  # Longer for full report
            temperature=0.7,
            response_format=None,  # Markdown, not JSON
        )

        # Extract and clean report text
        report_text = self._clean_report_text(response["content"])

        # Extract sections
        sections = self._extract_sections(report_text)

        # Count words
        word_count = len(report_text.split())

        generation_time = time.time() - start_time

        output = ReportWriterOutput(
            report_text=report_text,
            sections_generated=sections,
            word_count=word_count,
            generation_time=generation_time,
            llm_cost=response.get("cost", 0.0),
        )

        self.log_execution(input_data, output)
        self.logger.info(
            f"Generated report: {word_count} words, "
            f"{len(sections)} sections, cost: ${output.llm_cost:.4f}"
        )

        return output

    def _clean_report_text(self, text: str) -> str:
        """Clean and format report text.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned report text
        """
        # Remove any leading/trailing whitespace
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```markdown"):
            text = re.sub(r"^```markdown\n", "", text)
            text = re.sub(r"\n```$", "", text)
        elif text.startswith("```"):
            text = re.sub(r"^```\n", "", text)
            text = re.sub(r"\n```$", "", text)

        # Ensure proper spacing after headers
        text = re.sub(r"(#{1,6}\s+[^\n]+)\n([^\n#])", r"\1\n\n\2", text)

        # Normalize multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_sections(self, text: str) -> list:
        """Extract section names from report.

        Args:
            text: Report text

        Returns:
            List of section names
        """
        sections = []

        # Find all headers (## or ###)
        header_pattern = r"^#{2,3}\s+(.+)$"

        for line in text.split("\n"):
            match = re.match(header_pattern, line)
            if match:
                section_name = match.group(1).strip()
                # Remove any markdown formatting from section name
                section_name = re.sub(r"\*\*", "", section_name)
                section_name = re.sub(r"__", "", section_name)
                sections.append(section_name)

        return sections
