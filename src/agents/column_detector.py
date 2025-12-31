"""Column detection agent."""

import json
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.llm.mock_llm import MockLLM
from src.models.schemas import (
    ColumnCandidate,
    ColumnDetectorInput,
    ColumnDetectorOutput,
)
from src.utils.exceptions import ValidationError


class ColumnDetectorAgent(BaseAgent):
    """Detect text column in uploaded CSV using heuristic approach."""

    # Keywords that suggest text column
    TEXT_COLUMN_KEYWORDS = [
        "comment",
        "text",
        "review",
        "feedback",
        "message",
        "description",
        "content",
        "body",
        "post",
        "tweet",
        "response",
        "answer",
        "note",
        "remark",
    ]

    def __init__(self, llm=None):
        """Initialize column detector.

        Args:
            llm: Optional LLM instance for fallback (defaults to MockLLM)
        """
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="ColumnDetectorAgent"))
        self.llm = llm or MockLLM()

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Detect text column in DataFrame.

        Args:
            input_data: ColumnDetectorInput with dataframe

        Returns:
            ColumnDetectorOutput with detected column

        Raises:
            ValidationError: If DataFrame is invalid
        """
        # Cast to specific type for type checker
        if not isinstance(input_data, ColumnDetectorInput):
            raise ValidationError("Invalid input type")

        df = input_data.dataframe

        # Validate DataFrame
        if not self.validate_input(input_data):
            raise ValidationError("Invalid DataFrame")

        # Get candidates
        candidates = self._get_candidates(df, input_data.user_hint)

        if not candidates:
            raise ValidationError("No text columns found in DataFrame")

        # Select best candidate
        best_candidate = candidates[0]

        # Determine method
        method = "heuristic"
        reasoning = best_candidate.reason

        # LLM fallback for low confidence
        if best_candidate.confidence < 0.7:
            self.logger.info("Low confidence, using LLM fallback")
            llm_result = self._llm_fallback(df, candidates[:3])

            if llm_result:
                best_candidate = llm_result
                method = "llm"
                reasoning = llm_result.reason

        output = ColumnDetectorOutput(
            column_name=best_candidate.column,
            confidence=best_candidate.confidence,
            method=method,
            reasoning=reasoning,
            candidates=candidates[:3],  # Top 3 candidates
        )

        self.log_execution(input_data, output)
        return output

    def validate_input(self, input_data: BaseModel) -> bool:
        """Validate input DataFrame.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        # Cast to specific type
        if not isinstance(input_data, ColumnDetectorInput):
            return False

        df = input_data.dataframe

        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a DataFrame")
            return False

        if df.empty:
            self.logger.error("DataFrame is empty")
            return False

        if len(df.columns) == 0:
            self.logger.error("DataFrame has no columns")
            return False

        return True

    def _get_candidates(
        self, df: pd.DataFrame, user_hint: Optional[str] = None
    ) -> List[ColumnCandidate]:
        """Get candidate columns sorted by confidence.

        Args:
            df: Input DataFrame
            user_hint: Optional column name hint from user

        Returns:
            List of ColumnCandidate sorted by confidence (highest first)
        """
        candidates = []

        for col in df.columns:
            confidence, reason = self._calculate_confidence(df, col, user_hint)

            if confidence > 0:  # Only include potential text columns
                sample_values = self._get_sample_values(df, col)

                candidates.append(
                    ColumnCandidate(
                        column=str(col),
                        confidence=confidence,
                        reason=reason,
                        sample_values=sample_values,
                    )
                )

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        return candidates

    def _calculate_confidence(
        self, df: pd.DataFrame, col: str, user_hint: Optional[str] = None
    ) -> Tuple[float, str]:
        """Calculate confidence score for a column.

        Args:
            df: DataFrame
            col: Column name
            user_hint: User's suggested column name

        Returns:
            Tuple of (confidence_score, reason)
        """
        score = 0.0
        reasons = []

        # Check 1: User hint (highest priority)
        if user_hint and str(col).lower() == user_hint.lower():
            return 1.0, "Exact match with user hint"

        # Check 2: Data type - MUST be object/string
        if df[col].dtype != "object":
            return 0.0, "Non-string data type"  # Changed from 0.1 to 0.0

        # Check 3: Column name matches keywords
        col_lower = str(col).lower()
        for keyword in self.TEXT_COLUMN_KEYWORDS:
            if keyword in col_lower:
                score += 0.4
                reasons.append(f"Column name contains '{keyword}'")
                break

        # Check 4: String data type
        score += 0.2
        reasons.append("String/object data type")

        # Check 5: Average text length
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            avg_length = sample.astype(str).str.len().mean()

            if avg_length > 50:  # Typical comments are longer
                score += 0.3
                reasons.append(f"Long average text length ({avg_length:.0f} chars)")
            elif avg_length > 20:
                score += 0.15
                reasons.append(f"Medium text length ({avg_length:.0f} chars)")

        # Check 6: Contains spaces (sentences vs single words)
        if len(sample) > 0:
            has_spaces = sample.astype(str).str.contains(" ").mean()
            if has_spaces > 0.7:
                score += 0.1
                reasons.append("Contains multi-word text")

        # Cap score at 1.0
        score = min(score, 1.0)

        reason = "; ".join(reasons) if reasons else "Low confidence"

        return score, reason

    def _llm_fallback(
        self, df: pd.DataFrame, candidates: List[ColumnCandidate]
    ) -> Optional[ColumnCandidate]:
        """Use LLM to select best column when heuristic confidence is low.

        Args:
            df: DataFrame
            candidates: Top candidate columns

        Returns:
            ColumnCandidate selected by LLM, or None if LLM fails
        """
        try:
            # Prepare prompt
            prompt = self._create_llm_prompt(df, candidates)

            # Call LLM
            response = self.llm.generate(prompt, max_tokens=200, temperature=0.3)

            # Parse response
            result = self._parse_llm_response(response["content"], candidates)

            if result:
                self.logger.info(f"LLM selected column: {result.column}")
                return result

        except Exception as e:
            self.logger.error(f"LLM fallback failed: {e}")

        return None

    def _create_llm_prompt(self, df: pd.DataFrame, candidates: List[ColumnCandidate]) -> str:
        """Create prompt for LLM.

        Args:
            df: DataFrame
            candidates: Candidate columns

        Returns:
            Formatted prompt string
        """
        prompt = """You are analyzing a CSV file to detect which column contains text comments/reviews for sentiment analysis.

Available columns:
"""

        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n{i}. Column: '{candidate.column}'\n"
            prompt += f"   Confidence: {candidate.confidence:.2f}\n"
            prompt += f"   Reason: {candidate.reason}\n"
            prompt += "   Sample values:\n"
            for sample in candidate.sample_values[:2]:
                prompt += f"   - {sample}\n"

        prompt += """
Select the column most likely to contain text comments/reviews for sentiment analysis.

Respond ONLY with JSON in this exact format:
{"column": "column_name", "reasoning": "your reasoning here"}
"""

        return prompt

    def _parse_llm_response(
        self, response: str, candidates: List[ColumnCandidate]
    ) -> Optional[ColumnCandidate]:
        """Parse LLM JSON response.

        Args:
            response: LLM response string
            candidates: Available candidates

        Returns:
            ColumnCandidate from LLM selection, or None if parsing fails
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            selected_column = data.get("column")
            reasoning = data.get("reasoning", "Selected by LLM")

            # Find matching candidate
            for candidate in candidates:
                if candidate.column == selected_column:
                    # Update with LLM reasoning and boost confidence
                    return ColumnCandidate(
                        column=candidate.column,
                        confidence=0.85,  # LLM selection gets high confidence
                        reason=f"LLM: {reasoning}",
                        sample_values=candidate.sample_values,
                    )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")

        return None

    def _get_sample_values(self, df: pd.DataFrame, col: str, n: int = 3) -> List[str]:
        """Get sample values from column.

        Args:
            df: DataFrame
            col: Column name
            n: Number of samples

        Returns:
            List of sample values (max 100 chars each)
        """
        sample = df[col].dropna().head(n)

        values = []
        for val in sample:
            text = str(val)
            # Truncate long text
            if len(text) > 100:
                text = text[:97] + "..."
            values.append(text)

        return values
