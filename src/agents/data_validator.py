"""Data validation and cleaning agent."""

import re
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.models.schemas import (
    DataStats,
    DataValidatorInput,
    DataValidatorOutput,
)
from src.utils.exceptions import ValidationError


class DataValidatorAgent(BaseAgent):
    """Validate and clean CSV data before analysis."""

    # Spam patterns
    SPAM_PATTERNS = [
        r"(buy|click|visit|check)\s+(now|here|this)",
        r"http.*\s.*http",  # Multiple URLs
        r"(!!!){2,}",  # Excessive exclamation
        r"(\$\$\$|€€€|£££)",  # Money symbols
        r"(free|win|winner|prize).*now",
        r"(viagra|cialis|pharmacy)",
    ]

    def __init__(self, min_rows: int = 10, max_rows: int = 10000):
        """Initialize data validator.

        Args:
            min_rows: Minimum required rows
            max_rows: Maximum allowed rows
        """
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="DataValidatorAgent"))
        self.min_rows = min_rows
        self.max_rows = max_rows

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Validate and clean data.

        Args:
            input_data: DataValidatorInput with dataframe and text column

        Returns:
            DataValidatorOutput with cleaned data and stats

        Raises:
            ValidationError: If validation fails critically
        """
        # Type check
        if not isinstance(input_data, DataValidatorInput):
            raise ValidationError("Invalid input type")

        df = input_data.dataframe
        text_column = input_data.text_column

        # Validate structure
        self._validate_structure(df, text_column)

        # Collect issues and warnings
        issues: List[str] = []
        warnings: List[str] = []

        # Store original stats
        original_rows = len(df)

        # Clean data
        df_cleaned, cleaning_stats = self._clean_data(df, text_column)

        # Calculate final stats
        stats = self._calculate_stats(
            original_df=df,
            cleaned_df=df_cleaned,
            text_column=text_column,
            cleaning_stats=cleaning_stats,
        )

        # Determine status
        status = self._determine_status(stats, issues, warnings)

        # Add warnings based on stats
        if stats.removed_rows > original_rows * 0.3:
            warnings.append(f"Removed {stats.removed_rows} rows (>30% of data)")

        if stats.avg_text_length < 10:
            warnings.append(
                f"Average text length is very short ({stats.avg_text_length:.1f} chars)"
            )

        output = DataValidatorOutput(
            status=status, cleaned_data=df_cleaned, issues=issues, warnings=warnings, stats=stats
        )

        self.log_execution(input_data, output)
        return output

    def validate_input(self, input_data: BaseModel) -> bool:
        """Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, DataValidatorInput):
            return False

        df = input_data.dataframe

        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a DataFrame")
            return False

        if df.empty:
            self.logger.error("DataFrame is empty")
            return False

        return True

    def _validate_structure(self, df: pd.DataFrame, text_column: str) -> None:
        """Validate DataFrame structure.

        Args:
            df: DataFrame to validate
            text_column: Text column name

        Raises:
            ValidationError: If structure is invalid
        """
        # Check if column exists
        if text_column not in df.columns:
            raise ValidationError(f"Column '{text_column}' not found in DataFrame")

        # Check minimum rows
        if len(df) < self.min_rows:
            raise ValidationError(f"DataFrame has only {len(df)} rows (minimum: {self.min_rows})")

        # Check maximum rows
        if len(df) > self.max_rows:
            raise ValidationError(f"DataFrame has {len(df)} rows (maximum: {self.max_rows})")

    def _is_spam(self, text: str) -> bool:
        """Check if text is spam.

        Args:
            text: Text to check

        Returns:
            True if spam detected, False otherwise
        """
        text_lower = text.lower()

        # Check spam patterns
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Check excessive caps (>70% uppercase)
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.7:
                return True

        return False

    def _detect_language(self, df: pd.DataFrame, text_column: str) -> str:
        """Detect primary language in dataset.

        Args:
            df: DataFrame
            text_column: Text column

        Returns:
            Language code or 'unknown'
        """
        # Simple heuristic: check for common English words
        sample = df[text_column].head(20)
        english_words = ["the", "is", "and", "to", "a", "of", "in", "for"]

        english_count = 0
        for text in sample:
            text_lower = str(text).lower()
            if any(word in text_lower for word in english_words):
                english_count += 1

        if english_count / len(sample) > 0.5:
            return "en"

        return "unknown"

    def _clean_data(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, dict]:
        """Clean the data.

        Args:
            df: DataFrame to clean
            text_column: Text column name

        Returns:
            Tuple of (cleaned_df, cleaning_stats)
        """
        df_clean = df.copy()

        stats = {
            "missing_removed": 0,
            "duplicates_removed": 0,
            "empty_removed": 0,
            "short_removed": 0,
            "spam_removed": 0,  # ADD THIS
        }

        original_len = len(df_clean)

        # 1. Remove missing values in text column
        df_clean = df_clean.dropna(subset=[text_column])
        stats["missing_removed"] = original_len - len(df_clean)

        # 2. Clean text content
        df_clean[text_column] = df_clean[text_column].apply(self._clean_text)

        # 3. Remove empty strings (after cleaning)
        before_empty = len(df_clean)
        df_clean = df_clean[df_clean[text_column].str.strip() != ""]
        stats["empty_removed"] = before_empty - len(df_clean)

        # 4. Remove very short text (< 3 chars)
        before_short = len(df_clean)
        df_clean = df_clean[df_clean[text_column].str.len() >= 3]
        stats["short_removed"] = before_short - len(df_clean)

        # 5. Remove spam (ADD THIS SECTION)
        before_spam = len(df_clean)
        df_clean = df_clean[~df_clean[text_column].apply(self._is_spam)]
        stats["spam_removed"] = before_spam - len(df_clean)

        # 6. Remove duplicates (renumber from 5 to 6)
        before_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=[text_column])
        stats["duplicates_removed"] = before_dup - len(df_clean)

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        return df_clean, stats

    def _clean_text(self, text: str) -> str:
        """Clean individual text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        text = str(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _calculate_stats(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        text_column: str,
        cleaning_stats: dict,
    ) -> DataStats:
        """Calculate data statistics.

        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            text_column: Text column name
            cleaning_stats: Stats from cleaning process

        Returns:
            DataStats object
        """
        text_lengths = cleaned_df[text_column].str.len()

        return DataStats(
            total_rows=len(original_df),
            rows_after_cleaning=len(cleaned_df),
            removed_rows=len(original_df) - len(cleaned_df),
            missing_values=cleaning_stats["missing_removed"],
            duplicates=cleaning_stats["duplicates_removed"],
            empty_strings=cleaning_stats["empty_removed"],
            avg_text_length=float(text_lengths.mean()),
            min_text_length=int(text_lengths.min()),
            max_text_length=int(text_lengths.max()),
            # ADD THIS (but need to update schema first)
        )

    def _determine_status(self, stats: DataStats, issues: List[str], warnings: List[str]) -> str:
        """Determine validation status.

        Args:
            stats: Data statistics
            issues: List of critical issues
            warnings: List of warnings

        Returns:
            Status string: 'pass', 'warning', or 'fail'
        """
        # Fail if critical issues
        if issues:
            return "fail"

        # Fail if too few rows after cleaning
        if stats.rows_after_cleaning < self.min_rows:
            return "fail"

        # Warning if many rows removed or short text
        if warnings or stats.removed_rows > stats.total_rows * 0.5:
            return "warning"

        return "pass"
