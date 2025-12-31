"""Pydantic schemas for data validation."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

# ============================================================================
# Base Configuration Models
# ============================================================================


class AgentConfig(BaseModel):
    """Base configuration for agents."""

    name: str
    log_level: str = "INFO"


class ToolConfig(BaseModel):
    """Configuration for analysis tools."""

    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    topic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    entity_model: str = "dslim/bert-base-NER"
    keyphrase_model: str = "ml6team/keyphrase-extraction-distilbert-inspec"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1, le=128)


# ============================================================================
# Agent 1: ColumnDetector Schemas (Week 1 Day 3-4)
# ============================================================================


class ColumnDetectorInput(BaseModel):
    """Input for ColumnDetectorAgent."""

    dataframe: Any  # pd.DataFrame
    user_hint: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class ColumnCandidate(BaseModel):
    """Candidate column for text analysis."""

    column: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    sample_values: List[str] = Field(default_factory=list)


class ColumnDetectorOutput(BaseModel):
    """Output from ColumnDetectorAgent."""

    column_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str  # 'heuristic' or 'llm'
    reasoning: str
    candidates: List[ColumnCandidate] = Field(default_factory=list)


# ============================================================================
# Agent 2: DataValidator Schemas (Week 1 Day 5-7)
# ============================================================================


class DataValidatorInput(BaseModel):
    """Input for DataValidatorAgent."""

    dataframe: Any  # pd.DataFrame
    text_column: str

    class Config:
        arbitrary_types_allowed = True


class DataStats(BaseModel):
    """Statistics about the data."""

    total_rows: int
    rows_after_cleaning: int
    removed_rows: int
    missing_values: int
    duplicates: int
    empty_strings: int
    spam_removed: int = 0  # ADD THIS
    avg_text_length: float
    min_text_length: int
    max_text_length: int


class DataValidatorOutput(BaseModel):
    """Output from DataValidatorAgent."""

    status: str  # 'pass', 'warning', 'fail'
    cleaned_data: Any  # pd.DataFrame
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stats: DataStats

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# More schemas will be added in Week 2 & 3
# ============================================================================
