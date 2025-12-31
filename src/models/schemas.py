"""Pydantic schemas for data validation."""

from typing import Any, Dict, List, Optional

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
# Tool Result Models (Week 2)
# ============================================================================


class SentimentResult(BaseModel):
    """Result from sentiment analysis."""

    label: str  # 'positive', 'negative', 'neutral'
    score: float = Field(..., ge=0.0, le=1.0)
    scores: Dict[str, float]  # All class probabilities


class EmotionResult(BaseModel):
    """Result from emotion detection."""

    emotion: str  # Primary emotion
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: Dict[str, float]  # All emotion scores


class TopicResult(BaseModel):
    """Result from topic extraction."""

    topics: List[str]
    relevance_scores: Dict[str, float]
    primary_topic: str


class Entity(BaseModel):
    """Named entity."""

    text: str
    type: str  # Entity type
    start: int
    end: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityResult(BaseModel):
    """Result from entity extraction."""

    entities: List[Entity] = Field(default_factory=list)


class KeyphraseResult(BaseModel):
    """Result from keyphrase extraction."""

    keyphrases: List[str]
    scores: Dict[str, float]


class CommentAnalysis(BaseModel):
    """Complete analysis for single comment."""

    comment_id: str
    text: str
    sentiment: SentimentResult
    emotion: EmotionResult
    topics: TopicResult
    entities: EntityResult
    keyphrases: KeyphraseResult
    execution_time: float


# ============================================================================
# Agent 3: AnalysisOrchestrator Schemas
# ============================================================================


class AnalysisOrchestratorInput(BaseModel):
    """Input for AnalysisOrchestratorAgent."""

    comments: List[str] = Field(..., min_length=1)
    batch_size: int = Field(default=32, ge=1, le=128)


class TopicSummary(BaseModel):
    """Summary of a topic."""

    topic: str
    count: int
    avg_sentiment: float = Field(..., ge=-1.0, le=1.0)
    sample_comments: List[str] = Field(default_factory=list, max_length=5)


class EntitySummary(BaseModel):
    """Summary of an entity."""

    text: str
    type: str
    count: int
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    contexts: List[str] = Field(default_factory=list, max_length=3)


class AnalysisOrchestratorOutput(BaseModel):
    """Output from AnalysisOrchestratorAgent."""

    total_comments: int
    sentiment_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    top_topics: List[TopicSummary]
    entities: List[EntitySummary]
    keyphrases: List[str]
    individual_results: List[CommentAnalysis] = Field(default_factory=list)
    execution_time: float


# ============================================================================
# Agent 4: PrePromptEvaluator Schemas
# ============================================================================


class QualityCheck(BaseModel):
    """Individual quality check result."""

    name: str
    passed: bool
    score: float = Field(..., ge=0.0, le=100.0)
    message: str


class PrePromptEvaluatorInput(BaseModel):
    """Input for PrePromptEvaluatorAgent."""

    tool_results: AnalysisOrchestratorOutput


class PrePromptEvaluatorOutput(BaseModel):
    """Output from PrePromptEvaluatorAgent."""

    status: str  # 'pass', 'warning', 'fail'
    quality_score: float = Field(..., ge=0.0, le=100.0)
    checks: List[QualityCheck]
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    should_proceed: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LLM Configuration
# ============================================================================


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: str = Field(..., description="'anthropic' or 'openai'")
    model: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(default=60, ge=10, le=300)


# ============================================================================
# Agent 5: ReportPlanner Schemas
# ============================================================================


class ReportSection(BaseModel):
    """Single section in report."""

    name: str
    priority: str  # 'high', 'medium', 'low'
    key_points: List[str]
    data_sources: List[str]
    estimated_length: str  # e.g., '2-3 paragraphs'


class ReportOutline(BaseModel):
    """Structure of the report."""

    title: str
    sections: List[ReportSection]
    recommended_length: str  # e.g., '5-7 pages'


class ReportPlannerInput(BaseModel):
    """Input for ReportPlannerAgent."""

    tool_results: AnalysisOrchestratorOutput
    report_type: str
    quality_assessment: PrePromptEvaluatorOutput


class ReportPlannerOutput(BaseModel):
    """Output from ReportPlannerAgent."""

    report_outline: ReportOutline
    key_insights: List[str]
    recommended_focus: List[str]
    narrative_flow: List[str]
    estimated_sections: int
    llm_cost: float = 0.0
