"""API request and response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ============================================================================
# Upload Models
# ============================================================================


class UploadResponse(BaseModel):
    """Response after file upload."""

    upload_id: str
    filename: str
    rows: int
    columns: List[str]
    detected_column: Optional[str] = None
    preview: List[Dict[str, Any]] = Field(
        default_factory=list, max_length=5
    )  # Changed from str to any


# ============================================================================
# Analysis Models
# ============================================================================


class AnalysisRequest(BaseModel):
    """Request to start analysis."""

    upload_id: str
    report_type: str = Field(
        ..., pattern="^(executive|marketing|product|customer_service|comprehensive)$"
    )
    text_column: Optional[str] = None
    max_regenerations: int = Field(default=3, ge=1, le=5)


class AnalysisResponse(BaseModel):
    """Response when analysis starts."""

    job_id: str
    status: str
    estimated_time: int  # seconds


# ============================================================================
# Job Status Models
# ============================================================================


class JobStatus(BaseModel):
    """Status of analysis job."""

    job_id: str
    status: str  # queued|processing|completed|failed
    progress: float = Field(..., ge=0.0, le=100.0)
    current_stage: str
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict] = None


# ============================================================================
# Report Models
# ============================================================================


class ReportResponse(BaseModel):
    """Report data response."""

    job_id: str
    report_text: str
    quality_score: float
    word_count: int
    cost: float
    total_time: float
    created_at: datetime


# ============================================================================
# Health Check
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime: float
