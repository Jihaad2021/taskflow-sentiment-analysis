"""API route handlers."""

import io

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
)

# Add BackgroundTasks here
from fastapi.responses import StreamingResponse

from src.agents.column_detector import ColumnDetectorAgent
from src.api.jobs import process_job  # Add this
from src.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    JobStatus,
    ReportResponse,
    UploadResponse,
)
from src.api.storage import storage
from src.models.schemas import ColumnDetectorInput

router = APIRouter()


# ============================================================================
# Upload Endpoint
# ============================================================================


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for analysis."""

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Read file with robust error handling
    try:
        contents = await file.read()

        # Try multiple parsing strategies
        df = None
        errors = []

        # Strategy 1: Normal comma delimiter
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            errors.append(f"Comma: {str(e)[:50]}")

        # Strategy 2: Semicolon delimiter
        if df is None:
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=";")
            except Exception as e:
                errors.append(f"Semicolon: {str(e)[:50]}")

        # Strategy 3: Tab delimiter
        if df is None:
            try:
                df = pd.read_csv(io.BytesIO(contents), sep="\t")
            except Exception as e:
                errors.append(f"Tab: {str(e)[:50]}")

        # Strategy 4: Skip bad lines
        if df is None:
            try:
                df = pd.read_csv(io.BytesIO(contents), on_bad_lines="skip", encoding="utf-8")
            except Exception as e:
                errors.append(f"Skip bad lines: {str(e)[:50]}")

        # Strategy 5: Engine python (more flexible)
        if df is None:
            try:
                df = pd.read_csv(io.BytesIO(contents), engine="python", on_bad_lines="skip")
            except Exception as e:
                errors.append(f"Python engine: {str(e)[:50]}")

        if df is None or len(df) == 0:
            raise ValueError(
                "Could not parse CSV file. Tried multiple formats but all failed. "
                "Please ensure your CSV has consistent formatting."
            )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read CSV: {str(e)}. Please check file format."
        )

    # Validate data
    if len(df) < 10:
        raise HTTPException(
            status_code=400, detail=f"CSV must have at least 10 rows (found {len(df)})"
        )

    if len(df) > 10000:
        raise HTTPException(
            status_code=400, detail=f"CSV too large. Maximum 10,000 rows (found {len(df)})"
        )

    # Auto-detect text column
    try:
        col_detector = ColumnDetectorAgent()
        col_result = col_detector.execute(ColumnDetectorInput(dataframe=df))
        detected_column = col_result.column_name
    except Exception:
        detected_column = None

    # Save upload
    upload_id = storage.save_upload(
        filename=file.filename, dataframe=df, detected_column=detected_column
    )

    # Create preview (first 5 rows) - CLEAN FOR JSON
    preview_df = df.head(5).copy()

    # Replace NaN/Infinity with None for JSON serialization
    preview_df = preview_df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

    # Convert to dict
    preview = preview_df.to_dict("records")

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        rows=len(df),
        columns=df.columns.tolist(),
        detected_column=detected_column,
        preview=preview,
    )


# ============================================================================
# Analysis Endpoint
# ============================================================================


@router.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest, background_tasks: BackgroundTasks
):  # Add background_tasks
    """Start analysis job.

    Args:
        request: Analysis request with upload_id and options
        background_tasks: FastAPI background tasks

    Returns:
        AnalysisResponse with job_id
    """
    # Validate upload exists
    upload = storage.get_upload(request.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Create job
    job_id = storage.create_job(
        upload_id=request.upload_id,
        report_type=request.report_type,
        text_column=request.text_column,
        max_regenerations=request.max_regenerations,
    )

    # Estimate time (rough calculation)
    num_comments = len(upload["dataframe"])
    estimated_time = int(num_comments * 1.5)  # ~1.5s per comment

    # Start background processing
    background_tasks.add_task(process_job, job_id)

    return AnalysisResponse(job_id=job_id, status="queued", estimated_time=estimated_time)


# ============================================================================
# Job Status Endpoint
# ============================================================================


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status.

    Args:
        job_id: Job ID

    Returns:
        JobStatus with current status and progress
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**job)


# ============================================================================
# Report Endpoints
# ============================================================================


@router.get("/report/{job_id}", response_model=ReportResponse)
async def get_report(job_id: str):
    """Get report data.

    Args:
        job_id: Job ID

    Returns:
        ReportResponse with report text and metadata
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed yet. Current status: {job['status']}"
        )

    if not job["result"]:
        raise HTTPException(status_code=500, detail="Report data not found")

    return ReportResponse(**job["result"])


@router.get("/report/{job_id}/download")
async def download_report(job_id: str, format: str = "md"):
    """Download report file.

    Args:
        job_id: Job ID
        format: File format (md or pdf)

    Returns:
        File download
    """
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if not job["result"]:
        raise HTTPException(status_code=500, detail="Report data not found")

    report_text = job["result"]["report_text"]

    if format == "md":
        # Return markdown
        return StreamingResponse(
            io.BytesIO(report_text.encode()),
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=report_{job_id}.md"},
        )
    elif format == "pdf":
        # TODO: Generate PDF
        raise HTTPException(status_code=501, detail="PDF generation not yet implemented")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'md' or 'pdf'")
