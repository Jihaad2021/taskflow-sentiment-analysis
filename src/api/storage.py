"""In-memory storage for jobs and uploads."""

import uuid
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


class JobStorage:
    """In-memory job storage."""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.uploads: Dict[str, Dict] = {}

    # ========== Uploads ==========

    def save_upload(
        self, filename: str, dataframe: pd.DataFrame, detected_column: Optional[str] = None
    ) -> str:
        """Save uploaded file data.

        Returns:
            upload_id
        """
        upload_id = str(uuid.uuid4())

        self.uploads[upload_id] = {
            "upload_id": upload_id,
            "filename": filename,
            "dataframe": dataframe,
            "detected_column": detected_column,
            "uploaded_at": datetime.utcnow(),
        }

        return upload_id

    def get_upload(self, upload_id: str) -> Optional[Dict]:
        """Get upload data."""
        return self.uploads.get(upload_id)

    # ========== Jobs ==========

    def create_job(
        self,
        upload_id: str,
        report_type: str,
        text_column: Optional[str] = None,
        max_regenerations: int = 3,
    ) -> str:
        """Create new job.

        Returns:
            job_id
        """
        job_id = str(uuid.uuid4())

        self.jobs[job_id] = {
            "job_id": job_id,
            "upload_id": upload_id,
            "report_type": report_type,
            "text_column": text_column,
            "max_regenerations": max_regenerations,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "Queued",
            "started_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": None,
            "error": None,
            "result": None,
        }

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job data."""
        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict] = None,
    ):
        """Update job status."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]

        if status:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if current_stage:
            job["current_stage"] = current_stage
        if error:
            job["error"] = error
        if result:
            job["result"] = result

        job["updated_at"] = datetime.utcnow()

        if status == "completed" or status == "failed":
            job["completed_at"] = datetime.utcnow()


# Global storage instance
storage = JobStorage()
