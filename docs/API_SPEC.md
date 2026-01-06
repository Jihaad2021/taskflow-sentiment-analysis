# TaskFlow API Specification

## Endpoints Overview

### 1. Upload CSV
```
POST /api/upload
Content-Type: multipart/form-data

Body:
- file: CSV file (max 10MB)

Response:
{
  "upload_id": "uuid-string",
  "filename": "comments.csv",
  "rows": 1000,
  "columns": ["review", "rating", "date"],
  "detected_column": "review",
  "preview": [
    {"review": "Great product!", "rating": 5},
    ...
  ]
}
```

### 2. Start Analysis
```
POST /api/analyze
Content-Type: application/json

Body:
{
  "upload_id": "uuid-string",
  "report_type": "marketing",  // executive|marketing|product|customer_service|comprehensive
  "text_column": "review",     // optional, override auto-detection
  "max_regenerations": 3       // optional, default 3
}

Response:
{
  "job_id": "uuid-string",
  "status": "queued",
  "estimated_time": 90
}
```

### 3. Check Job Status
```
GET /api/job/{job_id}

Response:
{
  "job_id": "uuid-string",
  "status": "processing",  // queued|processing|completed|failed
  "progress": 45,          // 0-100
  "current_stage": "Analyzing comments...",
  "started_at": "2025-01-06T08:15:00Z",
  "updated_at": "2025-01-06T08:15:45Z",
  "result": null  // or result object when completed
}
```

### 4. Get Report
```
GET /api/report/{job_id}

Response (when completed):
{
  "job_id": "uuid-string",
  "report_text": "# Report...",
  "quality_score": 85.5,
  "word_count": 1250,
  "cost": 0.015,
  "total_time": 95.2,
  "created_at": "2025-01-06T08:17:00Z"
}
```

### 5. Download Report
```
GET /api/report/{job_id}/download?format=pdf
Query params:
- format: md|pdf (default: pdf)

Response:
File download (application/pdf or text/markdown)
```

### 6. Health Check
```
GET /api/health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600
}
```

---

## Project Structure for Week 4
```
src/api/
├── main.py              # FastAPI app
├── routes.py            # Route handlers
├── models.py            # Request/Response models
├── dependencies.py      # Shared dependencies
├── jobs.py              # Background job processing
└── storage.py           # File & job storage

src/export/
├── pdf_generator.py     # Markdown → PDF
└── charts.py            # Generate charts
```