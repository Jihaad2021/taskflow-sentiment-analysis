# TaskFlow API Documentation

> **Complete reference for TaskFlow Sentiment Analysis REST API**

**Base URL:** `http://localhost:8000/api`  
**Version:** 1.0.0  
**Protocol:** REST  
**Format:** JSON

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [Request/Response Models](#requestresponse-models)
4. [Endpoints](#endpoints)
5. [Error Handling](#error-handling)
6. [Rate Limits](#rate-limits)
7. [Examples](#examples)

---

## Authentication

Currently, the API does not require authentication for local development.

**Future:** API keys will be required for production deployment.

```http
# Future authentication
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload CSV file |
| POST | `/api/analyze` | Start analysis job |
| GET | `/api/job/{job_id}` | Get job status |
| GET | `/api/report/{job_id}` | Get report data |
| GET | `/api/report/{job_id}/download` | Download report file |

---

## Request/Response Models

### UploadResponse

```json
{
  "upload_id": "string",
  "filename": "string",
  "rows": 0,
  "columns": ["string"],
  "detected_column": "string",
  "preview": [
    {
      "column1": "value1",
      "column2": "value2"
    }
  ]
}
```

### AnalysisRequest

```json
{
  "upload_id": "string",
  "report_type": "executive",
  "text_column": "string (optional)",
  "max_regenerations": 3
}
```

### AnalysisResponse

```json
{
  "job_id": "string",
  "status": "queued",
  "estimated_time": 90
}
```

### JobStatus

```json
{
  "job_id": "string",
  "status": "processing",
  "progress": 45.5,
  "current_stage": "Analyzing comments...",
  "started_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:45Z",
  "completed_at": null,
  "error": null
}
```

### ReportResponse

```json
{
  "job_id": "string",
  "report_text": "# Report Title\n\n...",
  "quality_score": 85.5,
  "word_count": 1250,
  "cost": 0.0154,
  "total_time": 95.2
}
```

---

## Endpoints

### 1. Health Check

Check API health and status.

**Endpoint:** `GET /api/health`

**Parameters:** None

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600.5,
  "jobs_count": 10,
  "uploads_count": 8
}
```

**Example:**

```bash
curl http://localhost:8000/api/health
```

---

### 2. Upload CSV

Upload a CSV file for analysis.

**Endpoint:** `POST /api/upload`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | CSV file (max 10MB) |

**Response:** `UploadResponse`

**Validation:**
- File must be `.csv` extension
- Minimum 10 rows
- Maximum 10,000 rows
- Maximum file size: 10MB

**Example:**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@reviews.csv"
```

**Response:**

```json
{
  "upload_id": "abc-123-def",
  "filename": "reviews.csv",
  "rows": 100,
  "columns": ["review", "rating", "date"],
  "detected_column": "review",
  "preview": [
    {
      "review": "Great product!",
      "rating": "5",
      "date": "2024-01-01"
    }
  ]
}
```

**Error Responses:**

| Status | Error | Description |
|--------|-------|-------------|
| 400 | Invalid file type | Only CSV files supported |
| 400 | File too large | Maximum 10MB |
| 400 | Too few rows | Minimum 10 rows required |
| 400 | Too many rows | Maximum 10,000 rows |
| 400 | Parse error | CSV format invalid |

---

### 3. Start Analysis

Start sentiment analysis job.

**Endpoint:** `POST /api/analyze`

**Content-Type:** `application/json`

**Request Body:** `AnalysisRequest`

```json
{
  "upload_id": "abc-123-def",
  "report_type": "executive",
  "text_column": "review",
  "max_regenerations": 3
}
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| upload_id | string | Yes | ID from upload endpoint |
| report_type | string | Yes | One of: executive, marketing, product, customer_service, comprehensive |
| text_column | string | No | Column name (auto-detect if omitted) |
| max_regenerations | integer | No | Max report regeneration attempts (default: 3) |

**Report Types:**

| Type | Description | Length |
|------|-------------|--------|
| executive | High-level summary | 3-5 pages |
| marketing | Campaign & audience insights | 5-7 pages |
| product | Feature feedback & improvements | 7-10 pages |
| customer_service | Support issues & trends | 5-8 pages |
| comprehensive | Full analysis | 15-20 pages |

**Response:** `AnalysisResponse`

```json
{
  "job_id": "xyz-456-uvw",
  "status": "queued",
  "estimated_time": 90
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "abc-123-def",
    "report_type": "executive"
  }'
```

**Error Responses:**

| Status | Error | Description |
|--------|-------|-------------|
| 404 | Upload not found | Invalid upload_id |
| 400 | Invalid report type | Must be one of valid types |

---

### 4. Get Job Status

Poll job status and progress.

**Endpoint:** `GET /api/job/{job_id}`

**Parameters:**

| Name | Location | Required | Description |
|------|----------|----------|-------------|
| job_id | Path | Yes | Job ID from analyze endpoint |

**Response:** `JobStatus`

```json
{
  "job_id": "xyz-456-uvw",
  "status": "processing",
  "progress": 45.5,
  "current_stage": "Analyzing 100 comments...",
  "started_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:45Z",
  "completed_at": null,
  "error": null
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| queued | Job is queued |
| processing | Job is running |
| completed | Job finished successfully |
| failed | Job failed with error |

**Progress Stages:**

1. **0-10%:** Detecting text column
2. **10-20%:** Validating data
3. **20-65%:** Analyzing comments (5 ML models)
4. **65-70%:** Quality check
5. **70-100%:** Generating report (LLM)

**Example:**

```bash
curl http://localhost:8000/api/job/xyz-456-uvw
```

**Polling Strategy:**

```javascript
// Poll every 2 seconds until complete
const checkStatus = async () => {
  const response = await fetch(`/api/job/${jobId}`);
  const data = await response.json();
  
  if (data.status === 'completed') {
    // Fetch report
  } else if (data.status === 'failed') {
    // Handle error
  } else {
    // Update progress, poll again
    setTimeout(checkStatus, 2000);
  }
};
```

---

### 5. Get Report

Retrieve generated report data.

**Endpoint:** `GET /api/report/{job_id}`

**Parameters:**

| Name | Location | Required | Description |
|------|----------|----------|-------------|
| job_id | Path | Yes | Job ID |

**Response:** `ReportResponse`

```json
{
  "job_id": "xyz-456-uvw",
  "report_text": "# Executive Summary\n\n...",
  "quality_score": 85.5,
  "word_count": 1250,
  "cost": 0.0154,
  "total_time": 95.2
}
```

**Example:**

```bash
curl http://localhost:8000/api/report/xyz-456-uvw
```

**Error Responses:**

| Status | Error | Description |
|--------|-------|-------------|
| 404 | Job not found | Invalid job_id |
| 400 | Job not completed | Job still processing |

---

### 6. Download Report

Download report as file (Markdown or PDF).

**Endpoint:** `GET /api/report/{job_id}/download`

**Parameters:**

| Name | Location | Required | Description |
|------|----------|----------|-------------|
| job_id | Path | Yes | Job ID |
| format | Query | Yes | File format: `md` or `pdf` |

**Response:** File download

**Content-Type:**
- Markdown: `text/markdown`
- PDF: `application/pdf`

**Example:**

```bash
# Download Markdown
curl http://localhost:8000/api/report/xyz-456-uvw/download?format=md \
  -o report.md

# Download PDF
curl http://localhost:8000/api/report/xyz-456-uvw/download?format=pdf \
  -o report.pdf
```

**Error Responses:**

| Status | Error | Description |
|--------|-------|-------------|
| 404 | Job not found | Invalid job_id |
| 400 | Job not completed | Job still processing |
| 400 | Invalid format | Must be 'md' or 'pdf' |
| 500 | Generation failed | PDF generation error |

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

### Common Errors

**1. CSV Parse Error**

```json
{
  "detail": "Failed to read CSV: Error tokenizing data. Please check file format."
}
```

**Solution:** Ensure CSV has consistent columns, proper encoding (UTF-8)

**2. Column Not Found**

```json
{
  "detail": "Column 'review' not found in DataFrame"
}
```

**Solution:** Check column name, use auto-detect, or specify correct column

**3. Validation Failed**

```json
{
  "detail": "Data validation failed: Too many missing values"
}
```

**Solution:** Clean data, remove rows with missing text

**4. Job Failed**

```json
{
  "detail": "Analysis quality too low: Low confidence scores"
}
```

**Solution:** Check data quality, ensure text is meaningful

---

## Rate Limits

**Current:** No rate limits (local development)

**Future Production Limits:**
- 100 uploads per hour
- 50 analysis jobs per hour
- 1000 status checks per hour

**Rate Limit Headers:**

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## Examples

### Complete Workflow (Python)

```python
import requests
import time

API_BASE = "http://localhost:8000/api"

# 1. Upload CSV
with open("reviews.csv", "rb") as f:
    files = {"file": ("reviews.csv", f, "text/csv")}
    response = requests.post(f"{API_BASE}/upload", files=files)
    upload_data = response.json()
    print(f"Uploaded: {upload_data['upload_id']}")

# 2. Start Analysis
payload = {
    "upload_id": upload_data["upload_id"],
    "report_type": "executive"
}
response = requests.post(f"{API_BASE}/analyze", json=payload)
job_data = response.json()
job_id = job_data["job_id"]
print(f"Job started: {job_id}")

# 3. Poll Status
while True:
    response = requests.get(f"{API_BASE}/job/{job_id}")
    status_data = response.json()
    
    print(f"Progress: {status_data['progress']:.1f}% - {status_data['current_stage']}")
    
    if status_data["status"] == "completed":
        break
    elif status_data["status"] == "failed":
        print(f"Error: {status_data['error']}")
        exit(1)
    
    time.sleep(2)

# 4. Get Report
response = requests.get(f"{API_BASE}/report/{job_id}")
report_data = response.json()
print(f"Quality: {report_data['quality_score']}/100")
print(f"Cost: ${report_data['cost']:.4f}")

# 5. Download PDF
response = requests.get(f"{API_BASE}/report/{job_id}/download?format=pdf")
with open("report.pdf", "wb") as f:
    f.write(response.content)
print("Report saved: report.pdf")
```

### Complete Workflow (JavaScript)

```javascript
const API_BASE = 'http://localhost:8000/api';

async function analyzeCSV(file) {
  // 1. Upload
  const formData = new FormData();
  formData.append('file', file);
  
  const uploadRes = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData
  });
  const uploadData = await uploadRes.json();
  console.log('Uploaded:', uploadData.upload_id);
  
  // 2. Start Analysis
  const analyzeRes = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      upload_id: uploadData.upload_id,
      report_type: 'executive'
    })
  });
  const jobData = await analyzeRes.json();
  const jobId = jobData.job_id;
  console.log('Job started:', jobId);
  
  // 3. Poll Status
  while (true) {
    const statusRes = await fetch(`${API_BASE}/job/${jobId}`);
    const statusData = await statusRes.json();
    
    console.log(`Progress: ${statusData.progress}% - ${statusData.current_stage}`);
    
    if (statusData.status === 'completed') {
      break;
    } else if (statusData.status === 'failed') {
      throw new Error(statusData.error);
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  // 4. Get Report
  const reportRes = await fetch(`${API_BASE}/report/${jobId}`);
  const reportData = await reportRes.json();
  console.log('Quality:', reportData.quality_score);
  
  // 5. Download
  const downloadUrl = `${API_BASE}/report/${jobId}/download?format=pdf`;
  window.open(downloadUrl, '_blank');
}
```

### cURL Examples

```bash
# Complete workflow in bash

# 1. Upload
UPLOAD_ID=$(curl -s -X POST http://localhost:8000/api/upload \
  -F "file=@reviews.csv" | jq -r '.upload_id')
echo "Upload ID: $UPLOAD_ID"

# 2. Start analysis
JOB_ID=$(curl -s -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"upload_id\": \"$UPLOAD_ID\", \"report_type\": \"executive\"}" \
  | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 3. Poll status (loop until complete)
while true; do
  STATUS=$(curl -s http://localhost:8000/api/job/$JOB_ID | jq -r '.status')
  PROGRESS=$(curl -s http://localhost:8000/api/job/$JOB_ID | jq -r '.progress')
  echo "Status: $STATUS - Progress: $PROGRESS%"
  
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  
  sleep 2
done

# 4. Download report
curl http://localhost:8000/api/report/$JOB_ID/download?format=pdf \
  -o report.pdf
echo "Report saved: report.pdf"
```

---

## Swagger UI

Interactive API documentation available at:

**URL:** `http://localhost:8000/docs`

Features:
- Try all endpoints
- See request/response schemas
- Test with sample data
- Download OpenAPI spec

---

## OpenAPI Specification

Download OpenAPI 3.0 spec:

**URL:** `http://localhost:8000/openapi.json`

---

## Support

**Issues:** [GitHub Issues](https://github.com/yourusername/taskflow-sentiment-analysis/issues)  
**Documentation:** [Full Docs](https://github.com/yourusername/taskflow-sentiment-analysis/tree/main/docs)

---

**Last Updated:** January 2025  
**API Version:** 1.0.0