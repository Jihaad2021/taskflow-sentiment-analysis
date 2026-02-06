"""End-to-end testing script for TaskFlow API."""

import time
from pathlib import Path

import pandas as pd
import requests

# Configuration
API_BASE = "http://localhost:8000/api"
TEST_DATA_DIR = Path("test_data")


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health_check():
    """Test API health check."""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200, "Health check failed"
    
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Version: {data['version']}")
    print(f"✅ Uptime: {data['uptime']}s")


def test_upload_csv(csv_path: Path):
    """Test CSV upload.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Upload ID
    """
    print_section(f"TEST 2: Upload CSV - {csv_path.name}")
    
    with open(csv_path, 'rb') as f:
        files = {'file': (csv_path.name, f, 'text/csv')}
        response = requests.post(f"{API_BASE}/upload", files=files)
    
    assert response.status_code == 200, f"Upload failed: {response.text}"
    
    data = response.json()
    print(f"✅ Upload ID: {data['upload_id']}")
    print(f"✅ Rows: {data['rows']}")
    print(f"✅ Columns: {data['columns']}")
    print(f"✅ Detected Column: {data['detected_column']}")
    
    return data['upload_id']


def test_start_analysis(upload_id: str, report_type: str = "executive"):
    """Test starting analysis.
    
    Args:
        upload_id: Upload ID from previous step
        report_type: Type of report
        
    Returns:
        Job ID
    """
    print_section(f"TEST 3: Start Analysis - {report_type}")
    
    payload = {
        "upload_id": upload_id,
        "report_type": report_type,
        "max_regenerations": 2
    }
    
    response = requests.post(f"{API_BASE}/analyze", json=payload)
    assert response.status_code == 200, f"Analysis failed: {response.text}"
    
    data = response.json()
    print(f"✅ Job ID: {data['job_id']}")
    print(f"✅ Status: {data['status']}")
    print(f"✅ Estimated Time: {data['estimated_time']}s")
    
    return data['job_id']


def test_poll_job_status(job_id: str, timeout: int = 300):
    """Poll job status until complete.
    
    Args:
        job_id: Job ID to poll
        timeout: Maximum wait time in seconds
        
    Returns:
        Final job status
    """
    print_section("TEST 4: Poll Job Status")
    
    start_time = time.time()
    last_progress = 0
    
    while True:
        response = requests.get(f"{API_BASE}/job/{job_id}")
        assert response.status_code == 200, f"Job status failed: {response.text}"
        
        data = response.json()
        
        # Print progress if changed
        if data['progress'] > last_progress:
            print(f"📊 Progress: {data['progress']:.1f}% - {data['current_stage']}")
            last_progress = data['progress']
        
        # Check completion
        if data['status'] == 'completed':
            print(f"✅ Job completed in {time.time() - start_time:.1f}s")
            return data
        
        if data['status'] == 'failed':
            print(f"❌ Job failed: {data.get('error', 'Unknown error')}")
            raise Exception(f"Job failed: {data.get('error')}")
        
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job timed out after {timeout}s")
        
        time.sleep(2)


def test_get_report(job_id: str):
    """Test getting report data.
    
    Args:
        job_id: Job ID
        
    Returns:
        Report data
    """
    print_section("TEST 5: Get Report")
    
    response = requests.get(f"{API_BASE}/report/{job_id}")
    assert response.status_code == 200, f"Get report failed: {response.text}"
    
    data = response.json()
    print(f"✅ Quality Score: {data['quality_score']:.1f}/100")
    print(f"✅ Word Count: {data['word_count']:,}")
    print(f"✅ Processing Time: {data['total_time']:.1f}s")
    print(f"✅ Cost: ${data['cost']:.4f}")
    print(f"✅ Report Length: {len(data['report_text'])} chars")
    
    return data


def test_download_markdown(job_id: str, output_path: Path):
    """Test downloading markdown report.
    
    Args:
        job_id: Job ID
        output_path: Where to save file
    """
    print_section("TEST 6: Download Markdown")
    
    response = requests.get(f"{API_BASE}/report/{job_id}/download?format=md")
    assert response.status_code == 200, f"Download failed: {response.text}"
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Saved to: {output_path}")
    print(f"✅ File size: {output_path.stat().st_size:,} bytes")


def test_download_pdf(job_id: str, output_path: Path):
    """Test downloading PDF report.
    
    Args:
        job_id: Job ID
        output_path: Where to save file
    """
    print_section("TEST 7: Download PDF")
    
    response = requests.get(f"{API_BASE}/report/{job_id}/download?format=pdf")
    assert response.status_code == 200, f"PDF download failed: {response.text}"
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Saved to: {output_path}")
    print(f"✅ File size: {output_path.stat().st_size:,} bytes")


def run_full_test(csv_path: Path, report_type: str = "executive"):
    """Run full E2E test.
    
    Args:
        csv_path: Path to test CSV
        report_type: Type of report to generate
    """
    print("\n" + "🚀" * 30)
    print(f"  FULL E2E TEST: {csv_path.name}")
    print("🚀" * 30)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: Upload
        upload_id = test_upload_csv(csv_path)
        
        # Test 3: Start analysis
        job_id = test_start_analysis(upload_id, report_type)
        
        # Test 4: Poll status
        test_poll_job_status(job_id)
        
        # Test 5: Get report
        test_get_report(job_id)
        
        # Test 6: Download MD
        md_path = Path("outputs") / f"test_{job_id}.md"
        test_download_markdown(job_id, md_path)
        
        # Test 7: Download PDF
        pdf_path = Path("outputs") / f"test_{job_id}.pdf"
        test_download_pdf(job_id, pdf_path)
        
        print("\n" + "✅" * 30)
        print("  ALL TESTS PASSED!")
        print("✅" * 30)
        
        return True
        
    except Exception as e:
        print("\n" + "❌" * 30)
        print(f"  TEST FAILED: {e}")
        print("❌" * 30)
        return False


def main():
    """Run all E2E tests."""
    
    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    
    # Test 1: Small CSV (existing sample)
    csv_small = TEST_DATA_DIR / "sample_reviews.csv"
    if csv_small.exists():
        print("\n📋 Testing with SMALL CSV (12 rows)...")
        run_full_test(csv_small, "executive")
    
    # Test 2: Different report type
    if csv_small.exists():
        print("\n📋 Testing with MARKETING report...")
        run_full_test(csv_small, "marketing")
    
    print("\n" + "🎉" * 30)
    print("  E2E TESTING COMPLETE!")
    print("🎉" * 30)


if __name__ == "__main__":
    main()