// Global state
let uploadId = null;
let jobId = null;
let pollInterval = null;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const configSection = document.getElementById('config-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

const fileInfo = document.getElementById('file-info');
const filename = document.getElementById('filename');
const rows = document.getElementById('rows');
const textColumn = document.getElementById('text-column');
const textColumnSelect = document.getElementById('text-column-select');

const startBtn = document.getElementById('start-btn');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const stageText = document.getElementById('stage-text');

const qualityScore = document.getElementById('quality-score');
const wordCount = document.getElementById('word-count');
const processingTime = document.getElementById('processing-time');
const cost = document.getElementById('cost');

const downloadMdBtn = document.getElementById('download-md-btn');
const downloadPdfBtn = document.getElementById('download-pdf-btn');
const newReportBtn = document.getElementById('new-report-btn');
const retryBtn = document.getElementById('retry-btn');

const errorMessage = document.getElementById('error-message');

// API base URL
const API_BASE = '/api';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Buttons
    startBtn.addEventListener('click', startAnalysis);
    downloadMdBtn.addEventListener('click', downloadMarkdown);
    newReportBtn.addEventListener('click', reset);
    retryBtn.addEventListener('click', reset);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10MB');
        return;
    }

    // Upload file
    try {
        showLoading('Uploading file...');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        handleUploadSuccess(data);

    } catch (error) {
        showError(`Upload failed: ${error.message}`);
    }
}

function handleUploadSuccess(data) {
    uploadId = data.upload_id;

    // Show file info
    filename.textContent = data.filename;
    rows.textContent = data.rows;
    textColumn.textContent = data.detected_column || 'Not detected';
    fileInfo.classList.remove('hidden');

    // Populate column select
    textColumnSelect.innerHTML = '<option value="">Auto-detect</option>';
    data.columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        if (col === data.detected_column) {
            option.selected = true;
        }
        textColumnSelect.appendChild(option);
    });

    // Show config section
    configSection.classList.remove('hidden');
    uploadArea.style.display = 'none';
}

async function startAnalysis() {
    const reportType = document.getElementById('report-type').value;
    const textColumnValue = textColumnSelect.value;

    try {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';

        const requestBody = {
            upload_id: uploadId,
            report_type: reportType,
            max_regenerations: 3
        };

        // Add text_column if manually selected
        if (textColumnValue) {
            requestBody.text_column = textColumnValue;
        }

        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed to start');
        }

        const data = await response.json();
        jobId = data.job_id;

        // Hide config, show processing
        configSection.classList.add('hidden');
        processingSection.classList.remove('hidden');

        // Start polling
        startPolling();

    } catch (error) {
        startBtn.disabled = false;
        startBtn.textContent = 'Generate Report';
        showError(`Failed to start analysis: ${error.message}`);
    }
}

function startPolling() {
    pollInterval = setInterval(checkJobStatus, 2000); // Poll every 2 seconds
    checkJobStatus(); // Check immediately
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

async function checkJobStatus() {
    try {
        const response = await fetch(`${API_BASE}/job/${jobId}`);
        
        if (!response.ok) {
            throw new Error('Failed to check job status');
        }

        const data = await response.json();

        // Update progress
        updateProgress(data.progress, data.current_stage);

        // Check status
        if (data.status === 'completed') {
            stopPolling();
            await loadReport();
        } else if (data.status === 'failed') {
            stopPolling();
            showError(data.error || 'Job failed');
        }

    } catch (error) {
        stopPolling();
        showError(`Failed to check status: ${error.message}`);
    }
}

function updateProgress(progress, stage) {
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
    stageText.textContent = stage;
}

async function loadReport() {
    try {
        const response = await fetch(`${API_BASE}/report/${jobId}`);

        if (!response.ok) {
            throw new Error('Failed to load report');
        }

        const data = await response.json();

        // Hide processing, show results
        processingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Display results
        qualityScore.textContent = `${data.quality_score.toFixed(1)}/100`;
        wordCount.textContent = data.word_count.toLocaleString();
        processingTime.textContent = `${data.total_time.toFixed(1)}s`;
        cost.textContent = `$${data.cost.toFixed(4)}`;

    } catch (error) {
        showError(`Failed to load report: ${error.message}`);
    }
}

async function downloadMarkdown() {
    try {
        downloadMdBtn.disabled = true;
        downloadMdBtn.textContent = 'Downloading...';

        const response = await fetch(`${API_BASE}/report/${jobId}/download?format=md`);

        if (!response.ok) {
            throw new Error('Download failed');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${jobId}.md`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        downloadMdBtn.disabled = false;
        downloadMdBtn.textContent = '📄 Download Markdown';

    } catch (error) {
        downloadMdBtn.disabled = false;
        downloadMdBtn.textContent = '📄 Download Markdown';
        showError(`Download failed: ${error.message}`);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    
    uploadSection.classList.add('hidden');
    configSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.remove('hidden');
}

function showLoading(message) {
    stageText.textContent = message;
    uploadSection.classList.add('hidden');
    processingSection.classList.remove('hidden');
}

function reset() {
    // Reset state
    uploadId = null;
    jobId = null;
    stopPolling();

    // Reset UI
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    uploadArea.style.display = 'block';
    
    uploadSection.classList.remove('hidden');
    configSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    stageText.textContent = 'Initializing...';

    startBtn.disabled = false;
    startBtn.textContent = 'Generate Report';
}