#!/bin/bash

# Start the API server
echo "Starting FastAPI server..."
uvicorn src.api.main:app --reload --port 8000