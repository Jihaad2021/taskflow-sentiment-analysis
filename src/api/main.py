"""FastAPI application for TaskFlow sentiment analysis."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.storage import storage

# App startup time for health check
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    # Startup
    print("🚀 TaskFlow API starting...")
    yield
    # Shutdown
    print("👋 TaskFlow API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="TaskFlow Sentiment Analysis API",
    description="Generate professional sentiment analysis reports from CSV files",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TaskFlow Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time

    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": round(uptime, 2),
        "jobs_count": len(storage.jobs),
        "uploads_count": len(storage.uploads),
    }
