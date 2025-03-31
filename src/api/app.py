"""
Main FastAPI application.
"""
import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .routes import router as rag_router
from ..config.config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="PDF RAG API",
    description="API for querying a Retrieval Augmented Generation system built with LangChain, ChromaDB, and OpenAI.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the RAG router
app.include_router(rag_router)


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "PDF RAG API is running",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if OpenAI API key is set
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
        )
    
    return {"status": "healthy"} 