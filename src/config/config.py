"""
Configuration settings for the RAG application.
"""
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(Path(__file__).parents[2] / ".env")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # Options: "openai", "ollama"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-3.5-turbo")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# For Ollama, prefer embedding-capable models or fall back to LLMs that can also do embeddings
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.2:1b")  # Changed from nomic-embed-text to llama2
OLLAMA_COMPLETION_MODEL = os.getenv("OLLAMA_COMPLETION_MODEL", "llama3.2:1b")

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = "pdf_documents"

# PDF document settings
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./data/pdfs")

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Get active models based on provider
if LLM_PROVIDER == "openai":
    EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
    COMPLETION_MODEL = OPENAI_COMPLETION_MODEL
    logger.info(f"Using OpenAI provider with model: {COMPLETION_MODEL}")
    
elif LLM_PROVIDER == "ollama":
    EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL
    COMPLETION_MODEL = OLLAMA_COMPLETION_MODEL
    logger.info(f"Using Ollama provider with model: {COMPLETION_MODEL}")
    
else:
    # Default to OpenAI if invalid provider
    logger.warning(f"Invalid LLM provider: {LLM_PROVIDER}. Using OpenAI instead.")
    LLM_PROVIDER = "openai"
    EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
    COMPLETION_MODEL = OPENAI_COMPLETION_MODEL 