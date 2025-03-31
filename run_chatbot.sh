#!/bin/bash

# RAG Pipeline Chatbot runner script
# This script runs the RAG pipeline chatbot with optional document ingestion

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ to run this script."
    exit 1
fi

# Parse arguments
INGEST_FLAG=""
if [ "$1" == "--ingest" ]; then
    INGEST_FLAG="--ingest"
    echo "🔄 Running chatbot with document ingestion..."
else
    echo "▶️ Running chatbot without document ingestion..."
fi

# Ensure the .env file exists
if [ ! -f .env ]; then
    echo "⚠️ Warning: .env file not found. You may need to set up your environment variables."
fi

# Run the chatbot directly from its location
cd "$SCRIPT_DIR/src"
python rag_pipeline_chatbot.py $INGEST_FLAG

exit 0 