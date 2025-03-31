#!/usr/bin/env python3
"""
RAG Pipeline Chatbot

This script creates a terminal-based chatbot that uses the RAG (Retrieval-Augmented Generation)
pipeline to answer questions based on PDF documents.

Usage:
    python rag_pipeline_chatbot.py [--ingest]

Options:
    --ingest    Ingest PDFs from the configured directory before starting the chatbot
"""
import argparse
import sys
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules
from Rag.src.ingest.ingest_documents import ingest_documents
from Rag.src.models.rag_model import RAGModel
from Rag.src.config.config import PDF_DIRECTORY, LLM_PROVIDER


def process_query(query: str, model: RAGModel, show_sources: bool = False) -> None:
    """
    Process a user query and display the response.
    
    Args:
        query: The user's question
        model: The RAG model to use
        show_sources: Whether to display the source documents
    """
    print("\n🔍 Searching knowledge base and generating response...\n")
    
    if show_sources:
        result = model.query_with_sources(query)
        print(f"\n🤖 Answer: {result['result']}\n")
        
        if result.get("source_documents"):
            print("📚 Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "Unknown page")
                print(f"  {i}. {source} (Page {page})")
                print(f"     {doc.page_content[:100]}...\n")
    else:
        result = model.query(query)
        print(f"\n🤖 Answer: {result}\n")


def run_chatbot(ingest_docs: bool = False) -> None:
    """
    Run the interactive RAG chatbot in the terminal.
    
    Args:
        ingest_docs: Whether to ingest documents before starting
    """
    # Display provider information
    print(f"\n💬 Using LLM provider: {LLM_PROVIDER}")
    
    if LLM_PROVIDER == "ollama":
        print("ℹ️  Using Ollama provider. Make sure Ollama is running and models are pulled.")
        print("   You can check available models with: ollama list")
        print("   You can pull models with: ollama pull model_name")
    
    if ingest_docs:
        try:
            print(f"\n📥 Ingesting documents from {PDF_DIRECTORY}...")
            ingest_documents()
            print("✅ Document ingestion complete\n")
        except Exception as e:
            print(f"\n❌ Error ingesting documents: {str(e)}")
            print("   Make sure your PDFs are in the correct directory and properly formatted.")
            if "model not found" in str(e).lower() and LLM_PROVIDER == "ollama":
                print("   It looks like the Ollama model wasn't found. Try pulling it with:")
                print("   ollama pull llama2")
            return

    print("\n🤖 Welcome to the RAG Chatbot!")
    print("Ask questions about the documents in your knowledge base.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'sources on' to show source documents with answers.")
    print("Type 'sources off' to hide source documents.\n")

    # Initialize the RAG model
    try:
        model = RAGModel(temperature=0.2)
        show_sources = False
    except Exception as e:
        print(f"\n❌ Error initializing RAG model: {str(e)}")
        if "model not found" in str(e).lower() and LLM_PROVIDER == "ollama":
            print("   It looks like the Ollama model wasn't found. Try pulling it with:")
            print("   ollama pull llama2")
        return

    while True:
        try:
            query = input("🧠 Ask a question: ")
            
            # Check for exit command
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
                
            # Check for sources command
            elif query.lower() == "sources on":
                show_sources = True
                print("📚 Sources display enabled")
                continue
            elif query.lower() == "sources off":
                show_sources = False
                print("📚 Sources display disabled")
                continue
            
            # Skip empty queries
            elif not query.strip():
                continue
            
            # Process query
            process_query(query, model, show_sources)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if "model not found" in str(e).lower() and LLM_PROVIDER == "ollama":
                print("   It looks like the Ollama model wasn't found. Try pulling it with:")
                print("   ollama pull llama2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Chatbot")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents before starting")
    
    args = parser.parse_args()
    
    run_chatbot(ingest_docs=args.ingest) 