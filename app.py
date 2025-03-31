#!/usr/bin/env python3
"""
Test script for the RAG system with PDF files.
This script demonstrates:
1. How to ingest a PDF file into the RAG system
2. How to query the system about the PDF content
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import glob

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check pydantic version and other dependencies
try:
    import pkg_resources
    pydantic_version = pkg_resources.get_distribution("pydantic").version
    logger.info(f"Pydantic version: {pydantic_version}")
except Exception as e:
    logger.warning(f"Could not determine pydantic version: {e}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import RAG system components with error handling
try:
    from src.ingest.ingest_documents import ingest_single_document, ingest_documents
    from src.models.rag_model import RAGModel
    from src.config.config import PDF_DIRECTORY
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("This might be due to dependency version conflicts.")
    logger.error("Try running: pip install -r requirements.txt")
    sys.exit(1)

def ensure_pdf_directory():
    """Ensure the PDF directory exists"""
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    return Path(PDF_DIRECTORY)

def find_pdfs_in_data_folder():
    """
    Find all PDF files in the data directory.
    
    Returns:
        List of PDF file paths
    """
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        # Create the data directory
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Created data directory: {data_dir}")
        return []
    
    # Find all PDFs in data directory and its subdirectories
    pdf_files = list(data_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    return pdf_files

def ingest_pdf(pdf_path, threads=None):
    """
    Ingest a PDF file into the RAG system.
    
    Args:
        pdf_path: Path to the PDF file to ingest
        threads: Number of threads to use for processing
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Ingesting PDF: {pdf_path}")
    ingest_single_document(pdf_path, max_workers=threads)
    logger.info("PDF ingestion complete")

def ingest_all_pdfs_from_data(threads=None):
    """
    Ingest all PDF files found in the data directory
    
    Args:
        threads: Number of threads to use for processing
    
    Returns:
        Number of PDF files ingested
    """
    pdf_files = find_pdfs_in_data_folder()
    if not pdf_files:
        logger.warning("No PDF files found in data directory")
        return 0
    
    logger.info(f"Ingesting {len(pdf_files)} PDF files from data directory")
    data_dir = Path("./data")
    
    # Use multi-threading for ingestion
    ingest_documents(str(data_dir), max_workers=threads)
    
    return len(pdf_files)

def query_rag(query_text, with_sources=False, use_multi_query=True):
    """
    Query the RAG system.
    
    Args:
        query_text: The query text
        with_sources: Whether to include sources in the response
        use_multi_query: Whether to use MultiQueryRetriever for enhanced retrieval
        
    Returns:
        The response from the RAG system
    """
    logger.info(f"Querying RAG system: {query_text}")
    try:
        rag_model = RAGModel(use_multi_query=use_multi_query)
        
        if with_sources:
            result = rag_model.query_with_sources(query_text)
            logger.info(f"Answer: {result['result']}")
            logger.info("Sources:")
            for i, doc in enumerate(result['source_documents']):
                logger.info(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'Unknown')}")
            return result
        else:
            result = rag_model.query(query_text)
            logger.info(f"Answer: {result}")
            return result
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        return f"Error: {str(e)}"

def handle_command(command, current_settings):
    """
    Handle command inputs in interactive mode.
    
    Args:
        command: The user command
        current_settings: Dictionary containing current settings
        
    Returns:
        Tuple of (handled_flag, updated_settings, new_model)
    """
    handled = True
    rag_model = None
    settings = current_settings.copy()
    
    if command.lower() == 'sources on':
        settings['with_sources'] = True
        print("Source citations enabled.")
    elif command.lower() == 'sources off':
        settings['with_sources'] = False
        print("Source citations disabled.")
    elif command.lower() == 'multiquery on':
        if not settings['use_multi_query']:
            settings['use_multi_query'] = True
            print("MultiQueryRetriever enabled. Recreating RAG model...")
            rag_model = RAGModel(use_multi_query=settings['use_multi_query'])
            print("MultiQueryRetriever is now active.")
        else:
            print("MultiQueryRetriever is already enabled.")
    elif command.lower() == 'multiquery off':
        if settings['use_multi_query']:
            settings['use_multi_query'] = False
            print("MultiQueryRetriever disabled. Recreating RAG model...")
            rag_model = RAGModel(use_multi_query=settings['use_multi_query'])
            print("MultiQueryRetriever is now inactive.")
        else:
            print("MultiQueryRetriever is already disabled.")
    else:
        handled = False
        
    return handled, settings, rag_model

def process_query(query, rag_model, settings):
    """
    Process a user query and display the result.
    
    Args:
        query: The user query text
        rag_model: The RAG model instance
        settings: Dictionary containing current settings
    """
    try:
        if settings['with_sources']:
            result = rag_model.query_with_sources(query)
            print(f"\nAnswer: {result['result']}")
            print("\nSources:")
            for i, doc in enumerate(result['source_documents']):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                print(f"  {i+1}. {source}, page {page}")
        else:
            result = rag_model.query(query)
            print(f"\nAnswer: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")

def print_help():
    """Print help information for interactive mode."""
    print("\n=== RAG Interactive Mode ===")
    print("You can ask questions about your ingested documents.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("Type 'sources on' to enable source citations.")
    print("Type 'sources off' to disable source citations.")
    print("Type 'multiquery on' to enable MultiQueryRetriever (generates multiple queries for better search).")
    print("Type 'multiquery off' to disable MultiQueryRetriever.")
    print("===============================\n")

def interactive_mode(with_sources=False, use_multi_query=True):
    """
    Run the RAG system in interactive mode, allowing continuous questioning.
    
    Args:
        with_sources: Whether to include sources in responses
        use_multi_query: Whether to use MultiQueryRetriever for enhanced retrieval
    """
    print_help()
    
    # Initialize settings
    settings = {
        'with_sources': with_sources,
        'use_multi_query': use_multi_query
    }
    
    # Create RAG model once for the entire session
    try:
        rag_model = RAGModel(use_multi_query=settings['use_multi_query'])
    except Exception as e:
        logger.error(f"Error initializing RAG model: {e}")
        print(f"Error: Could not initialize RAG model. {str(e)}")
        print("Make sure you've ingested documents first.")
        return
    
    try:
        while True:
            query = input("\nQuestion: ").strip()
            
            # Check for exit command
            if query.lower() in ('exit', 'quit', 'q'):
                print("Exiting interactive mode.")
                break
                
            # Skip empty input
            if not query:
                continue
                
            # Handle special commands
            handled, settings, new_model = handle_command(query, settings)
            if handled:
                if new_model:
                    rag_model = new_model
                continue
            
            # Process regular query
            process_query(query, rag_model, settings)
                
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

def main():
    parser = argparse.ArgumentParser(description="Test the RAG system with PDF files from data folder")
    parser.add_argument("--pdf", "-p", type=str, help="Path to a specific PDF file to ingest (optional)")
    parser.add_argument("--query", "-q", type=str, help="Query to ask about the PDF content")
    parser.add_argument("--sources", "-s", action="store_true", help="Include sources in the response")
    parser.add_argument("--ingest-all", "-a", action="store_true", help="Ingest all PDFs from data folder")
    parser.add_argument("--install-deps", "-i", action="store_true", help="Install or update dependencies")
    parser.add_argument("--threads", "-t", type=int, help="Number of threads to use for processing (default: CPU count)")
    parser.add_argument("--interactive", "-I", action="store_true", help="Run in interactive mode for continuous questioning")
    parser.add_argument("--no-multi-query", action="store_true", help="Disable MultiQueryRetriever (improves quality but uses more tokens)")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        try:
            import subprocess
            logger.info("Installing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logger.info("Dependencies installed successfully")
            return
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return
    
    try:
        # Ensure PDF directory exists
        pdf_dir = ensure_pdf_directory()
        
        # Ingest specific PDF if provided
        if args.pdf:
            ingest_pdf(args.pdf, threads=args.threads)
        # Otherwise ingest all PDFs from data folder if requested or if no query
        elif args.ingest_all or (not args.query and not args.interactive):
            ingested = ingest_all_pdfs_from_data(threads=args.threads)
            if ingested == 0 and not args.query and not args.interactive:
                print("\nNo PDF files found in data directory.")
                print("Please place PDF files in the 'data' folder or specify a path with --pdf.")
                print("\nExample usage:")
                print("1. Ingest all PDFs from data folder: python tst.py --ingest-all")
                print("2. Ingest all PDFs with multi-threading: python tst.py --ingest-all --threads 4")
                print("3. Ingest a specific PDF: python tst.py --pdf path/to/your/document.pdf")
                print("4. Query the system: python tst.py --query 'What is the main topic of the document?'")
                print("5. Query with sources: python tst.py --query 'What is the main topic?' --sources")
                print("6. Interactive mode: python tst.py --interactive")
                print("7. Disable multi-query retrieval: python tst.py --interactive --no-multi-query")
                print("8. Update dependencies: python tst.py --install-deps\n")
                return
        
        # Run in interactive mode if requested
        if args.interactive:
            interactive_mode(with_sources=args.sources, use_multi_query=not args.no_multi_query)
        # Single query mode
        elif args.query:
            query_rag(args.query, args.sources, use_multi_query=not args.no_multi_query)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\nError: {e}")
        print("If this is a dependency issue, try: python tst.py --install-deps")

if __name__ == "__main__":
    main() 