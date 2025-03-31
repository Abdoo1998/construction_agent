"""
Script for ingesting documents into the vector database.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Union, List
import concurrent.futures
import os

from ..database.chroma_db import ChromaDBConnector
from ..ingest.document_processor import DocumentProcessor
from ..config.config import PDF_DIRECTORY
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_documents(pdf_directory: Optional[str] = None, max_workers: int = None):
    """
    Ingest all PDF documents from the specified directory into the vector database.
    Uses multi-threading to process multiple documents in parallel.
    
    Args:
        pdf_directory: Directory containing PDF files to ingest.
        max_workers: Maximum number of worker threads (default: CPU count)
    """
    pdf_dir = pdf_directory or PDF_DIRECTORY
    logger.info(f"Ingesting documents from {pdf_dir}")
    
    # Use multi-threading for processing multiple documents
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        logger.error(f"Directory {pdf_dir} does not exist")
        return
    
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return
    
    # Process documents in parallel
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(pdf_files))
    
    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Processing documents using {max_workers} threads")
        future_to_file = {
            executor.submit(_process_single_file, pdf_file): pdf_file
            for pdf_file in pdf_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            pdf_file = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                logger.info(f"Processed {pdf_file.name} ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
    
    logger.info(f"Processed {len(all_chunks)} total document chunks")
    
    # Add to database with parallel embedding
    db_connector = ChromaDBConnector()
    db_connector.add_documents(all_chunks, max_workers=max_workers)
    
    logger.info("Successfully added documents to vector database")


def _process_single_file(file_path: Path) -> List[Document]:
    """
    Process a single PDF file and return its chunks.
    Helper function for multi-threading.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of document chunks
    """
    document_processor = DocumentProcessor()
    return document_processor.process_single_document(file_path)


def ingest_single_document(file_path: Union[str, Path], max_workers: int = None):
    """
    Ingest a single PDF document into the vector database.
    
    Args:
        file_path: Path to the PDF file to ingest.
        max_workers: Maximum number of worker threads for embedding (default: CPU count)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist")
        return
    
    logger.info(f"Ingesting document {file_path}")
    
    # Process document
    document_processor = DocumentProcessor()
    chunks = document_processor.process_single_document(file_path)
    
    logger.info(f"Processed {len(chunks)} document chunks")
    
    # Add to database with parallel embedding
    db_connector = ChromaDBConnector()
    db_connector.add_documents(chunks, max_workers=max_workers)
    
    logger.info("Successfully added document to vector database")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the vector database")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--directory", "-d", type=str, help="Directory containing PDF files to ingest")
    group.add_argument("--file", "-f", type=str, help="Single PDF file to ingest")
    
    parser.add_argument("--threads", "-t", type=int, help="Number of threads to use for processing (default: CPU count)")
    
    args = parser.parse_args()
    
    if args.directory:
        ingest_documents(args.directory, max_workers=args.threads)
    elif args.file:
        ingest_single_document(args.file, max_workers=args.threads) 