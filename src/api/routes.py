"""
API routes for the RAG system.
"""
import logging
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict, Any

from ..models.rag_model import RAGModel
from ..ingest.ingest_documents import ingest_single_document, ingest_documents
from ..ingest.document_processor import DocumentProcessor
from .schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, SourceDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the API router
router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query_rag_model(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG model with a question.
    
    Args:
        request: Query request with the question.
        
    Returns:
        Query response with the answer.
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        rag_model = RAGModel()
        
        if request.include_sources:
            result = rag_model.query_with_sources(request.query)
            
            # Extract source documents
            sources = []
            for doc in result.get("source_documents", []):
                sources.append(
                    SourceDocument(
                        content=doc.page_content,
                        source=doc.metadata.get("source"),
                        metadata=doc.metadata
                    )
                )
            
            return QueryResponse(
                answer=result.get("result", "No result found"),
                sources=sources
            )
        else:
            answer = rag_model.query(request.query)
            return QueryResponse(answer=answer)
            
    except Exception as e:
        logger.error(f"Error querying RAG model: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying RAG model: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Ingest a PDF document into the vector database.
    
    Args:
        request: Ingest request with the file path.
        
    Returns:
        Ingest response indicating success or failure.
    """
    file_path = Path(request.file_path)
    logger.info(f"Received ingest request for file: {file_path}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {file_path} not found")
    
    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail=f"File {file_path} is not a PDF file")
    
    try:
        # First process the document to count chunks
        processor = DocumentProcessor()
        chunks = processor.process_single_document(file_path)
        chunk_count = len(chunks)
        
        # Then ingest the document
        ingest_single_document(file_path)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested document {file_path}",
            document_chunks=chunk_count
        )
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")


@router.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(directory_path: str) -> IngestResponse:
    """
    Ingest all PDF documents from a directory into the vector database.
    
    Args:
        directory_path: Path to the directory containing PDFs.
        
    Returns:
        Ingest response indicating success or failure.
    """
    dir_path = Path(directory_path)
    logger.info(f"Received ingest request for directory: {dir_path}")
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory {dir_path} not found")
    
    try:
        # First count the number of PDF files and chunks
        processor = DocumentProcessor(pdf_directory=str(dir_path))
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            raise HTTPException(status_code=400, detail=f"No PDF files found in directory {dir_path}")
        
        # Process all documents to count chunks
        chunks = processor.process_documents()
        chunk_count = len(chunks)
        
        # Then ingest all documents
        ingest_documents(str(dir_path))
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(pdf_files)} PDF files from {dir_path}",
            document_chunks=chunk_count
        )
    except Exception as e:
        logger.error(f"Error ingesting documents from directory: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}") 