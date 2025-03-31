"""
Document processor for PDF ingestion and chunking.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config.config import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """
    Class for processing PDF documents for RAG.
    """
    def __init__(self, 
                 pdf_directory: Optional[str] = None, 
                 chunk_size: int = CHUNK_SIZE, 
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            pdf_directory: Directory containing PDF files to process.
            chunk_size: Size of chunks to split documents into.
            chunk_overlap: Overlap size between chunks.
        """
        self.pdf_directory = pdf_directory or PDF_DIRECTORY
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def load_single_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of loaded document pages.
        """
        loader = PyMuPDFLoader(str(file_path))
        return loader.load()
    
    def load_documents_from_directory(self) -> List[Document]:
        """
        Load all PDF documents from the specified directory.
        
        Returns:
            List of loaded document pages from all PDFs.
        """
        pdf_dir = Path(self.pdf_directory)
        all_documents = []
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory {pdf_dir} does not exist")
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            documents = self.load_single_pdf(pdf_file)
            all_documents.extend(documents)
            
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of document chunks.
        """
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self) -> List[Document]:
        """
        Load and process all documents from the PDF directory.
        
        Returns:
            List of processed document chunks.
        """
        documents = self.load_documents_from_directory()
        return self.chunk_documents(documents)
    
    def process_single_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Process a single PDF document.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of processed document chunks.
        """
        documents = self.load_single_pdf(file_path)
        return self.chunk_documents(documents) 