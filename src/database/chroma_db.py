"""
ChromaDB connector for vector database operations.
"""
import chromadb
from chromadb.config import Settings
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional
import concurrent.futures
from math import ceil

# Don't even try to use AVX instructions
os.environ["DISABLE_AVX"] = "1"
os.environ["DISABLE_AVX2"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization

# Import necessary modules after setting environment variables
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ..config.config import (
    CHROMA_PERSIST_DIRECTORY, 
    COLLECTION_NAME, 
    LLM_PROVIDER,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL
)

# Configure logging
logger = logging.getLogger(__name__)

class SimpleEmbeddings:
    """
    A simple fallback embedding class that doesn't use TensorFlow.
    It uses a very basic approach based on word hashing.
    """
    def __init__(self, dimension=1536):
        self.dimension = dimension
    
    def _hash_text(self, text):
        """Simple word hashing function"""
        import hashlib
        words = text.lower().split()
        hashes = [int(hashlib.md5(word.encode()).hexdigest(), 16) for word in words]
        return hashes
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents"""
        embeddings = []
        for text in texts:
            hashes = self._hash_text(text)
            # Create a pseudo-random embedding based on word hashes
            np.random.seed(sum(hashes))
            embedding = np.random.uniform(-1, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text):
        """Generate embedding for a query"""
        return self.embed_documents([text])[0]


class ChromaDBConnector:
    """
    Connector class for ChromaDB operations.
    """
    def __init__(self):
        """Initialize the ChromaDB connector."""
        # Configure embeddings based on provider
        if LLM_PROVIDER == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
        elif LLM_PROVIDER == "ollama":
            # Try multiple embedding strategies with fallbacks
            
                try:
                    # Try importing HuggingFace embeddings - might fail if TensorFlow has AVX issues
                    logger.info("Trying to use HuggingFaceEmbeddings...")
                    from langchain_huggingface import HuggingFaceEmbeddings
                    
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="paraphrase-MiniLM-L3-v2",  # Smaller model
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    logger.info("Successfully initialized HuggingFaceEmbeddings")
                except Exception as e2:
                    logger.warning(f"Failed to initialize HuggingFaceEmbeddings: {str(e2)}")
                    logger.info("Falling back to SimpleEmbeddings (basic fallback)")
                    self.embeddings = SimpleEmbeddings(dimension=384)  # Small dimension for efficiency
                    logger.warning("Using SimpleEmbeddings fallback - limited semantic search capabilities")
        
        self.persist_directory = CHROMA_PERSIST_DIRECTORY
        self.collection_name = COLLECTION_NAME
        
    def get_vector_store(self) -> Chroma:
        """
        Initialize and return a ChromaDB vector store.
        
        Returns:
            Chroma: The initialized vector store.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Document], batch_size: int = 100, max_workers: int = None) -> None:
        """
        Add documents to the vector store with parallel embedding.
        
        Args:
            documents: List of document objects to add.
            batch_size: Number of documents to process in each batch.
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Get the vector store once
        vector_store = self.get_vector_store()
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 max to avoid overload
        
        # Calculate optimal batch size based on document count and workers
        if len(documents) < batch_size:
            batches = [documents]
        else:
            num_batches = max(1, min(ceil(len(documents) / batch_size), max_workers * 2))
            batch_size = ceil(len(documents) / num_batches)
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        logger.info(f"Processing {len(batches)} batches with {max_workers} workers")
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                futures.append(executor.submit(self._add_batch, vector_store, batch, i+1, len(batches)))
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error adding documents batch: {e}")
        
        logger.info(f"Successfully added {len(documents)} documents to vector store")
    
    def _add_batch(self, vector_store: Chroma, batch: List[Document], batch_num: int, total_batches: int) -> None:
        """
        Add a batch of documents to the vector store.
        Helper function for parallel processing.
        
        Args:
            vector_store: The Chroma vector store.
            batch: Batch of documents to add.
            batch_num: Current batch number.
            total_batches: Total number of batches.
        """
        try:
            vector_store.add_documents(batch)
            logger.info(f"Added batch {batch_num}/{total_batches} ({len(batch)} documents)")
        except Exception as e:
            logger.error(f"Error adding batch {batch_num}/{total_batches}: {e}")
            raise
        
    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query string to search for.
            k: Number of similar documents to retrieve.
            
        Returns:
            List of similar documents.
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string to search for.
            k: Number of similar documents to retrieve.
            
        Returns:
            List of tuples containing documents and their relevance scores.
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search_with_score(query, k=k) 