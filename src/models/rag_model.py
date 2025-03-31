"""
RAG model that combines retrieval with generation.
"""
from typing import List, Dict, Any, Optional
import logging

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from ..database.chroma_db import ChromaDBConnector
from ..config.config import (
    LLM_PROVIDER, 
    COMPLETION_MODEL, 
    OPENAI_API_KEY,
    OLLAMA_BASE_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the custom prompt for RAG
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant that provides information based on the documents you have access to.
Use the following context to answer the question. If you don't know the answer or if the answer 
cannot be determined from the context, say "I don't have enough information to answer this question." 
Don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

# Define MultiQueryRetriever prompt template
MULTI_QUERY_PROMPT_TEMPLATE = """
You are an AI assistant helping to generate multiple search queries for retrieving relevant documents.
Your task is to generate 3 different versions of the given user question to retrieve relevant information
from a vector database. By generating multiple perspectives on the user question, your goal is to help
find the most useful documents that might answer the original question.

Provide these alternative questions separated by newlines. Don't include any numbering or prefixes.
Each line should contain just one complete question.

Original question: {question}
"""


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse the LLM output into a list of queries."""
    
    def parse(self, text: str) -> List[str]:
        """Parse the output text into a list of strings, one per line."""
        queries = text.strip().split("\n")
        # Filter out empty lines and remove any numbering
        cleaned_queries = []
        for query in queries:
            # Remove numbering patterns like "1.", "1:", "1)" etc.
            query = query.strip()
            if not query:
                continue
            # Remove any numbering patterns at the beginning
            import re
            query = re.sub(r"^\d+[\.\)\:-]\s*", "", query)
            cleaned_queries.append(query)
        return cleaned_queries


class RAGModel:
    """
    RAG model that combines retrieval with generation.
    """
    
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024, use_multi_query: bool = True):
        """
        Initialize the RAG model.
        
        Args:
            temperature: Temperature for the LLM.
            max_tokens: Maximum tokens for the LLM response.
            use_multi_query: Whether to use MultiQueryRetriever for enhanced retrieval.
        """
        self.db_connector = ChromaDBConnector()
        self.use_multi_query = use_multi_query
        
        # Initialize LLM based on provider
        if LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(
                model=COMPLETION_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=OPENAI_API_KEY
            )
        elif LLM_PROVIDER == "ollama":
            self.llm = OllamaLLM(
                model=COMPLETION_MODEL,
                temperature=temperature,
            )
        else:
            # Default to OpenAI if invalid provider
            logger.warning(f"Invalid LLM provider: {LLM_PROVIDER}. Using OpenAI instead.")
            self.llm = ChatOpenAI(
                model=COMPLETION_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=OPENAI_API_KEY
            )
        
        self.prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
    def get_retriever(self):
        """
        Get the appropriate retriever based on configuration.
        
        Returns:
            A retriever instance (either standard or multi-query).
        """
        base_retriever = self.db_connector.get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        if not self.use_multi_query:
            return base_retriever
        
        try:
            # Set up the multi-query retriever
            logger.info("Setting up MultiQueryRetriever")
            
            # Create the multi-query chain
            multi_query_prompt = PromptTemplate(
                template=MULTI_QUERY_PROMPT_TEMPLATE,
                input_variables=["question"]
            )
            output_parser = LineListOutputParser()
            
            multi_query_llm_chain = multi_query_prompt | self.llm | output_parser
            
            # Create the multi-query retriever
            multi_query_retriever = MultiQueryRetriever(
                retriever=base_retriever,
                llm_chain=multi_query_llm_chain,
                parser_key="lines"
            )
            
            return multi_query_retriever
            
        except Exception as e:
            logger.error(f"Error setting up MultiQueryRetriever: {e}")
            logger.warning("Falling back to standard retriever")
            return base_retriever
    
    def setup_retrieval_chain(self):
        """
        Set up the retrieval chain with the RAG pattern.
        
        Returns:
            The configured retrieval chain.
        """
        retriever = self.get_retriever()
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def setup_retrieval_qa_chain(self):
        """
        Set up a RetrievalQA chain.
        
        Returns:
            RetrievalQA: The configured QA chain.
        """
        retriever = self.get_retriever()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )
    
    def query(self, query_text: str) -> str:
        """
        Query the RAG model with a question.
        
        Args:
            query_text: The question to ask.
            
        Returns:
            str: The generated answer.
        """
        chain = self.setup_retrieval_chain()
        try:
            result = chain.invoke(query_text)
            return result
        except Exception as e:
            logger.error(f"Error querying RAG model: {e}")
            return f"An error occurred: {str(e)}"
    
    def query_with_sources(self, query_text: str) -> Dict[str, Any]:
        """
        Query the RAG model and return the answer with source documents.
        
        Args:
            query_text: The question to ask.
            
        Returns:
            Dict containing the answer and source documents.
        """
        chain = self.setup_retrieval_qa_chain()
        try:
            result = chain({"query": query_text})
            return result
        except Exception as e:
            logger.error(f"Error querying RAG model with sources: {e}")
            return {"result": f"An error occurred: {str(e)}", "source_documents": []} 