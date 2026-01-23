"""Main RAG engine combining all components."""
from pathlib import Path
from typing import List, Dict, Optional
from src.core.document_processor import DocumentProcessor
from src.core.vector_store import VectorStoreBase, create_vector_store
from src.core.llm import LLMBase, create_llm
from src.core.prompts import PromptTemplates
from src.config import settings
from src.logger import logger


class RAGEngine:
    """Main RAG engine for document Q&A."""
    
    def __init__(
        self,
        llm_backend: str = "ollama",
        vector_store_type: str = None,
        language: str = None
    ):
        """
        Initialize RAG engine.
        
        Args:
            llm_backend: LLM backend to use ('ollama' or 'transformers')
            vector_store_type: Vector store type ('faiss' or 'chroma')
            language: Default language for prompts
        """
        logger.info("Initializing RAG engine")
        
        self.language = language or settings.default_language
        self.document_processor = DocumentProcessor()
        self.vector_store = create_vector_store(vector_store_type)
        self.llm = create_llm(llm_backend)
        self.prompts = PromptTemplates()
        
        logger.info(f"RAG engine initialized (backend: {llm_backend}, language: {self.language})")
    
    def ingest_documents(self, path: Path) -> int:
        """
        Ingest documents from file or directory.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Number of chunks processed
        """
        logger.info(f"Ingesting documents from: {path}")
        
        path = Path(path)
        
        if path.is_file():
            chunks = self.document_processor.process_document(path)
        elif path.is_dir():
            chunks = self.document_processor.process_directory(path)
        else:
            raise ValueError(f"Invalid path: {path}")
        
        if chunks:
            self.vector_store.add_documents(chunks)
            logger.info(f"Ingested {len(chunks)} chunks")
        else:
            logger.warning("No chunks to ingest")
        
        return len(chunks)
    
    def save_index(self, path: Path = None) -> None:
        """
        Save vector store index to disk.
        
        Args:
            path: Path to save to (defaults to config setting)
        """
        path = path or settings.get_vector_store_path()
        logger.info(f"Saving vector store to: {path}")
        self.vector_store.save(path)
    
    def load_index(self, path: Path = None) -> None:
        """
        Load vector store index from disk.
        
        Args:
            path: Path to load from (defaults to config setting)
        """
        path = path or settings.get_vector_store_path()
        logger.info(f"Loading vector store from: {path}")
        self.vector_store.load(path)
    
    def retrieve_context(self, query: str, k: int = None) -> tuple[str, List[Dict]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (formatted context string, list of source documents)
        """
        k = k or settings.top_k_results
        
        logger.info(f"Retrieving context for query: {query[:100]}...")
        results = self.vector_store.search(query, k=k)
        
        if not results:
            logger.warning("No relevant documents found")
            return "", []
        
        # Format context with sources
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            text = result["text"]
            context_parts.append(f"[Source: {source}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        logger.info(f"Retrieved {len(results)} relevant documents")
        return context, results
    
    def answer_question(
        self,
        question: str,
        language: str = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            language: Language for response (uses default if not specified)
            include_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        language = language or self.language
        
        logger.info(f"Answering question in {language}: {question[:100]}...")
        
        # Retrieve context
        context, sources = self.retrieve_context(question)
        
        if not context:
            # No context found
            answer = self.prompts.get_no_context_response(language)
            return {
                "answer": answer,
                "sources": [],
                "language": language,
                "has_context": False
            }
        
        # Format prompt
        system_prompt = self.prompts.get_system_prompt(language)
        user_prompt = self.prompts.format_prompt(question, context, language)
        
        # Generate answer
        answer = self.llm.generate(user_prompt, system_prompt)
        
        # Prepare response
        response = {
            "answer": answer,
            "language": language,
            "has_context": True
        }
        
        if include_sources:
            response["sources"] = [
                {
                    "text": src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"],
                    "metadata": src["metadata"],
                    "score": src.get("score", 0.0)
                }
                for src in sources
            ]
        
        logger.info(f"Generated answer: {len(answer)} characters")
        return response
    
    def chat(
        self,
        question: str,
        language: str = None
    ) -> str:
        """
        Simple chat interface that returns just the answer text.
        
        Args:
            question: User question
            language: Language for response
            
        Returns:
            Answer text
        """
        response = self.answer_question(question, language, include_sources=False)
        return response["answer"]
