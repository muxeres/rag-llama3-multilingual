"""Vector store module for document embeddings and retrieval."""
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle
from src.config import settings
from src.logger import logger


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = None) -> List[Dict]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load vector store from disk."""
        pass


class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store implementation."""
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Dict] = []
        logger.info(f"Initialized FAISS vector store (dimension: {self.dimension})")
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to FAISS index.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        logger.info(f"Adding {len(chunks)} documents to FAISS index")
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store document metadata
        self.documents.extend(chunks)
        
        logger.info(f"Added {len(chunks)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Search for similar documents using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata and scores
        """
        k = k or settings.top_k_results
        
        if len(self.documents) == 0:
            logger.warning("No documents in index")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        
        # Search in FAISS
        k = min(k, len(self.documents))
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                result = {
                    **self.documents[idx],
                    "score": float(distances[0][i]),
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self, path: Path) -> None:
        """
        Save FAISS index and documents to disk.
        
        Args:
            path: Directory path to save to
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "embedding_model": self.embedding_model_name,
                "dimension": self.dimension,
                "num_documents": len(self.documents),
            }, f, indent=2)
        
        logger.info(f"Saved FAISS vector store to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load FAISS index and documents from disk.
        
        Args:
            path: Directory path to load from
        """
        logger.info(f"Loading FAISS vector store from {path}")
        
        # Load FAISS index
        index_path = path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load documents
        docs_path = path / "documents.pkl"
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        
        # Load and verify metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded FAISS index: {metadata['num_documents']} documents")


class ChromaVectorStore(VectorStoreBase):
    """Chroma-based vector store implementation."""
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize Chroma vector store.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb is required for ChromaVectorStore")
        
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        # Initialize Chroma client
        self.client = None
        self.collection = None
        logger.info("Initialized Chroma vector store")
    
    def _init_collection(self, path: Path):
        """Initialize Chroma collection."""
        import chromadb
        
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to Chroma.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        logger.info(f"Adding {len(chunks)} documents to Chroma")
        
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call load() or save() first.")
        
        # Extract texts and prepare data
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to Chroma
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} documents to Chroma")
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Search for similar documents using Chroma.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata and scores
        """
        k = k or settings.top_k_results
        
        if self.collection is None:
            raise RuntimeError("Collection not initialized")
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        
        # Search in Chroma
        results_data = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Format results
        results = []
        if results_data["documents"]:
            for i in range(len(results_data["documents"][0])):
                result = {
                    "text": results_data["documents"][0][i],
                    "metadata": results_data["metadatas"][0][i],
                    "score": results_data["distances"][0][i] if results_data["distances"] else 0.0,
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self, path: Path) -> None:
        """
        Save Chroma collection to disk.
        
        Args:
            path: Directory path to save to
        """
        path.mkdir(parents=True, exist_ok=True)
        
        if self.collection is None:
            self._init_collection(path)
        
        logger.info(f"Chroma collection persisted to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load Chroma collection from disk.
        
        Args:
            path: Directory path to load from
        """
        logger.info(f"Loading Chroma collection from {path}")
        self._init_collection(path)
        logger.info(f"Loaded Chroma collection")


def create_vector_store(store_type: str = None) -> VectorStoreBase:
    """
    Factory function to create vector store.
    
    Args:
        store_type: Type of vector store ('faiss' or 'chroma')
        
    Returns:
        VectorStore instance
    """
    store_type = store_type or settings.vector_store_type
    
    if store_type == "faiss":
        return FAISSVectorStore()
    elif store_type == "chroma":
        return ChromaVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
