"""Tests for vector store."""
import pytest
from pathlib import Path
import tempfile
import shutil
from src.core.vector_store import FAISSVectorStore


@pytest.fixture
def vector_store():
    """Create FAISS vector store instance."""
    return FAISSVectorStore()


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        {
            "text": "Python is a high-level programming language.",
            "metadata": {"source": "doc1.txt", "chunk_id": 0}
        },
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "doc2.txt", "chunk_id": 0}
        },
        {
            "text": "Natural language processing deals with text and language.",
            "metadata": {"source": "doc3.txt", "chunk_id": 0}
        }
    ]


def test_add_documents(vector_store, sample_chunks):
    """Test adding documents to vector store."""
    vector_store.add_documents(sample_chunks)
    
    assert len(vector_store.documents) == len(sample_chunks)


def test_search(vector_store, sample_chunks):
    """Test searching in vector store."""
    vector_store.add_documents(sample_chunks)
    
    results = vector_store.search("What is Python?", k=2)
    
    assert len(results) > 0
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)


def test_search_empty_store(vector_store):
    """Test searching in empty vector store."""
    results = vector_store.search("test query")
    
    assert len(results) == 0


def test_save_and_load(vector_store, sample_chunks):
    """Test saving and loading vector store."""
    # Add documents
    vector_store.add_documents(sample_chunks)
    
    # Create temporary directory for saving
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save
        vector_store.save(temp_dir)
        
        # Verify files exist
        assert (temp_dir / "faiss.index").exists()
        assert (temp_dir / "documents.pkl").exists()
        assert (temp_dir / "metadata.json").exists()
        
        # Load into new instance
        new_store = FAISSVectorStore()
        new_store.load(temp_dir)
        
        # Verify loaded data
        assert len(new_store.documents) == len(sample_chunks)
        
        # Test search on loaded store
        results = new_store.search("Python programming", k=1)
        assert len(results) > 0
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
