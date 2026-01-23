"""Tests for configuration module."""
import pytest
from src.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3"
    assert settings.vector_store_type in ["faiss", "chroma"]
    assert settings.default_language in ["en", "es", "pt"]
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.top_k_results > 0


def test_get_vector_store_path():
    """Test vector store path method."""
    settings = Settings()
    path = settings.get_vector_store_path()
    
    assert path is not None
    assert hasattr(path, 'exists')  # Path object
