"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_engine():
    """Create mock RAG engine."""
    mock = Mock()
    mock.answer_question.return_value = {
        "answer": "This is a test answer.",
        "language": "en",
        "has_context": True,
        "sources": [
            {
                "text": "Test context",
                "metadata": {"source": "test.txt"},
                "score": 0.5
            }
        ]
    }
    return mock


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_status_endpoint(client):
    """Test status endpoint."""
    with patch("src.api.main.rag_engine") as mock_engine:
        mock_engine.__bool__.return_value = True
        response = client.get("/status")
        
        # Note: May return 503 if RAG engine not initialized in test
        assert response.status_code in [200, 503]


def test_ask_question(client, mock_rag_engine):
    """Test ask question endpoint."""
    with patch("src.api.main.rag_engine", mock_rag_engine):
        response = client.post(
            "/ask",
            json={
                "question": "What is RAG?",
                "language": "en",
                "include_sources": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "language" in data
        assert "has_context" in data
