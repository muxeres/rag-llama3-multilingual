"""Tests for document processor."""
import pytest
from pathlib import Path
import tempfile
from src.core.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Create document processor instance."""
    return DocumentProcessor()


@pytest.fixture
def temp_txt_file():
    """Create temporary text file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\n")
        f.write("It has multiple lines.\n")
        f.write("This is for testing purposes.")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_md_file():
    """Create temporary markdown file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\n")
        f.write("This is a **test** markdown file.\n\n")
        f.write("## Section 1\n\n")
        f.write("Some content here.")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


def test_load_txt(processor, temp_txt_file):
    """Test loading text file."""
    text = processor.load_txt(temp_txt_file)
    
    assert len(text) > 0
    assert "test document" in text.lower()


def test_load_md(processor, temp_md_file):
    """Test loading markdown file."""
    text = processor.load_md(temp_md_file)
    
    assert len(text) > 0
    assert "Test Document" in text


def test_chunk_text(processor):
    """Test text chunking."""
    text = "This is a test. " * 100  # Long text
    chunks = processor.chunk_text(text)
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_process_document(processor, temp_txt_file):
    """Test processing a complete document."""
    chunks = processor.process_document(temp_txt_file)
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    assert all("source" in chunk["metadata"] for chunk in chunks)


def test_unsupported_file_type(processor):
    """Test handling unsupported file types."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError):
            processor.load_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
