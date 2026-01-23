"""Document ingestion and processing module."""
import os
from pathlib import Path
from typing import List, Dict
import pypdf
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.logger import logger


class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: Path) -> str:
        """
        Load and extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Loading PDF: {file_path}")
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            logger.info(f"Successfully loaded PDF: {file_path} ({len(text)} characters)")
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
        return text
    
    def load_txt(self, file_path: Path) -> str:
        """
        Load text from TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Text content
        """
        logger.info(f"Loading TXT: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            logger.info(f"Successfully loaded TXT: {file_path} ({len(text)} characters)")
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise
        return text
    
    def load_md(self, file_path: Path) -> str:
        """
        Load and convert Markdown to text.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Text content
        """
        logger.info(f"Loading MD: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                md_content = file.read()
            # Keep markdown as text (could be converted to HTML if needed)
            text = md_content
            logger.info(f"Successfully loaded MD: {file_path} ({len(text)} characters)")
        except Exception as e:
            logger.error(f"Error loading MD {file_path}: {e}")
            raise
        return text
    
    def load_document(self, file_path: Path) -> str:
        """
        Load document based on file extension.
        
        Args:
            file_path: Path to document
            
        Returns:
            Text content
            
        Raises:
            ValueError: If file type is not supported
        """
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            return self.load_pdf(file_path)
        elif extension == ".txt":
            return self.load_txt(file_path)
        elif extension in [".md", ".markdown"]:
            return self.load_md(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of dictionaries with text and metadata
        """
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "text": chunk,
                "metadata": {
                    **(metadata or {}),
                    "chunk_id": i,
                }
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def process_document(self, file_path: Path) -> List[Dict]:
        """
        Process a document: load and chunk.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Processing document: {file_path}")
        
        # Load document
        text = self.load_document(file_path)
        
        # Create metadata
        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower(),
        }
        
        # Chunk document
        chunks = self.chunk_text(text, metadata)
        
        logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        return chunks
    
    def process_directory(self, directory: Path) -> List[Dict]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of all text chunks from all documents
        """
        logger.info(f"Processing directory: {directory}")
        
        supported_extensions = {".pdf", ".txt", ".md", ".markdown"}
        all_chunks = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Processed directory: {len(all_chunks)} total chunks from {directory}")
        return all_chunks
