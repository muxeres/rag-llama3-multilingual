"""Configuration module for RAG chatbot."""
import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # Hugging Face Configuration
    hf_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token: str = ""
    
    # Vector Store Configuration
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store_path: str = "./vector_store"
    
    # Application Configuration
    default_language: Literal["en", "es", "pt"] = "en"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    temperature: float = 0.7
    max_tokens: int = 512
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    def get_vector_store_path(self) -> Path:
        """Get vector store path as Path object."""
        return Path(self.vector_store_path)


# Global settings instance
settings = Settings()
