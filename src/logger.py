"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path
from src.config import settings


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("rag_chatbot")
    return logger


# Global logger instance
logger = setup_logging()
