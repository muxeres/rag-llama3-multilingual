"""FastAPI backend for RAG chatbot."""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from pathlib import Path
import aiofiles
import tempfile
import shutil
from src.core.rag_engine import RAGEngine
from src.config import settings
from src.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Multilingual RAG chatbot API with Llama 3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine: Optional[RAGEngine] = None


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="Question to ask")
    language: Optional[str] = Field("en", description="Language code (en, es, pt)")
    include_sources: bool = Field(True, description="Include source documents")


class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str
    language: str
    has_context: bool
    sources: Optional[List[Dict]] = None


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    chunks_processed: int
    message: str


class StatusResponse(BaseModel):
    """Response model for status check."""
    status: str
    backend: str
    vector_store: str
    language: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup."""
    global rag_engine
    try:
        logger.info("Starting FastAPI application")
        rag_engine = RAGEngine(llm_backend="ollama")
        
        # Try to load existing index
        try:
            rag_engine.load_index()
            logger.info("Loaded existing vector store index")
        except Exception as e:
            logger.warning(f"No existing index found: {e}")
        
        logger.info("FastAPI application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
    """Get API status."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return StatusResponse(
        status="ready",
        backend="ollama",
        vector_store=settings.vector_store_type,
        language=settings.default_language
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer with citations.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        logger.info(f"Received question: {request.question[:100]}")
        response = rag_engine.answer_question(
            question=request.question,
            language=request.language,
            include_sources=request.include_sources
        )
        return AnswerResponse(**response)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Documents"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a single document (PDF, TXT, or MD).
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".pdf", ".txt", ".md", ".markdown"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .txt, .md"
        )
    
    try:
        # Save uploaded file to temporary location
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Ingest document
        chunks_processed = rag_engine.ingest_documents(tmp_path)
        
        # Save updated index
        rag_engine.save_index()
        
        # Clean up temporary file
        tmp_path.unlink()
        
        return IngestResponse(
            success=True,
            chunks_processed=chunks_processed,
            message=f"Successfully ingested {file.filename}"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        # Clean up on error
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Documents"])
async def ingest_directory(directory_path: str = Form(...)):
    """
    Ingest all documents from a directory on the server.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    path = Path(directory_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")
    
    try:
        logger.info(f"Processing directory: {directory_path}")
        chunks_processed = rag_engine.ingest_documents(path)
        
        # Save updated index
        rag_engine.save_index()
        
        return IngestResponse(
            success=True,
            chunks_processed=chunks_processed,
            message=f"Successfully ingested documents from {directory_path}"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["Index"])
async def clear_index():
    """
    Clear the vector store index.
    """
    global rag_engine
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        logger.info("Clearing vector store index")
        # Reinitialize RAG engine to clear the index
        rag_engine = RAGEngine(llm_backend="ollama")
        return {"message": "Index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
