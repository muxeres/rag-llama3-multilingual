# 🤖 RAG Llama 3 Multilingual Chatbot

A production-ready multilingual (EN/ES/PT) RAG (Retrieval-Augmented Generation) chatbot powered by Llama 3. Ask questions about your documents in English, Spanish, or Portuguese with cited sources.

## ✨ Features

- **🌍 Multilingual Support**: Ask questions in English, Spanish, or Portuguese
- **📚 Multiple Document Formats**: Support for PDF, TXT, and Markdown files
- **🔍 Vector Search**: FAISS and ChromaDB support for fast similarity search
- **📖 Citations**: Responses include source references for transparency
- **🎯 Two LLM Backends**: 
  - Ollama (recommended, easy setup)
  - Hugging Face Transformers (for custom deployments)
- **🖥️ Multiple Interfaces**:
  - Streamlit UI for interactive chat
  - FastAPI REST API for integration
  - CLI tool for scripting
- **🐳 Docker Support**: Complete Docker Compose setup
- **✅ Production Ready**: Logging, error handling, and comprehensive tests

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ├─── Streamlit UI (Port 8501)
       ├─── FastAPI API (Port 8000)
       └─── CLI Tool
              │
       ┌──────▼──────────┐
       │  RAG Engine     │
       │  - Prompts      │
       │  - Retrieval    │
       └──────┬──────────┘
              │
       ┌──────┼──────────┐
       │      │          │
   ┌───▼───┐ ┌▼────┐ ┌──▼────┐
   │ Docs  │ │Vector│ │ LLM   │
   │Process│ │Store │ │Ollama/│
   │       │ │FAISS/│ │Trans. │
   │       │ │Chroma│ │       │
   └───────┘ └──────┘ └───────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running (for Ollama backend)
- Docker & Docker Compose (for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/muxeres/rag-llama3-multilingual.git
cd rag-llama3-multilingual
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Pull Llama 3 model** (if using Ollama)
```bash
ollama pull llama3
```

### 🐳 Docker Quick Start

The easiest way to run the entire stack:

```bash
# Start all services (Ollama, API, UI)
docker-compose up -d

# Pull Llama 3 model into Ollama container
docker exec -it rag-ollama ollama pull llama3

# Access the UI at http://localhost:8501
# Access the API at http://localhost:8000/docs
```

## 💻 Usage

### 1. Streamlit UI (Recommended for beginners)

```bash
streamlit run src/ui/streamlit_app.py
```

Visit `http://localhost:8501` to:
- Upload documents (PDF, TXT, MD)
- Switch between languages (EN/ES/PT)
- Chat with your documents
- View source citations

### 2. FastAPI Backend

```bash
python -m uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

**Example API calls:**

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is the main topic?",
        "language": "en",
        "include_sources": True
    }
)
print(response.json())

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ingest/file",
        files={"file": f}
    )
```

### 3. CLI Tool

```bash
# Ingest documents
python cli.py ingest ./data/documents/

# Ask a single question
python cli.py query "What is RAG?" --language en

# Interactive chat mode
python cli.py chat --language es
```

## 📁 Project Structure

```
rag-llama3-multilingual/
├── src/
│   ├── core/                   # Core RAG components
│   │   ├── document_processor.py   # PDF/TXT/MD ingestion
│   │   ├── vector_store.py         # FAISS/Chroma implementations
│   │   ├── llm.py                  # Ollama/Transformers LLM
│   │   ├── prompts.py              # Multilingual prompts
│   │   └── rag_engine.py           # Main RAG orchestration
│   ├── api/                    # FastAPI backend
│   │   └── main.py
│   └── ui/                     # Streamlit frontend
│       └── streamlit_app.py
├── tests/                      # Comprehensive tests
├── data/                       # Your documents (gitignored)
├── vector_store/               # Vector index (gitignored)
├── logs/                       # Application logs
├── cli.py                      # CLI tool
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚙️ Configuration

Edit `.env` file to customize:

```bash
# LLM Backend
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Vector Store (faiss or chroma)
VECTOR_STORE_TYPE=faiss
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Retrieval Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3

# Generation Settings
TEMPERATURE=0.7
MAX_TOKENS=512
DEFAULT_LANGUAGE=en
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_document_processor.py -v
```

## 🌐 Multilingual Prompts

The system includes carefully crafted prompts for each language:

- **English**: Clear, professional responses with citations
- **Spanish**: Natural Spanish phrasing and terminology
- **Portuguese**: Brazilian Portuguese conventions

All prompts emphasize:
- Answering based on provided context
- Including source citations
- Being honest when information is not available

## 🔧 Advanced Usage

### Using Transformers Backend

If you prefer to run models locally without Ollama:

```python
from src.core.rag_engine import RAGEngine

engine = RAGEngine(llm_backend="transformers")
# Note: Requires HF_TOKEN in .env for gated models
```

### Using ChromaDB Instead of FAISS

In `.env`:
```bash
VECTOR_STORE_TYPE=chroma
```

### Custom Embedding Models

Any sentence-transformers model can be used:
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 📊 Example Workflows

### Workflow 1: Research Assistant

1. Upload research papers (PDFs)
2. Ask questions in your preferred language
3. Get answers with paper citations
4. Export conversation for documentation

### Workflow 2: Documentation Search

1. Ingest technical documentation (MD files)
2. Use CLI for quick queries
3. Integrate API into internal tools
4. Team members access via Streamlit

### Workflow 3: Customer Support

1. Upload product manuals and FAQs
2. Support agents use Streamlit UI
3. Multiple language support for international customers
4. Audit trail via logging

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
# On macOS: restart Ollama app
# On Linux: systemctl restart ollama
```

### Memory Issues with Large PDFs
Adjust chunk size in `.env`:
```bash
CHUNK_SIZE=300  # Smaller chunks
```

### Model Download Errors
```bash
# Manually pull the model
ollama pull llama3

# Or use a smaller model
OLLAMA_MODEL=llama3.1
```

## 📚 API Documentation

FastAPI automatically generates interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- **Meta** for Llama 3
- **Ollama** for easy LLM deployment
- **LangChain** for RAG components
- **FAISS** for efficient vector search
- **Sentence Transformers** for embeddings

## 📞 Support

- 📖 [Documentation](https://github.com/muxeres/rag-llama3-multilingual/wiki)
- 🐛 [Issue Tracker](https://github.com/muxeres/rag-llama3-multilingual/issues)
- 💬 [Discussions](https://github.com/muxeres/rag-llama3-multilingual/discussions)

## 🗺️ Roadmap

- [ ] Add more languages (FR, DE, IT)
- [ ] Support for more document formats (DOCX, HTML)
- [ ] Conversation memory and context
- [ ] Advanced filtering and metadata search
- [ ] Integration with cloud storage (S3, GCS)
- [ ] Fine-tuning support for domain-specific models
- [ ] Web scraping capabilities
- [ ] Audio/video transcription support

---

**Made with ❤️ for the open-source community**
