"""Streamlit UI for RAG chatbot."""
import streamlit as st
from pathlib import Path
import tempfile
from src.core.rag_engine import RAGEngine
from src.config import settings
from src.logger import logger

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag_engine():
    """Initialize and cache RAG engine."""
    logger.info("Initializing RAG engine for Streamlit")
    engine = RAGEngine(llm_backend="ollama")
    
    # Try to load existing index
    try:
        engine.load_index()
        logger.info("Loaded existing vector store")
    except Exception as e:
        logger.warning(f"No existing vector store: {e}")
    
    return engine


def main():
    """Main Streamlit application."""
    st.title("🤖 Multilingual RAG Chatbot")
    st.markdown("*Ask questions about your documents in English, Spanish, or Portuguese*")
    
    # Sidebar for configuration and document upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        language = st.selectbox(
            "Language / Idioma / Língua",
            options=["en", "es", "pt"],
            format_func=lambda x: {
                "en": "🇬🇧 English",
                "es": "🇪🇸 Español",
                "pt": "🇧🇷 Português"
            }[x]
        )
        
        st.divider()
        
        # Document upload section
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                process_documents(uploaded_files)
            else:
                st.warning("Please upload at least one document")
        
        st.divider()
        
        # Index management
        st.header("🗂️ Index Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Index"):
                save_index()
        with col2:
            if st.button("Clear Index"):
                clear_index()
        
        st.divider()
        
        # System info
        st.header("ℹ️ System Info")
        st.info(f"""
        **Backend:** Ollama  
        **Vector Store:** {settings.vector_store_type.upper()}  
        **Model:** {settings.ollama_model}  
        **Embeddings:** {settings.embedding_model.split('/')[-1]}
        """)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['metadata']['source']}")
                        st.text(source["text"])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_engine = initialize_rag_engine()
                    response = rag_engine.answer_question(
                        question=prompt,
                        language=language,
                        include_sources=True
                    )
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response.get("sources"):
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:** {source['metadata']['source']}")
                                st.text(source["text"])
                                st.markdown("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error in chat: {e}")


def process_documents(uploaded_files):
    """Process uploaded documents."""
    try:
        rag_engine = initialize_rag_engine()
        
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            total_chunks = 0
            
            for uploaded_file in uploaded_files:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = Path(tmp_file.name)
                
                # Process document
                try:
                    chunks = rag_engine.ingest_documents(tmp_path)
                    total_chunks += chunks
                    logger.info(f"Processed {uploaded_file.name}: {chunks} chunks")
                finally:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)
            
            # Save index
            rag_engine.save_index()
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s) ({total_chunks} chunks)")
    
    except Exception as e:
        st.error(f"❌ Error processing documents: {e}")
        logger.error(f"Error processing documents: {e}")


def save_index():
    """Save the vector store index."""
    try:
        rag_engine = initialize_rag_engine()
        rag_engine.save_index()
        st.success("✅ Index saved successfully")
    except Exception as e:
        st.error(f"❌ Error saving index: {e}")
        logger.error(f"Error saving index: {e}")


def clear_index():
    """Clear the vector store index."""
    try:
        # Clear the cache to reinitialize engine
        st.cache_resource.clear()
        st.success("✅ Index cleared successfully. Please refresh the page.")
    except Exception as e:
        st.error(f"❌ Error clearing index: {e}")
        logger.error(f"Error clearing index: {e}")


if __name__ == "__main__":
    main()
