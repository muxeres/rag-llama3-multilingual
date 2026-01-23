"""
Example script demonstrating RAG chatbot usage.

This script shows how to use the RAG engine programmatically.
Note: Requires Ollama running with llama3 model OR embedding models downloaded.
"""

from pathlib import Path
from src.core.rag_engine import RAGEngine
from src.logger import logger

def main():
    """Demonstrate RAG engine usage."""
    print("=" * 60)
    print("RAG Chatbot Example")
    print("=" * 60)
    
    # Initialize RAG engine
    print("\n1. Initializing RAG engine...")
    try:
        engine = RAGEngine(llm_backend="ollama", language="en")
        print("✓ RAG engine initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nNote: This requires:")
        print("  - Ollama running with llama3 model")
        print("  - Embedding model downloaded")
        return
    
    # Ingest sample documents
    print("\n2. Ingesting sample documents...")
    docs_path = Path("data/sample_docs")
    
    if not docs_path.exists():
        print(f"✗ Sample documents not found at {docs_path}")
        return
    
    try:
        num_chunks = engine.ingest_documents(docs_path)
        print(f"✓ Ingested {num_chunks} chunks from documents")
    except Exception as e:
        print(f"✗ Failed to ingest documents: {e}")
        return
    
    # Save the index
    print("\n3. Saving vector store index...")
    try:
        engine.save_index()
        print("✓ Index saved")
    except Exception as e:
        print(f"✗ Failed to save index: {e}")
    
    # Example queries
    questions = [
        ("What is RAG?", "en"),
        ("What are Python best practices?", "en"),
        ("Tell me about Llama 3", "en"),
    ]
    
    print("\n4. Testing queries:")
    print("-" * 60)
    
    for question, lang in questions:
        print(f"\nQ: {question}")
        try:
            response = engine.answer_question(question, language=lang)
            print(f"A: {response['answer'][:200]}...")
            if response.get('sources'):
                print(f"   Sources: {len(response['sources'])} documents")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
