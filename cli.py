"""CLI tool for managing the RAG chatbot."""
import argparse
from pathlib import Path
from src.core.rag_engine import RAGEngine
from src.config import settings
from src.logger import logger


def ingest_command(args):
    """Handle document ingestion."""
    print("🔄 Initializing RAG engine...")
    engine = RAGEngine(llm_backend=args.backend)
    
    # Load existing index if available
    try:
        engine.load_index()
        print("✅ Loaded existing index")
    except:
        print("📝 Creating new index")
    
    # Ingest documents
    path = Path(args.path)
    print(f"📚 Ingesting documents from: {path}")
    chunks = engine.ingest_documents(path)
    print(f"✅ Ingested {chunks} chunks")
    
    # Save index
    print("💾 Saving index...")
    engine.save_index()
    print("✅ Index saved successfully")


def query_command(args):
    """Handle query."""
    print("🔄 Initializing RAG engine...")
    engine = RAGEngine(llm_backend=args.backend)
    
    # Load index
    try:
        engine.load_index()
        print("✅ Loaded index\n")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("Please ingest documents first using: python cli.py ingest <path>")
        return
    
    # Query
    print(f"❓ Question: {args.question}\n")
    response = engine.answer_question(args.question, language=args.language)
    
    print(f"💬 Answer ({response['language']}):")
    print(response['answer'])
    
    if response.get('sources'):
        print("\n📚 Sources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n{i}. {source['metadata']['source']}")
            print(f"   {source['text'][:150]}...")


def chat_command(args):
    """Handle interactive chat."""
    print("🔄 Initializing RAG engine...")
    engine = RAGEngine(llm_backend=args.backend)
    
    # Load index
    try:
        engine.load_index()
        print("✅ Loaded index")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("Please ingest documents first using: python cli.py ingest <path>")
        return
    
    print(f"\n💬 Interactive Chat Mode (Language: {args.language})")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...\n")
            response = engine.answer_question(question, language=args.language)
            print(f"Assistant: {response['answer']}\n")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--backend',
        choices=['ollama', 'transformers'],
        default='ollama',
        help='LLM backend to use'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('path', help='Path to document or directory')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a single question')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument(
        '--language', '-l',
        choices=['en', 'es', 'pt'],
        default='en',
        help='Response language'
    )
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument(
        '--language', '-l',
        choices=['en', 'es', 'pt'],
        default='en',
        help='Response language'
    )
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        ingest_command(args)
    elif args.command == 'query':
        query_command(args)
    elif args.command == 'chat':
        chat_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
