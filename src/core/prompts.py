"""Multilingual prompt templates for RAG chatbot."""
from typing import Dict


class PromptTemplates:
    """Multilingual prompt templates for different languages."""
    
    SYSTEM_PROMPTS: Dict[str, str] = {
        "en": """You are a helpful AI assistant that answers questions based on the provided context. 
Always cite the source of your information using [Source: filename] format.
If you cannot find the answer in the context, say so clearly.
Be concise and accurate.""",
        
        "es": """Eres un asistente de IA útil que responde preguntas basándose en el contexto proporcionado.
Siempre cita la fuente de tu información usando el formato [Fuente: nombre_archivo].
Si no puedes encontrar la respuesta en el contexto, dilo claramente.
Sé conciso y preciso.""",
        
        "pt": """Você é um assistente de IA útil que responde perguntas com base no contexto fornecido.
Sempre cite a fonte de suas informações usando o formato [Fonte: nome_arquivo].
Se você não conseguir encontrar a resposta no contexto, diga isso claramente.
Seja conciso e preciso."""
    }
    
    QUERY_PROMPTS: Dict[str, str] = {
        "en": """Context information from relevant documents:
{context}

Question: {question}

Answer based on the context above. Include citations using [Source: filename] format:""",
        
        "es": """Información de contexto de documentos relevantes:
{context}

Pregunta: {question}

Responde basándote en el contexto anterior. Incluye citas usando el formato [Fuente: nombre_archivo]:""",
        
        "pt": """Informação de contexto de documentos relevantes:
{context}

Pergunta: {question}

Responda com base no contexto acima. Inclua citações usando o formato [Fonte: nome_arquivo]:"""
    }
    
    NO_CONTEXT_RESPONSES: Dict[str, str] = {
        "en": "I don't have enough information in the provided documents to answer this question.",
        "es": "No tengo suficiente información en los documentos proporcionados para responder esta pregunta.",
        "pt": "Não tenho informações suficientes nos documentos fornecidos para responder a esta pergunta."
    }
    
    @classmethod
    def get_system_prompt(cls, language: str = "en") -> str:
        """Get system prompt for specified language."""
        return cls.SYSTEM_PROMPTS.get(language, cls.SYSTEM_PROMPTS["en"])
    
    @classmethod
    def get_query_prompt(cls, language: str = "en") -> str:
        """Get query prompt template for specified language."""
        return cls.QUERY_PROMPTS.get(language, cls.QUERY_PROMPTS["en"])
    
    @classmethod
    def get_no_context_response(cls, language: str = "en") -> str:
        """Get no context response for specified language."""
        return cls.NO_CONTEXT_RESPONSES.get(language, cls.NO_CONTEXT_RESPONSES["en"])
    
    @classmethod
    def format_prompt(cls, question: str, context: str, language: str = "en") -> str:
        """
        Format a complete prompt with question and context.
        
        Args:
            question: User question
            context: Retrieved context from documents
            language: Language code (en, es, pt)
            
        Returns:
            Formatted prompt string
        """
        template = cls.get_query_prompt(language)
        return template.format(context=context, question=question)
