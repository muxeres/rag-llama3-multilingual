"""Tests for prompt templates."""
import pytest
from src.core.prompts import PromptTemplates


def test_get_system_prompt():
    """Test getting system prompts."""
    # Test English
    prompt_en = PromptTemplates.get_system_prompt("en")
    assert "helpful" in prompt_en.lower()
    assert "cite" in prompt_en.lower() or "source" in prompt_en.lower()
    
    # Test Spanish
    prompt_es = PromptTemplates.get_system_prompt("es")
    assert len(prompt_es) > 0
    
    # Test Portuguese
    prompt_pt = PromptTemplates.get_system_prompt("pt")
    assert len(prompt_pt) > 0


def test_get_query_prompt():
    """Test getting query prompt templates."""
    for lang in ["en", "es", "pt"]:
        template = PromptTemplates.get_query_prompt(lang)
        assert "{context}" in template
        assert "{question}" in template


def test_format_prompt():
    """Test formatting prompts with context and question."""
    question = "What is RAG?"
    context = "RAG stands for Retrieval Augmented Generation"
    
    for lang in ["en", "es", "pt"]:
        formatted = PromptTemplates.format_prompt(question, context, lang)
        assert question in formatted
        assert context in formatted


def test_get_no_context_response():
    """Test no context responses."""
    for lang in ["en", "es", "pt"]:
        response = PromptTemplates.get_no_context_response(lang)
        assert len(response) > 0
