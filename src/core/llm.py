"""LLM integration module supporting Ollama and Hugging Face Transformers."""
from typing import Optional, Dict
from abc import ABC, abstractmethod
from src.config import settings
from src.logger import logger


class LLMBase(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate response from LLM."""
        pass


class OllamaLLM(LLMBase):
    """Ollama-based LLM implementation."""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
        """
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is required for OllamaLLM")
        
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = ollama.Client(host=self.base_url)
        logger.info(f"Initialized Ollama LLM: {self.model_name} at {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"Generating response with Ollama ({self.model_name})")
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            answer = response["message"]["content"]
            logger.info(f"Generated response: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise


class TransformersLLM(LLMBase):
    """Hugging Face Transformers-based LLM implementation."""
    
    def __init__(self, model_name: str = None, token: str = None):
        """
        Initialize Transformers LLM.
        
        Args:
            model_name: Name of the HuggingFace model
            token: HuggingFace API token
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for TransformersLLM")
        
        self.model_name = model_name or settings.hf_model_name
        self.token = token or settings.hf_token or None
        
        logger.info(f"Loading Transformers model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        logger.info(f"Initialized Transformers LLM: {self.model_name} on {self.device}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate response using Transformers.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        import torch
        
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        # Format prompt (Llama 3 chat format)
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        try:
            logger.info(f"Generating response with Transformers ({self.model_name})")
            
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                answer = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                answer = full_response[len(full_prompt):].strip()
            
            logger.info(f"Generated response: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response with Transformers: {e}")
            raise


def create_llm(backend: str = "ollama") -> LLMBase:
    """
    Factory function to create LLM instance.
    
    Args:
        backend: LLM backend to use ('ollama' or 'transformers')
        
    Returns:
        LLM instance
    """
    if backend == "ollama":
        return OllamaLLM()
    elif backend == "transformers":
        return TransformersLLM()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
