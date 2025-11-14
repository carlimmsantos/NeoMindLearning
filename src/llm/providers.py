"""LLM provider implementations and comparison utilities."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import time
import logging
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    response_time: float = 0.0
    cost: Optional[float] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[LLMResponse]:
        """Generate responses for a batch of prompts."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000, temperature: float = 0.7):
        """Initialize OpenAI provider."""
        self.model = model
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from OpenAI."""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                provider="openai",
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[LLMResponse]:
        """Generate responses for a batch of prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        return responses


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro", max_tokens: int = 1000, temperature: float = 0.7):
        """Initialize Gemini provider."""
        self.model = model
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from Gemini."""
        start_time = time.time()
        
        # Gemini handles system prompts differently
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                provider="gemini",
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[LLMResponse]:
        """Generate responses for a batch of prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        return responses


def create_llm_providers(
    openai_key: str, 
    gemini_key: str, 
    openai_model: str = "gpt-3.5-turbo", 
    gemini_model: str = "gemini-pro"
) -> Dict[str, BaseLLMProvider]:
    """Factory function to create LLM providers."""
    providers = {}
    
    if openai_key:
        providers["openai"] = OpenAIProvider(openai_key, openai_model)
    
    if gemini_key:
        providers["gemini"] = GeminiProvider(gemini_key, gemini_model)
    
    return providers
