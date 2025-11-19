"""LLM provider implementations and comparison utilities."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from .cache import LLMCache

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
    def get_llm(self):
        """Return the underlying LLM instance."""
        pass
    
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
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000, temperature: float = 0.7, use_cache: bool = True):
        """Initialize OpenAI provider."""
        self.model = model
        
        # Initialize LangChain ChatOpenAI wrapper with specified configuration
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Conditionally initialize response cache for cost optimization
        self.cache = LLMCache() if use_cache else None 
        logger.info(f"OpenAIProvider initialized with model {model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from OpenAI."""
        
        if self.cache:
            cached = self.cache.get(prompt, self.model)
            if cached:
                logger.info("Cache HIT - API call avoided!")
                return LLMResponse(
                    content=cached,
                    model=self.model,
                    provider="openai",
                    response_time=0.0
                )
        
        # Measure API call execution time
        start_time = time.time()
        
        # Construct message structure with optional system prompt
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            # Invoke LLM with message structure
            response = self.llm.invoke(messages)
            response_time = time.time() - start_time

            # Cache successful response for future queries
            if self.cache:
                self.cache.set(prompt, self.model, response.content)
                logger.info("Response cached for future use")
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                provider="openai",
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None, batch_size: int = 5) -> List[LLMResponse]:
        """Generate responses for a batch of prompts."""
        responses = []
        
        # Process prompts in configured batch sizes
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Combine multiple prompts into single request
            combined_prompt = "Process the following items and return results as a JSON array:\n\n"
            for j, prompt in enumerate(batch, 1):
                combined_prompt += f"{j}. {prompt}\n"
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} items...")
            
            # Measure batch execution time
            start_time = time.time()
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=combined_prompt))
            
            try:
                # Invoke LLM with combined batch prompt
                response = self.llm.invoke(messages)
                response_time = time.time() - start_time
                
                responses.append(LLMResponse(
                    content=response.content,
                    model=self.model,
                    provider="openai",
                    response_time=response_time
                ))
            except Exception as e:
                logger.error(f"OpenAI API error in batch: {e}")
                raise
        
        logger.info(f"Batch processing complete: {len(responses)} batches processed")
        return responses

    def get_llm(self):
        """Return the ChatOpenAI instance for use in agents."""
        return self.llm


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro", max_tokens: int = 1000, temperature: float = 0.7):
        """Initialize Gemini provider."""
        self.model = model
        
        # Initialize LangChain ChatGoogleGenerativeAI wrapper
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.info(f"GeminiProvider initialized with model {model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from Google Gemini.
        
        Note: Gemini integrates system prompts differently than OpenAI,
        combining them with the main prompt in the message.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system instruction context
            
        Returns:
            LLMResponse containing generated text and performance metrics
            
        Raises:
            Exception: If Gemini API call fails
        """
        # Measure API call execution time
        start_time = time.time()
        
        # Combine system and user prompts for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            # Invoke Gemini with combined prompt
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
        
        # Process each prompt individually
        for prompt in prompts:
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        
        logger.info(f"Batch processing complete: {len(responses)} prompts processed")
        return responses

    def get_llm(self):
        """Return the ChatGoogleGenerativeAI instance for use in agents."""
        return self.llm


def create_llm_providers(
    openai_key: str, 
    gemini_key: str, 
    openai_model: str = "gpt-3.5-turbo", 
    gemini_model: str = "gemini-pro"
) -> Dict[str, BaseLLMProvider]:
    """Factory function to create LLM providers."""
    providers = {}
    
    # Initialize OpenAI provider if API key is available
    if openai_key:
        providers["openai"] = OpenAIProvider(openai_key, openai_model)
        logger.info("OpenAI provider registered")
    
    # Initialize Gemini provider if API key is available
    if gemini_key:
        providers["gemini"] = GeminiProvider(gemini_key, gemini_model)
        logger.info("Gemini provider registered")
    
    if not providers:
        logger.warning("No LLM providers configured. Provide at least one API key.")
    
    return providers
