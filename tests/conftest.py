import tempfile
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, provider_name: str = "mock"):
        self.provider_name = provider_name
    
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Generate a mock response."""
        return LLMResponse(
            content=f"Mock response to: {prompt[:50]}...",
            model="mock-model",
            provider=self.provider_name,
            response_time=0.1
        )
    
    def generate_batch(self, prompts: list, system_prompt: str = None) -> list:
        """Generate mock responses for batch."""
        return [self.generate(prompt, system_prompt) for prompt in prompts]


@pytest.fixture
def mock_llm_providers():
    """Fixture providing mock LLM providers."""
    return {
        "openai": MockLLMProvider("openai"),
        "gemini": MockLLMProvider("gemini")
    }

@pytest.fixture
def sample_comment_data():
    """Fixture providing sample comment data."""
    
    data = {
        "Review Title": [
            "Great product! Highly recommend.",
            "Poor quality, not worth the money.",
            "Average product, does the job.",
            "Excellent customer service!",
            "Shipping was very slow."
        ],
        "Customer name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Date": pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"
        ]),
        "Category": ["electronics", "home", "electronics", "service", "shipping"],
        "Comment": [
            "Great product! Highly recommend.",
            "Poor quality, not worth the money.",
            "Average product, does the job.",
            "Excellent customer service!",
            "Shipping was very slow."
        ],
        "Rating": ["4.0 out of 5", "2.0 out of 5", "3.0 out of 5", "5.0 out of 5", "2.0 out of 5"],
        "Useful": ["7 people found this helpful", "3 people found this helpful", None, "10 people found this helpful", None]
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_dir():
    """Fixture providing a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
