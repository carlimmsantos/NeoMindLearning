from unittest.mock import Mock, patch
from src.llm.providers import OpenAIProvider, GeminiProvider, LLMResponse


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_llm_response_creation(self):
        """Test creating an LLMResponse."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai",
            response_time=1.5
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.response_time == 1.5


class TestOpenAIProvider:
    """Test OpenAI provider (with mocking)."""
    
    @patch('src.llm.providers.ChatOpenAI')
    def test_generate_response(self, mock_chat_openai):
        """Test generating a response with OpenAI provider."""
        # Mock the ChatOpenAI response
        mock_response = Mock()
        mock_response.content = "Test response from OpenAI"
        mock_chat_openai.return_value.invoke.return_value = mock_response
        
        provider = OpenAIProvider("fake-api-key")
        response = provider.generate("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response from OpenAI"
        assert response.provider == "openai"
        assert response.response_time > 0


class TestGeminiProvider:
    """Test Gemini provider (with mocking)."""
    
    @patch('src.llm.providers.ChatGoogleGenerativeAI')
    def test_generate_response(self, mock_chat_gemini):
        """Test generating a response with Gemini provider."""
        # Mock the ChatGoogleGenerativeAI response
        mock_response = Mock()
        mock_response.content = "Test response from Gemini"
        mock_chat_gemini.return_value.invoke.return_value = mock_response
        
        provider = GeminiProvider("fake-api-key")
        response = provider.generate("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response from Gemini"
        assert response.provider == "gemini"
        assert response.response_time > 0
