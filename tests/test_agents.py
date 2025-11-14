from src.agents.core import DataAnalysisAgent

class TestDataAnalysisAgentStructure:
    """Test DataAnalysisAgent structure and initialization."""
    
    def test_agent_initialization(self, mock_llm_providers):
        """Test agent initialization."""
        from src.data import DataProcessor
        
        data_processor = DataProcessor()
        agent = DataAnalysisAgent(mock_llm_providers, data_processor)
        
        assert agent._llm_providers == mock_llm_providers
        assert agent._data_processor == data_processor
        assert agent._graph is not None