"""
AI Agents module for customer comment analysis.

This module provides a complete LangChain-based agent system for analyzing
customer feedback data using React Agent for automatic tool calling.

Main Components:
- DataAnalysisAgent: Main React Agent for analysis
- Tools: LangChain tools for data analysis, sentiment aggregation, and insights
- CustomerAnalysisReactAgent: Direct React Agent implementation

Usage:
    from src.agents import DataAnalysisAgent
    from src.data import DataProcessor
    from src.llm import OpenAIProvider, GeminiProvider
    
    # Initialize components
    data_processor = DataProcessor()
    llm_providers = {
        "openai": OpenAIProvider(api_key="your-key"),
        "gemini": GeminiProvider(api_key="your-key")
    }
    
    # Create and run agent
    agent = DataAnalysisAgent(llm_providers, data_processor)
    results = agent.analyze()
"""

from .core import DataAnalysisAgent
from .tools import (
    DataStatsTool,
    SentimentAggregationTool, 
    InsightGenerationTool,
    DataStatsInput,
    SentimentAggregationInput,
    InsightGenerationInput
)
from .react_agent import CustomerAnalysisReactAgent

__all__ = [
    "DataAnalysisAgent",
    "CustomerAnalysisReactAgent",
    "DataStatsTool",
    "SentimentAggregationTool",
    "InsightGenerationTool",
    "DataStatsInput",
    "SentimentAggregationInput", 
    "InsightGenerationInput"
]
