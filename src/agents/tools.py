"""
LangChain Tools for Customer Comments Analysis

This module contains all the tools used by the DataAnalysisAgent for analyzing
customer feedback data.
"""

import logging
from typing import Dict, Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for tool inputs
class DataStatsInput(BaseModel):
    """Input schema for data statistics tool."""
    metric: str = Field(description="Type of statistic to calculate: 'count', 'avg_length', 'word_frequency'")
    column: str = Field(default="comment", description="Column name to analyze")


class SentimentAggregationInput(BaseModel):
    """Input schema for sentiment aggregation tool."""
    aggregation_type: str = Field(description="Type of aggregation: 'summary', 'distribution', 'trends'")


class InsightGenerationInput(BaseModel):
    """Input schema for insight generation tool."""
    insight_type: str = Field(description="Type of insights: 'recommendations', 'trends', 'priorities'")


class DataStatsTool(BaseTool):
    """Tool for calculating statistical metrics on customer comments data."""
    
    name: str = "calculate_data_stats"
    description: str = """Calculate statistical metrics for customer comments dataset.
    Use this tool to get basic statistics like count, average length, or word frequency analysis."""
    args_schema: Type[BaseModel] = DataStatsInput
    
    def __init__(self, data_processor=None):
        super().__init__()
        self.data_processor = data_processor
        self._current_data = None
    
    def set_data(self, data):
        """Set the current dataset for analysis."""
        self._current_data = data
    
    def _run(self, metric: str, column: str = "comment") -> Dict[str, Any]:
        """
        Calculate statistical metrics for customer comments data.
        
        Args:
            metric: The type of metric to calculate
            column: Column to analyze
            
        Returns:
            Dict with statistical results
        """
        if self._current_data is None:
            return {"error": "No data available for analysis"}
        
        try:
            if metric == "count":
                return {
                    "count": len(self._current_data), 
                    "metric": metric,
                    "message": f"Dataset contains {len(self._current_data)} customer comments"
                }
            elif metric == "avg_length":
                avg_len = self._current_data[column].str.len().mean()
                return {
                    "avg_length": round(avg_len, 2), 
                    "metric": metric,
                    "message": f"Average comment length is {avg_len:.2f} characters"
                }
            elif metric == "word_frequency":
                # Simple word frequency analysis
                all_text = " ".join(self._current_data[column].astype(str))
                words = all_text.lower().split()
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                # Get top 10 words
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                return {
                    "word_frequency": dict(top_words),
                    "metric": metric,
                    "message": "Top 10 most frequent words in customer comments"
                }
            else:
                return {"error": f"Unknown metric: {metric}"}
        except Exception as e:
            return {"error": f"Error calculating {metric}: {str(e)}"}
    
    async def _arun(self, metric: str, column: str = "comment") -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(metric, column)


class SentimentAggregationTool(BaseTool):
    """Tool for aggregating sentiment analysis results."""
    
    name: str = "aggregate_sentiment" 
    description: str = """Aggregate sentiment analysis results across all comments.
    Use this tool to summarize sentiment patterns and distributions."""
    args_schema: Type[BaseModel] = SentimentAggregationInput
    
    def __init__(self):
        super().__init__()
        self._sentiment_results = None
    
    def set_sentiment_data(self, sentiment_data):
        """Set sentiment analysis results."""
        self._sentiment_results = sentiment_data
    
    def _run(self, aggregation_type: str) -> Dict[str, Any]:
        """
        TODO: Implement sentiment aggregation logic.
        
        The intern should implement:
        1. Process individual sentiment scores/labels
        2. Calculate overall sentiment distribution  
        3. Identify sentiment trends or patterns
        4. Generate summary statistics
        
        Args:
            aggregation_type: How to aggregate the data
            
        Returns:
            Dict with aggregated sentiment insights
        """
        if self._sentiment_results is None:
            return {"error": "No sentiment data available"}
        
        # TODO: Intern must implement sentiment aggregation
        if aggregation_type == "summary":
            raise NotImplementedError("Intern must implement sentiment summary aggregation")
        elif aggregation_type == "distribution":
            raise NotImplementedError("Intern must implement sentiment distribution calculation")
        elif aggregation_type == "trends":
            raise NotImplementedError("Intern must implement sentiment trend analysis")
        else:
            return {"error": f"Unknown aggregation type: {aggregation_type}"}
    
    async def _arun(self, aggregation_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(aggregation_type)


class InsightGenerationTool(BaseTool):
    """Tool for generating business insights from analysis results."""
    
    name: str = "generate_insights"
    description: str = """Generate actionable business insights from customer feedback analysis.
    Use this tool to create recommendations and identify key business opportunities."""
    args_schema: Type[BaseModel] = InsightGenerationInput
    
    def __init__(self):
        super().__init__()
        self._analysis_results = None
    
    def set_analysis_results(self, results):
        """Set the analysis results for insight generation."""
        self._analysis_results = results
    
    def _run(self, insight_type: str) -> Dict[str, Any]:
        """
        TODO: Implement insight generation logic.
        
        The intern should implement:
        1. Analyze patterns across sentiment, topics, and statistics
        2. Identify key business opportunities or issues
        3. Generate actionable recommendations
        4. Prioritize findings by impact/importance
        
        Args:
            insight_type: Type of insights to focus on
            
        Returns:
            Dict with structured business insights
        """
        if self._analysis_results is None:
            return {"error": "No analysis results available"}
        
        # TODO: Intern must implement insight generation
        if insight_type == "recommendations":
            raise NotImplementedError("Intern must implement recommendation generation")
        elif insight_type == "trends":
            raise NotImplementedError("Intern must implement trend analysis")
        elif insight_type == "priorities":
            raise NotImplementedError("Intern must implement priority analysis")
        else:
            return {"error": f"Unknown insight type: {insight_type}"}
    
    async def _arun(self, insight_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(insight_type)
