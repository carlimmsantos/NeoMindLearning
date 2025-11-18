"""
LangChain Tools for Customer Comments Analysis

This module contains all the tools used by the DataAnalysisAgent for analyzing
customer feedback data.
"""

import logging
import json
from typing import Dict, Any, Type, List, Optional
from collections import Counter

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from ..prompts.templates import (
    SentimentAnalysisPrompts,
    TopicExtractionPrompts,
    SummaryPrompts,
)
from ..llm.providers import BaseLLMProvider

logger = logging.getLogger(__name__)


# Pydantic models for tool inputs
class DataStatsInput(BaseModel):
    """Input schema for data statistics tool."""
    metric: str = Field(
        description="Type of statistic to calculate: 'count', 'avg_length', 'word_frequency', 'rating_distribution'"
    )
    column: str = Field(default="comment", description="Column name to analyze")


class SentimentAggregationInput(BaseModel):
    """Input schema for sentiment aggregation tool."""
    aggregation_type: str = Field(
        description="Type of aggregation: 'summary', 'distribution', 'trends'"
    )


class InsightGenerationInput(BaseModel):
    """Input schema for insight generation tool."""
    insight_type: str = Field(
        description="Type of insights: 'recommendations', 'trends', 'priorities'"
    )


class DataStatsTool(BaseTool):
    """Tool for calculating statistical metrics on customer comments data."""

    name: str = "calculate_data_stats"
    description: str = """Calculate statistical metrics for customer comments dataset.
    Use this tool to get basic statistics like count, average length, or word frequency analysis."""
    args_schema: Type[BaseModel] = DataStatsInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_processor: Any = Field(default=None, exclude=True)
    current_data: Any = Field(default=None, exclude=True)

    def __init__(self, data_processor=None, **kwargs):
        super().__init__(**kwargs)
        self.data_processor = data_processor
        self.current_data = None

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the current dataset for analysis."""
        data.columns = data.columns.str.lower().str.replace(" ", "_")
        self.current_data = data
        logger.debug(f"DataStatsTool: Set data with {len(data)} records")

    def _run(self, metric: str, column: str = "comment") -> Dict[str, Any]:
        """
        Calculate statistical metrics for customer comments data.

        Args:
            metric: The type of metric to calculate
            column: Column to analyze

        Returns:
            Dict with statistical results
        """
        if self.current_data is None:
            return {"error": "No data available for analysis"}

        try:
            if metric == "count":
                return {
                    "count": len(self.current_data),
                    "metric": metric,
                    "message": f"Dataset contains {len(self.current_data)} customer comments",
                }

            elif metric == "avg_length":
                if column not in self.current_data.columns:
                    return {"error": f"Column '{column}' not found in dataset"}

                avg_len = self.current_data[column].astype(str).str.len().mean()
                median_len = self.current_data[column].astype(str).str.len().median()

                return {
                    "avg_length": round(avg_len, 2),
                    "median_length": round(median_len, 2),
                    "metric": metric,
                    "message": f"Average comment length is {avg_len:.2f} characters",
                }

            elif metric == "word_frequency":
                if column not in self.current_data.columns:
                    return {"error": f"Column '{column}' not found in dataset"}

                all_text = " ".join(self.current_data[column].astype(str))
                words = all_text.lower().split()
                word_counts = Counter(words)

                # Get top 10 words (excluding very common words)
                top_words = word_counts.most_common(10)

                return {
                    "word_frequency": dict(top_words),
                    "total_unique_words": len(word_counts),
                    "metric": metric,
                    "message": "Top 10 most frequent words in customer comments",
                }

            elif metric == "rating_distribution":
                if "rating" not in self.current_data.columns:
                    return {"error": "Rating column not found in dataset"}

                ratings = self.current_data["rating"].value_counts().sort_index()

                return {
                    "distribution": ratings.to_dict(),
                    "metric": metric,
                    "message": "Distribution of customer ratings",
                }

            else:
                return {"error": f"Unknown metric: {metric}"}

        except Exception as e:
            logger.error(f"Error calculating {metric}: {str(e)}")
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sentiment_results: Any = Field(default=None, exclude=True)
    llm_provider: Optional[BaseLLMProvider] = Field(default=None, exclude=True)

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, **kwargs):
        super().__init__(**kwargs)
        self.sentiment_results = None
        self.llm_provider = llm_provider 

    def set_sentiment_data(self, sentiment_data: List[Dict[str, Any]]) -> None:
        """Set sentiment analysis results."""
        self.sentiment_results = sentiment_data
        logger.debug(
            f"SentimentAggregationTool: Set sentiment data with {len(sentiment_data)} items"
        )

    def _run(self, aggregation_type: str) -> Dict[str, Any]:
        """
        Aggregate sentiment analysis results.

        Implements:
        1. Process individual sentiment scores/labels
        2. Calculate overall sentiment distribution
        3. Identify sentiment trends or patterns
        4. Generate summary statistics

        Args:
            aggregation_type: How to aggregate the data ('summary', 'distribution', 'trends')

        Returns:
            Dict with aggregated sentiment insights
        """
        if self.sentiment_results is None:
            return {"error": "No sentiment data available"}

        try:
            if aggregation_type == "summary":
                return self._aggregate_summary()
            elif aggregation_type == "distribution":
                return self._aggregate_distribution()
            elif aggregation_type == "trends":
                return self._aggregate_trends()
            else:
                return {"error": f"Unknown aggregation type: {aggregation_type}"}

        except Exception as e:
            logger.error(f"Error aggregating sentiment ({aggregation_type}): {str(e)}")
            return {"error": f"Error aggregating sentiment: {str(e)}"}

    def _aggregate_summary(self) -> Dict[str, Any]:
        """Calculate overall sentiment summary."""
        if not self.sentiment_results:
            return {"error": "No sentiment data available"}

        positive_count = sum(
            1 for item in self.sentiment_results if item.get("sentiment") == "positive"
        )
        negative_count = sum(
            1 for item in self.sentiment_results if item.get("sentiment") == "negative"
        )
        neutral_count = sum(
            1 for item in self.sentiment_results if item.get("sentiment") == "neutral"
        )

        total = len(self.sentiment_results)

        if total == 0:
            return {"error": "No valid sentiment data"}

        confidences = [item.get("confidence", 0) for item in self.sentiment_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        if positive_count > negative_count:
            overall = "positive"
        elif negative_count > positive_count:
            overall = "negative"
        else:
            overall = "neutral"

        feedback_summary = f"Analyzed {total} comments: {positive_count} positive, {negative_count} negative, {neutral_count} neutral"
        
        try:
            # ✅ Format the prompt template
            formatted_prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(
                feedback=feedback_summary
            )
            logger.info(f"SentimentAggregationTool: Using BASIC_SENTIMENT template")
            
            # ✅ If LLM provider available, use it
            if self.llm_provider:
                logger.info("SentimentAggregationTool: Calling LLM for sentiment analysis...")
                llm_response = self.llm_provider.generate(formatted_prompt)
                prompt_preview = llm_response.content[:200]
            else:
                logger.warning("SentimentAggregationTool: No LLM provider, using template preview only")
                prompt_preview = formatted_prompt[:200]
            
        except Exception as e:
            logger.error(f"Error using BASIC_SENTIMENT template: {e}")
            prompt_preview = ""

        logger.info(
            f"SentimentAggregationTool: Summary - Overall: {overall}, Confidence: {avg_confidence:.3f}"
        )

        return {
            "aggregation_type": "summary",
            "total_comments": total,
            "positive": {
                "count": positive_count,
                "percentage": round((positive_count / total * 100), 2) if total > 0 else 0,
            },
            "negative": {
                "count": negative_count,
                "percentage": round((negative_count / total * 100), 2) if total > 0 else 0,
            },
            "neutral": {
                "count": neutral_count,
                "percentage": round((neutral_count / total * 100), 2) if total > 0 else 0,
            },
            "overall_sentiment": overall,
            "average_confidence": round(avg_confidence, 3),
            "prompt_template_used": "SentimentAnalysisPrompts.BASIC_SENTIMENT",
            "prompt_preview": prompt_preview,
            "message": f"Sentiment analysis complete. Overall sentiment: {overall}",
        }

    def _aggregate_distribution(self) -> Dict[str, Any]:
        """Analyze confidence distribution of sentiments."""
        if not self.sentiment_results:
            return {"error": "No sentiment data available"}

        distribution = {
            "high_confidence_positive": 0,
            "medium_confidence_positive": 0,
            "low_confidence_positive": 0,
            "high_confidence_negative": 0,
            "medium_confidence_negative": 0,
            "low_confidence_negative": 0,
            "neutral": 0,
        }

        for item in self.sentiment_results:
            sentiment = item.get("sentiment", "neutral")
            confidence = item.get("confidence", 0)

            if sentiment == "positive":
                if confidence >= 0.8:
                    distribution["high_confidence_positive"] += 1
                elif confidence >= 0.6:
                    distribution["medium_confidence_positive"] += 1
                else:
                    distribution["low_confidence_positive"] += 1
            elif sentiment == "negative":
                if confidence >= 0.8:
                    distribution["high_confidence_negative"] += 1
                elif confidence >= 0.6:
                    distribution["medium_confidence_negative"] += 1
                else:
                    distribution["low_confidence_negative"] += 1
            else:
                distribution["neutral"] += 1

        return {
            "aggregation_type": "distribution",
            "distribution": distribution,
            "total": len(self.sentiment_results),
            "message": "Sentiment distribution by confidence levels calculated",
        }

    def _aggregate_trends(self) -> Dict[str, Any]:
        """Identify sentiment trends and patterns."""
        if not self.sentiment_results:
            return {"error": "No sentiment data available"}

        positive_words = []
        negative_words = []

        for item in self.sentiment_results:
            if item.get("sentiment") == "positive" and "key_emotions" in item:
                positive_words.extend(item.get("key_emotions", []))
            elif item.get("sentiment") == "negative" and "key_emotions" in item:
                negative_words.extend(item.get("key_emotions", []))

        pos_counter = Counter(positive_words) if positive_words else Counter()
        neg_counter = Counter(negative_words) if negative_words else Counter()

        sentiment_summary = self._aggregate_summary()
        positive_pct = sentiment_summary.get("positive", {}).get("percentage", 0)
        negative_pct = sentiment_summary.get("negative", {}).get("percentage", 0)

        trend = (
            "improving"
            if positive_pct > negative_pct
            else "declining"
            if negative_pct > positive_pct
            else "stable"
        )

        logger.info(f"SentimentAggregationTool: Trend - {trend}")

        return {
            "aggregation_type": "trends",
            "trend": trend,
            "positive_percentage": positive_pct,
            "negative_percentage": negative_pct,
            "top_positive_emotions": dict(pos_counter.most_common(5)),
            "top_negative_emotions": dict(neg_counter.most_common(5)),
            "message": f"Sentiment trend is {trend} overall",
        }

    async def _arun(self, aggregation_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(aggregation_type)

class InsightGenerationTool(BaseTool):
    """
    Tool for generating business insights by calling an LLM
    with the final summary prompt.
    """

    name: str = "generate_insights"
    description: str = """Generate actionable business insights from customer feedback analysis.
    Use this tool to create the final business report, including recommendations, 
    key findings, and priority issues."""
    args_schema: Type[BaseModel] = InsightGenerationInput
    model_config = ConfigDict(arbitrary_types_allowed=True)


    analysis_results: Any = Field(default=None, exclude=True)
    llm_provider: Optional[BaseLLMProvider] = Field(default=None, exclude=True)

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, **kwargs):
        """
        Initialize the tool with an LLM provider.
        """
        super().__init__(**kwargs)
        self.analysis_results = None
        self.llm_provider = llm_provider 

    def set_analysis_results(self, results: Dict[str, Any]) -> None:
        """Set the analysis results for insight generation."""
        self.analysis_results = results
        logger.debug("InsightGenerationTool: Set analysis results")

    def _run(self, insight_type: str) -> Dict[str, Any]:
        """
        Generates the final business report by calling the LLM
        using the FEEDBACK_SUMMARY prompt.
        """
        if self.llm_provider is None:
            return {"error": "LLM provider not configured for InsightGenerationTool"}

        if self.analysis_results is None:
            return {"error": "No analysis results available to generate insights"}

        if insight_type not in ["recommendations", "trends", "priorities"]:
            return {
                "error": f"Unknown insight_type: '{insight_type}'. Try 'recommendations', 'trends', or 'priorities'"
            }
            
        logger.info(f"Generating LLM-based report for insight_type='{insight_type}'...")
        
        try:

            data_summary = self.analysis_results.get("data_summary", {})
            sentiment_results = self.analysis_results.get("sentiment_analysis", {})
            topic_results = self.analysis_results.get("topic_extraction", {})
            

            formatted_prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
                data_summary=str(data_summary),
                sentiment_results=str(sentiment_results),
                topic_results=str(topic_results)
            )
            

            response = self.llm_provider.generate(formatted_prompt)
            
            logger.info("Successfully generated insights from LLM.")

            return {
                "insight_type": insight_type,
                "report": response.content,
                "model_used": response.model,
                "response_time_seconds": response.response_time,
                "generation_method": "LLM-based generation",
                "message": "Final business report generated successfully."
            }
        
        except Exception as e:
            logger.error(f"Error generating insights via LLM: {str(e)}")
            return {"error": f"Error generating insights: {str(e)}"}

    async def _arun(self, insight_type: str) -> Dict[str, Any]:
        """Async version of the tool."""

        return self._run(insight_type)
