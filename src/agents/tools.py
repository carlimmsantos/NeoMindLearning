"""
LangChain Tools for Customer Comments Analysis

This module contains all the tools used by the DataAnalysisAgent for analyzing
customer feedback data.
"""

import logging
import pandas as pd
from typing import Dict, Any, Type, List
from collections import Counter

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..prompts.templates import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts

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
    
    def __init__(self, data_processor=None):
        super().__init__()
        self.data_processor = data_processor
        self._current_data = None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the current dataset for analysis."""
        data.columns = data.columns.str.lower().str.replace(' ', '_')
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
                if column not in self._current_data.columns:
                    return {"error": f"Column '{column}' not found in dataset"}
                
                avg_len = self._current_data[column].astype(str).str.len().mean()
                median_len = self._current_data[column].astype(str).str.len().median()
                
                return {
                    "avg_length": round(avg_len, 2),
                    "median_length": round(median_len, 2),
                    "metric": metric,
                    "message": f"Average comment length is {avg_len:.2f} characters"
                }
            
            elif metric == "word_frequency":
                if column not in self._current_data.columns:
                    return {"error": f"Column '{column}' not found in dataset"}
                
                all_text = " ".join(self._current_data[column].astype(str))
                words = all_text.lower().split()
                word_counts = Counter(words)
                
                # Get top 10 words (excluding very common words)
                top_words = word_counts.most_common(10)
                
                return {
                    "word_frequency": dict(top_words),
                    "total_unique_words": len(word_counts),
                    "metric": metric,
                    "message": "Top 10 most frequent words in customer comments"
                }
            
            elif metric == "rating_distribution":
                if "rating" not in self._current_data.columns:
                    return {"error": "Rating column not found in dataset"}
                
                ratings = self._current_data["rating"].value_counts().sort_index()
                
                return {
                    "distribution": ratings.to_dict(),
                    "metric": metric,
                    "message": "Distribution of customer ratings"
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
    
    def __init__(self):
        super().__init__()
        self._sentiment_results = None
    
    def set_sentiment_data(self, sentiment_data: List[Dict[str, Any]]) -> None:
        """Set sentiment analysis results."""
        self._sentiment_results = sentiment_data
    
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
        if self._sentiment_results is None:
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
        if not self._sentiment_results:
            return {"error": "No sentiment data available"}
        
        # Contar sentimentos
        positive_count = sum(1 for item in self._sentiment_results 
                            if item.get("sentiment") == "positive")
        negative_count = sum(1 for item in self._sentiment_results 
                            if item.get("sentiment") == "negative")
        neutral_count = sum(1 for item in self._sentiment_results 
                           if item.get("sentiment") == "neutral")
        
        total = len(self._sentiment_results)
        
        if total == 0:
            return {"error": "No valid sentiment data"}
        
        # Calcular confiança média
        confidences = [item.get("confidence", 0) for item in self._sentiment_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Determinar sentimento geral
        if positive_count > negative_count:
            overall = "positive"
        elif negative_count > positive_count:
            overall = "negative"
        else:
            overall = "neutral"
        
        # INTEGRAÇÃO COM PROMPT (SEGUINDO README)
        # Formatar dados para o prompt template
        formatted_prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(
            feedback=f"Analyzed {total} comments with {positive_count} positive, "
                    f"{negative_count} negative, and {neutral_count} neutral sentiments"
        )
        
        return {
            "aggregation_type": "summary",
            "total_comments": total,
            "positive": {
                "count": positive_count,
                "percentage": round((positive_count / total * 100), 2) if total > 0 else 0
            },
            "negative": {
                "count": negative_count,
                "percentage": round((negative_count / total * 100), 2) if total > 0 else 0
            },
            "neutral": {
                "count": neutral_count,
                "percentage": round((neutral_count / total * 100), 2) if total > 0 else 0
            },
            "overall_sentiment": overall,
            "average_confidence": round(avg_confidence, 3),
            "prompt_used": formatted_prompt,
            "message": f"Sentiment analysis complete. Overall sentiment: {overall}"
        }
    
    def _aggregate_distribution(self) -> Dict[str, Any]:
        """Analyze confidence distribution of sentiments."""
        if not self._sentiment_results:
            return {"error": "No sentiment data available"}
        
        distribution = {
            "high_confidence_positive": 0,
            "medium_confidence_positive": 0,
            "low_confidence_positive": 0,
            "high_confidence_negative": 0,
            "medium_confidence_negative": 0,
            "low_confidence_negative": 0,
            "neutral": 0
        }
        
        for item in self._sentiment_results:
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
            "total": len(self._sentiment_results),
            "message": "Sentiment distribution by confidence levels calculated"
        }
    
    def _aggregate_trends(self) -> Dict[str, Any]:
        """Identify sentiment trends and patterns."""
        if not self._sentiment_results:
            return {"error": "No sentiment data available"}
        
        # Extrair palavras-chave de sentimentos
        positive_words = []
        negative_words = []
        
        for item in self._sentiment_results:
            if item.get("sentiment") == "positive" and "key_emotions" in item:
                positive_words.extend(item.get("key_emotions", []))
            elif item.get("sentiment") == "negative" and "key_emotions" in item:
                negative_words.extend(item.get("key_emotions", []))
        
        # Contar frequência
        pos_counter = Counter(positive_words) if positive_words else Counter()
        neg_counter = Counter(negative_words) if negative_words else Counter()
        
        # Determinar tendência usando summary
        sentiment_summary = self._aggregate_summary()
        positive_pct = sentiment_summary.get("positive", {}).get("percentage", 0)
        negative_pct = sentiment_summary.get("negative", {}).get("percentage", 0)
        
        trend = "improving" if positive_pct > negative_pct else "declining" if negative_pct > positive_pct else "stable"
        
        return {
            "aggregation_type": "trends",
            "trend": trend,
            "positive_percentage": positive_pct,
            "negative_percentage": negative_pct,
            "top_positive_emotions": dict(pos_counter.most_common(5)),
            "top_negative_emotions": dict(neg_counter.most_common(5)),
            "message": f"Sentiment trend is {trend} overall"
        }
    
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
    
    def set_analysis_results(self, results: Dict[str, Any]) -> None:
        """Set the analysis results for insight generation."""
        self._analysis_results = results
    
    def _run(self, insight_type: str) -> Dict[str, Any]:
        """
        Generate business insights from analysis results.
        
        Implements:
        1. Analyze patterns across sentiment, topics, and statistics
        2. Identify key business opportunities or issues
        3. Generate actionable recommendations
        4. Prioritize findings by impact/importance
        
        Args:
            insight_type: Type of insights to focus on ('recommendations', 'trends', 'priorities')
            
        Returns:
            Dict with structured business insights
        """
        if self._analysis_results is None:
            return {"error": "No analysis results available"}
        
        try:
            if insight_type == "recommendations":
                return self._generate_recommendations()
            
            elif insight_type == "trends":
                return self._generate_trend_insights()
            
            elif insight_type == "priorities":
                return self._generate_priorities()
            
            else:
                return {"error": f"Unknown insight type: {insight_type}"}
        
        except Exception as e:
            logger.error(f"Error generating insights ({insight_type}): {str(e)}")
            return {"error": f"Error generating insights: {str(e)}"}
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        logger.info("Generating recommendations using Python logic")
        
        sentiment = self._analysis_results.get("sentiment_analysis", {})
        topics = self._analysis_results.get("topic_extraction", {})
        data_summary = self._analysis_results.get("data_summary", {})
        
        recommendations = []
        
        negative_pct = sentiment.get("negative", {}).get("percentage", 0)
        negative_count = sentiment.get("negative", {}).get("count", 0)
        
        if negative_pct > 30:
            recommendations.append({
                "priority": "HIGH",
                "category": "Negative Feedback Management",
                "action": "Investigate and address negative feedback",
                "reason": f"{negative_count} comments ({negative_pct}%) express negative sentiment",
                "impact": "high",
                "timeline": "Urgent (1-2 weeks)"
            })
            logger.warning(f"HIGH priority alert: {negative_pct}% negative sentiment detected")
        
        elif 10 <= negative_pct <= 30:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Quality Improvement",
                "action": "Monitor and improve areas with complaints",
                "reason": f"{negative_count} comments ({negative_pct}%) indicate areas for improvement",
                "impact": "medium",
                "timeline": "Important (2-4 weeks)"
            })
            logger.info(f"MEDIUM priority: {negative_pct}% negative sentiment - areas for improvement identified")
        
        elif negative_pct < 10:
            recommendations.append({
                "priority": "LOW",
                "category": "Maintenance",
                "action": "Maintain current product quality and standards",
                "reason": f"Only {negative_count} comments ({negative_pct}%) are negative - strong performance",
                "impact": "low",
                "timeline": "Ongoing"
            })
            logger.info(f"Excellent: Only {negative_pct}% negative sentiment - maintain current approach")
        
        positive_pct = sentiment.get("positive", {}).get("percentage", 0)
        positive_count = sentiment.get("positive", {}).get("count", 0)
        
        if positive_pct >= 50:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Marketing & Growth",
                "action": "Leverage positive feedback for marketing campaigns",
                "reason": f"{positive_count} comments ({positive_pct}%) praise the product/service",
                "impact": "medium",
                "timeline": "Important (2-4 weeks)",
                "suggested_tactics": [
                    "Collect customer testimonials",
                    "Create case studies from positive reviews",
                    "Use positive feedback in marketing materials"
                ]
            })
            logger.info(f"Marketing opportunity: {positive_pct}% positive sentiment - strong brand sentiment")
        
        avg_confidence = sentiment.get("average_confidence", 0)
        
        if avg_confidence <= 0.75:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Data Quality",
                "action": "Review sentiment analysis model accuracy",
                "reason": f"Average confidence is {avg_confidence:.2f} - lower than expected (target: >0.8)",
                "impact": "medium",
                "timeline": "Important (1-2 weeks)"
            })
            logger.warning(f"Low confidence detected: {avg_confidence:.2f} - may need model review")
        
        try:
            formatted_prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
                data_summary=str(data_summary),
                sentiment_results=str(sentiment),
                topic_results=str(topics)
            )
            prompt_integrated = True
        except Exception as e:
            logger.warning(f"Could not format summary prompt: {str(e)}")
            formatted_prompt = ""
            prompt_integrated = False
        
        return {
            "insight_type": "recommendations",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "prompt_template_used": "SummaryPrompts.FEEDBACK_SUMMARY",
            "prompt_integrated": prompt_integrated,
            "prompt_preview": formatted_prompt[:150] + "..." if formatted_prompt else "N/A",
            "generation_method": "Python-based logic analysis",
            "message": f"Generated {len(recommendations)} actionable recommendations based on data analysis"
        }
    
    def _generate_trend_insights(self) -> Dict[str, Any]:
        """Generate trend analysis insights."""
        logger.info("Generating trend insights")
        
        sentiment = self._analysis_results.get("sentiment_analysis", {})
        
        positive_pct = sentiment.get("positive", {}).get("percentage", 0)
        negative_pct = sentiment.get("negative", {}).get("percentage", 0)
        neutral_pct = sentiment.get("neutral", {}).get("percentage", 0)
        

        if positive_pct > negative_pct:
            sentiment_direction = "positive"
            interpretation = "Customer sentiment is predominantly positive"
        elif negative_pct > positive_pct:
            sentiment_direction = "negative"
            interpretation = "Customer sentiment is concerning with more negative than positive feedback"
        else:
            sentiment_direction = "neutral"
            interpretation = "Customer sentiment is balanced between positive and negative"
        

        sentiment_balance = abs(positive_pct - negative_pct)
        
        if sentiment_balance > 40:
            sentiment_strength = "strong"
        elif sentiment_balance > 20:
            sentiment_strength = "moderate"
        else:
            sentiment_strength = "weak"
        
        logger.info(f"Trend: {sentiment_direction} ({sentiment_strength}) - balance: {sentiment_balance:.1f}%")

        if sentiment_direction == "positive":
            if sentiment_strength in ["strong", "moderate"]:
                growth_potential = "Strong market position"
            else:
                growth_potential = "Moderate market position"
        else:
            growth_potential = "Challenged market position"

        trends = {
            "sentiment_direction": sentiment_direction,
            "sentiment_strength": sentiment_strength,
            "positive_momentum": round(positive_pct, 2),
            "negative_momentum": round(negative_pct, 2),
            "neutral_momentum": round(neutral_pct, 2),
            "sentiment_balance": round(sentiment_balance, 2),
            "interpretation": interpretation,
            "business_implication": {
                "positive": "Continue current strategies - customers are satisfied" if sentiment_direction == "positive" else "Implement improvement initiatives",
                "customer_retention": "High priority" if sentiment_direction == "negative" else "Normal monitoring",
                "growth_potential": growth_potential
            }
        }
        
        return {
            "insight_type": "trends",
            "trends": trends,
            "message": f"Sentiment trend is {sentiment_direction} with {sentiment_strength} momentum"
        }
    
    def _generate_priorities(self) -> Dict[str, Any]:
        """Generate priority analysis."""
        logger.info("Generating priority analysis")
        
        sentiment = self._analysis_results.get("sentiment_analysis", {})
        
        priorities = []
        
        negative_count = sentiment.get("negative", {}).get("count", 0)
        negative_pct = sentiment.get("negative", {}).get("percentage", 0)
        
        if negative_count > 0:
            priorities.append({
                "priority": 1,
                "category": "Negative Feedback",
                "issue": "Address customer complaints and negative feedback",
                "affected_customers": negative_count,
                "percentage": round(negative_pct, 2),
                "impact": "HIGH",
                "business_risk": "Customer churn, reputation damage, revenue loss",
                "action_items": [
                    "Analyze patterns in negative comments",
                    "Identify root causes",
                    "Create corrective action plan",
                    "Track resolution progress"
                ],
                "timeline": "Urgent (1-2 weeks)",
                "success_metric": "Reduce negative feedback by 20%"
            })
            logger.warning(f"Priority 1: {negative_count} negative comments require immediate attention")
        
        positive_count = sentiment.get("positive", {}).get("count", 0)
        positive_pct = sentiment.get("positive", {}).get("percentage", 0)
        
        if positive_count > 0:
            priorities.append({
                "priority": 2,
                "category": "Positive Feedback",
                "issue": "Leverage positive feedback for business growth",
                "affected_customers": positive_count,
                "percentage": round(positive_pct, 2),
                "impact": "MEDIUM",
                "business_opportunity": "Customer retention, brand advocacy, market expansion",
                "action_items": [
                    "Collect customer testimonials",
                    "Request case studies and reviews",
                    "Create content for marketing",
                    "Build referral program"
                ],
                "timeline": "Important (2-4 weeks)",
                "success_metric": "Convert 30% of positive customers to advocates"
            })
            logger.info(f"Priority 2: {positive_count} satisfied customers - opportunity for growth")
        
        total_comments = sentiment.get("total_comments", 0)
        
        if total_comments > 0:
            priorities.append({
                "priority": 3,
                "category": "Quality Assurance",
                "issue": "Maintain and improve overall product/service quality",
                "affected_customers": total_comments,
                "percentage": 100.0,
                "impact": "MEDIUM",
                "business_benefit": "Long-term competitiveness, customer satisfaction",
                "action_items": [
                    "Establish quality benchmarks",
                    "Regular quality audits",
                    "Continuous improvement process",
                    "Staff training programs"
                ],
                "timeline": "Ongoing",
                "success_metric": "Maintain >80% positive sentiment"
            })
            logger.info(f"Priority 3: Continuous quality maintenance for {total_comments} customers")
        
        logger.info(f"Generated {len(priorities)} prioritized action items")
        
        return {
            "insight_type": "priorities",
            "priorities": priorities,
            "priority_count": len(priorities),
            "total_affected_customers": sentiment.get("total_comments", 0),
            "generation_method": "Impact-based prioritization",
            "message": f"Prioritized {len(priorities)} action items ranked by business impact"
        }
    
    async def _arun(self, insight_type: str) -> Dict[str, Any]:
        """Async version of the tool."""
        logger.info(f"Async call to _arun with insight_type={insight_type}")
        return self._run(insight_type)
