"""
Workflow nodes for the DataAnalysisAgent.

This module contains all the individual workflow steps (nodes) that are
executed as part of the LangGraph workflow.
"""

import logging
from typing import Dict, Any, List
import json

from .state import AgentState, update_state_with_error
from .tools import DataStatsTool, SentimentAggregationTool, InsightGenerationTool
from ..data import DataProcessor
from ..prompts.templates import (
    SentimentAnalysisPrompts,
    TopicExtractionPrompts,
)
from ..llm.providers import BaseLLMProvider

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Container class for all workflow node implementations."""

    MAX_SENTIMENT_PROMPTS = 10
    MAX_TOPIC_PROMPTS = 10
    MAX_REQUESTS_PER_RUN = 50

    def __init__(
        self,
        data_processor: DataProcessor,
        llm_providers: Dict[str, BaseLLMProvider], 
        llm_to_use: str,
        batch_size: int = 5,
    ):
        """
        Initialize workflow nodes with all required dependencies.
        
        Establishes LLM provider selection, tool instantiation, and resource
        configuration for the complete analysis pipeline.
        
        Args:
            data_processor: Data loading and preprocessing utility
            llm_providers: Dictionary mapping provider names to LLM instances
            llm_to_use: Key identifying the primary LLM provider
            batch_size: Number of requests to process per batch operation
        """
        self.data_processor = data_processor
        self.llm_providers = llm_providers
        self.llm_to_use = llm_to_use
        self.batch_size = batch_size
        self.request_count = 0
        self.max_requests = self.MAX_REQUESTS_PER_RUN

        # Step 1: Select and retrieve the specified LLM provider
        try:
            self.llm_provider = llm_providers.get(llm_to_use)
            if self.llm_provider is None:
                logger.warning(
                    f"LLM provider '{llm_to_use}' not found. Using first available..."
                )
                self.llm_provider = (
                    next(iter(llm_providers.values())) if llm_providers else None
                )

            if self.llm_provider is None:
                raise ValueError("No LLM providers available.")

        except Exception as e:
            logger.error(f"Error selecting LLM provider: {e}")
            self.llm_provider = None

        # Step 2: Initialize all analysis tools with proper dependency injection
        self.data_stats_tool = DataStatsTool(data_processor=data_processor)
        self.sentiment_aggregation_tool = SentimentAggregationTool()

        # Critical: Inject LLM provider into insight generation tool
        self.insight_generation_tool = InsightGenerationTool(
            llm_provider=self.llm_provider
        )

        logger.info(
            f"WorkflowNodes initialized with LLM: {self.llm_provider.model if self.llm_provider else 'None'}"
        )

    def load_data(self, state: AgentState) -> AgentState:
        """
        Node 1: Data Loading and Preprocessing.
        
        Orchestrates initial data ingestion, validation, and preparation:
        1. Load customer comments from data source (CSV/database)
        2. Perform data cleaning and standardization
        3. Generate descriptive summary statistics
        4. Distribute dataset to all analysis tools
        
        Args:
            state: Current workflow state to be updated
            
        Returns:
            Updated state with loaded data and summary statistics
        """
        try:
            # Load raw data from configured source
            df = self.data_processor.load_customer_comments()
            logger.info(f"Data loaded: {len(df)} records")

            # Generate and cache data summary for reference
            data_summary = self.data_processor.get_data_summary(df)
            logger.info(f"Data summary: {data_summary['total_records']} records")

            # Distribute data to all dependent tools
            self.data_stats_tool.set_data(df)

            # Propagate loaded data through workflow state
            state["data"] = df
            state["analysis_results"]["data_summary"] = data_summary
            state["current_step"] = "sentiment_analysis"

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            state = update_state_with_error(state, error_msg, "load_data")

        return state

    def analyze_sentiment(self, state: AgentState) -> AgentState:
        """Analyze sentiment of comments."""
        try:
            # Validate data availability
            df = state.get("data")
            if df is None or len(df) == 0:
                raise ValueError("No data available for sentiment analysis")

            logger.info("Analyzing sentiment of customer comments...")

            # Extract comment texts for batch processing
            comments = df["comment"].tolist()

            # Format batch prompts using sentiment analysis template
            prompts = [
                SentimentAnalysisPrompts.BASIC_SENTIMENT.format(feedback=comment)
                for comment in comments[:10]
            ]

            # Invoke LLM for batch sentiment analysis
            responses = self.llm_provider.generate_batch(prompts)

            # Parse sentiment results from LLM responses
            sentiments = []
            for response in responses:
                try:
                    # Parse JSON-formatted sentiment classification
                    sentiment_data = json.loads(response.content)
                    sentiments.append(sentiment_data)
                except json.JSONDecodeError:
                    logger.warning("Could not parse sentiment response as JSON")
                    # Fallback to neutral classification on parse error
                    sentiments.append(
                        {
                            "sentiment": "neutral",
                            "confidence": 0.5,
                            "reasoning": "Parse error",
                        }
                    )

            # Count sentiment classifications
            positive_count = sum(
                1 for s in sentiments if s.get("sentiment") == "positive"
            )
            negative_count = sum(
                1 for s in sentiments if s.get("sentiment") == "negative"
            )
            neutral_count = len(sentiments) - positive_count - negative_count

            # Calculate average confidence score
            avg_confidence = (
                sum(s.get("confidence", 0) for s in sentiments) / len(sentiments)
                if sentiments
                else 0
            )

            # Structure sentiment results for state propagation
            state["sentiment_analysis"] = {
                "total_analyzed": len(sentiments),
                "positive": {
                    "count": positive_count,
                    "percentage": round(positive_count / len(sentiments) * 100, 2)
                    if sentiments
                    else 0,
                },
                "negative": {
                    "count": negative_count,
                    "percentage": round(negative_count / len(sentiments) * 100, 2)
                    if sentiments
                    else 0,
                },
                "neutral": {
                    "count": neutral_count,
                    "percentage": round(neutral_count / len(sentiments) * 100, 2)
                    if sentiments
                    else 0,
                },
                "average_confidence": round(avg_confidence, 3),
                "overall_sentiment": "positive"
                if positive_count > negative_count
                else "negative"
                if negative_count > positive_count
                else "neutral",
            }

            # Store raw sentiment classifications for downstream processing
            state["raw_sentiments"] = sentiments

            logger.info(
                f"Sentiment analysis complete: {positive_count} positive, {negative_count} negative, {neutral_count} neutral"
            )

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            state["sentiment_analysis"] = {"error": str(e)}

        return state

    def calculate_statistics(self, state: AgentState) -> AgentState:
        """
        Node 3: Calculate statistical metrics on customer data.

        README Phase 4, Task 3:
        - Use DataStatsTool (from Phase 3)
        - Calculate: count, avg_length, word_frequency, rating_distribution
        """
        try:
            # Initialize statistics container
            stats = {}

            logger.info("Calculating statistical metrics on dataset...")

            # Execute all statistical calculations through DataStatsTool
            stats["count"] = self.data_stats_tool._run(metric="count")
            stats["avg_length"] = self.data_stats_tool._run(metric="avg_length")
            stats["word_frequency"] = self.data_stats_tool._run(metric="word_frequency")
            stats["rating_distribution"] = self.data_stats_tool._run(
                metric="rating_distribution"
            )

            # Propagate statistical results through workflow state
            state["analysis_results"]["statistics"] = stats
            state["current_step"] = "topic_extraction"

            # Log summary of calculated statistics
            logger.info("Statistical calculations completed successfully")
            logger.info(f"  Total comments: {stats['count']['count']}")
            logger.info(
                f"  Average comment length: {stats['avg_length']['avg_length']} characters"
            )

        except Exception as e:
            error_msg = f"Error calculating statistics: {str(e)}"
            logger.error(error_msg)
            state = update_state_with_error(state, error_msg, "calculate_statistics")

        return state

    def generate_insights(self, state: AgentState) -> AgentState:
        """
        Node 4: Extract topics from customer comments using LLM.

        README Phase 4, Task 4:
        - Use TopicExtractionPrompts.EXTRACT_TOPICS (from Phase 2)
        - Call LLM to extract topics from comments
        - Parse and aggregate topic results
        """
        try:
            # Retrieve loaded dataset
            df = state.get("data")
            if df is None or len(df) == 0:
                logger.warning("No data available for topic extraction")
                return state

            logger.info("Extracting topics from customer comments...")

            # Sample comments for topic extraction (processing first 10 for efficiency)
            comments = df["comment"].tolist()[:10]
            
            # Format batch prompts using topic extraction template
            prompts = [
                TopicExtractionPrompts.EXTRACT_TOPICS.format(comment=comment)
                for comment in comments
            ]

            # Invoke LLM for batch topic extraction
            responses = self.llm_provider.generate_batch(prompts)

            # Containers for aggregating extracted topics
            all_topics = []
            all_issues = []
            all_praise = []

            # Parse topic extraction results from LLM responses
            for response in responses:
                try:
                    # Parse JSON-formatted topic classification
                    topic_data = json.loads(response.content)
                    all_topics.extend(topic_data.get("main_topics", []))
                    all_issues.extend(topic_data.get("issues", []))
                    all_praise.extend(topic_data.get("praise", []))
                except json.JSONDecodeError:
                    logger.warning("Could not parse topic extraction response as JSON")

            # Deduplicate and rank topics by frequency
            from collections import Counter

            topic_counter = Counter(all_topics)

            # Structure topic extraction results for state propagation
            state["topic_extraction"] = {
                "aggregated_topics": [
                    topic for topic, _ in topic_counter.most_common(5)
                ],
                "top_issues": dict(Counter(all_issues).most_common(3))
                if all_issues
                else {},
                "top_praise": dict(Counter(all_praise).most_common(3))
                if all_praise
                else {},
                "total_topics_extracted": len(all_topics),
            }

            logger.info(f"Topic extraction complete: {len(all_topics)} topics identified")

        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            state["topic_extraction"] = {"error": str(e)}

        return state

    def _aggregate_topics(self, responses: List[Any]) -> Dict[str, Any]:
        """
        Parse and aggregate topic extraction responses from LLM.

        Args:
            responses: List of LLM responses containing topic data

        Returns:
            Aggregated topic results
        """
        # Containers for collecting extracted information
        all_topics = []
        all_issues = []
        all_praise = []

        # Parse each LLM response
        for res in responses:
            try:
                content = res.content.strip()

                # Handle markdown code block formatting from LLM output
                if content.startswith("```json"):
                    content = content.replace("```json\n", "").replace("```", "")
                elif content.startswith("```"):
                    content = content.replace("```\n", "").replace("```", "")

                # Parse JSON-formatted response
                json_data = json.loads(content.strip())

                # Extract and aggregate topic components
                all_topics.extend(json_data.get("main_topics", []))
                all_issues.extend(json_data.get("issues", []))
                all_praise.extend(json_data.get("praise", []))

            except Exception as e:
                logger.warning(f"Failed to parse topic extraction response: {e}")

        # Deduplicate and rank extracted items by frequency
        from collections import Counter

        topic_counts = Counter(all_topics)
        issue_counts = Counter(all_issues)
        praise_counts = Counter(all_praise)

        # Return ranked aggregation results
        return {
            "aggregated_topics": [t for t, _ in topic_counts.most_common(5)],
            "top_issues": dict(issue_counts.most_common(5)),
            "top_praise": dict(praise_counts.most_common(5)),
            "total_topics_extracted": len(all_topics),
        }

    def generate_final_summary(self, state: AgentState) -> AgentState:
        """
        Node 5 (Final): Comprehensive Business Report Generation.
        
        Synthesizes all analysis outputs into actionable business report:
        1. Aggregate all analysis results from prior workflow stages
        2. Invoke InsightGenerationTool for Python-based analytics
        3. Format data using FEEDBACK_SUMMARY prompt template
        4. Call LLM to generate comprehensive business report and recommendations
        5. Integrate final report into workflow state for delivery
        
        Returns:
            Updated state with final business report and recommendations
        """
        try:
            # Validate LLM provider availability
            if self.llm_provider is None:
                raise ValueError("LLM provider not initialized for report generation")

            # Retrieve aggregated analysis results from all workflow stages
            analysis_data = state.get("analysis_results", {})

            # Inject collected analysis results into insight generation tool
            self.insight_generation_tool.set_analysis_results(analysis_data)

            # Generate insights using Python-based analysis logic
            logger.info("Generating business insights and recommendations...")

            # Invoke LLM through InsightGenerationTool for report generation
            report_result = self.insight_generation_tool._run(
                insight_type="recommendations"
            )

            # Validate report generation success
            if "error" in report_result:
                raise Exception(report_result["error"])

            # Integrate final report into workflow state
            state["analysis_results"]["final_summary"] = report_result
            state["current_step"] = "completed"

            logger.info("Final business report generated successfully")
            logger.info(f"LLM model used: {report_result.get('model_used')}")

        except Exception as e:
            error_msg = f"Error generating final summary: {str(e)}"
            logger.error(error_msg)
            state = update_state_with_error(state, error_msg, "generate_final_summary")

        return state
