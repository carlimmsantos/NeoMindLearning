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
    SummaryPrompts,
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
        llm_providers: Dict[str, BaseLLMProvider], # <-- Use o tipo BaseLLMProvider
        llm_to_use: str,
        batch_size: int = 5
        ):
        """Initialize workflow nodes with required dependencies."""
        self.data_processor = data_processor
        self.llm_providers = llm_providers
        self.llm_to_use = llm_to_use
        self.batch_size = batch_size
        self.request_count = 0
        self.max_requests = self.MAX_REQUESTS_PER_RUN

        # 1. Selecionar o LLM Provider PRIMEIRO
        try:
            self.llm_provider = llm_providers.get(llm_to_use)
            if self.llm_provider is None:
                logger.warning(f"LLM provider '{llm_to_use}' not found. Using first available...")
                self.llm_provider = next(iter(llm_providers.values())) if llm_providers else None

            if self.llm_provider is None:
                raise ValueError("No LLM providers available.")

        except Exception as e:
            logger.error(f"Error selecting LLM provider: {e}")
            self.llm_provider = None

        # 2. Inicializar as ferramentas (APENAS UMA VEZ)
        self.data_stats_tool = DataStatsTool(data_processor=data_processor)
        self.sentiment_aggregation_tool = SentimentAggregationTool()
                
                # CORREÇÃO CRÍTICA: Passe o llm_provider para a ferramenta
        self.insight_generation_tool = InsightGenerationTool(llm_provider=self.llm_provider)

        logger.info(f"WorkflowNodes initialized with LLM: {self.llm_provider.model if self.llm_provider else 'None'}")
    

    def load_data(self, state: AgentState) -> AgentState:
        """
        Nó 1: Carrega e pré-processa os dados de comentários de clientes.

        Este nó:
        1. Carrega dados do CSV
        2. Limpa e processa os dados
        3. Gera resumo dos dados
        4. Inicializa as ferramentas com os dados

        Args:
            state: Estado atual do workflow

        Returns:
            Estado atualizado com dados carregados
        """
        try:
            df = self.data_processor.load_customer_comments()
            logger.info(f"  Dados carregados: {len(df)} registros")

            # Get data summary
            data_summary = self.data_processor.get_data_summary(df)
            logger.info(f"  Data summary: {data_summary['total_records']} records")
            
            # Set data for tools
            self.data_stats_tool.set_data(df)

            # Update state
            state["data"] = df
            state["analysis_results"]["data_summary"] = data_summary
            state["current_step"] = "sentiment_analysis"

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(f"{error_msg}")
            state = update_state_with_error(state, error_msg, "load_data")

        return state

    def analyze_sentiment(self, state: AgentState) -> AgentState:
        """Analyze sentiment of comments."""
        try:
            df = state.get("data")
            if df is None or len(df) == 0:
                raise ValueError("No data available")

            logger.info("Analyzing sentiment of comments...")
            
            # Flatten list of lists
            comments = df["comment"].tolist()
            
            prompts = [
                SentimentAnalysisPrompts.BASIC_SENTIMENT.format(feedback=comment)
                for comment in comments[:10] 
            ]
            
            responses = self.llm_provider.generate_batch(prompts)
            
            sentiments = []
            for response in responses:
                try:
                    import json
                    # Parse JSON from LLM response
                    sentiment_data = json.loads(response.content)
                    sentiments.append(sentiment_data)
                except json.JSONDecodeError:
                    logger.warning("Could not parse sentiment response")
                    sentiments.append({
                        "sentiment": "neutral",
                        "confidence": 0.5,
                        "reasoning": "Parse error"
                    })
            
            positive_count = sum(1 for s in sentiments if s.get("sentiment") == "positive")
            negative_count = sum(1 for s in sentiments if s.get("sentiment") == "negative")
            neutral_count = len(sentiments) - positive_count - negative_count
            
            avg_confidence = sum(s.get("confidence", 0) for s in sentiments) / len(sentiments) if sentiments else 0
            
            state["sentiment_analysis"] = {
                "total_analyzed": len(sentiments),
                "positive": {"count": positive_count, "percentage": round(positive_count/len(sentiments)*100, 2) if sentiments else 0},
                "negative": {"count": negative_count, "percentage": round(negative_count/len(sentiments)*100, 2) if sentiments else 0},
                "neutral": {"count": neutral_count, "percentage": round(neutral_count/len(sentiments)*100, 2) if sentiments else 0},
                "average_confidence": round(avg_confidence, 3),
                "overall_sentiment": "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
            }
            
            state["raw_sentiments"] = sentiments
            
            logger.info(f"Sentiment analysis complete: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
            
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
            stats = {}
            
            logger.info("  Calculating statistics...")
            
            stats["count"] = self.data_stats_tool._run(metric="count")
            stats["avg_length"] = self.data_stats_tool._run(metric="avg_length")
            stats["word_frequency"] = self.data_stats_tool._run(metric="word_frequency")
            stats["rating_distribution"] = self.data_stats_tool._run(metric="rating_distribution")
            
            state["analysis_results"]["statistics"] = stats
            state["current_step"] = "topic_extraction"
            
            logger.info("  Statistics Calculated")
            logger.info(f"     Total comments: {stats['count']['count']}")
            logger.info(f"     Average length: {stats['avg_length']['avg_length']} chars\n")

        except Exception as e:
            error_msg = f"Error calculating statistics: {str(e)}"
            logger.error(f"❌ {error_msg}")
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
            df = state.get("data")
            if df is None or len(df) == 0:
                logger.warning("No data for topic extraction")
                return state

            logger.info("Extracting topics from comments...")

            comments = df["comment"].tolist()[:10] 
            prompts = [
                TopicExtractionPrompts.EXTRACT_TOPICS.format(comment=comment)
                for comment in comments
            ]
            
            responses = self.llm_provider.generate_batch(prompts)

            all_topics = []
            all_issues = []
            all_praise = []
            
            for response in responses:
                try:
                    import json
                    topic_data = json.loads(response.content)
                    all_topics.extend(topic_data.get("main_topics", []))
                    all_issues.extend(topic_data.get("issues", []))
                    all_praise.extend(topic_data.get("praise", []))
                except json.JSONDecodeError:
                    logger.warning("Could not parse topic response")
            
            from collections import Counter
            topic_counter = Counter(all_topics)
            
            state["topic_extraction"] = {
                "aggregated_topics": [topic for topic, _ in topic_counter.most_common(5)],
                "top_issues": dict(Counter(all_issues).most_common(3)) if all_issues else {},
                "top_praise": dict(Counter(all_praise).most_common(3)) if all_praise else {},
                "total_topics_extracted": len(all_topics)
            }
            
            logger.info(f"Topics extracted: {len(all_topics)} total")
            
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
        all_topics = []
        all_issues = []
        all_praise = []
        
        for res in responses:
            try:
                content = res.content.strip()
                
                # Handle markdown code blocks
                if content.startswith("```json"):
                    content = content.replace("```json\n", "").replace("```", "")
                elif content.startswith("```"):
                    content = content.replace("```\n", "").replace("```", "")
                
                json_data = json.loads(content.strip())
                
                # Agregar tópicos
                all_topics.extend(json_data.get("main_topics", []))
                all_issues.extend(json_data.get("issues", []))
                all_praise.extend(json_data.get("praise", []))
                
            except Exception as e:
                logger.warning(f"   Failed to parse topic JSON: {e}")
        
        # Deduplicate and count
        from collections import Counter
        topic_counts = Counter(all_topics)
        issue_counts = Counter(all_issues)
        praise_counts = Counter(all_praise)
        
        return {
            "aggregated_topics": [t for t, _ in topic_counts.most_common(5)],
            "top_issues": dict(issue_counts.most_common(5)),
            "top_praise": dict(praise_counts.most_common(5)),
            "total_topics_extracted": len(all_topics)
        }

    def generate_final_summary(self, state: AgentState) -> AgentState:
        """
        Node 5 (Final): Generate final business report using LLM and InsightGenerationTool.
        
        README Phase 4, Task 5:
        - Collect all analysis results
        - Use InsightGenerationTool to generate insights (Python logic)
        - Use SummaryPrompts.FEEDBACK_SUMMARY (from Phase 2) to create final report
        - Call LLM to generate business summary
        """
        
        try:
            if self.llm_provider is None:
                raise ValueError("LLM provider not initialized.")
            
            # Collect all analysis results
            analysis_data = state.get("analysis_results", {})

            self.insight_generation_tool.set_analysis_results(analysis_data)
            
            # Get Python-based insights
            logger.info("  Generating insights using Python logic...")

            report_result = self.insight_generation_tool._run(insight_type="recommendations")
            
            if "error" in report_result:
                 raise Exception(report_result["error"])
            
            # 4. Salvar o relatório final no estado
            state["analysis_results"]["final_summary"] = report_result
            state["current_step"] = "completed"
            
            logger.info("  Relatório final gerado com sucesso!")
            logger.info(f"     Modelo usado: {report_result.get('model_used')}\n")

        except Exception as e:
            error_msg = f"Error generating final summary: {str(e)}"
            logger.error(f"{error_msg}")
            state = update_state_with_error(state, error_msg, "generate_final_summary")

        return state