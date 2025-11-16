"""
Workflow nodes for the DataAnalysisAgent.

This module contains all the individual workflow steps (nodes) that are
executed as part of the LangGraph workflow.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState, update_state_with_error
from .tools import DataStatsTool, SentimentAggregationTool, InsightGenerationTool
from ..data import DataProcessor
from ..prompts.templates import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Container class for all workflow node implementations."""
    
    def __init__(self, data_processor: DataProcessor, llm_providers: Dict[str, Any], llm_to_use: str):
        """Initialize workflow nodes with required dependencies."""
        self.data_processor = data_processor
        self.llm_providers = llm_providers
        self.llm_to_use = llm_to_use

        try:
            self.llm_provider = llm_providers.get(llm_to_use)
            if self.llm_provider is None:
                logger.warning(f"LLM provider '{llm_to_use}' not found")
                self.llm_provider = next(iter(llm_providers.values())) if llm_providers else None
        except Exception as e:
            logger.error(f"Error selecting LLM provider: {e}")
            self.llm_provider = None
        
        # Initialize tools
        self.data_stats_tool = DataStatsTool(data_processor=data_processor)
        self.sentiment_aggregation_tool = SentimentAggregationTool()
        self.insight_generation_tool = InsightGenerationTool()

        logger.info(f"WorkflowNodes initialized with LLM provider: {llm_to_use}")
    
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
        logger.info("=" * 70)
        logger.info("NÓ 1: load_data - Carregando dados")
        logger.info("=" * 70)
        
        try:
            # 1. Carregar dados do arquivo CSV
            df = self.data_processor.load_customer_comments()
            logger.info(f"Dados carregados: {len(df)} registros")
            
            # 2. Armazenar no estado
            state["data"] = df
            state["current_step"] = "data_loaded"
            
            # 3. Gerar resumo dos dados
            data_summary = self.data_processor.get_data_summary(df)
            state["analysis_results"]["data_summary"] = data_summary
            logger.info("Resumo dos dados gerado")
            logger.info(f"   - Total de registros: {data_summary['total_records']}")
            logger.info(f"   - Categorias: {list(data_summary.get('categories', {}).keys())}")
            
            # 4. Inicializar ferramentas com dados
            self.data_stats_tool.set_data(df)
            logger.info("Ferramentas inicializadas com dados")
            
            # 5. Limpar histórico de mensagens
            state["messages"] = []
            state["tool_calls_made"] = []
            
            logger.info("Nó load_data concluído com sucesso\n")
            
        except FileNotFoundError:
            error_msg = "Arquivo de dados não encontrado. Verifique o caminho do arquivo."
            logger.error(f"{error_msg}")
            state = update_state_with_error(state, error_msg, "load_data")
        
        except Exception as e:
            error_msg = f"Erro ao carregar dados: {str(e)}"
            logger.error(f"{error_msg}")
            state = update_state_with_error(state, error_msg, "load_data")
        
        return state
    
    def analyze_sentiment(self, state: AgentState) -> AgentState:
        """
        Nó 2: Realiza análise de sentimento nos comentários.
        
        Este nó:
        1. Extrai comentários dos dados
        2. Formata prompts de sentimento usando templates
        3. Processa sentimentos com a ferramenta
        4. Agrega resultados
        
        Args:
            state: Estado atual do workflow
            
        Returns:
            Estado atualizado com análise de sentimento
        """
        logger.info("=" * 70)
        logger.info("😊 NÓ 2: analyze_sentiment - Analisando sentimentos")
        logger.info("=" * 70)
        
        try:
            df = state.get("data")
            if df is None or len(df) == 0:
                raise ValueError("Nenhum dado disponível para análise de sentimento")
            
            # 1. Preparar dados simulados de sentimento
            # (Em um cenário real, você chamaria o LLM com o prompt)
            sentiment_results = []
            
            # Simular análise de sentimento com base em palavras-chave
            positive_words = ["ótimo", "excelente", "bom", "gostei", "adorei", "amo"]
            negative_words = ["ruim", "péssimo", "horrível", "não gostei", "decepcionante"]
            
            for idx, comment in enumerate(df["comment"].head(50)):  # Limitar a 50 para performance
                comment_lower = str(comment).lower()
                
                # Contar palavras positivas e negativas
                pos_count = sum(1 for word in positive_words if word in comment_lower)
                neg_count = sum(1 for word in negative_words if word in comment_lower)
                
                # Determinar sentimento
                if pos_count > neg_count:
                    sentiment = "positive"
                    confidence = min(0.95, 0.7 + (pos_count * 0.1))
                elif neg_count > pos_count:
                    sentiment = "negative"
                    confidence = min(0.95, 0.7 + (neg_count * 0.1))
                else:
                    sentiment = "neutral"
                    confidence = 0.6
                
                sentiment_results.append({
                    "text": comment,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "key_emotions": ["satisfaction"] if sentiment == "positive" else ["concern"] if sentiment == "negative" else []
                })
            
            # 2. Integrar com a ferramenta de agregação
            self.sentiment_aggregation_tool.set_sentiment_data(sentiment_results)
            
            # 3. Agregar sentimentos
            sentiment_summary = self.sentiment_aggregation_tool._run("summary")
            state["analysis_results"]["sentiment_analysis"] = sentiment_summary
            
            logger.info(f"✅ Análise de sentimento concluída")
            logger.info(f"   - Positivos: {sentiment_summary['positive']['count']} ({sentiment_summary['positive']['percentage']}%)")
            logger.info(f"   - Negativos: {sentiment_summary['negative']['count']} ({sentiment_summary['negative']['percentage']}%)")
            logger.info(f"   - Neutros: {sentiment_summary['neutral']['count']} ({sentiment_summary['neutral']['percentage']}%)")
            logger.info(f"   - Sentimento geral: {sentiment_summary['overall_sentiment'].upper()}\n")
            
            state["current_step"] = "sentiment_analyzed"
        
        except Exception as e:
            error_msg = f"Erro na análise de sentimento: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state = update_state_with_error(state, error_msg, "analyze_sentiment")
        
        return state

    def agent_with_tools(self, state: AgentState) -> AgentState:
        """
        Main agent logic with tool calling capabilities.
        
        TODO: Intern should implement this method to:
        1. Analyze the current state and determine what analysis is needed
        2. Create appropriate prompts for the LLM
        3. Handle LLM responses and tool calls
        4. Manage the conversation flow with tools
        5. Store results appropriately in the state
        """
        try:
            messages = state["messages"]
            current_step = state.get("current_step", "starting")
            
            # TODO: Intern must implement the agent logic
            # This should include:
            # 1. Determining what analysis to perform based on current step
            # 2. Creating prompts that encourage tool usage
            # 3. Calling the LLM with tool descriptions
            # 4. Processing LLM responses and tool calls
            
            if current_step == "data_loaded":
                # Create a prompt for data analysis
                prompt = """You are a data analysis expert. You have access to customer comments data and several tools to analyze it.

Available tools:
- calculate_data_stats: Get statistical metrics about the data
- aggregate_sentiment: Aggregate sentiment analysis results  
- generate_insights: Generate business insights from analysis

Start by getting basic statistics about the dataset, then proceed with sentiment analysis.
The data is already loaded and ready for analysis."""
                
                message = HumanMessage(content=prompt)
                messages.append(message)
                
                # TODO: Intern should implement LLM call with tools
                # This is where the LLM would be called with tool descriptions
                # and the response would be processed for tool calls
                
                # Placeholder response - intern should replace with actual LLM call
                ai_response = AIMessage(content="I'll analyze the data using the available tools.")
                messages.append(ai_response)
                
                state["current_step"] = "analyzing"
                
            state["messages"] = messages
            
        except Exception as e:
            logger.error(f"Error in agent with tools: {e}")
            state = update_state_with_error(state, str(e), "agent_with_tools")
        
        return state
    
    def generate_final_summary(self, state: AgentState) -> AgentState:
        """
        Generate final summary of all analysis results.
        
        TODO: Intern should implement comprehensive summary generation.
        """
        try:
            # TODO: Intern should implement final summary generation
            # This should combine all analysis results into a comprehensive summary
            
            state["analysis_results"]["final_summary"] = {
                "status": "completed",
                "note": "TODO: Intern must implement comprehensive summary generation"
            }
            state["current_step"] = "completed"
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            state = update_state_with_error(state, str(e), "generate_final_summary")
        
        return state
    
    def should_continue_with_tools(self, state: AgentState) -> str:
        """
        Determine if we should continue with tool calling or end the workflow.
        
        TODO: Intern should implement logic to:
        1. Check if the last message contains tool calls
        2. Determine if more analysis is needed
        3. Handle maximum tool call limits
        """
        messages = state["messages"]
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # TODO: Intern should implement proper tool call detection
        # This is a simplified check - the intern should implement:
        # - Proper tool call message detection
        # - Logic for when to stop calling tools
        # - Error handling for failed tool calls
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "end"
