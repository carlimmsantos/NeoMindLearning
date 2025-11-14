import logging
from typing import Dict, Any

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from .tools import DataStatsTool, SentimentAggregationTool, InsightGenerationTool
from ..llm import BaseLLMProvider
from ..data import DataProcessor

logger = logging.getLogger(__name__)


class DataAnalysisAgent:
    """
    Simplified AI agent for analyzing customer comments data using React Agent.
    
    This agent uses LangChain's built-in React Agent instead of complex LangGraph workflows
    to automatically determine which tools to use and when to use them for:
    1. Loading and preprocessing customer comment data
    2. Statistical analysis using custom tools
    3. Sentiment analysis and aggregation
    4. Business insight generation
    5. Comprehensive summary creation
    """
    
    def __init__(
        self, 
        llm_providers: Dict[str, BaseLLMProvider], 
        data_processor: DataProcessor,
        llm_to_use: str = "openai"
    ):
        """
        Initialize the data analysis agent with React Agent.
        
        Args:
            llm_providers: Dictionary of available LLM providers
            data_processor: Data processing utilities
            llm_to_use: Which LLM provider to use for analysis
        """
        self._llm_providers = llm_providers
        self._data_processor = data_processor
        self._llm_to_use = llm_to_use
        
        # Get the LLM instance
        self._llm = self._llm_providers[llm_to_use].get_llm()
        
        # Initialize tools
        self.data_stats_tool = DataStatsTool(data_processor)
        self.sentiment_aggregation_tool = SentimentAggregationTool()
        self.insight_generation_tool = InsightGenerationTool()
        
        # Load data and set up tools
        self._setup_tools()
        
        # Create the React Agent
        self._create_agent()
    
    def _setup_tools(self):
        """Load data and configure tools."""
        try:
            df = self._data_processor.load_customer_comments()
            self.data_stats_tool.set_data(df)
            logger.info(f"Loaded {len(df)} customer comments for analysis")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _create_agent(self):
        """Create the React Agent with tools."""
        # Define the tools available to the agent
        tools: list[BaseTool] = [
            self.data_stats_tool,
            self.sentiment_aggregation_tool,
            self.insight_generation_tool
        ]
        
        # Create a prompt template for the React Agent
        prompt = PromptTemplate.from_template("""
You are a customer feedback analysis expert. Your goal is to analyze customer comments data and provide comprehensive insights.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Always start by getting basic statistics about the dataset, then proceed with sentiment analysis, and finally generate business insights.
Be thorough in your analysis and provide actionable recommendations.

Question: {input}
Thought: {agent_scratchpad}
""")
        
        # Create the React Agent
        agent = create_react_agent(
            llm=self._llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create the AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
    
    
    def analyze(self, query: str = None) -> Dict[str, Any]:
        """
        Execute the complete data analysis workflow using React Agent.
        
        This method runs the entire analysis pipeline using the React Agent which
        automatically determines which tools to call and when:
        1. Loads customer comment data (done in initialization)
        2. Performs statistical analysis using tools
        3. Conducts sentiment analysis and aggregation
        4. Generates business insights and recommendations
        5. Creates a comprehensive summary
        
        Args:
            query: Optional specific question to analyze. If None, runs full analysis.
        
        Returns:
            Dict containing all analysis results and final summary
        """
        if query is None:
            query = """
            Please perform a comprehensive analysis of the customer comments data:
            
            1. First, get basic statistics about the dataset (count, average length, word frequency)
            2. Then, analyze the sentiment distribution and trends
            3. Finally, generate actionable business insights and recommendations
            
            Provide a summary of key findings and prioritized recommendations for business action.
            """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            
            logger.info("Analysis workflow completed successfully")
            return {
                "final_answer": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "success": False,
                "analysis_results": {}
            }
    
    def get_available_tools(self) -> list:
        """
        Get list of available tools for debugging and inspection.
        
        Returns:
            List of tool names and descriptions
        """
        tools = [
            self.data_stats_tool,
            self.sentiment_aggregation_tool,
            self.insight_generation_tool
        ]
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if tool.args_schema else None
            }
            for tool in tools
        ]