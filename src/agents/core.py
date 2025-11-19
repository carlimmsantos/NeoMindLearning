import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .workflow import WorkflowNodes
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
        llm_providers: Dict[str, Any],
        data_processor: DataProcessor,
        llm_to_use: str = "openai",
    ):
        """
        Initialize the data analysis agent with React Agent.

        Args:
            llm_providers: Dictionary of available LLM providers
            data_processor: Data processing utilities
            llm_to_use: Which LLM provider to use for analysis
        """
        self.llm_providers = llm_providers
        self.data_processor = data_processor
        self.llm_to_use = llm_to_use

        # Retrieve the specified LLM provider instance
        self.llm_provider = llm_providers.get(self.llm_to_use)

        # Initialize workflow nodes with LLM and data processing capabilities
        self.workflow_nodes = WorkflowNodes(
            data_processor=data_processor,
            llm_providers=llm_providers,
            llm_to_use=llm_to_use,
        )

        # Obtain the LLM instance and configure tools and workflow
        self._llm = llm_providers[llm_to_use].get_llm()
        self._setup_tools()
        self.graph = self._create_workflow_graph()
        self._create_agent()

    def _setup_tools(self):
        """
        Load customer comment data and initialize tools with dataset.
        
        Attempts to load data from CSV or falls back to sample data if unavailable.
        Propagates data to all workflow tools for consistent analysis.
        """
        try:
            # Check if data is already loaded, otherwise load from source
            if self.data_processor._data is None:
                df = self.data_processor.load_customer_comments()
                self.data_processor._data = df
            else:
                df = self.data_processor._data
            
            # Set data in tools
            self.workflow_nodes.data_stats_tool.set_data(df)
            logger.info(f"Loaded {len(df)} customer comments for analysis")
            
        except FileNotFoundError:
            logger.warning("CSV file not found. Using sample data.")
            df = self.data_processor._create_sample_comment_data()
            self.data_processor._data = df
            self.workflow_nodes.data_stats_tool.set_data(df)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _create_workflow_graph(self):
        """
        Construct the LangGraph workflow defining the analysis pipeline.
        
        Establishes a linear execution flow through five sequential analysis stages:
        load_data → analyze_sentiment → calculate_statistics → generate_insights → generate_final_summary
        
        Returns:
            Compiled LangGraph representing the complete analysis workflow.
        """
        workflow = StateGraph(AgentState)

        # Register workflow nodes for each analysis stage
        workflow.add_node("load_data", self.workflow_nodes.load_data)
        workflow.add_node("analyze_sentiment", self.workflow_nodes.analyze_sentiment)
        workflow.add_node(
            "calculate_statistics", self.workflow_nodes.calculate_statistics
        )
        workflow.add_node("generate_insights", self.workflow_nodes.generate_insights)
        workflow.add_node(
            "generate_final_summary", self.workflow_nodes.generate_final_summary
        )

        # Define entry point for workflow execution
        workflow.set_entry_point("load_data")

        # Establish sequential edges connecting analysis stages
        workflow.add_edge("load_data", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "calculate_statistics")
        workflow.add_edge("calculate_statistics", "generate_insights")
        workflow.add_edge("generate_insights", "generate_final_summary")
        workflow.add_edge("generate_final_summary", END)

        return workflow.compile()

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
            Dictionary containing analysis results, status, and success indicators.
        """
        try:
            logger.info("Starting analysis workflow...")
            
            # Initialize state with default values
            initial_state = create_initial_state()

            # Inject loaded data into workflow state
            initial_state["data"] = self.data_processor._data
            
            # Execute the compiled workflow graph
            final_state = self.graph.invoke(initial_state)
            
            logger.info("Analysis completed successfully")
            
            return {
                "analysis_results": final_state.get("analysis_results", {}),
                "success": True,
                "status": "completed"
            }
        
        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "success": False,
                "analysis_results": {},
            }

    def get_available_tools(self) -> list:
        """
        Retrieve metadata about available analysis tools.

        Provides introspection capabilities for debugging and tool inspection.
        Documents tool names, descriptions, and supported capabilities.

        Returns:
            List of dictionaries containing tool specifications and available metrics.
        """
        return [
            {
                "name": "calculate_data_stats",
                "description": "Calculate statistical metrics on customer comments",
                "available_metrics": [
                    "count",
                    "avg_length",
                    "word_frequency",
                    "rating_distribution",
                ],
            },
            {
                "name": "aggregate_sentiment",
                "description": "Aggregate sentiment analysis results",
                "available_aggregations": ["summary", "distribution", "trends"],
            },
            {
                "name": "generate_insights",
                "description": "Generate actionable business insights",
                "available_insights": ["recommendations", "trends", "priorities"],
            },
        ]

    def _create_agent(self):
        """
        Initialize the React Agent with configured tools and prompt template.

        Constructs a LangChain React Agent that automatically selects and executes
        appropriate tools based on analysis requirements. Implements error handling
        and iteration limits for robust execution.

        Note: Skips agent creation when MockLLMProvider is detected for testing environments.
        """
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import PromptTemplate

        # Skip agent creation in testing environments with mock providers
        if not hasattr(self.llm_provider, 'invoke'):
            logger.warning("MockLLMProvider detected. Skipping ReAct agent creation for testing.")
            self.agent = None
            return        

        try:
            # Aggregate all analysis tools for agent access
            tools = [
                self.workflow_nodes.data_stats_tool,
                self.workflow_nodes.sentiment_aggregation_tool,
                self.workflow_nodes.insight_generation_tool,
            ]

            # Define the React Agent prompt template with structured format
            prompt = PromptTemplate.from_template("""
            You are a customer feedback analysis expert. Your goal is to analyze customer comments data.

            You have access to the following tools:
            {tools}

            Use this format:
            Question: [{input}]
            Thought: You should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            {agent_scratchpad}
            """)
            

            # Create React Agent with LLM and tools
            agent = create_react_agent(llm=self._llm, tools=tools, prompt=prompt)
            

            # Wrap agent with executor for robust execution and error handling
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True,
            )
            
            logger.info("React Agent created successfully")

        except Exception as e:
            logger.error(f"Failed to create React Agent: {e}")
            self.agent_executor = None