import argparse
import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.settings import LLMConfig
from src.data.processor import DataProcessor
from src.llm.providers import OpenAIProvider, GeminiProvider
from src.agents.core import DataAnalysisAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Main class for running the customer feedback analysis workflow."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the analysis runner with configuration."""
        self.config = config or self._load_config()
        self.data_processor = DataProcessor()
        self.llm_providers = self._initialize_llm_providers()
        self.agent = None
        
    def _load_config(self) -> LLMConfig:
        """Load configuration from environment variables."""
        try:
            return LLMConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                temperature=float(os.getenv("TEMPERATURE", "0.7"))
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Please ensure your .env file is properly configured")
            raise
    
    def _initialize_llm_providers(self) -> Dict[str, Any]:
        """Initialize LLM providers based on available API keys."""
        providers = {}
        
        # Initialize OpenAI provider
        if self.config.openai_api_key:
            try:
                providers["openai"] = OpenAIProvider(
                    api_key=self.config.openai_api_key,
                    model=self.config.openai_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                logger.info("✅ OpenAI provider initialized successfully")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize OpenAI provider: {e}")
        
        # Initialize Gemini provider
        if self.config.google_api_key:
            try:
                providers["gemini"] = GeminiProvider(
                    api_key=self.config.google_api_key,
                    model=self.config.gemini_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                logger.info("✅ Gemini provider initialized successfully")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize Gemini provider: {e}")
        
        if not providers:
            raise ValueError("No LLM providers could be initialized. Check your API keys.")
        
        return providers
    
    def run_data_analysis(self, use_sample_data: bool = False) -> Dict[str, Any]:
        """
        Run the complete data analysis workflow.
        
        Args:
            use_sample_data: If True, use generated sample data instead of CSV
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("🚀 Starting customer feedback analysis workflow...")
        
        try:
            # Initialize the agent
            primary_llm = list(self.llm_providers.keys())[0]
            self.agent = DataAnalysisAgent(
                llm_providers=self.llm_providers,
                data_processor=self.data_processor,
                llm_to_use=primary_llm
            )
            
            # Display available tools
            logger.info("🛠️  Available analysis tools:")
            for tool in self.agent.get_available_tools():
                logger.info(f"   - {tool['name']}: {tool['description'][:60]}...")
            
            # Run the analysis
            start_time = time.time()
            results = self.agent.analyze()
            analysis_time = time.time() - start_time
            
            # Add metadata
            results["metadata"] = {
                "analysis_time_seconds": round(analysis_time, 2),
                "primary_llm": primary_llm,
                "available_providers": list(self.llm_providers.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "metadata": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            }
    
    def save_results(self, results: Dict[str, Any], output_path: str = "results") -> None:
        """Save analysis results to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = output_dir / f"analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {results_file}")
        
        # Save human-readable summary if available
        if "analysis_results" in results and "final_summary" in results["analysis_results"]:
            summary_file = output_dir / f"summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write("Customer Feedback Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(str(results["analysis_results"]["final_summary"]))
            
            logger.info(f"📄 Summary saved to {summary_file}")
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of results to console."""
        print("\n" + "="*60)
        print("🎯  CUSTOMER FEEDBACK ANALYSIS RESULTS")
        print("="*60)
        
        # Print metadata
        if "metadata" in results:
            metadata = results["metadata"]
            print(f"📊 Analysis Time: {metadata.get('analysis_time_seconds', 'N/A')} seconds")
            print(f"🤖 Primary LLM: {metadata.get('primary_llm', 'N/A')}")
            print(f"🔧 Available Providers: {', '.join(metadata.get('available_providers', []))}")
            print(f"⏰ Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        # Print key results
        if "analysis_results" in results:
            analysis = results["analysis_results"]
            
            if "data_summary" in analysis:
                print(f"\n📈 Dataset Summary:")
                summary = analysis["data_summary"]
                for key, value in summary.items():
                    print(f"   {key}: {value}")
            
            if "final_summary" in analysis:
                print(f"\n📋 Final Summary:")
                print(f"   {analysis['final_summary']}")
        
        # Print errors if any
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
        
        print("\n" + "="*60)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Customer Feedback Analysis Tool")
    parser.add_argument("--demo", action="store_true", 
                       help="Run with sample data only")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to files")
    parser.add_argument("--output-dir", default="results",
                       help="Directory to save results (default: results)")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize the analysis runner
        logger.info("🎬 Initializing AI Analysis System...")
        runner = AnalysisRunner()
        
        # Run main analysis
        logger.info("🔍 Running customer feedback analysis...")
        results = runner.run_data_analysis(use_sample_data=args.demo)
        
        # Print results summary
        runner.print_results_summary(results)
        
        # Save results if requested
        if args.save_results:
            runner.save_results(results, args.output_dir)
        
        logger.info("🎉 Analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
