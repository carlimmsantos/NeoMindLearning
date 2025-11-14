import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    openai_api_key: str
    google_api_key: str
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash-lite"
    max_tokens: int = 1000
    temperature: float = 1.0
    

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = "data"
    output_path: str = "output"
    batch_size: int = 100
    

@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig
    data: DataConfig
    debug: bool = True
    

def load_config() -> Config:
    """Load configuration from environment variables."""
    llm_config = LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    data_config = DataConfig(
        data_path=os.getenv("DATA_PATH", "data"),
        output_path=os.getenv("OUTPUT_PATH", "output"),
        batch_size=int(os.getenv("BATCH_SIZE", "100"))
    )
    
    return Config(
        llm=llm_config,
        data=data_config,
        debug=os.getenv("DEBUG", "True").lower() == "true"
    )
