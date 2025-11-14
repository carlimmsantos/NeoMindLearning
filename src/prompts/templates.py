from typing import List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Base class for prompt templates."""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)


class SentimentAnalysisPrompts:
    """Prompts for sentiment analysis tasks."""
    
    BASIC_SENTIMENT = PromptTemplate(
        template="""""",
        input_variables=["feedback"]
    )


class TopicExtractionPrompts:
    """Prompts for topic extraction and categorization."""
    
    EXTRACT_TOPICS = PromptTemplate(
        template="""""",
        input_variables=["comment"]
    )
    

class SummaryPrompts:
    """Prompts for generating summaries and insights."""
    
    FEEDBACK_SUMMARY = PromptTemplate(
        template="""""",
        input_variables=["?????"]
    )

