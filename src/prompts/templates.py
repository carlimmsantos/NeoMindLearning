from typing import List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Base prompt template class."""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")


class SentimentAnalysisPrompts:
    """Prompts for sentiment analysis tasks."""
    
    BASIC_SENTIMENT = PromptTemplate(
        template="""Analyze the sentiment of this customer feedback:

                    Customer Review: {feedback}

                    Provide a JSON response with:
                        - sentiment: "positive", "negative", or "neutral"
                        - confidence: 0.0 to 1.0
                        - reasoning: brief explanation
                        - key_emotions: list of detected emotions

                    Response:""",
        input_variables=["feedback"]
    )


class TopicExtractionPrompts:
    """Prompts for topic extraction tasks."""
    
    EXTRACT_TOPICS = PromptTemplate(
        template="""Extract key topics and themes from this customer comment:

                    Comment: {comment}

                    Identify:
                        - main_topics: primary subjects discussed
                        - product_aspects: specific features/aspects mentioned
                        - issues: problems or complaints raised
                        - praise: positive aspects highlighted

                    Format as JSON:""",
        input_variables=["comment"]
    )


class SummaryPrompts:
    """Prompts for generating business summaries and insights."""

    FEEDBACK_SUMMARY = PromptTemplate(
        template="""Create a business summary from these customer insights:

            Data Summary: {data_summary}
            Sentiment Analysis: {sentiment_results}
            Topic Analysis: {topic_results}

            Generate:
                    1. Executive Summary (2-3 sentences)
                    2. Key Findings (3-5 bullet points)
                    3. Actionable Recommendations (3-4 specific actions)
                    4. Priority Issues (ranked by impact)

            Format as structured business report:""",
        input_variables=["data_summary", "sentiment_results", "topic_results"]
    )