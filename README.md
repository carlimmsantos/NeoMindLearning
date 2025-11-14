# AI Engineer Intern Take-Home Case Study

## Overview
This take-home project is designed to evaluate an AI Engineer intern candidate across six key competency areas:

1. **Code Writing Ability** - Python programming and software engineering practices
2. **Data Manipulation** - Working with datasets, preprocessing, and analysis
3. **LLM Integration** - Connecting and working with Large Language Models
4. **LLM Testing** - Comparing and evaluating different LLM providers
5. **AI Agent Creation** - Building intelligent agents using LangGraph
6. **Prompt Engineering** - Crafting effective prompts for various tasks

## Project Structure
```
takehome/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── config/          # Setting dataclasses
│   ├── data/            # Data processing modules
│   ├── llm/             # LLM integration and testing
│   └── prompts/         # Prompt templates and engineering
├── data/                # Sample datasets
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for exploration
└── requirements.txt     # Python dependencies
```

## Setup Instructions

1. **Environment Setup**
   ```bash
   python -m venv venv
   pip install -r requirements.txt
   ```

2. **API Keys Configuration**
   Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_gemini_key
   ```

3. **Running the Project**
   ```bash
   python -m src.main
   ```

4. **Running Tests**
   ```bash
   pytest tests/
   ```

## Tasks Overview

This take-home project involves building a complete AI agent system for customer feedback analysis. The system uses LangChain tools, LangGraph workflows, and multiple LLM providers to provide comprehensive business insights.

**Dataset**: `data/customer_comment.csv` contains 570+ customer reviews with fields:
- Review Title, Customer name, Rating (1-5 stars)
- Date, Category, Comments, Useful votes

**Architecture**: The codebase is organized into 4 main modules:
- `src/data/`: Data processing and cleaning
- `src/llm/`: LLM provider integrations (OpenAI, Gemini)
- `src/prompts/`: Prompt templates and engineering
- `src/agents/`: LangGraph agents with tool calling capabilities

---

## Step-by-Step Implementation Guide

*Note: You are free to follow whatever order you like, and change anything about the code*

### 🔧 **Phase 1: Environment Setup & Data Foundation**

#### Step 1: Environment Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env  # Edit with your API keys
```

#### Step 2: Data Processing Implementation
**File**: `src/data/processor.py`

**Objective**: Implement methods to load, clean, and analyze customer comment data.

**Tasks**:
1. **Complete `load_customer_comments()`**:
   ```python
   def load_customer_comments(self) -> pd.DataFrame:
       # Load data/customer_comment.csv
       # Handle encoding issues (CSV may have special characters)
       # Return cleaned DataFrame
   ```

2. **Implement `_clean_comments_data()`**:
   ```python
   def _clean_comments_data(self, df: pd.DataFrame) -> pd.DataFrame:
       # Standardize column names (lowercase, underscores)
       # Convert date strings to datetime objects
       # Parse rating strings ("4.0 out of 5 stars" → 4)
       # Handle missing values appropriately
       # Remove duplicates if any
   ```

3. **Complete `get_data_summary()`**:
   ```python
   def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
       # Return: total_records, rating_distribution, category_breakdown
       # avg_rating, date_range, comment_length_stats
   ```

**Testing**: Run `pytest tests/test_data.py -v` to verify implementation.

---

### 🎯 **Phase 2: Prompt Engineering**

#### Step 3: Prompt Template Implementation
**File**: `src/prompts/templates.py`

**Objective**: Create effective prompts for sentiment analysis, topic extraction, and summarization.

**Tasks**:
1. **Sentiment Analysis Prompt**:
   ```python
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
   ```

2. **Topic Extraction Prompt**:
   ```python
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
   ```

3. **Summary Generation Prompt**:
   ```python
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
   ```

**Testing**: Test prompts with sample data and refine based on LLM outputs.

### 🔗 **CRITICAL: Prompt Integration with Tools & Agents**

**⚠️ IMPORTANT**: The prompt templates you create in Step 3 must be integrated into your tools and workflow. Here's exactly how:

#### **Integration Points:**

1. **In Tools** (`src/agents/tools.py`):
   ```python
   # Import the prompts at the top of the file
   from ..prompts import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts
   
   class SentimentAggregationTool(BaseTool):
       def _run(self, aggregation_type: str) -> Dict[str, Any]:
           # Use the prompt template you created
           formatted_prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(
               feedback=self._sentiment_results
           )
           # Call LLM with the formatted prompt
           # Process and return results
   ```

2. **In Workflow** (`src/agents/workflow.py`):
   ```python
   # Import prompts
   from ..prompts import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts
   
   def generate_final_summary(self, state: AgentState) -> AgentState:
       # Use your summary prompt template
       summary_prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
           data_summary=state["analysis_results"]["data_summary"],
           sentiment_results=state["analysis_results"]["sentiment_analysis"],
           topic_results=state["analysis_results"]["topic_extraction"]
       )
       # Call LLM and process results
   ```

3. **Tool-LLM Integration Pattern**:
   ```python
   # Standard pattern for using prompts in tools
   def _run(self, param: str) -> Dict[str, Any]:
       # 1. Format the prompt template
       prompt = MyPromptTemplate.TEMPLATE_NAME.format(param=param)
       
       # 2. Call LLM (you'll need to access LLM provider)
       # This is where you'll need to implement LLM integration
       
       # 3. Parse and return structured results
       return {"result": "parsed_output", "prompt_used": prompt}
   ```

**🚨 Key Integration Requirements:**
- Tools must import and use the prompt templates (not hardcoded strings)
- Prompts must be formatted with actual data before calling LLMs
- LLM responses must be parsed and returned in structured format
- Each tool should use the appropriate prompt template for its function

---

### 🛠️ **Phase 3: Tool Implementation** 

#### Step 4: LangChain Tools Implementation
**File**: `src/agents/tools.py`

**Objective**: Implement three analysis tools that USE THE PROMPT TEMPLATES you created in Step 3.

**🔥 CRITICAL**: Your tools MUST use the prompt templates from `src/prompts/templates.py`. Do NOT use hardcoded strings.

**Tasks**:
1. **DataStatsTool Implementation**:
   ```python
   def _run(self, metric: str, column: str = "comment") -> Dict[str, Any]:
       if metric == "count":
           return {"count": len(self._current_data), "metric": metric}
       elif metric == "avg_length":
           return {
               "avg_length": self._current_data[column].str.len().mean(),
               "median_length": self._current_data[column].str.len().median(),
               "metric": metric
           }
       elif metric == "word_frequency":
           # Implement word frequency analysis using pandas/collections
           from collections import Counter
           all_text = ' '.join(self._current_data[column].astype(str))
           words = all_text.lower().split()
           word_freq = Counter(words).most_common(20)
           return {"word_frequency": dict(word_freq), "metric": metric}
       # Handle rating_distribution, category_breakdown, etc.
   ```

2. **SentimentAggregationTool Implementation - MUST USE PROMPTS**:
   ```python
   from ..prompts import SentimentAnalysisPrompts
   
   def _run(self, aggregation_type: str) -> Dict[str, Any]:
       if aggregation_type == "summary":
           # REQUIRED: Use the sentiment prompt template you created
           prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(
               feedback=str(self._sentiment_results)
           )
           
           # TODO: Call LLM with the formatted prompt
           # llm_response = self.llm_provider.generate(prompt)
           # Parse the JSON response and aggregate results
           
           # For now, return placeholder that shows prompt integration
           return {
               "aggregation_type": aggregation_type,
               "prompt_used": prompt,
               "status": "TODO: Implement LLM call and response parsing"
           }
   ```

3. **InsightGenerationTool Implementation - MUST USE PROMPTS**:
   ```python
   from ..prompts import SummaryPrompts
   
   def _run(self, insight_type: str) -> Dict[str, Any]:
       if insight_type == "recommendations":
           # REQUIRED: Use the summary prompt template you created
           prompt = SummaryPrompts.FEEDBACK_SUMMARY.format(
               data_summary=self._analysis_results.get("data_summary", {}),
               sentiment_results=self._analysis_results.get("sentiment_analysis", {}),
               topic_results=self._analysis_results.get("topic_extraction", {})
           )
           
           # TODO: Call LLM and generate business insights
           # llm_response = self.llm_provider.generate(prompt)
           
           return {
               "insight_type": insight_type,
               "prompt_used": prompt,
               "status": "TODO: Implement LLM call and insight generation"
           }
   ```

**🎯 Implementation Requirements:**
- **Import prompt templates**: `from ..prompts import SentimentAnalysisPrompts, TopicExtractionPrompts, SummaryPrompts`
- **Use `.format()` method**: Always format prompts with actual data
- **Access LLM providers**: You'll need to get access to LLM providers in tools (hint: pass via constructor or state)
- **Parse LLM responses**: Convert LLM text responses to structured dictionaries
- **Handle errors**: What if LLM returns malformed JSON or fails?

**Testing**: Test each tool individually AND verify they use your prompt templates correctly.

---

### 📊 **Phase 4: Analysis & Documentation**

#### Step 5: Performance Analysis & Documentation

**Tasks**:
1. **Create `docs/ANALYSIS.md`**:
   - LLM comparison results (accuracy, speed, cost)
   - Tool performance metrics
   - Business insights quality assessment

2. **Document challenges and solutions**:
   - Data quality issues encountered
   - Prompt engineering iterations
   - Tool calling implementation decisions

3. **Generate sample results**:
   - Run analysis on full dataset
   - Create example outputs for demonstration

---

## 🎯 **Success Criteria**

### Minimum Viable Product (4-5 hours):
- ✅ Data processing pipeline working
- ✅ All three tools implemented and functional
- ✅ Basic LLM integration with one provider
- ✅ Simple workflow executing end-to-end
- ✅ Unit tests passing

### Complete Implementation (5-8 hours):
- ✅ All tools with comprehensive functionality
- ✅ Both OpenAI and Gemini integration
- ✅ Advanced workflow with proper tool calling
- ✅ Comparative analysis documentation
- ✅ Error handling and edge cases covered

### Excellence (8+ hours):
- ✅ Performance optimizations (batch processing, caching)
- ✅ Advanced prompt engineering with iterations
- ✅ Comprehensive test coverage
- ✅ Business-ready insights and visualizations
- ✅ Cost optimization strategies

---

## 💡 **Pro Tips for Success**

1. **Start Small**: Implement basic functionality first, then enhance
2. **Test Early**: Test each component as you build it
3. **Use Logging**: Add logging statements to debug workflow issues
4. **Handle Errors**: Customer data can be messy - plan for edge cases
5. **Prompt Iterate**: Test prompts with real data and refine
6. **Time Management**: Allocate time for testing and documentation

---

## 📋 **Submission Checklist**

### **Required Deliverables**:
- [ ] **Working codebase** with all core functionality implemented
- [ ] **Data processing pipeline** (`src/data/processor.py`) fully functional
- [ ] **Three LangChain tools** (`src/agents/tools.py`) implemented
- [ ] **Prompt templates** (`src/prompts/templates.py`) for all three tasks
- [ ] **Unit tests passing** (minimum 80% of provided tests)
- [ ] **Integration test** demonstrating end-to-end workflow

### **Documentation Requirements**:
- [ ] **Analysis report** (`docs/ANALYSIS.md`) with LLM comparison results
- [ ] **Implementation notes** documenting key decisions and challenges
- [ ] **Sample outputs** from running the analysis on the dataset
- [ ] **Code comments** explaining complex logic, especially in tool calling

### **Quality Standards**:
- [ ] **Error handling** for API failures, data issues, malformed inputs
- [ ] **Input validation** for all tool parameters and data processing
- [ ] **Logging** throughout the application for debugging
- [ ] **PEP 8 compliance** and clean, readable code structure

### **Bonus Achievements** (Extra Credit):
- [ ] **Performance optimizations** (batch processing, caching, async calls)
- [ ] **Advanced prompt engineering** with documented iteration process
- [ ] **Comprehensive test coverage** (>90% with edge cases)
- [ ] **Cost analysis** and optimization strategies for API usage
- [ ] **Visualization dashboard** for analysis results
- [ ] **Streaming responses** implementation
- [ ] **Multiple evaluation metrics** for comparing LLM outputs

---

## ⏰ **Time Allocation Guide**

| **Phase** | **Time** | **Priority** | **Key Deliverables** |
|-----------|----------|--------------|---------------------|
| **Phase 1: Setup & Data** | 1-1.5 hrs | 🔴 Critical | Working data pipeline, tests passing |
| **Phase 2: Prompts** | 1-2 hrs | 🔴 Critical | All three prompt templates implemented |
| **Phase 3: Tool Implementation** | 2-3 hrs | 🔴 Critical | All three tools functional |
| **Phase 4: Documentation** | 1-1.5 hrs | 🟡 Important | Analysis report, code documentation |

---

## 🎯 **Evaluation Criteria**

Your submission will be evaluated on these dimensions:

### **Technical Implementation (40%)**
- **Code Quality**: Clean, readable, well-structured code following Python best practices
- **Functionality**: All required features implemented and working correctly
- **Error Handling**: Robust handling of edge cases, API failures, and data issues
- **Testing**: Comprehensive test coverage with meaningful assertions

### **AI Engineering (30%)**
- **LangChain Integration**: Proper use of LangChain tools and patterns
- **Tool Calling**: Effective implementation of agent-tool interaction
- **LLM Integration**: Successful integration with multiple providers
- **Workflow Design**: Logical and efficient agent workflow structure

### **Prompt Engineering (15%)**
- **Prompt Quality**: Effective prompts that generate useful outputs
- **Template Design**: Well-structured, reusable prompt templates
- **Iteration Process**: Evidence of prompt testing and refinement
- **Business Relevance**: Prompts generate actionable business insights

### **Data Engineering (10%)**
- **Data Processing**: Clean, efficient data loading and preprocessing
- **Data Quality**: Proper handling of real-world data messiness
- **Performance**: Efficient processing of the dataset
- **Validation**: Appropriate data validation and error checking

### **Documentation & Analysis (5%)**
- **Code Documentation**: Clear comments and docstrings
- **Analysis Quality**: Insightful comparison of LLM performance
- **Problem Solving**: Clear explanation of challenges and solutions
- **Business Value**: Demonstration of practical business applications

---

## 🆘 **Getting Help & Troubleshooting**

### **Common Issues & Solutions**:

1. **API Key Issues**:
   ```bash
   # Check if .env file is properly loaded
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY')[:10] if os.getenv('OPENAI_API_KEY') else 'Not found')"
   ```

2. **Import Errors**:
   ```bash
   # Ensure you're running from project root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python -m src.main
   ```

3. **Tool Calling Issues**:
   - Check that tools inherit from `BaseTool` correctly
   - Verify `args_schema` Pydantic models are properly defined
   - Ensure `_run()` method returns proper Dict format

4. **LLM Integration Problems**:
   - Test with simple prompts first
   - Check rate limits and API quotas
   - Implement retry logic for transient failures

### **Debug Commands**:
```bash
# Test data processing in isolation
python -c "from src.data import DataProcessor; dp = DataProcessor(); print(dp.load_customer_comments().head())"

# Test tool imports
python -c "from src.agents.tools import DataStatsTool; print('Success')"

# Validate prompt templates (CRITICAL TEST)
python -c "from src.prompts.templates import SentimentAnalysisPrompts; print('Template length:', len(SentimentAnalysisPrompts.BASIC_SENTIMENT.template))"

# Test prompt formatting (VERIFY INTEGRATION)
python -c "
from src.prompts.templates import SentimentAnalysisPrompts
prompt = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(feedback='Test review')
print('Formatted prompt:', prompt)
"

# Check if tools are using prompts (INTEGRATION TEST)
python -c "
from src.agents.tools import SentimentAggregationTool
tool = SentimentAggregationTool()
tool.set_sentiment_data({'test': 'data'})
try:
    result = tool._run('summary')
    print('Tool result contains prompt:', 'prompt_used' in result)
except NotImplementedError:
    print('Tool not yet implemented - this is expected')
"
```

### **Prompt Integration Troubleshooting**:

1. **Empty Prompt Templates**:
   ```bash
   # Check if your templates are still empty
   python -c "
   from src.prompts.templates import SentimentAnalysisPrompts
   template = SentimentAnalysisPrompts.BASIC_SENTIMENT.template
   if not template or template == '':
       print('❌ BASIC_SENTIMENT template is empty!')
   else:
       print('✅ Template has content:', len(template), 'characters')
   "
   ```

2. **Import Errors in Tools**:
   ```bash
   # Verify tools can import prompts
   python -c "
   try:
       from src.agents.tools import DataStatsTool
       from src.prompts import SentimentAnalysisPrompts
       print('✅ Imports working')
   except ImportError as e:
       print('❌ Import error:', e)
   "
   ```

3. **Prompt Formatting Issues**:
   ```bash
   # Test prompt formatting with sample data
   python -c "
   from src.prompts.templates import SentimentAnalysisPrompts
   try:
       formatted = SentimentAnalysisPrompts.BASIC_SENTIMENT.format(feedback='sample review')
       print('✅ Prompt formatting works')
       print('Preview:', formatted[:100])
   except KeyError as e:
       print('❌ Missing variable in template:', e)
   "
   ```

4. **Tool-Prompt Integration**:
   ```bash
   # Verify tools are using prompts (not hardcoded strings)
   python -c "
   import inspect
   from src.agents.tools import SentimentAggregationTool
   source = inspect.getsource(SentimentAggregationTool._run)
   if 'SentimentAnalysisPrompts' in source:
       print('✅ Tool imports prompts')
   else:
       print('❌ Tool not using prompt templates')
   "
   ```

### **Questions & Clarifications**:
If you encounter setup issues or need clarification on requirements, document your questions and assumptions in your submission. Show your problem-solving process even if you can't fully resolve an issue.

**Remember**: This is as much about demonstrating your approach to problem-solving as it is about the final implementation. Good luck! 🚀
