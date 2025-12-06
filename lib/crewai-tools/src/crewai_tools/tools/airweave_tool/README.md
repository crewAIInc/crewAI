# AirweaveSearchTool

## Description

The **AirweaveSearchTool** enables CrewAI agents to search across your organization's data that has been synced through Airweave. Airweave makes any app searchable for your agent by syncing data from various sources with minimal configuration for agentic search.

This tool provides powerful search capabilities including:
- **Natural language queries** - Search using plain English
- **AI-powered answers** - Get generated answers based on search results
- **Query expansion** - Automatically generate query variations for better recall
- **Reranking** - Reorder results for improved relevance
- **Temporal relevance** - Weight recent content higher than older content
- **Advanced filtering** - Filter by metadata and other criteria

## Installation

Install CrewAI with tools support:

```bash
pip install 'crewai[tools]'
```

Or install the Airweave SDK directly:

```bash
pip install airweave-sdk
# or with uv
uv add airweave-sdk
```

## Configuration

### API Key

Set your Airweave API key as an environment variable:

```bash
export AIRWEAVE_API_KEY="your-api-key-here"
```

Or pass it directly when initializing the tool:

```python
from crewai_tools import AirweaveSearchTool

search_tool = AirweaveSearchTool(api_key="your-api-key-here")
```

### Environment Variables

- `AIRWEAVE_API_KEY` (required): Your Airweave API key

## Usage

### Basic Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import AirweaveSearchTool

# Initialize the tool
airweave_search = AirweaveSearchTool()

# Create an agent with the tool
research_agent = Agent(
    role='Research Analyst',
    goal='Find relevant information from company data',
    backstory='Expert at finding insights in organizational data',
    tools=[airweave_search],
    verbose=True
)

# Create a task
research_task = Task(
    description="""
    Search for information about Q4 sales performance in the sales-data-2024 collection.
    Provide a comprehensive summary of the findings.
    """,
    agent=research_agent,
    expected_output='Detailed summary of Q4 sales performance with key metrics and trends'
)

# Create and run the crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

### Advanced Example with Answer Generation

```python
from crewai import Agent, Task, Crew
from crewai_tools import AirweaveSearchTool

# Initialize with custom configuration
airweave_search = AirweaveSearchTool(
    max_content_length_per_result=2000,  # Longer content per result
    timeout=90  # Longer timeout for complex queries
)

# Create a specialized agent
data_analyst = Agent(
    role='Senior Data Analyst',
    goal='Extract actionable insights from company data',
    backstory='Expert analyst with deep knowledge of business metrics',
    tools=[airweave_search],
    verbose=True
)

# Task with specific parameters
analysis_task = Task(
    description="""
    Using the AirweaveSearchTool:
    1. Search the 'finance-reports' collection for "revenue trends Q4 2024"
    2. Use generate_answer=True to get an AI-generated summary
    3. Enable query expansion for better recall
    4. Set temporal_relevance=0.3 to slightly favor recent documents
    
    Provide a detailed analysis with key findings and recommendations.
    """,
    agent=data_analyst,
    expected_output='Comprehensive analysis with AI-generated insights and actionable recommendations'
)

crew = Crew(
    agents=[data_analyst],
    tasks=[analysis_task],
    verbose=True
)

result = crew.kickoff()
```

### Example with Multiple Collections

```python
from crewai import Agent, Task, Crew
from crewai_tools import AirweaveSearchTool

airweave_search = AirweaveSearchTool()

# Multi-source research agent
researcher = Agent(
    role='Multi-Source Researcher',
    goal='Gather comprehensive information from multiple data sources',
    backstory='Skilled at synthesizing information from diverse sources',
    tools=[airweave_search],
    verbose=True
)

# Task searching multiple collections
multi_collection_task = Task(
    description="""
    Research the impact of our Q4 marketing campaigns by:
    1. Searching 'marketing-campaigns-2024' collection for campaign details
    2. Searching 'sales-data-2024' collection for sales impact
    3. Searching 'customer-feedback' collection for customer sentiment
    
    Synthesize findings into a comprehensive report.
    """,
    agent=researcher,
    expected_output='Integrated report combining insights from all three collections'
)

crew = Crew(
    agents=[researcher],
    tasks=[multi_collection_task],
    verbose=True
)

result = crew.kickoff()
```

## Parameters

### Tool Initialization Parameters

- **`api_key`** (str, optional): Airweave API key. Defaults to `AIRWEAVE_API_KEY` environment variable.
- **`base_url`** (str, optional): Custom base URL for Airweave API. Defaults to production.
- **`framework_name`** (str, optional): Framework identifier. Defaults to "crewai".
- **`framework_version`** (str, optional): Framework version for tracking.
- **`timeout`** (int, optional): Request timeout in seconds. Defaults to 60.
- **`max_content_length_per_result`** (int, optional): Maximum content length per result. Defaults to 1000.

### Search Parameters (Passed to Agent)

When the agent uses the tool, it can specify:

- **`query`** (str, required): The search query text
- **`collection_id`** (str, required): The unique collection identifier (e.g., 'finance-data-2024')
- **`limit`** (int, optional): Maximum number of results to return. Defaults to 10.
- **`generate_answer`** (bool, optional): Generate an AI-powered answer. Defaults to False.
- **`expand_query`** (bool, optional): Generate query variations for better recall. Defaults to False.
- **`rerank`** (bool, optional): Reorder results for improved relevance. Defaults to True.
- **`temporal_relevance`** (float, optional): Weight for recent content (0-1). 0 = no recency effect, 1 = only recent items. Defaults to None.

## Response Format

The tool returns a JSON string with the following structure:

```json
{
  "query": "original search query",
  "results_count": 5,
  "completion": "AI-generated answer (if generate_answer=True)",
  "results": [
    {
      "content": "Document content...",
      "metadata": {
        "source": "source-name",
        "updated_at": "2024-12-01T10:00:00Z",
        ...
      },
      "score": 0.95
    }
  ]
}
```

## Best Practices

### 1. Collection Management

- Use descriptive collection IDs (e.g., `finance-reports-2024` instead of `collection-1`)
- Organize data into logical collections by source, department, or topic
- Keep collection IDs consistent across your organization

### 2. Query Optimization

- Use `expand_query=True` for exploratory searches to improve recall
- Enable `generate_answer=True` when you need summarized insights
- Use `rerank=True` (default) for most searches to improve relevance
- Set `temporal_relevance` (0.2-0.5) when recent information is more important

### 3. Performance

- Start with `limit=10` and adjust based on your needs
- Use `max_content_length_per_result` to control context window usage
- Monitor API timeouts and adjust the `timeout` parameter if needed

### 4. Error Handling

The tool includes built-in error handling and will return error information in the response:

```json
{
  "error": "Error message",
  "query": "original query",
  "collection_id": "collection-id"
}
```

### 5. Integration with CrewAI Memory

Combine with CrewAI's memory systems for enhanced context:

```python
from crewai import Agent, Crew
from crewai_tools import AirweaveSearchTool

airweave_search = AirweaveSearchTool()

agent = Agent(
    role='Knowledge Manager',
    goal='Build and maintain comprehensive knowledge',
    tools=[airweave_search],
    memory=True,  # Enable CrewAI memory
    verbose=True
)

# The agent will remember previous searches and insights
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `AIRWEAVE_API_KEY` is set or passed to the tool
2. **Collection Not Found**: Verify the collection ID exists in your Airweave account
3. **Timeout Errors**: Increase the `timeout` parameter for complex queries
4. **Import Error**: Install airweave-sdk with `pip install airweave-sdk`

### Debug Mode

Enable verbose logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Your code here
```

## Related Resources

- [Airweave Documentation](https://docs.airweave.ai)
- [Airweave Python SDK](https://github.com/airweave-ai/python-sdk)
- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI Tools Documentation](https://docs.crewai.com/core-concepts/tools/)

## Support

For issues related to:
- **The tool integration**: Open an issue in the CrewAI repository
- **Airweave API or SDK**: Contact Airweave support or open an issue in the python-sdk repository
- **Data syncing**: Refer to Airweave platform documentation

## License

This tool is part of the CrewAI Tools package and follows the same license as CrewAI.

