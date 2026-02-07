# Valyu Search Tool

## Description

The `ValyuSearchTool` provides an interface to the Valyu Search API, enabling CrewAI agents to perform unified searches across web, academic, financial, and proprietary data sources. It allows for specifying search types, relevance thresholds, date ranges, included/excluded sources, and response lengths. The tool returns the search results as a JSON string.

## Installation

To use the `ValyuSearchTool`, you need to install the `valyu` library:

```shell
pip install 'crewai[tools]' valyu
```

## Environment Variables

Ensure your Valyu API key is set as an environment variable:

```bash
export VALYU_API_KEY='your_valyu_api_key'
```

Get an API key at https://platform.valyu.ai/ (sign up, then create a key from the dashboard).

## Example

Here's how to initialize and use the `ValyuSearchTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import ValyuSearchTool

# Ensure the VALYU_API_KEY environment variable is set
# os.environ["VALYU_API_KEY"] = "YOUR_VALYU_API_KEY"

# Initialize the tool
valyu_tool = ValyuSearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role='Research Analyst',
    goal='Find comprehensive information from web and academic sources',
    backstory='An expert research analyst with access to diverse data sources.',
    tools=[valyu_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description='Search for the latest research on large language models.',
    expected_output='A comprehensive report summarizing key findings from web and academic sources.',
    agent=researcher
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
print(result)

# Example of using specific parameters
detailed_search_result = valyu_tool.run(
    query="What are the recent advancements in large language models?",
    search_type="all",
    max_num_results=10,
    relevance_threshold=0.7
)
print(detailed_search_result)
```

## Arguments

The `ValyuSearchTool` accepts the following arguments during initialization or when calling the `run` method:

- `query` (str): **Required**. The search query string.
- `search_type` (Literal["all", "web", "proprietary", "news"], optional): The type of search to perform. Defaults to `"all"`.
- `max_num_results` (int, optional): The maximum number of search results to return (1-20). Defaults to `10`.
- `relevance_threshold` (float, optional): Minimum relevance score for results (0.0-1.0). Higher values return more relevant results. Defaults to `0.5`.
- `included_sources` (Sequence[str], optional): A list of specific sources or domains to include in the search. Defaults to `None`.
- `excluded_sources` (Sequence[str], optional): A list of specific sources or domains to exclude from the search. Defaults to `None`.
- `start_date` (str, optional): Start date for filtering results (YYYY-MM-DD format). Defaults to `None`.
- `end_date` (str, optional): End date for filtering results (YYYY-MM-DD format). Defaults to `None`.
- `response_length` (Literal["short", "medium", "large", "max"], optional): Content length per result. Defaults to `"short"`.
- `country_code` (str, optional): 2-letter ISO country code to bias results geographically. Defaults to `None`.
- `api_key` (str, optional): Your Valyu API key. If not provided, it's read from the `VALYU_API_KEY` environment variable.

## Custom Configuration

You can configure the tool during initialization:

```python
# Example: Initialize with specific parameters for academic research
academic_valyu_tool = ValyuSearchTool(
    search_type='proprietary',
    max_num_results=15,
    relevance_threshold=0.7,
    response_length='medium'
)

# The agent will use these defaults
agent_with_custom_tool = Agent(
    role="Academic Researcher",
    goal="Find high-quality academic and research content",
    tools=[academic_valyu_tool]
)

# Example: Initialize for news search with date filtering
news_valyu_tool = ValyuSearchTool(
    search_type='news',
    max_num_results=10,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

## Data Sources

Valyu provides access to diverse data sources including:

- **Web**: Real-time web search across the internet
- **Academic**: arXiv, PubMed, scholarly papers, and research publications
- **Financial**: Stock prices, SEC filings, market metrics, and financial reports
- **Medical**: Peer-reviewed literature, clinical trials, FDA drug labels
- **Proprietary**: Licensed datasets and specialized content providers

Refer to the [Valyu API documentation](https://docs.valyu.ai/search/quickstart) for more details on the search API.