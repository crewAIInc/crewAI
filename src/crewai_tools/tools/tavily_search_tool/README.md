# Tavily Search Tool

## Description

The `TavilySearchTool` provides an interface to the Tavily Search API, enabling CrewAI agents to perform comprehensive web searches. It allows for specifying search depth, topics, time ranges, included/excluded domains, and whether to include direct answers, raw content, or images in the results. The tool returns the search results as a JSON string.

## Installation

To use the `TavilySearchTool`, you need to install the `tavily-python` library:

```shell
pip install 'crewai[tools]' tavily-python
```

## Environment Variables

Ensure your Tavily API key is set as an environment variable:

```bash
export TAVILY_API_KEY='your_tavily_api_key'
```

## Example

Here's how to initialize and use the `TavilySearchTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool

# Ensure the TAVILY_API_KEY environment variable is set
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# Initialize the tool
tavily_tool = TavilySearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role='Market Researcher',
    goal='Find information about the latest AI trends',
    backstory='An expert market researcher specializing in technology.',
    tools=[tavily_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description='Search for the top 3 AI trends in 2024.',
    expected_output='A JSON report summarizing the top 3 AI trends found.',
    agent=researcher
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

result = crew.kickoff()
print(result)

# Example of using specific parameters
detailed_search_result = tavily_tool.run(
    query="What are the recent advancements in large language models?",
    search_depth="advanced",
    topic="general",
    max_results=5,
    include_answer=True
)
print(detailed_search_result)
```

## Arguments

The `TavilySearchTool` accepts the following arguments during initialization or when calling the `run` method:

- `query` (str): **Required**. The search query string.
- `search_depth` (Literal["basic", "advanced"], optional): The depth of the search. Defaults to `"basic"`.
- `topic` (Literal["general", "news", "finance"], optional): The topic to focus the search on. Defaults to `"general"`.
- `time_range` (Literal["day", "week", "month", "year"], optional): The time range for the search. Defaults to `None`.
- `days` (int, optional): The number of days to search back. Relevant if `time_range` is not set. Defaults to `7`.
- `max_results` (int, optional): The maximum number of search results to return. Defaults to `5`.
- `include_domains` (Sequence[str], optional): A list of domains to prioritize in the search. Defaults to `None`.
- `exclude_domains` (Sequence[str], optional): A list of domains to exclude from the search. Defaults to `None`.
- `include_answer` (Union[bool, Literal["basic", "advanced"]], optional): Whether to include a direct answer synthesized from the search results. Defaults to `False`.
- `include_raw_content` (bool, optional): Whether to include the raw HTML content of the searched pages. Defaults to `False`.
- `include_images` (bool, optional): Whether to include image results. Defaults to `False`.
- `timeout` (int, optional): The request timeout in seconds. Defaults to `60`.
- `api_key` (str, optional): Your Tavily API key. If not provided, it's read from the `TAVILY_API_KEY` environment variable.
- `proxies` (dict[str, str], optional): A dictionary of proxies to use for the API request. Defaults to `None`.

## Custom Configuration

You can configure the tool during initialization:

```python
# Example: Initialize with a default max_results and specific API key
custom_tavily_tool = TavilySearchTool(
    api_key="YOUR_SPECIFIC_TAVILY_KEY",
    config={
        'max_results': 10,
        'search_depth': 'advanced'
    }
)

# The agent will use these defaults unless overridden in the task input
agent_with_custom_tool = Agent(
    # ... agent configuration ...
    tools=[custom_tavily_tool]
)
```

Note: The `config` dictionary allows setting default values for the arguments defined in `TavilySearchToolSchema`. These defaults can be overridden when the tool is executed if the specific parameters are provided in the agent's action input.
