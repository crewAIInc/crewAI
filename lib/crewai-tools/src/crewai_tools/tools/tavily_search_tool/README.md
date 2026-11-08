# Tavily Search Tool

## Description

The `TavilySearchTool` provides an interface to the Tavily Search API, enabling CrewAI agents to perform comprehensive web searches. It allows for specifying search depth, topics, time ranges, included/excluded domains, and whether to include direct answers, raw content, or images in the results. The tool returns the search results as a JSON string.

## Installation

To use the `TavilySearchTool`, you need to install the `tavily-python` library:

```shell
uv add 'crewai[tools]' 'tavily-python>=0.7.24,<0.8'
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
    verbose=True
)

result = crew.kickoff()
print(result)

# Example of using specific parameters
detailed_tool = TavilySearchTool(
    search_depth="advanced",
    topic="general",
    max_results=5,
    include_answer=True
)
print(detailed_tool.run(query="What are the recent advancements in large language models?"))
```

## Arguments

Pass only `query` (required, str) to `run`, `arun`, or the agent tool call. Configure the following settings on `TavilySearchTool(...)`:
- `search_depth` (Literal["basic", "advanced", "fast", "ultra-fast"], optional): The depth of the search. Defaults to `"basic"`.
- `topic` (Literal["general", "news", "finance"], optional): The topic to focus the search on. Defaults to `"general"`.
- `time_range` (Literal["day", "week", "month", "year", "d", "w", "m", "y"], optional): The time range for the search. Defaults to `None`.
- `days` (int, optional): The number of days to search back. Relevant if `time_range` is not set. Defaults to `7`. The implicit default is omitted when `start_date` or `end_date` is configured; explicitly setting `days` together with either date bound raises a validation error.
- `max_results` (int, optional): The maximum number of search results to return. Defaults to `5`.
- `include_domains` (Sequence[str], optional): A list of domains to restrict the search to, unless `include_domains_mode="prefer"`. Defaults to `None`.
- `exclude_domains` (Sequence[str], optional): A list of domains to exclude from the search. Defaults to `None`.
- `include_answer` (Union[bool, Literal["basic", "advanced"]], optional): Whether to include a direct answer synthesized from the search results. Defaults to `False`.
- `include_raw_content` (bool or `"markdown"`/`"text"`, optional): Whether to include extracted page content, optionally selecting its format. Defaults to `False`.
- `include_images` (bool, optional): Whether to include image results. Defaults to `False`.
- `timeout` (float, optional): The request timeout in seconds. Defaults to `60`.
- `api_key` (str, optional): Your Tavily API key. If not provided, it's read from the `TAVILY_API_KEY` environment variable.
- `proxies` (dict[str, str], optional): A dictionary of proxies to use for the API request. Defaults to `None`.

## Optional Search Controls

These settings default to `None` and are omitted from requests unless configured. Explicit `False` is forwarded in both sync and async calls.

| Setting | Purpose |
| --- | --- |
| `chunks_per_source` | 1–3 content chunks per source for basic, advanced, or fast searches. Unavailable for ultra-fast. |
| `include_domains_mode` | `"restrict"` or `"prefer"`; requires `include_domains`. |
| `start_date`, `end_date` | Date boundaries in `YYYY-MM-DD` format. |
| `country` | Country to boost; only available for the general topic. |
| `language`, `filter_by_language` | Boost a language, or require it with `filter_by_language=True`. Filtering requires `language`. |
| `auto_parameters` | Let Tavily choose unspecified settings. The tool still sends its configured defaults, including basic depth and five results, which take precedence. |
| `exact_match` | Require phrases quoted in the query. |
| `safe_search` | Filter unsafe content; unavailable for fast and ultra-fast. |
| `include_image_descriptions` | Include image descriptions when `include_images=True`. |
| `include_favicon`, `include_usage` | Include favicons or credit usage metadata. |
| `include_published_date`, `filter_by_published_date` | Beta, best-effort publication dates. With a date filter, the latter removes undated results and also enables publication dates. |

```python
custom_tool = TavilySearchTool(
    search_depth="fast",
    max_results=5,
    include_domains=["example.com"],
    include_domains_mode="prefer",
    language="en",
    filter_by_language=True,
    include_usage=True,
)
print(custom_tool.run(query="Recent product announcements"))
# In async code: await custom_tool.arun(query="Recent product announcements")
```

The agent supplies the query; it does not override these constructor settings. Each result's `content` is truncated to `max_content_length_per_result` (default 1000 characters) plus `...` when needed. Other response fields, including raw content and usage, are preserved in the JSON string. SDK errors propagate to the caller.

See the [Search API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search) for parameter semantics and response fields.
