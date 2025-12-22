# TavilyCrawlTool

## Description

The `TavilyCrawlTool` allows CrewAI agents to crawl websites starting from a base URL using the Tavily API. It intelligently traverses links and extracts structured content from multiple pages at scale.

## Installation

To use the `TavilyCrawlTool`, you need to install the `tavily-python` library:

```shell
pip install 'crewai[tools]' tavily-python
```

You also need to set your Tavily API key as an environment variable:

```bash
export TAVILY_API_KEY='your-tavily-api-key'
```

## Example

Here's how to initialize and use the `TavilyCrawlTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilyCrawlTool

# Ensure TAVILY_API_KEY is set in your environment
# os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

# Initialize the tool
crawl_tool = TavilyCrawlTool()

# Create an agent that uses the tool
crawler_agent = Agent(
    role='Web Crawler',
    goal='Crawl websites and extract comprehensive content',
    backstory='You are an expert at crawling websites and extracting relevant content using the Tavily API.',
    tools=[crawl_tool],
    verbose=True
)

# Define a task for the agent
crawl_task = Task(
    description='Crawl the documentation at https://docs.example.com and extract all relevant content.',
    expected_output='A JSON string containing the crawled content from the website.',
    agent=crawler_agent
)

# Create and run the crew
crew = Crew(
    agents=[crawler_agent],
    tasks=[crawl_task],
    verbose=2
)

result = crew.kickoff()
print(result)
```

## Arguments

The `TavilyCrawlTool` accepts the following arguments:

### Agent-Settable (Runtime)

These parameters can be provided by the agent at runtime:

- `url` (str): **Required**. The base URL to start crawling from.
- `max_depth` (int, 1-5): Maximum depth to crawl (number of link hops from the base URL).
- `extract_depth` (Literal["basic", "advanced"]): Extraction depth for page content.
- `instructions` (str): Natural language instructions to guide the crawler.
- `allow_external` (bool): Whether to allow crawling external domains.

### User-Settable (Initialization)

These parameters are configured when creating the tool:

- `api_key` (str): Your Tavily API key. Defaults to the `TAVILY_API_KEY` environment variable.
- `proxies` (dict[str, str]): Proxies to use for the API requests.
- `max_breadth` (int): Maximum number of links to follow per page.
- `limit` (int): Total number of links to process before stopping.
- `select_paths` (Sequence[str]): Regex patterns to select specific paths.
- `select_domains` (Sequence[str]): Regex patterns to select specific domains.
- `exclude_paths` (Sequence[str]): Regex patterns to exclude specific paths.
- `exclude_domains` (Sequence[str]): Regex patterns to exclude specific domains.
- `include_images` (bool): Whether to include images in the results.
- `format` (Literal["markdown", "text"]): The format of the extracted content.
- `timeout` (float): The timeout for the crawl request in seconds. Default is 150.
- `include_favicon` (bool): Whether to include favicon URLs.
- `include_usage` (bool): Whether to include credit usage information.
- `chunks_per_source` (int, 1-5): Maximum number of content chunks per source.
- `extra_kwargs` (dict): Additional keyword arguments to pass to tavily-python.

## Advanced Usage

```python
# Configure the tool with custom parameters
custom_crawler = TavilyCrawlTool(
    max_breadth=10,
    limit=100,
    select_paths=[r'/docs/.*', r'/api/.*'],
    exclude_paths=[r'/private/.*'],
    include_images=True,
    format='markdown',
    timeout=120
)

agent_with_custom_tool = Agent(
    role="Documentation Crawler",
    goal="Crawl and extract documentation content",
    tools=[custom_crawler]
)
```

## Response Format

The tool returns a JSON string containing the crawled data from the website. The response includes extracted content from each page discovered during the crawl.

Refer to the [Tavily API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/crawl) for details on the response structure.

