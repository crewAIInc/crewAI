# TavilyMapTool

## Description

The `TavilyMapTool` allows CrewAI agents to map the structure of websites starting from a base URL using the Tavily API. It discovers pages, links, and site hierarchy without extracting full content, making it ideal for understanding site architecture.

## Installation

To use the `TavilyMapTool`, you need to install the `tavily-python` library:

```shell
pip install 'crewai[tools]' tavily-python
```

You also need to set your Tavily API key as an environment variable:

```bash
export TAVILY_API_KEY='your-tavily-api-key'
```

## Example

Here's how to initialize and use the `TavilyMapTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilyMapTool

# Ensure TAVILY_API_KEY is set in your environment
# os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

# Initialize the tool
map_tool = TavilyMapTool()

# Create an agent that uses the tool
mapper_agent = Agent(
    role='Site Mapper',
    goal='Map website structures and discover all pages',
    backstory='You are an expert at mapping website structures using the Tavily API.',
    tools=[map_tool],
    verbose=True
)

# Define a task for the agent
map_task = Task(
    description='Map the structure of https://docs.example.com to understand the site hierarchy.',
    expected_output='A JSON string containing the site structure and discovered URLs.',
    agent=mapper_agent
)

# Create and run the crew
crew = Crew(
    agents=[mapper_agent],
    tasks=[map_task],
    verbose=2
)

result = crew.kickoff()
print(result)
```

## Arguments

The `TavilyMapTool` accepts the following arguments:

### Agent-Settable (Runtime)

These parameters can be provided by the agent at runtime:

- `url` (str): **Required**. The base URL to start mapping from.
- `max_depth` (int, 1-5): Maximum depth to map (number of link hops from the base URL).
- `instructions` (str): Natural language instructions to guide the mapping.
- `allow_external` (bool): Whether to allow mapping external domains.

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
- `timeout` (float): The timeout for the map request in seconds. Default is 150.
- `include_usage` (bool): Whether to include credit usage information.
- `extra_kwargs` (dict): Additional keyword arguments to pass to tavily-python.

## Advanced Usage

```python
# Configure the tool with custom parameters
custom_mapper = TavilyMapTool(
    max_breadth=15,
    limit=200,
    select_paths=[r'/docs/.*', r'/guides/.*'],
    exclude_paths=[r'/api/.*', r'/internal/.*'],
    timeout=120
)

agent_with_custom_tool = Agent(
    role="Documentation Mapper",
    goal="Map documentation site structure",
    tools=[custom_mapper]
)
```

## Response Format

The tool returns a JSON string containing the site map with discovered URLs and their relationships. This includes the site hierarchy and link structure.

Refer to the [Tavily API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/map) for details on the response structure.

