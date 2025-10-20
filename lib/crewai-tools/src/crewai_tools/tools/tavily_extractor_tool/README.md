# TavilyExtractorTool

## Description

The `TavilyExtractorTool` allows CrewAI agents to extract structured content from web pages using the Tavily API. It can process single URLs or lists of URLs and provides options for controlling the extraction depth and including images.

## Installation

To use the `TavilyExtractorTool`, you need to install the `tavily-python` library:

```shell
pip install 'crewai[tools]' tavily-python
```

You also need to set your Tavily API key as an environment variable:

```bash
export TAVILY_API_KEY='your-tavily-api-key'
```

## Example

Here's how to initialize and use the `TavilyExtractorTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilyExtractorTool

# Ensure TAVILY_API_KEY is set in your environment
# os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY"

# Initialize the tool
tavily_tool = TavilyExtractorTool()

# Create an agent that uses the tool
extractor_agent = Agent(
    role='Web Content Extractor',
    goal='Extract key information from specified web pages',
    backstory='You are an expert at extracting relevant content from websites using the Tavily API.',
    tools=[tavily_tool],
    verbose=True
)

# Define a task for the agent
extract_task = Task(
    description='Extract the main content from the URL https://example.com using basic extraction depth.',
    expected_output='A JSON string containing the extracted content from the URL.',
    agent=extractor_agent,
    tool_inputs={
        'urls': 'https://example.com',
        'extract_depth': 'basic'
    }
)

# Create and run the crew
crew = Crew(
    agents=[extractor_agent],
    tasks=[extract_task],
    verbose=2
)

result = crew.kickoff()
print(result)

# Example with multiple URLs and advanced extraction
extract_multiple_task = Task(
    description='Extract content from https://example.com and https://anotherexample.org using advanced extraction.',
    expected_output='A JSON string containing the extracted content from both URLs.',
    agent=extractor_agent,
    tool_inputs={
        'urls': ['https://example.com', 'https://anotherexample.org'],
        'extract_depth': 'advanced',
        'include_images': True
    }
)

result_multiple = crew.kickoff(inputs={'urls': ['https://example.com', 'https://anotherexample.org'], 'extract_depth': 'advanced', 'include_images': True}) # If task doesn't specify inputs directly
print(result_multiple)

```

## Arguments

The `TavilyExtractorTool` accepts the following arguments during initialization or when running the tool:

- `api_key` (Optional[str]): Your Tavily API key. If not provided during initialization, it defaults to the `TAVILY_API_KEY` environment variable.
- `proxies` (Optional[dict[str, str]]): Proxies to use for the API requests. Defaults to `None`.

When running the tool (`_run` or `_arun` methods, or via agent execution), it uses the `TavilyExtractorToolSchema` and expects the following inputs:

- `urls` (Union[List[str], str]): **Required**. A single URL string or a list of URL strings to extract data from.
- `include_images` (Optional[bool]): Whether to include images in the extraction results. Defaults to `False`.
- `extract_depth` (Literal["basic", "advanced"]): The depth of extraction. Use `"basic"` for faster, surface-level extraction or `"advanced"` for more comprehensive extraction. Defaults to `"basic"`.
- `timeout` (int): The maximum time in seconds to wait for the extraction request to complete. Defaults to `60`.

## Response Format

The tool returns a JSON string representing the structured data extracted from the provided URL(s). The exact structure depends on the content of the pages and the `extract_depth` used. Refer to the [Tavily API documentation](https://docs.tavily.com/docs/tavily-api/python-sdk#extract) for details on the response structure.
