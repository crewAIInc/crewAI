# Valyu Extractor Tool

## Description

The `ValyuExtractorTool` allows CrewAI agents to extract clean, structured content from web pages using the Valyu API. It can process single URLs or lists of URLs (up to 10) and provides options for controlling content length, extraction quality, screenshots, and AI-powered summarization.

## Installation

To use the `ValyuExtractorTool`, you need to install the `valyu` library:

```shell
pip install 'crewai[tools]' valyu
```

You also need to set your Valyu API key as an environment variable:

```bash
export VALYU_API_KEY='your-valyu-api-key'
```

Get an API key at https://platform.valyu.ai/ (sign up, then create a key from the dashboard).

## Example

Here's how to initialize and use the `ValyuExtractorTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import ValyuExtractorTool

# Ensure VALYU_API_KEY is set in your environment
# os.environ["VALYU_API_KEY"] = "YOUR_API_KEY"

# Initialize the tool
valyu_extractor = ValyuExtractorTool()

# Create an agent that uses the tool
extractor_agent = Agent(
    role='Web Content Extractor',
    goal='Extract key information from specified web pages',
    backstory='You are an expert at extracting relevant content from websites using the Valyu API.',
    tools=[valyu_extractor],
    verbose=True
)

# Define a task for the agent
extract_task = Task(
    description='Extract the main content from the URL https://example.com.',
    expected_output='A JSON string containing the extracted content from the URL.',
    agent=extractor_agent
)

# Create and run the crew
crew = Crew(
    agents=[extractor_agent],
    tasks=[extract_task],
    verbose=True
)

result = crew.kickoff()
print(result)

# Example with multiple URLs and high extraction effort
extract_multiple_task = Task(
    description='Extract content from https://example.com and https://anotherexample.org.',
    expected_output='A JSON string containing the extracted content from both URLs.',
    agent=extractor_agent
)
```

## Arguments

The `ValyuExtractorTool` accepts the following arguments during initialization or when running the tool:

- `api_key` (Optional[str]): Your Valyu API key. If not provided during initialization, it defaults to the `VALYU_API_KEY` environment variable.

When running the tool (`_run` or `_arun` methods, or via agent execution), it uses the `ValyuExtractorToolSchema` and expects the following inputs:

- `urls` (Union[List[str], str]): **Required**. A single URL string or a list of URL strings to extract data from. Maximum 10 URLs per request.
- `response_length` (Literal["short", "medium", "large", "max"], optional): Content length per result. `"short"` (25K chars), `"medium"` (50K), `"large"` (100K), or `"max"` (unlimited). Defaults to `"short"`.
- `extract_effort` (Literal["normal", "high", "auto"], optional): Processing quality level. Use `"normal"` for fastest extraction, `"high"` for better quality, or `"auto"` for automatic selection. Defaults to `"normal"`.
- `screenshot` (bool, optional): Whether to request page screenshots as pre-signed URLs. Defaults to `False`.
- `summary` (Union[bool, str], optional): Enable AI-powered summarization. Pass `True` for default summary, or a string with custom instructions. Defaults to `False`.

## Custom Configuration

You can configure the tool during initialization:

```python
# Example: Initialize with high extraction effort and screenshots
custom_extractor = ValyuExtractorTool(
    extract_effort='high',
    response_length='medium',
    screenshot=True
)

# Example: Initialize with AI summarization
summarizing_extractor = ValyuExtractorTool(
    summary=True,  # Enable default summarization
    response_length='large'
)

# Or with custom summarization instructions
custom_summary_extractor = ValyuExtractorTool(
    summary="Extract key points and main arguments from the article",
    response_length='medium'
)
```

## Response Format

The tool returns a JSON string representing the structured data extracted from the provided URL(s).

Common response elements include:
- **title**: The page title
- **url**: The processed URL
- **content**: Main text content in markdown format
- **description**: Page meta description
- **source**: Source identifier
- **price**: Cost for this extraction
- **length**: Character count of extracted content
- **screenshot_url**: Pre-signed screenshot URL (when `screenshot=True`)

Refer to the [Valyu API documentation](https://docs.valyu.ai/api-reference/endpoint/contents) for details on the response structure.
