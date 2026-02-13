# YouContentsTool

Extracts content from web pages using the You.com Contents API.

## Installation

1. Get your API key from https://you.com/platform/api-keys

2. Install dependencies and set your API key:

```bash
uv add requests aiohttp
export YOU_API_KEY="your-api-key-here"
```

## Usage

```python
from crewai_tools import YouContentsTool

# Basic usage - single URL
contents_tool = YouContentsTool()
content = contents_tool.run(urls="https://example.com")

# Multiple URLs
content = contents_tool.run(urls=[
    "https://example.com/page1",
    "https://example.com/page2"
])

# With advanced options
contents_tool = YouContentsTool(
    formats=["markdown", "metadata"],
    crawl_timeout=45
)
content = contents_tool.run(urls="https://example.com")
```

## Parameters

- **urls** (required): Single URL string or list of URL strings (must be valid URLs)
- **formats**: List of output formats - "markdown", "html", or "metadata" (default: ["markdown"])
- **crawl_timeout**: Page crawling timeout in seconds, 1-60 (default: 10)
- **timeout**: API request timeout in seconds (default: 60)

## Output Formats

### Markdown
Clean text extraction, best for reading and analysis.

```python
contents_tool = YouContentsTool(formats=["markdown"])
result = contents_tool.run(urls="https://example.com")
# Returns: { "url": "...", "title": "...", "markdown": "# Content..." }
```

### HTML
Preserves layout and structure, best for rendering.

```python
contents_tool = YouContentsTool(formats=["html"])
result = contents_tool.run(urls="https://example.com")
# Returns: { "url": "...", "title": "...", "html": "<html>..." }
```

### Metadata
Structured data including OpenGraph and JSON-LD information.

```python
contents_tool = YouContentsTool(formats=["metadata"])
result = contents_tool.run(urls="https://example.com")
# Returns: {
#   "url": "...",
#   "title": "...",
#   "metadata": {
#     "site_name": "Example Site",
#     "favicon_url": "https://example.com/favicon.ico"
#   }
# }
```

### Multiple Formats
Request multiple formats in a single API call.

```python
contents_tool = YouContentsTool(formats=["markdown", "html", "metadata"])
result = contents_tool.run(urls="https://example.com")
# Returns all requested formats in the response
```

## With CrewAI

```python
from crewai import Agent, Task
from crewai_tools import YouContentsTool

contents_tool = YouContentsTool(formats=["markdown"])

analyzer = Agent(
    role="Content Analyzer",
    goal="Extract and analyze web content",
    backstory="Expert at content extraction and analysis",
    tools=[contents_tool]
)

task = Task(
    description="Extract content from https://example.com and summarize key points",
    agent=analyzer,
    expected_output="Summary of key points from the webpage"
)
```

## Combining with YouSearchTool

```python
from crewai_tools import YouSearchTool, YouContentsTool

search_tool = YouSearchTool(count=5)
contents_tool = YouContentsTool()

# Search for pages
results = search_tool.run(query="Python async programming")

# Extract content from top results
# (you would parse the JSON results and extract URLs)
content = contents_tool.run(urls=["url1", "url2"])
```
