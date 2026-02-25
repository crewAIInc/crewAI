# NimbleSearchTool

## Description
Real-time web search tool powered by the Nimble Search API. Provides AI agents with high-quality search capabilities including fast metadata search and deep content extraction.

## Features
- **Fast Mode** (default): Token-efficient metadata search (titles, URLs, descriptions)
- **Deep Search**: Full content extraction with markdown formatting
- **Time-Based Filtering**: Search within specific time ranges (hour, day, week, month, year)
- **Domain Filtering**: Include or exclude specific domains
- **LLM-Generated Answers**: Optional AI-generated answers to queries
- **Geo-Targeting**: Country and locale-specific results

## Installation

```bash
uv add nimble-python
```

## Setup

Get your API key from [Nimble](https://nimbleway.com) and set it as an environment variable:

```bash
export NIMBLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
NIMBLE_API_KEY=your-api-key-here
```

## Usage

### Basic Example

```python
from crewai_tools import NimbleSearchTool

# Create search tool with defaults
search_tool = NimbleSearchTool()

# Basic search
result = search_tool.run(query="latest AI developments")

# Override parameters at runtime
result = search_tool.run(
    query="detailed research topic",
    deep_search=True,        # Get full content
    max_results=5,           # More results
    time_range="week"        # Recent results only
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `NIMBLE_API_KEY` env var | Your Nimble API key |
| `max_results` | `int` | `3` | Maximum number of results (runtime override available) |
| `deep_search` | `bool` | `False` | Full content extraction vs metadata only (runtime override available) |
| `include_answer` | `bool` | `False` | Include LLM-generated answer (runtime override available) |
| `time_range` | `str` | `None` | Filter by time: 'hour', 'day', 'week', 'month', 'year' (runtime only) |
| `include_domains` | `list[str]` | `None` | Only search within these domains (runtime only) |
| `exclude_domains` | `list[str]` | `None` | Exclude these domains (runtime only) |
| `parsing_type` | `str` | `"markdown"` | Format: 'markdown', 'plain_text', or 'simplified_html' |
| `locale` | `str` | `None` | Locale for results (e.g., 'en-US') |
| `country` | `str` | `None` | Country code for geo-targeting (e.g., 'US') |
| `max_content_length_per_result` | `int` | `1000` | Maximum content length per result |

### Using with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import NimbleSearchTool

search_tool = NimbleSearchTool()

researcher = Agent(
    role="Research Analyst",
    goal="Find and analyze information efficiently",
    backstory="Expert at web research.",
    tools=[search_tool],
    verbose=True
)

task = Task(
    description="Research the latest AI trends from this week.",
    agent=researcher,
    expected_output="Summary of recent AI trends"
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

Agents can override parameters at runtime based on task needs:
- Use `deep_search=False` for quick fact-checking
- Use `deep_search=True` for detailed research
- Add `time_range="week"` for recent news
- Add `include_domains=["github.com"]` for specific sources

## Response Structure

Returns JSON with search results:

```json
{
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com",
      "description": "Short snippet",
      "content": "Full content (only in deep mode)",
      "metadata": {
        "country": "US",
        "locale": "en",
        "position": 1
      }
    }
  ],
  "query": "your search query"
}
```

## Troubleshooting

**"Nimble client is not initialized" Error**
- Install `nimble-python` package
- Set `NIMBLE_API_KEY` environment variable
- Verify API key is valid

**Empty Results**
- Adjust query specificity
- Increase `max_results`
- Remove restrictive domain filters
- Check API key quota

## Support

- [Nimble Documentation](https://docs.nimbleway.com)
- [CrewAI Documentation](https://docs.crewai.com)
