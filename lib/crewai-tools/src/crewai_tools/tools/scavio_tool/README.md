# Scavio Tools

## Description

The Scavio tools provide an interface to the [Scavio Search API](https://scavio.dev), enabling CrewAI agents to perform real-time web searches, Amazon product searches, and YouTube video searches. The API returns structured JSON data covering Google, Amazon, YouTube, and more.

Available tools:

- **ScavioSearchTool** — Web search with knowledge graphs, related questions, and news results.
- **ScavioAmazonSearchTool** — Amazon product search with prices, ratings, and availability across 20+ marketplaces.
- **ScavioYouTubeSearchTool** — YouTube video search with view counts, channel info, and thumbnails.

## Installation

Install with the `scavio` extra:

```shell
uv add 'crewai-tools[scavio]'
```

## Environment Variables

Set your Scavio API key as an environment variable. Get a free key at [dashboard.scavio.dev](https://dashboard.scavio.dev):

```bash
export SCAVIO_API_KEY='sk_live_...'
```

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import ScavioSearchTool

# Initialize the tool
search_tool = ScavioSearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role="Research Analyst",
    goal="Find the latest information on any topic",
    backstory="An expert researcher who finds accurate, up-to-date information.",
    tools=[search_tool],
    verbose=True,
)

# Create a task for the agent
research_task = Task(
    description="Research the top 3 trends in AI agents for 2026.",
    expected_output="A summary of the top 3 AI agent trends with sources.",
    agent=researcher,
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
)

result = crew.kickoff()
print(result)
```

### Amazon Product Search

```python
from crewai_tools import ScavioAmazonSearchTool

amazon_tool = ScavioAmazonSearchTool(
    domain="com",
    max_results=5,
)

result = amazon_tool.run("wireless noise cancelling headphones")
print(result)
```

### YouTube Video Search

```python
from crewai_tools import ScavioYouTubeSearchTool

youtube_tool = ScavioYouTubeSearchTool(
    max_results=5,
    sort_by="relevance",
)

result = youtube_tool.run("CrewAI tutorial")
print(result)
```

## Arguments

### ScavioSearchTool

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | `str` | **Required** | The search query string. |
| `search_type` | `str` | `"classic"` | Type of search: `"classic"`, `"news"`, `"maps"`, `"images"`. |
| `max_results` | `int` | `5` | Maximum number of results to return. |
| `country_code` | `str` | `None` | ISO 3166-1 alpha-2 country code (e.g., `"us"`, `"gb"`). |
| `language` | `str` | `None` | ISO 639-1 language code (e.g., `"en"`, `"fr"`). |
| `device` | `str` | `"desktop"` | Device type: `"desktop"` or `"mobile"`. |
| `include_knowledge_graph` | `bool` | `True` | Include knowledge graph in results. |
| `include_questions` | `bool` | `True` | Include related questions in results. |

### ScavioAmazonSearchTool

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | `str` | **Required** | Product search query. |
| `max_results` | `int` | `5` | Maximum number of products to return. |
| `domain` | `str` | `"com"` | Amazon marketplace (e.g., `"com"`, `"co.uk"`, `"de"`). |
| `sort_by` | `str` | `None` | Sort order: `"most_recent"`, `"price_low_to_high"`, `"price_high_to_low"`, `"featured"`, `"average_review"`, `"bestsellers"`. |

### ScavioYouTubeSearchTool

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | `str` | **Required** | YouTube search query. |
| `max_results` | `int` | `5` | Maximum number of results to return. |
| `upload_date` | `str` | `None` | Filter: `"hour"`, `"today"`, `"week"`, `"month"`, `"year"`. |
| `sort_by` | `str` | `None` | Sort: `"relevance"`, `"date"`, `"views"`, `"rating"`. |
| `type` | `str` | `None` | Content type: `"video"`, `"channel"`, `"playlist"`. |
| `duration` | `str` | `None` | Duration filter: `"short"`, `"medium"`, `"long"`. |
