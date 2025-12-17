# OlostepTool

The most reliable and cost-effective web search, scraping and crawling API for AI. Build intelligent agents that can search, scrape, analyze, and structure data from any website using the [Olostep API](https://olostep.com).

## Features

- **Scrape Mode**: Extract clean content from any webpage
  - Markdown, HTML, text, and JSON formats
  - LLM-powered structured data extraction
  - Built-in parsers for Google, LinkedIn, etc.
  - JavaScript rendering support
  - Country-based geolocation

- **Crawl Mode**: Follow links and extract content from multiple pages
  - Configurable max pages (up to 1000)
  - Include/exclude URL patterns

- **Map Mode**: Discover all URLs on a website
  - Comprehensive sitemap generation

- **Answer Mode**: AI-powered web search
  - Ask questions in natural language
  - Get structured JSON responses
  - Sources included for verification

## Installation

```bash
pip install crewai-tools olostep
```

## Environment Variables

```bash
export OLOSTEP_API_KEY="your-api-key-here"
```

Get your API key at [olostep.com/dashboard](https://olostep.com/dashboard)

## Usage

### Basic Scraping

```python
from crewai_tools import OlostepTool

# Create the tool
scrape_tool = OlostepTool()

# Scrape a webpage
content = scrape_tool.run(
    url="https://example.com/article",
    mode="scrape"
)
print(content)
```

### Using Built-in Parsers

```python
from crewai_tools import OlostepTool

tool = OlostepTool()

# Parse Google search results
results = tool.run(
    url="https://www.google.com/search?q=python+programming",
    mode="scrape",
    parser="@olostep/google-search"
)

# Parse LinkedIn profile
profile = tool.run(
    url="https://www.linkedin.com/in/username",
    mode="scrape",
    parser="@olostep/linkedin-profile"
)
```

**Available parsers:**
- `@olostep/google-search` - Google search results
- `@olostep/google-news` - Google News articles
- `@olostep/google-maps` - Google Maps places
- `@olostep/linkedin-profile` - LinkedIn profiles
- `@olostep/linkedin-company` - LinkedIn companies
- `@olostep/perplexity-search` - Perplexity AI search
- `@olostep/brave-search` - Brave search results

### LLM Extraction with JSON Schema

```python
from crewai_tools import OlostepTool

tool = OlostepTool()

# Extract structured data using LLM
event_data = tool.run(
    url="https://example.com/event-page",
    mode="scrape",
    llm_extract_schema={
        "event_name": {"type": "string"},
        "date": {"type": "string"},
        "venue": {"type": "string"},
        "price": {"type": "number"}
    }
)
```

### Crawling Websites

```python
from crewai_tools import OlostepCrawlTool

crawl_tool = OlostepCrawlTool()

# Crawl a website
result = crawl_tool.run(
    url="https://example.com",
    max_pages=50,
    include_patterns=["/blog/*", "/docs/*"]
)
```

### Discovering URLs (Sitemap)

```python
from crewai_tools import OlostepMapTool

map_tool = OlostepMapTool()

# Get all URLs on a website
urls = map_tool.run(url="https://example.com")
```

### AI-Powered Web Search

```python
from crewai_tools import OlostepAnswerTool

answer_tool = OlostepAnswerTool()

# Ask a question
answer = answer_tool.run(
    task="What is the latest version of Python?"
)

# Get structured answer
structured_answer = answer_tool.run(
    task="What are the top 3 programming languages in 2024?",
    json_schema={
        "languages": [{"name": "", "rank": 0}]
    }
)
```

## Specialized Tools

For convenience, you can use pre-configured tools for each mode:

| Tool | Mode | Use Case |
|------|------|----------|
| `OlostepTool` | All modes | General purpose |
| `OlostepScrapeTool` | Scrape | Single page extraction |
| `OlostepCrawlTool` | Crawl | Multi-page crawling |
| `OlostepMapTool` | Map | URL discovery |
| `OlostepAnswerTool` | Answer | AI web search |

## Arguments

### Common Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `url` | str | URL to process (required for scrape/crawl/map) |
| `mode` | str | Operation mode: "scrape", "crawl", "map", "answer" |

### Scrape Mode Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `formats` | list[str] | Output formats: "markdown", "html", "text", "json" |
| `wait_before_scraping` | int | Milliseconds to wait for JS content |
| `country` | str | Country code for geolocation (e.g., "US", "UK") |
| `parser` | str | Parser ID for structured extraction |
| `llm_extract_schema` | dict | JSON schema for LLM extraction |

### Crawl Mode Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `max_pages` | int | Maximum pages to crawl (1-1000) |
| `include_patterns` | list[str] | URL patterns to include (regex) |
| `exclude_patterns` | list[str] | URL patterns to exclude (regex) |

### Answer Mode Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `task` | str | Question or task for AI to answer |
| `json_schema` | dict | JSON schema for structured output |

## Integration with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import OlostepTool, OlostepAnswerTool

# Create tools
scrape_tool = OlostepTool()
answer_tool = OlostepAnswerTool()

# Create a researcher agent
researcher = Agent(
    role="Web Researcher",
    goal="Find and analyze information from the web",
    backstory="Expert at finding and extracting information from websites",
    tools=[scrape_tool, answer_tool]
)

# Create a task
task = Task(
    description="Research the latest trends in AI and summarize key findings",
    agent=researcher,
    expected_output="A summary of AI trends with sources"
)

# Run the crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Links

- [Olostep Website](https://olostep.com)
- [Olostep API Documentation](https://docs.olostep.com)
- [Get API Key](https://olostep.com/dashboard)
