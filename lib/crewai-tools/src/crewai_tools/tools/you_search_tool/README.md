# YouSearchTool

Performs web searches using the You.com Search API.

## Installation

1. Get your API key from https://you.com/platform/api-keys

2. Install dependencies and set your API key:

```bash
uv add requests aiohttp
export YOU_API_KEY="your-api-key-here"
```

## Usage

```python
from crewai_tools import YouSearchTool

# Basic usage
search_tool = YouSearchTool()
results = search_tool.run(query="latest AI developments")

# With advanced options
search_tool = YouSearchTool(
    count=20,
    country="US",
    freshness="week",
    safesearch="moderate",
    livecrawl="web",
    livecrawl_formats="markdown"
)
results = search_tool.run(query="Python best practices")
```

## Parameters

- **query** (required): The search query string
- **count**: Maximum number of results per section (1-100, default: 10)
- **offset**: Pagination offset (0-9, default: None). Works with count for pagination
- **country**: Country code - "US", "GB", "CA", "FR", "DE", "JP", etc. (default: "US")
- **language**: Language code in BCP 47 format - "EN", "EN-GB", "FR", "ZH-HANS", "PT-BR", etc.
- **freshness**: Time filter - "day", "week", "month", "year", or "YYYY-MM-DDtoYYYY-MM-DD"
- **safesearch**: "off", "moderate", or "strict" (default: "moderate")
- **livecrawl**: Live-crawl sections - "web", "news", or "all"
- **livecrawl_formats**: "html" or "markdown" (default: "markdown")
- **timeout**: Request timeout in seconds (default: 60)

## Search Operators

- `site:domain.com` - Filter by domain
- `filetype:pdf` - Filter by file type
- `+term` / `-term` - Include/exclude terms
- `AND` / `OR` / `NOT` - Boolean logic
- `lang:en` - Language filter

Example:
```python
results = search_tool.run(query="machine learning (Python OR PyTorch) -TensorFlow")
```

### Pagination

Use `offset` with `count` to paginate through results:

```python
search_tool = YouSearchTool(count=10)

# First page (results 0-9)
page1 = search_tool.run(query="AI news", offset=0)

# Second page (results 10-19)
page2 = search_tool.run(query="AI news", offset=1)

# Third page (results 20-29)
page3 = search_tool.run(query="AI news", offset=2)
```

### Language-Specific Search

Search for content in specific languages using BCP 47 language codes:

```python
# Search for content in British English
search_tool = YouSearchTool(language="EN-GB", country="GB")
results = search_tool.run(query="football news")

# Search for content in Simplified Chinese
search_tool = YouSearchTool(language="ZH-HANS", country="CN")
results = search_tool.run(query="人工智能")

# Search for content in Brazilian Portuguese
search_tool = YouSearchTool(language="PT-BR", country="BR")
results = search_tool.run(query="notícias de tecnologia")
```

**Common language codes:**
- English: `EN`, `EN-GB` (British), `EN-US` (American)
- Chinese: `ZH-HANS` (Simplified), `ZH-HANT` (Traditional)
- Portuguese: `PT-PT` (Portugal), `PT-BR` (Brazil)
- Spanish: `ES`
- French: `FR`
- German: `DE`
- Japanese: `JP`
- Korean: `KO`

## With CrewAI

```python
from crewai import Agent
from crewai_tools import YouSearchTool

search_tool = YouSearchTool(count=10, freshness="week")

researcher = Agent(
    role="Research Analyst",
    goal="Find relevant information",
    backstory="Expert at web research",
    tools=[search_tool]
)
```
