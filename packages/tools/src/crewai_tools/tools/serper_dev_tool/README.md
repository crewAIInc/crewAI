# SerperDevTool Documentation

## Description
The SerperDevTool is a powerful search tool that interfaces with the `serper.dev` API to perform internet searches. It supports multiple search types including general search and news search, with features like knowledge graph integration, organic results, "People Also Ask" questions, and related searches.

## Features
- Multiple search types: 'search' (default) and 'news'
- Knowledge graph integration for enhanced search context
- Organic search results with sitelinks
- "People Also Ask" questions and answers
- Related searches suggestions
- News search with date, source, and image information
- Configurable number of results
- Optional result saving to file

## Installation
```shell
pip install 'crewai[tools]'
```

## Usage
```python
from crewai_tools import SerperDevTool

# Initialize the tool
tool = SerperDevTool(
    n_results=10,  # Optional: Number of results to return (default: 10)
    save_file=False,  # Optional: Save results to file (default: False)
    search_type="search",  # Optional: Type of search - "search" or "news" (default: "search")
    country="us",  # Optional: Country for search (default: "")
    location="New York",  # Optional: Location for search (default: "")
    locale="en-US"  # Optional: Locale for search (default: "")
)

# Execute a search
results = tool._run(search_query="your search query")
```

## Configuration
1. **API Key Setup**:
   - Sign up for an account at `serper.dev`
   - Obtain your API key
   - Set the environment variable: `SERPER_API_KEY`

## Response Format
The tool returns structured data including:
- Search parameters
- Knowledge graph data (for general search)
- Organic search results
- "People Also Ask" questions
- Related searches
- News results (for news search type)
