# SerpexTool Documentation

## Description
The SerpexTool is a powerful multi-engine search tool that interfaces with the Serpex API to perform internet searches across multiple search engines including Google, Bing, DuckDuckGo, Brave, Yahoo, and Yandex. It features automatic engine routing, time-based filtering, and built-in handling of captchas and blocking.

## Features
- **Multi-Engine Support**: Search across 6 different search engines (Google, Bing, DuckDuckGo, Brave, Yahoo, Yandex)
- **Auto Routing**: Automatically selects the best available search engine
- **Time Filtering**: Filter results by time ranges (day, week, month, year)
- **Structured Results**: Clean JSON responses with titles, URLs, snippets, and metadata
- **Captcha Handling**: Built-in proxy rotation and retry logic
- **Affordable**: 200 free credits to start, $0.0008 per request
- **Optional result saving to file**

## Installation
```shell
pip install 'crewai[tools]'
```

## Usage
```python
from crewai_tools import SerpexTool

# Initialize the tool
tool = SerpexTool(
    n_results=10,  # Optional: Number of results to return (default: 10)
    save_file=False,  # Optional: Save results to file (default: False)
    engine="auto",  # Optional: Search engine - "auto", "google", "bing", "duckduckgo", "brave", "yahoo", "yandex" (default: "auto")
    time_range="all"  # Optional: Time filter - "all", "day", "week", "month", "year" (default: "all")
)

# Execute a search
results = tool._run(search_query="your search query")
```

## Configuration
1. **API Key Setup**:
   - Sign up for an account at [serpex.dev](https://serpex.dev)
   - Obtain your API key from the dashboard
   - Set the environment variable: `SERPEX_API_KEY`

2. **Environment Variable**:
   ```bash
   export SERPEX_API_KEY="your-api-key-here"
   ```

## Parameters

### Tool Initialization
- `n_results` (int): Number of results to return (default: 10)
- `save_file` (bool): Whether to save results to a file (default: False)
- `engine` (str): Search engine to use (default: "auto")
  - Options: "auto", "google", "bing", "duckduckgo", "brave", "yahoo", "yandex"
- `time_range` (str): Filter results by time (default: "all")
  - Options: "all", "day", "week", "month", "year"

### Runtime Arguments
- `search_query` (str): The search query (required)
- `engine` (str): Override the default search engine (optional)
- `time_range` (str): Override the default time range (optional)
- `save_file` (bool): Override the default save behavior (optional)

## Response Format
The tool returns structured data including:

```python
{
    "query": "your search query",
    "engines": ["google"],  # List of engines used
    "metadata": {
        "number_of_results": 10,
        "response_time": 0.5,
        "timestamp": "2025-10-26T12:00:00Z",
        "credits_used": 1
    },
    "results": [
        {
            "title": "Result Title",
            "url": "https://example.com",
            "snippet": "Result description...",
            "position": 1,
            "engine": "google",
            "published_date": "2025-10-25"
        }
    ],
    "suggestions": ["related query 1", "related query 2"]
}
```

## Examples

### Basic Search
```python
from crewai_tools import SerpexTool

tool = SerpexTool()
results = tool._run(search_query="latest AI developments")
print(results)
```

### Multi-Engine Search with Time Filter
```python
from crewai_tools import SerpexTool

tool = SerpexTool(
    engine="google",
    time_range="week",
    n_results=20
)
results = tool._run(search_query="quantum computing news")
```

### Auto Engine Selection
```python
from crewai_tools import SerpexTool

tool = SerpexTool(engine="auto")
results = tool._run(search_query="machine learning trends")
```

### Save Results to File
```python
from crewai_tools import SerpexTool

tool = SerpexTool(save_file=True)
results = tool._run(search_query="AI research papers")
# Results will be saved to search_results_YYYY-MM-DD_HH-MM-SS.txt
```

## Integration with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerpexTool

# Initialize the tool
search_tool = SerpexTool(
    engine="auto",
    time_range="month",
    n_results=15
)

# Create an agent with the tool
researcher = Agent(
    role="Research Analyst",
    goal="Find the latest information on AI developments",
    backstory="Expert at finding and analyzing current information",
    tools=[search_tool],
    verbose=True
)

# Create a task
research_task = Task(
    description="Research the latest developments in artificial intelligence",
    agent=researcher,
    expected_output="A comprehensive summary of recent AI developments"
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

## Error Handling
The tool includes comprehensive error handling for:
- Invalid API keys
- Network errors
- Rate limiting
- Invalid parameters
- Malformed responses

## Pricing
- **Free Tier**: 200 free credits to get started
- **Paid Plans**: Starting at $0.0008 per request
- Monitor usage at [serpex.dev/dashboard](https://serpex.dev/dashboard)

## Advantages Over Single-Engine Tools
- **Multi-Engine Support**: Access 6 engines vs single engine
- **Auto Routing**: No manual engine switching needed
- **No Captchas**: Built-in handling vs manual solving
- **Affordable**: Competitive pricing with generous free tier
- **Structured Data**: Consistent JSON format across all engines
- **Fast Response**: Optimized infrastructure

## Troubleshooting

### API Key Issues
- Ensure `SERPEX_API_KEY` environment variable is set correctly
- Verify your API key is active at [serpex.dev](https://serpex.dev)

### No Results
- Check your account has available credits
- Try using `engine="auto"` for automatic engine selection
- Verify your search query is valid

### Rate Limits
- Monitor your usage in the dashboard
- Upgrade to a paid plan for higher limits
- Implement caching for frequently searched queries

## Support
- Documentation: [serpex.dev/docs](https://serpex.dev/docs)
- API Reference: [serpex.dev/docs/api](https://serpex.dev/docs/api)
- Support: Contact through [serpex.dev](https://serpex.dev)
