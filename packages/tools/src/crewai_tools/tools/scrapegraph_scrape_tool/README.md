# ScrapegraphScrapeTool

## Description
A tool that leverages Scrapegraph AI's SmartScraper API to intelligently extract content from websites. This tool provides advanced web scraping capabilities with AI-powered content extraction, making it ideal for targeted data collection and content analysis tasks.

## Installation
Install the required packages:
```shell
pip install 'crewai[tools]'
```

## Example Usage

### Basic Usage
```python
from crewai_tools import ScrapegraphScrapeTool

# Basic usage with API key
tool = ScrapegraphScrapeTool(api_key="your_api_key")
result = tool.run(
    website_url="https://www.example.com",
    user_prompt="Extract the main heading and summary"
)
```

### Fixed Website URL
```python
# Initialize with a fixed website URL
tool = ScrapegraphScrapeTool(
    website_url="https://www.example.com",
    api_key="your_api_key"
)
result = tool.run()
```

### Custom Prompt
```python
# With custom prompt
tool = ScrapegraphScrapeTool(
    api_key="your_api_key",
    user_prompt="Extract all product prices and descriptions"
)
result = tool.run(website_url="https://www.example.com")
```

### Error Handling
```python
try:
    tool = ScrapegraphScrapeTool(api_key="your_api_key")
    result = tool.run(
        website_url="https://www.example.com",
        user_prompt="Extract the main heading"
    )
except ValueError as e:
    print(f"Configuration error: {e}")  # Handles invalid URLs or missing API keys
except RuntimeError as e:
    print(f"Scraping error: {e}")  # Handles API or network errors
```

## Arguments
- `website_url`: The URL of the website to scrape (required if not set during initialization)
- `user_prompt`: Custom instructions for content extraction (optional)
- `api_key`: Your Scrapegraph API key (required, can be set via SCRAPEGRAPH_API_KEY environment variable)

## Environment Variables
- `SCRAPEGRAPH_API_KEY`: Your Scrapegraph API key, you can obtain one [here](https://scrapegraphai.com)

## Rate Limiting
The Scrapegraph API has rate limits that vary based on your subscription plan. Consider the following best practices:
- Implement appropriate delays between requests when processing multiple URLs
- Handle rate limit errors gracefully in your application
- Check your API plan limits on the Scrapegraph dashboard

## Error Handling
The tool may raise the following exceptions:
- `ValueError`: When API key is missing or URL format is invalid
- `RuntimeError`: When scraping operation fails (network issues, API errors)
- `RateLimitError`: When API rate limits are exceeded

## Best Practices
1. Always validate URLs before making requests
2. Implement proper error handling as shown in examples
3. Consider caching results for frequently accessed pages
4. Monitor your API usage through the Scrapegraph dashboard
