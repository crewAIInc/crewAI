# ScrapegraphScrapeTool

## Description
A tool that leverages Scrapegraph AI's SmartScraper API to intelligently extract content from websites. This tool provides advanced web scraping capabilities with AI-powered content extraction, making it ideal for targeted data collection and content analysis tasks.

## Installation
Install the required packages:
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import ScrapegraphScrapeTool

# Basic usage with API key
tool = ScrapegraphScrapeTool(api_key="your_api_key")
result = tool.run(
    website_url="https://www.example.com",
    user_prompt="Extract the main heading and summary"
)

# Initialize with a fixed website URL
tool = ScrapegraphScrapeTool(
    website_url="https://www.example.com",
    api_key="your_api_key"
)
result = tool.run()

# With custom prompt
tool = ScrapegraphScrapeTool(
    api_key="your_api_key",
    user_prompt="Extract all product prices and descriptions"
)
```

## Arguments
- `website_url`: The URL of the website to scrape (required if not set during initialization)
- `user_prompt`: Custom instructions for content extraction (optional)
- `api_key`: Your Scrapegraph API key (required, can be set via SCRAPEGRAPH_API_KEY environment variable)

## Environment Variables
- `SCRAPEGRAPH_API_KEY`: Your Scrapegraph API key, you can buy it [here](https://scrapegraphai.com)
