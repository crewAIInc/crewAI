# BrightData Tools Documentation

## Description

A comprehensive suite of CrewAI tools that leverage Bright Data's powerful infrastructure for web scraping, data extraction, and search operations. These tools provide three distinct capabilities:

- **BrightDataDatasetTool**: Extract structured data from popular data feeds (Amazon, LinkedIn, Instagram, etc.) using pre-built datasets
- **BrightDataSearchTool**: Perform web searches across multiple search engines with geo-targeting and device simulation
- **BrightDataWebUnlockerTool**: Scrape any website content while bypassing bot protection mechanisms

## Installation

To incorporate these tools into your project, follow the installation instructions below:

```shell
pip install crewai[tools] aiohttp requests
```

## Examples

### Dataset Tool - Extract Amazon Product Data
```python
from crewai_tools import BrightDataDatasetTool

# Initialize with specific dataset and URL
tool = BrightDataDatasetTool(
    dataset_type="amazon_product",
    url="https://www.amazon.com/dp/B08QB1QMJ5/"
)
result = tool.run()
```

### Search Tool - Perform Web Search
```python
from crewai_tools import BrightDataSearchTool

# Initialize with search query
tool = BrightDataSearchTool(
    query="latest AI trends 2025",
    search_engine="google",
    country="us"
)
result = tool.run()
```

### Web Unlocker Tool - Scrape Website Content
```python
from crewai_tools import BrightDataWebUnlockerTool

# Initialize with target URL
tool = BrightDataWebUnlockerTool(
    url="https://example.com",
    data_format="markdown"
)
result = tool.run()
```

## Steps to Get Started

To effectively use the BrightData Tools, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.

2. **API Key Acquisition**: Register for a Bright Data account at `https://brightdata.com/` and obtain your API credentials from your account settings.

3. **Environment Configuration**: Set up the required environment variables:
   ```bash
   export BRIGHT_DATA_API_KEY="your_api_key_here"
   export BRIGHT_DATA_ZONE="your_zone_here"
   ```

4. **Tool Selection**: Choose the appropriate tool based on your needs:
   - Use **DatasetTool** for structured data from supported platforms
   - Use **SearchTool** for web search operations
   - Use **WebUnlockerTool** for general website scraping

## Conclusion

By integrating BrightData Tools into your CrewAI agents, you gain access to enterprise-grade web scraping and data extraction capabilities. These tools handle complex challenges like bot protection, geo-restrictions, and data parsing, allowing you to focus on building your applications rather than managing scraping infrastructure.