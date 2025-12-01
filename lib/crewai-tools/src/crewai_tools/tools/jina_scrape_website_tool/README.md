# JinaScrapeWebsiteTool

## Description
A tool designed to extract and read the content of a specified website by using Jina.ai reader. It is capable of handling various types of web pages by making HTTP requests and parsing the received HTML content. This tool can be particularly useful for web scraping tasks, data collection, or extracting specific information from websites.

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import JinaScrapeWebsiteTool

# To enable scraping any website it finds during its execution
tool = JinaScrapeWebsiteTool(api_key='YOUR_API_KEY')

# Initialize the tool with the website URL, so the agent can only scrape the content of the specified website
tool = JinaScrapeWebsiteTool(website_url='https://www.example.com')

# With custom headers
tool = JinaScrapeWebsiteTool(
    website_url='https://www.example.com',
    custom_headers={'X-Target-Selector': 'body, .class, #id'}
)
```

## Authentication
The tool uses Jina.ai's reader service. While it can work without an API key, Jina.ai may apply rate limiting or blocking to unauthenticated requests. For production use, it's recommended to provide an API key.

## Arguments
- `website_url`: Mandatory website URL to read the file. This is the primary input for the tool, specifying which website's content should be scraped and read.
- `api_key`: Optional Jina.ai API key for authenticated access to the reader service.
- `custom_headers`: Optional dictionary of HTTP headers to use when making requests.

## Note
This tool is an alternative to the standard `ScrapeWebsiteTool` that specifically uses Jina.ai's reader service for enhanced content extraction. Choose this tool when you need more sophisticated content parsing capabilities.