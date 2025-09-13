# ScrapeWebsiteTool

## Description
A tool designed to extract and read the content of a specified website. It is capable of handling various types of web pages by making HTTP requests and parsing the received HTML content. This tool can be particularly useful for web scraping tasks, data collection, or extracting specific information from websites.

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import ScrapeWebsiteTool

# To enable scrapping any website it finds during it's execution
tool = ScrapeWebsiteTool()

# Initialize the tool with the website URL, so the agent can only scrap the content of the specified website
tool = ScrapeWebsiteTool(website_url='https://www.example.com')
```

## Arguments
- `website_url` : Mandatory website URL to read the file. This is the primary input for the tool, specifying which website's content should be scraped and read.