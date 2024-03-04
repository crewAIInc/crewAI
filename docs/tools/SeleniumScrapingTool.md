# SeleniumScrapingTool

## Description
This tool is designed for efficient web scraping, enabling users to extract content from web pages. It supports targeted scraping by allowing the specification of a CSS selector for desired elements. The flexibility of the tool enables it to be used on any website URL provided by the user, making it a versatile tool for various web scraping needs.

## Installation
Install the crewai_tools package
```
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import SeleniumScrapingTool

# Example 1: Scrape any website it finds during its execution
tool = SeleniumScrapingTool()

# Example 2: Scrape the entire webpage
tool = SeleniumScrapingTool(website_url='https://example.com')

# Example 3: Scrape a specific CSS element from the webpage
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content')

# Example 4: Scrape using optional parameters for customized scraping
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content', cookie={'name': 'user', 'value': 'John Doe'})
```

## Arguments
- `website_url`: Mandatory. The URL of the website to scrape.
- `css_element`: Mandatory. The CSS selector for a specific element to scrape from the website.
- `cookie`: Optional. A dictionary containing cookie information. This parameter allows the tool to simulate a session with cookie information, providing access to content that may be restricted to logged-in users.
- `wait_time`: Optional. The number of seconds the tool waits after loading the website and after setting a cookie, before scraping the content. This allows for dynamic content to load properly.
