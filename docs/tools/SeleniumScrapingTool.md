# SeleniumScrapingTool

!!! note "Experimental"
    This tool is currently in development. As we refine its capabilities, users may encounter unexpected behavior. Your feedback is invaluable to us for making improvements.

## Description
The SeleniumScrapingTool is crafted for high-efficiency web scraping tasks. It allows for precise extraction of content from web pages by using CSS selectors to target specific elements. Its design caters to a wide range of scraping needs, offering flexibility to work with any provided website URL.

## Installation
To get started with the SeleniumScrapingTool, install the crewai_tools package using pip:

```
pip install 'crewai[tools]'
```

## Usage Examples
Below are some scenarios where the SeleniumScrapingTool can be utilized:

```python
from crewai_tools import SeleniumScrapingTool

# Example 1: Initialize the tool without any parameters to scrape the current page it navigates to
tool = SeleniumScrapingTool()

# Example 2: Scrape the entire webpage of a given URL
tool = SeleniumScrapingTool(website_url='https://example.com')

# Example 3: Target and scrape a specific CSS element from a webpage
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content')

# Example 4: Perform scraping with additional parameters for a customized experience
tool = SeleniumScrapingTool(website_url='https://example.com', css_element='.main-content', cookie={'name': 'user', 'value': 'John Doe'}, wait_time=10)
```

## Arguments
The following parameters can be used to customize the SeleniumScrapingTool's scraping process:

- `website_url`: **Mandatory**. Specifies the URL of the website from which content is to be scraped.
- `css_element`: **Mandatory**. The CSS selector for a specific element to target on the website. This enables focused scraping of a particular part of a webpage.
- `cookie`: **Optional**. A dictionary that contains cookie information. Useful for simulating a logged-in session, thereby providing access to content that might be restricted to non-logged-in users.
- `wait_time`: **Optional**. Specifies the delay (in seconds) before the content is scraped. This delay allows for the website and any dynamic content to fully load, ensuring a successful scrape.

!!! attention
    Since the SeleniumScrapingTool is under active development, the parameters and functionality may evolve over time. Users are encouraged to keep the tool updated and report any issues or suggestions for enhancements.