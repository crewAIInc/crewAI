# SpiderTool

## Description
[Spider](https://spider.cloud/?ref=crewai) is a high-performance web scraping and crawling tool that delivers optimized markdown for LLMs and AI agents. It intelligently switches between HTTP requests and JavaScript rendering based on page requirements. Perfect for both single-page scraping and website crawling—making it ideal for content extraction and data collection.

## Installation
To use the Spider API you need to download the [Spider SDK](https://pypi.org/project/spider-client/) and the crewai[tools] SDK, too:

```python
pip install spider-client 'crewai[tools]'
```

## Example
This example shows you how you can use the Spider tool to enable your agent to scrape and crawl websites. The data returned from the Spider API is LLM-ready.

```python
from crewai_tools import SpiderTool

# To enable scraping any website it finds during its execution
spider_tool = SpiderTool(api_key='YOUR_API_KEY')

# Initialize the tool with the website URL, so the agent can only scrape the content of the specified website
spider_tool = SpiderTool(website_url='https://spider.cloud')

# Pass in custom parameters, see below for more details
spider_tool = SpiderTool(
    website_url='https://spider.cloud',
    custom_params={"depth": 2, "anti_bot": True, "proxy_enabled": True}
)

# Advanced usage using css query selector to extract content
css_extraction_map = {
    "/": [ # pass in path (main index in this case)
        {
            "name": "headers", # give it a name for this element
            "selectors": [
                "h1"
            ]
        }
    ]
}

spider_tool = SpiderTool(
    website_url='https://spider.cloud',
    custom_params={"anti_bot": True, "proxy_enabled": True, "metadata": True, "css_extraction_map": css_extraction_map}
)

### Response (extracted text will be in the metadata)
"css_extracted": {
    "headers": [
        "The Web Crawler for AI Agents and LLMs!"
    ]
}
```
## Agent setup
```yaml
researcher:
  role: >
    You're a researcher that is tasked with researching a website and it's content (use crawl mode). The website is to crawl is: {website_url}.
```

## Arguments

- `api_key` (string, optional): Specifies Spider API key. If not specified, it looks for `SPIDER_API_KEY` in environment variables.
- `website_url` (string): The website URL. Will be used as a fallback if passed when the tool is initialized.
- `log_failures` (bool): Log scrape failures or fail silently. Defaults to `true`.
- `custom_params` (object, optional): Optional parameters for the request.
    - `return_format` (string): The return format of the website's content. Defaults to `markdown`.
    - `request` (string): The request type to perform. Possible values are `http`, `chrome`, and `smart`. Use `smart` to perform an HTTP request by default until JavaScript rendering is needed for the HTML.
    - `limit` (int): The maximum number of pages allowed to crawl per website. Remove the value or set it to `0` to crawl all pages.
    - `depth` (int): The crawl limit for maximum depth. If `0`, no limit will be applied.
    - `locale` (string): The locale to use for request, example `en-US`.
    - `cookies` (string): Add HTTP cookies to use for request.
    - `stealth` (bool): Use stealth mode for headless chrome request to help prevent being blocked. The default is `true` on chrome.
    - `headers` (object): Forward HTTP headers to use for all request. The object is expected to be a map of key value pairs.
    - `metadata` (bool): Boolean to store metadata about the pages and content found. Defaults to `false`.
    - `subdomains` (bool): Allow subdomains to be included. Default is `false`.
    - `user_agent` (string): Add a custom HTTP user agent to the request. By default this is set to a random agent.
    - `proxy_enabled` (bool): Enable high performance premium proxies for the request to prevent being blocked at the network level.
    - `css_extraction_map` (object): Use CSS or XPath selectors to scrape contents from the web page. Set the paths and the extraction object map to perform extractions per path or page.
    - `request_timeout` (int): The timeout to use for request. Timeouts can be from `5-60`. The default is `30` seconds.
    - `return_headers` (bool): Return the HTTP response headers with the results. Defaults to `false`.
    - `filter_output_main_only` (bool): Filter the nav, aside, and footer from the output.
    - `headers` (object): Forward HTTP headers to use for all request. The object is expected to be a map of key value pairs.

Learn other parameters that can be used: [https://spider.cloud/docs/api](https://spider.cloud/docs/api)

