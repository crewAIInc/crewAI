# SpiderTool

## Description

[Spider](https://spider.cloud/?ref=crewai) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md#benchmark-results) open source scraper and crawler that returns LLM-ready data. It converts any website into pure HTML, markdown, metadata or text while enabling you to crawl with custom actions using AI.

## Installation

To use the Spider API you need to download the [Spider SDK](https://pypi.org/project/spider-client/) and the crewai[tools] SDK too:

```python
pip install spider-client 'crewai[tools]'
```

## Example

This example shows you how you can use the Spider tool to enable your agent to scrape and crawl websites. The data returned from the Spider API is already LLM-ready, so no need to do any cleaning there.

```python
from crewai_tools import SpiderTool

def main():
    spider_tool = SpiderTool()
    
    searcher = Agent(
        role="Web Research Expert",
        goal="Find related information from specific URL's",
        backstory="An expert web researcher that uses the web extremely well",
        tools=[spider_tool],
        verbose=True,
    )

    return_metadata = Task(
        description="Scrape https://spider.cloud with a limit of 1 and enable metadata",
        expected_output="Metadata and 10 word summary of spider.cloud",
        agent=searcher
    )

    crew = Crew(
        agents=[searcher],
        tasks=[
            return_metadata, 
        ],
        verbose=2
    )
    
    crew.kickoff()

if __name__ == "__main__":
    main()
```

## Arguments

- `api_key` (string, optional): Specifies Spider API key. If not specified, it looks for `SPIDER_API_KEY` in environment variables.
- `params` (object, optional): Optional parameters for the request. Defaults to `{"return_format": "markdown"}` to return the website's content in a format that fits LLMs better.
    - `request` (string): The request type to perform. Possible values are `http`, `chrome`, and `smart`. Use `smart` to perform an HTTP request by default until JavaScript rendering is needed for the HTML.
    - `limit` (int): The maximum number of pages allowed to crawl per website. Remove the value or set it to `0` to crawl all pages.
    - `depth` (int): The crawl limit for maximum depth. If `0`, no limit will be applied.
    - `cache` (bool): Use HTTP caching for the crawl to speed up repeated runs. Default is `true`.
    - `budget` (object): Object that has paths with a counter for limiting the amount of pages example `{"*":1}` for only crawling the root page.
    - `locale` (string): The locale to use for request, example `en-US`.
    - `cookies` (string): Add HTTP cookies to use for request.
    - `stealth` (bool): Use stealth mode for headless chrome request to help prevent being blocked. The default is `true` on chrome.
    - `headers` (object): Forward HTTP headers to use for all request. The object is expected to be a map of key value pairs.
    - `metadata` (bool): Boolean to store metadata about the pages and content found. This could help improve AI interopt. Defaults to `false` unless you have the website already stored with the configuration enabled.
    - `viewport` (object): Configure the viewport for chrome. Defaults to `800x600`.
    - `encoding` (string): The type of encoding to use like `UTF-8`, `SHIFT_JIS`, or etc.
    - `subdomains` (bool): Allow subdomains to be included. Default is `false`.
    - `user_agent` (string): Add a custom HTTP user agent to the request. By default this is set to a random agent.
    - `store_data` (bool): Boolean to determine if storage should be used. If set this takes precedence over `storageless`. Defaults to `false`.
    - `gpt_config` (object): Use AI to generate actions to perform during the crawl. You can pass an array for the `"prompt"` to chain steps.
    - `fingerprint` (bool): Use advanced fingerprint for chrome.
    - `storageless` (bool): Boolean to prevent storing any type of data for the request including storage and AI vectors embedding. Defaults to `false` unless you have the website already stored.
    - `readability` (bool): Use [readability](https://github.com/mozilla/readability) to pre-process the content for reading. This may drastically improve the content for LLM usage.
    `return_format` (string): The format to return the data in. Possible values are `markdown`, `raw`, `text`, and `html2text`. Use `raw` to return the default format of the page like HTML etc.
    - `proxy_enabled` (bool): Enable high performance premium proxies for the request to prevent being blocked at the network level.
    - `query_selector` (string): The CSS query selector to use when extracting content from the markup.
    - `full_resources` (bool): Crawl and download all the resources for a website.
    - `request_timeout` (int): The timeout to use for request. Timeouts can be from `5-60`. The default is `30` seconds.
    - `run_in_background` (bool): Run the request in the background. Useful if storing data and wanting to trigger crawls to the dashboard. This has no effect if storageless is set.
