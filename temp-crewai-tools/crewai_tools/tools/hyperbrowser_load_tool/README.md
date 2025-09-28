# HyperbrowserLoadTool

## Description

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:
- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).

## Installation

- Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the `HYPERBROWSER_API_KEY` environment variable or you can pass it to the `HyperbrowserLoadTool` constructor.
- Install the [Hyperbrowser SDK](https://github.com/hyperbrowserai/python-sdk):

```
pip install hyperbrowser 'crewai[tools]'
```

## Example

Utilize the HyperbrowserLoadTool as follows to allow your agent to load websites:

```python
from crewai_tools import HyperbrowserLoadTool

tool = HyperbrowserLoadTool()
```

## Arguments

`__init__` arguments:
- `api_key`: Optional. Specifies Hyperbrowser API key. Defaults to the `HYPERBROWSER_API_KEY` environment variable.

`run` arguments:
- `url`: The base URL to start scraping or crawling from.
- `operation`: Optional. Specifies the operation to perform on the website. Either 'scrape' or 'crawl'. Defaults is 'scrape'.
- `params`: Optional. Specifies the params for the operation. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait.
