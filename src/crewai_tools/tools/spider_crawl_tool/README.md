# SpiderTool

## Description

[Spider](https://spider.cloud) is the [fastest]([Spider](https://spider.cloud/?ref=crewai) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md#benchmark-results) open source scraper and crawler that returns LLM-ready data. It converts any website into pure HTML, markdown, metadata or text while enabling you to crawl with custom actions using AI.

## Installation

To use the Spider API you need to download the [Spider SDK](https://pypi.org/project/spider-client/) and the crewai[tools] SDK too:

```python
pip install spider-client 'crewai[tools]'
```

## Example

This example shows you how you can use the Spider tool to enable your agent to scrape and crawl websites. The data returned from the Spider API is already LLM-ready, so no need to do any cleaning there.

```python
from crewai_tools import SpiderTool

tool = SpiderTool()
```

## Arguments

- `api_key`: Optional. Specifies Spider API key. If not specified it looks for `SPIDER_API_KEY` in environment variables.
