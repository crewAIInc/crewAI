# FirecrawlInteractTool

## Description

[Firecrawl](https://firecrawl.dev) is a platform for crawling and converting any website into clean markdown or structured data. This tool runs an autonomous browser **agent** that navigates and interacts with web pages to accomplish a natural-language task, then returns the result.

## Installation

- Get an API key from [firecrawl.dev](https://firecrawl.dev) and set it in environment variables (`FIRECRAWL_API_KEY`).
- Install the [Firecrawl SDK](https://github.com/firecrawl/firecrawl) along with `crewai[tools]` package:

```shell
pip install firecrawl-py 'crewai[tools]'
```

## Example

Utilize the FirecrawlInteractTool as follows to let your agent act on the web:

```python
from crewai_tools import FirecrawlInteractTool

tool = FirecrawlInteractTool(config={"model": "spark-1-mini"})
tool.run(prompt="Find the pricing page on firecrawl.dev and return the plan names")
```

## Arguments

- `prompt`: Required. A natural-language description of the task for the agent to carry out.
- `urls`: Optional. A list of URLs to start from or constrain the agent to.
- `api_key`: Optional. Specifies the Firecrawl API key. Defaults to the `FIRECRAWL_API_KEY` environment variable.
- `config`: Optional. It contains Firecrawl v2 agent parameters.

This is the default configuration

```python
{
    "model": "spark-1-mini",
    "max_credits": None,
    "strict_constrain_to_urls": None,
    "poll_interval": 2,
    "timeout": None,
}
```
