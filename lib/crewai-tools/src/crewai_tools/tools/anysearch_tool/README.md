# AnySearchTool

## Description

`AnySearchTool` performs web searches through the [AnySearch](https://anysearch.com)
API and returns the most relevant results as a JSON string.

AnySearch accepts anonymous requests, so the tool works without any API key.
Set `ANYSEARCH_API_KEY` to send authenticated requests instead; the applicable
quota and permissions are determined by the AnySearch service.

## Installation

```shell
pip install 'crewai[tools]'
```

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `ANYSEARCH_API_KEY` | No | Sends authenticated requests. Omit it to send anonymous requests. |

## Example

```python
from crewai import Agent
from crewai_tools import AnySearchTool


# Works with no configuration at all
tool = AnySearchTool()

# Or tune the tool
tool = AnySearchTool(max_results=5, timeout=15)

agent = Agent(
    role="Researcher",
    goal="Find current information on the web",
    backstory="An analyst who always checks primary sources.",
    tools=[tool],
)
```

## Arguments

- `query` (str, required): the search query.


## Parameters

- `search_url` (str): AnySearch search endpoint. Default `https://api.anysearch.com/v1/search`.
  Must use `https://` whenever an API key is configured.
- `api_key` (str | None): AnySearch API key. Defaults to `ANYSEARCH_API_KEY`.
- `max_results` (int): number of results to return, between 1 and 10. Default `10`.
- `result_format` (`"json"` | `"markdown"`): content format. Default `"json"`.
- `timeout` (int): request timeout in seconds. Default `30`.
- `max_content_length_per_result` (int): truncation limit for each result's
  `content` field. Default `1000`.
