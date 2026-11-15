# SpecStreamExtractTool

## Description

The `SpecStreamExtractTool` reads the main text of one or more web pages with the [SpecStream](https://specstream.ai) Extract API. It returns a JSON list with `url`, `status`, `title` and `text` for each page.

## Installation

The tool uses `requests`, which `crewai-tools` already installs.

```shell
uv add 'crewai[tools]'
```

Set your SpecStream API key:

```bash
export SPECSTREAM_API_KEY='your-specstream-api-key'
```

## Example

```python
from crewai import Agent
from crewai_tools import SpecStreamExtractTool

extract_tool = SpecStreamExtractTool()

reader = Agent(
    role="Reader",
    goal="Read web pages and pull out the facts asked for",
    backstory="You read sources closely.",
    tools=[extract_tool],
)

# Or call it directly
print(extract_tool.run(urls=["https://example.com"]))
```

## Arguments

- `urls` (required): one URL or a list of up to 20 URLs.

## Constructor options

- `api_key`: defaults to the `SPECSTREAM_API_KEY` environment variable.
- `format`: `"markdown"` (default) or `"text"`.
- `timeout`: request timeout in seconds (120).
