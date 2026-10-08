# StringWebAccessScrapeTool / StringWebAccessSearchTool

## Description

[String Web Access](https://usestring.ai) gives an agent the live web. Search the web and fetch any
URL, all returned as clean, LLM-ready Markdown. Proxy rotation, anti-bot handling, CAPTCHA solving
and JavaScript rendering happen server-side, so the agent gets the page instead of a block screen.
Best for sites that rate-limit, geo-gate or block automated traffic.

Two tools:

- `StringWebAccessScrapeTool` — fetch a URL as Markdown, the verbatim body, or a JSON envelope with
  the destination's status code and headers.
- `StringWebAccessSearchTool` — search Google, DuckDuckGo, Brave or Mojeek and return the organic
  results as structured JSON.

## Installation

Both tools ship with `crewai-tools` and use `requests`, so there is nothing extra to install.

```shell
uv add crewai-tools
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `STRING_API_KEY` | yes | Your String API key. Get one at [usestring.ai](https://usestring.ai). |

You can pass the key directly instead: `StringWebAccessScrapeTool(api_key="...")`.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import StringWebAccessScrapeTool, StringWebAccessSearchTool

researcher = Agent(
    role="Market researcher",
    goal="Find and read what competitors publish about their pricing",
    backstory="You read the live web and quote what the page actually says.",
    tools=[StringWebAccessSearchTool(), StringWebAccessScrapeTool()],
)

task = Task(
    description="Find the pricing page for {competitor} and summarize every tier and its price.",
    expected_output="A table of tiers with prices, and the URL you read them from.",
    agent=researcher,
)

Crew(agents=[researcher], tasks=[task]).kickoff(
    inputs={"competitor": "an example vendor"}
)
```

## Arguments

### `StringWebAccessScrapeTool`

| Argument | Default | Description |
| --- | --- | --- |
| `url` | — | **Required.** The http/https URL to fetch. |
| `format` | `markdown` | `markdown`, `raw` for the verbatim body, or `json` for a `{statusCode, headers, data}` envelope. |
| `main_content_only` | `None` | Strip navigation, footers and other page chrome from the Markdown. |
| `execute_js` | `None` | Render the page in a browser first. Use when a fetch of a JavaScript-rendered site comes back empty. |
| `country_code` | `None` | ISO 3166-1 alpha-2 country to route the request through, e.g. `"GB"`. |

### `StringWebAccessSearchTool`

| Argument | Default | Description |
| --- | --- | --- |
| `query` | — | **Required.** The search query to run. |
| `engine` | `google` | `google`, `duckduckgo`, `brave` or `mojeek`. |
| `country` | `US` | ISO 3166-1 alpha-2 country code used to localize results. |
| `language` | `None` | Language tag such as `"en"` or `"pt-br"`. |
| `max_results` | `10` | Maximum number of organic results to return. |

### Constructor arguments

Both tools accept `api_key`, `timeout` (seconds, default `120`) and `ignore_failures`. With
`ignore_failures=True` a failed call is logged and returns `None` instead of raising, so one bad URL
does not stop a crew.
