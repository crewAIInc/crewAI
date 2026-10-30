# StringWebAccessScrapeTool / StringWebAccessSearchTool

## Description

[String Web Access](https://usestring.ai) gives an agent the live web. Search the web and fetch any
URL, all returned as clean, LLM-ready Markdown.

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
| `country_code` | `None` | ISO 3166-1 alpha-2 country code, e.g. `"GB"`. The page comes back as it is served in that country. |
| `solve_captcha` | `False` | A challenge page fails fast and is reported as an error. Set `True` to have the request attempt the challenge. |

### `StringWebAccessSearchTool`

| Argument | Default | Description |
| --- | --- | --- |
| `query` | — | **Required.** The search query to run. |
| `engine` | `google` | `google`, `duckduckgo`, `brave` or `mojeek`. |
| `country` | `US` | ISO 3166-1 alpha-2 country code used to localize results. |
| `language` | `None` | Language tag such as `"en"` or `"pt-br"`. |
| `max_results` | `None` | Maximum number of organic results to return, 1 to 300. Unset returns one results page. When set, Google is paged until that many are collected, and each page is billed as one search. |

### Constructor arguments

Both tools accept `api_key`, `timeout` (seconds, default `120`) and `ignore_failures`. With
`ignore_failures=True` a failed call is logged and returns `None` instead of raising, so one bad URL
does not stop a crew.
