# SearchApi Tool

## Description

The `SearchApiTool` searches the internet through [SearchApi](https://www.searchapi.io). SearchApi puts many engines behind one endpoint, so a single tool covers Google web search, news, scholar, jobs, Bing, YouTube, Baidu and the rest of the [supported engines](https://www.searchapi.io/docs) by changing the `engine` argument. The tool returns the engine's own JSON, so results line up with the engine's documentation.

Two things happen to the response before an agent sees it:

- Inline `data:` URIs are dropped. SearchApi returns favicons and thumbnails as base64 strings, and a single one can run to tens of kilobytes of context that means nothing to an agent.
- Long strings are truncated at `max_string_length`, and every `*_results` list is capped at `n_results`.

## Installation

No extra package is needed:

```shell
uv add 'crewai[tools]'
```

## Environment Variables

Set your SearchApi key:

```bash
export SEARCHAPI_API_KEY='your_searchapi_key'
```

The key is sent in the `Authorization` header rather than the query string, so it stays out of request logs and out of the `request_url` SearchApi echoes back in `search_metadata`.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import SearchApiTool

search_tool = SearchApiTool()

researcher = Agent(
    role="Market Researcher",
    goal="Find what people are saying about a company right now",
    backstory="An analyst who checks the record before forming a view.",
    tools=[search_tool],
    verbose=True,
)

research_task = Task(
    description="Search for recent coverage of CrewAI.",
    expected_output="A short report on what the search returned.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[research_task], verbose=True)
result = crew.kickoff()
print(result)
```

Point the same tool at a different engine:

```python
news_tool = SearchApiTool(engine="google_news")
scholar_tool = SearchApiTool(engine="google_scholar", n_results=5)
uk_tool = SearchApiTool(country="uk", locale="en", location="London,England")
```

## Arguments

- `engine` (str, optional): The SearchApi engine to query, such as `"google"`, `"google_news"`, `"google_scholar"`, `"google_jobs"`, `"bing"`, `"youtube"` or `"baidu"`. Defaults to `"google"`. Can also be passed per call.
- `n_results` (int, optional): Cap on the length of each `*_results` list in the response. Defaults to `10`.
- `country` (str, optional): Country of the search, sent as `gl` (for example `"uk"`). Defaults to `None`.
- `locale` (str, optional): Interface language, sent as `hl` (for example `"en"`). Defaults to `None`.
- `location` (str, optional): Canonical location of the search, for example `"London,England"`. Defaults to `None`.
- `max_string_length` (int, optional): Longest string kept intact in the response. Defaults to `1000`.
- `timeout` (int, optional): Request timeout in seconds. Defaults to `30`.
- `api_key` (str, optional): Your SearchApi key. Falls back to the `SEARCHAPI_API_KEY` environment variable.
- `search_url` (str, optional): The endpoint to call. Defaults to `https://www.searchapi.io/api/v1/search`.

## Errors

A failed request raises a `RuntimeError` carrying SearchApi's own message, for example `SearchApi request failed (HTTP 401): Invalid API key.`.

A successful search that found nothing is not an error: SearchApi returns HTTP 200 with an `error` message such as `"Google didn't return any results."`, and the tool passes that through so the agent can read why the page was empty.
