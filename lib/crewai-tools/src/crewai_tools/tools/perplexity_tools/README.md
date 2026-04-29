# PerplexitySearchTool

Search the web with Perplexity's Search API. Returns a ranked list of web
results (title, URL, snippet, date) for a given query, suitable for grounding
LLM answers.

- **Quickstart**: [Search API Quickstart](https://docs.perplexity.ai/docs/search/quickstart)
- **API reference**: [Search API reference](https://docs.perplexity.ai/api-reference/search-post)
- **Domain filters**: [Domain filter docs](https://docs.perplexity.ai/docs/search/filters/domain-filter)
- **Recency filters**: [Date/time filter docs](https://docs.perplexity.ai/docs/search/filters/date-time-filters)

## Why this tool

- **Ranked web results**: A direct line to Perplexity's web index, returning
  ranked results with titles, URLs, snippets, and dates.
- **Filterable**: Restrict results by domain (allow OR deny) and by recency
  window.
- **No model coupling**: This is the Search API, not chat — useful as a search
  primitive for any agent or RAG pipeline.

## Environment

- `PERPLEXITY_API_KEY` (required) — also accepts `PPLX_API_KEY`. Get a key at
  https://www.perplexity.ai/account/api/keys.

## Parameters

- `query` (str, required): The search query.
- `max_results` (int, default `5`): Maximum number of results to return.
- `search_domain_filter` (list[str], optional): Allow- or deny-list of domains.
  Prefix with `-` to exclude (e.g. `"-pinterest.com"`). Do not mix allow and
  deny entries in the same list.
- `search_recency_filter` (str, optional): One of `"hour"`, `"day"`, `"week"`,
  `"month"`, `"year"`.

## Direct usage

```python
from crewai_tools import PerplexitySearchTool

tool = PerplexitySearchTool()
output = tool.run(
    query="When was the United Nations established?",
    max_results=5,
    search_domain_filter=["un.org"],
    search_recency_filter="year",
)
print(output)
```

The tool returns a numbered, human-readable string of results. Each entry
includes title, URL, optional date, and snippet.

## Example with an agent

```python
import os
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import PerplexitySearchTool

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.5,
    api_key=os.getenv("GEMINI_API_KEY"),
)

search = PerplexitySearchTool()

researcher = Agent(
    role="Web Researcher",
    backstory="You are an expert web researcher.",
    goal="Find cited, high-quality sources and provide a brief answer.",
    tools=[search],
    llm=llm,
    verbose=True,
)

task = Task(
    description="Research recent advances in retrieval-augmented generation and produce a short, cited answer.",
    expected_output="A concise, sourced answer with URLs.",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()
print(result)
```

## Behavior

- Single-request web search; no scraping or post-processing needed.
- Returns ranked `results` with `title`, `url`, `snippet`, and optional `date`.
- Clear error handling on HTTP errors and timeouts.

## References

- Search API quickstart: https://docs.perplexity.ai/docs/search/quickstart
- Search API reference: https://docs.perplexity.ai/api-reference/search-post
- Domain filter: https://docs.perplexity.ai/docs/search/filters/domain-filter
- Recency/date filter: https://docs.perplexity.ai/docs/search/filters/date-time-filters
