# KeenableSearchTool

## Description

`KeenableSearchTool` searches the internet using the [Keenable](https://keenable.ai)
Search API, a web search API built for AI agents.

Unlike most search tools, it works **without an API key** by default: with no
`KEENABLE_API_KEY` set it uses the keyless public endpoint. Set `KEENABLE_API_KEY`
to use the authenticated endpoint and lift rate limits.

## Installation

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import KeenableSearchTool

# Works with no API key (keyless free tier)
tool = KeenableSearchTool()

results = tool.run(query="latest developments in AI agents")
print(results)
```

## Configuration

- `KEENABLE_API_KEY` (env, optional): lifts rate limits. Not required.
- `KEENABLE_API_URL` (env, optional): base-URL override. Must be `https://`;
  plain `http://` is accepted only for loopback hosts (`localhost`,
  `127.0.0.1`, `::1`). Defaults to `https://api.keenable.ai`.

`run()` takes a single input, `query`. Everything else is constructor
configuration:

```python
tool = KeenableSearchTool(
    n_results=5,  # results to return (default 10)
    max_snippet_chars=300,  # cap per result description, 0 = no cap (default 500)
    timeout=15,  # request timeout in seconds (default 30)
)
```
