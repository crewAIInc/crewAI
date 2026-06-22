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

- `KEENABLE_API_KEY` (env, optional): lifts rate limits and enables `realtime` mode.
- `KEENABLE_API_URL` (env, optional): base-URL override (HTTPS). Defaults to
  `https://api.keenable.ai`.

Tool arguments:

- `mode` (`"pro"` default, or `"realtime"` — requires a key)
- `n_results` (default `10`)
- `timeout` (default `30`)
