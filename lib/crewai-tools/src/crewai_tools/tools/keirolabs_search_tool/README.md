# KeirolabsSearchTool

## Description

`KeirolabsSearchTool` searches the web through [Keirolabs](https://keirolabs.cloud), a search API built for AI agents. Each result has a title, a link and a snippet. Keirolabs charges a flat price per search whatever the result count, from $0.10 per 1,000 searches.

## Installation

```shell
pip install 'crewai[tools]'
```

Get an API key at [platform.keirolabs.cloud](https://platform.keirolabs.cloud/register) and set it:

```shell
export KEIROLABS_API_KEY="keiro_..."
```

## Usage

```python
from crewai import Agent
from crewai_tools import KeirolabsSearchTool

search = KeirolabsSearchTool(n_results=5)

researcher = Agent(
    role="Researcher",
    goal="Find current information on the web",
    backstory="You look things up before answering.",
    tools=[search],
)
```

## Parameters

- `n_results`: how many results to return (default 10). The endpoint returns at most about 15.
- `timeout`: request timeout in seconds (default 30).

The tool returns a list of `{"title", "link", "snippet"}` dicts.
