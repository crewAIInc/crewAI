# SERPdive Search Tool

## Description

The `SerpdiveSearchTool` connects CrewAI agents to [SERPdive](https://serpdive.com), an AI search API that returns answer-ready web content: every result carries the actual text of the page (url, title, date, content), already extracted, cleaned, and sized for LLM consumption, so agents can quote and cite straight from the tool output. On a [public, replayable 1,000-question benchmark](https://github.com/edendalexis/serpdive-benchmark), SERPdive runs at the same speed as Tavily, feeds your LLM 20.2% fewer tokens, and wins 60.7% of decided quality duels.

The tool returns the search results as a JSON string.

## Installation

To use the `SerpdiveSearchTool`, you need to install the `serpdive` library:

```shell
pip install 'crewai[tools]' serpdive
```

## Environment Variables

Get a free API key (no card required) at [serpdive.com/dashboard/keys](https://serpdive.com/dashboard/keys) and set it as an environment variable:

```bash
export SERPDIVE_API_KEY='sd_live_...'
```

## Example

Here's how to initialize and use the `SerpdiveSearchTool` within a CrewAI agent:

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerpdiveSearchTool

# Initialize the tool
serpdive_tool = SerpdiveSearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role='Market Researcher',
    goal='Find current information about the solid state battery market',
    backstory='An expert market researcher specializing in energy technology.',
    tools=[serpdive_tool],
    verbose=True,
)

# Create a task for the agent
research_task = Task(
    description='Search for the main obstacles to mass production of solid state batteries.',
    expected_output='A short report citing the sources found.',
    agent=researcher,
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
)

result = crew.kickoff()
print(result)
```

## Arguments

Initialization arguments:

- `model` (str, optional): Retrieval depth. `"mako"` (default) returns the fact-carrying sentences of each page, fast. `"moby"` returns the full readable text of each page, for deep research.
- `answer` (bool, optional): When `True`, the output also carries an `answer` field: a direct answer synthesized from the sources (concise on `mako`, cited on `moby`). Defaults to `False`.
- `max_results` (int, optional): Hard cap on delivered results (1-10). By default the engine picks its calibrated mix.
- `api_key` (str, optional): Your SERPdive API key. If not provided, it's read from the `SERPDIVE_API_KEY` environment variable.

Run argument:

- `query` (str): **Required**. The search query, in any language.

There is no country or locale parameter to configure: the language of the query picks where the search runs, automatically. Failed searches are never billed.

## Links

- [API reference](https://serpdive.com/docs)
- [Public benchmark](https://github.com/edendalexis/serpdive-benchmark) (replayable end to end: same questions, same judge, your machine)
