# EdgepediaTool

## Description

`EdgepediaTool` searches [Edgepedia](https://www.edgepedia.org) and retrieves cited Markdown encyclopedia articles with attribution lines. It connects directly to the public keyless Edgepedia REST API (`https://www.edgepedia.org/api/v1`) and requires no API key or sign-up.

- More than 519,000 cited articles across people, places, events, science, history, and culture.
- Stored translations across 31 languages, including Chinese, Spanish, Arabic, Hindi, French, Japanese, and 25 additional languages.
- Optional compact citation mode (`citations=False`) using `[n]` markers.

## Installation

`EdgepediaTool` is included in `crewai-tools` with no extra dependencies:

```bash
pip install crewai-tools
```

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import EdgepediaTool

edgepedia_tool = EdgepediaTool()

researcher = Agent(
    role="Encyclopedia Researcher",
    goal="Verify factual details with primary citations from Edgepedia",
    backstory="You look up authoritative reference facts and preserve source credits.",
    tools=[edgepedia_tool],
)

task = Task(
    description="Look up Mount Fuji in Edgepedia and summarize its elevation and last eruption.",
    expected_output="A concise factual summary with the Edgepedia credit line.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

### Multilingual Lookup

```python
from crewai_tools import EdgepediaTool

tool = EdgepediaTool(default_lang="zh", default_citations=False)
print(tool.run(query="Mount Fuji", limit=1))
```

## Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str` | Required | Subject name or title words to search in Edgepedia. |
| `lang` | `str \| None` | `None` | Optional language code across 31 languages (for example, `lang="zh"` for Chinese or `lang="es"` for Spanish). |
| `limit` | `int` | `1` | Number of matching articles to fetch (1 to 5). |
| `citations` | `bool \| None` | `None` | `True` for inline source URLs, or `False` for shorter `[n]` citation markers. |

## License and Attribution

Edgepedia articles are published under the [Edgepedia Community License 1.0](https://www.edgepedia.org/edgepedia/license) (free with credit, commercial use included, and AI training open to everyone). `EdgepediaTool` includes each article's `credit` line in its output.
