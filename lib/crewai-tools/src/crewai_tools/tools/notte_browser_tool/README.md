# NotteBrowserTool

Drive a managed remote [Notte](https://notte.cc) browser session from any CrewAI agent. Notte provides hosted browser infrastructure plus a perception layer that lets agents act on the live web using natural language; sessions are remote-hosted, so there is no local browser to install or babysit.

The tool mirrors the shape of `StagehandTool`: a single tool class that accepts a `command_type` of `navigate`, `act`, `extract`, or `observe`.

## Installation

```bash
uv add notte-sdk
# or
pip install notte-sdk
```

## Authentication

Sign up at [notte.cc](https://notte.cc) for an API key, then export it:

```bash
export NOTTE_API_KEY="your-notte-api-key"
```

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools import NotteBrowserTool

with NotteBrowserTool() as notte_tool:
    researcher = Agent(
        role="Web Researcher",
        goal="Find and summarise information from websites",
        backstory="I am an expert at finding things online.",
        tools=[notte_tool],
    )
    research = Task(
        description=(
            "Navigate to https://news.ycombinator.com and report the titles "
            "of the top three posts as a numbered list."
        ),
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[research])
    print(crew.kickoff())
```

## Command types

| `command_type` | Purpose | Required input |
| --- | --- | --- |
| `navigate` | Open a URL in the session. | `url` |
| `act` | Perform a natural-language action on the current page (e.g. "click the sign-up button"). | `instruction` |
| `extract` | Extract data from the current page. | `instruction` describing what to extract |
| `observe` | List the interactive elements on the current page. | (none) |

`act` is executed via Notte's autonomous agent with a tight step budget (default `max_act_steps=3`) because Notte's low-level act surface is typed rather than free-text.

## Links

- Notte product: <https://notte.cc>
- Notte API and SDK reference: <https://docs.notte.cc>
- Notte Python SDK on PyPI: [`notte-sdk`](https://pypi.org/project/notte-sdk/)
