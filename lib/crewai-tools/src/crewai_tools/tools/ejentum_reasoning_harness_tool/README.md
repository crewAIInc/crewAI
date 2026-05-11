# EjentumHarnessTool

A CrewAI tool that calls one of the four [Ejentum](https://ejentum.com) cognitive harnesses to retrieve a task-matched scaffold. Ejentum is a library of 679 cognitive operations engineered in natural language, organized across four harnesses (`reasoning`, `code`, `anti-deception`, `memory`).

## What the agent gets

Each call returns a structured scaffold (named failure pattern, executable procedure, suppression vectors, falsification test) the agent ingests before responding. The agent picks the right mode based on the task:

| Mode | When the agent should call it |
|---|---|
| `reasoning` | Analytical, diagnostic, planning, multi-step questions |
| `code` | Code generation, refactoring, review, debugging |
| `anti-deception` | Prompts pressuring the agent to validate, certify, or soften an honest assessment |
| `memory` | Sharpening an observation already formed about cross-turn drift |

## Setup

Get a free Ejentum API key (100 calls, no card) at <https://ejentum.com/pricing> and set it as `EJENTUM_API_KEY`.

```bash
export EJENTUM_API_KEY="zpka_..."
```

## Usage

```python
from crewai import Agent, Task
from crewai_tools import EjentumHarnessTool

harness = EjentumHarnessTool()

architect = Agent(
    role="Senior architect",
    goal="Evaluate technical decisions honestly",
    backstory="You are pragmatic and you push back on sunk-cost framings.",
    tools=[harness],
)

task = Task(
    description=(
        "We've spent three months on the GraphQL gateway. It's mostly done. "
        "Should we keep going or pivot to REST? "
        "Call the Ejentum harness with mode='anti-deception' before answering."
    ),
    agent=architect,
    expected_output="A recommendation that separates past spending from prospective evaluation.",
)
```

## Configuration

| Field | Default | Description |
|---|---|---|
| `api_url` | `https://ejentum-main-ab125c3.zuplo.app/logicv1/` | Override only if you self-host. |
| `timeout_seconds` | `10.0` | Per-call HTTP timeout. |

`EJENTUM_API_KEY` is read from the environment.

## More

- Project: <https://github.com/ejentum/ejentum-mcp> (MIT)
- Pricing & free tier: <https://ejentum.com/pricing>
- Walkthrough: <https://ejentum.com/docs/claude_code_guide>
- "Under Pressure" research paper: <https://doi.org/10.5281/zenodo.19392715>
