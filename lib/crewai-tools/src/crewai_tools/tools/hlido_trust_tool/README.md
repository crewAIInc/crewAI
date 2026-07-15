# HlidoTrustTool

Independent, evidence-backed trust checks on AI agents — so a crew can vet a
collaborator or tool **before** delegating work to it.

[Hlido](https://hlido.eu) publishes independent reviews of AI agents (score,
tier, verified claim audits, strengths and red flags). This module exposes two
tools:

- **`HlidoTrustCheckTool`** — given an agent slug (e.g. `aider`), returns its
  Hlido score (0-100), tier (`VITAL`/`STEADY`/`FADING`/`FLATLINE`), a `PASS`/`FAIL`
  gate against a `min_score`, and the reviewed strengths / red flags.
- **`HlidoRecommendTool`** — given a free-text need (e.g. "AI coding agent"),
  returns matching Hlido-reviewed agents ranked by trust score.

Both call the **public** Hlido API — **no API key required**, and no new
dependencies beyond `requests` (already a CrewAI dependency).

## Installation

`HlidoTrustTool` ships with `crewai-tools`; no extra install is needed.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import HlidoRecommendTool, HlidoTrustCheckTool

router = Agent(
    role="Delegation Router",
    goal="Only delegate coding work to agents that pass an independent Hlido trust check.",
    backstory="You refuse to rely on any agent Hlido has not vetted to STEADY tier or above.",
    tools=[HlidoRecommendTool(), HlidoTrustCheckTool()],
)

task = Task(
    description=(
        "Find a reliable AI coding agent with Hlido Recommend, then run Hlido Trust "
        "Check on it with min_score=70. Only approve it if the gate PASSES."
    ),
    expected_output="The chosen agent's name, its Hlido verdict, and APPROVE or REJECT.",
    agent=router,
)

Crew(agents=[router], tasks=[task]).kickoff()
```

## Tool inputs

### `HlidoTrustCheckTool`

| Argument    | Type    | Default | Description                                             |
| ----------- | ------- | ------- | ------------------------------------------------------- |
| `slug`      | `str`   | —       | Hlido agent slug (e.g. `aider`, `crewai`, `langchain`). |
| `min_score` | `float` | `70.0`  | Minimum acceptable Hlido score; sets the PASS/FAIL gate. |

### `HlidoRecommendTool`

| Argument    | Type    | Default | Description                                        |
| ----------- | ------- | ------- | -------------------------------------------------- |
| `need`      | `str`   | —       | Free-text description of what you need an agent for. |
| `min_score` | `float` | `70.0`  | Only recommend agents at or above this score.       |
| `limit`     | `int`   | `5`     | Maximum number of agents to return.                 |

## Notes

- Data source: the public Hlido reviews API (`hlido.eu/data/scorecards/<slug>.json`
  and `hlido.eu/data/review-registry.json`).
- Errors (unknown slug, network issues) are returned as plain, actionable strings
  rather than raised, so an agent can react to them.
