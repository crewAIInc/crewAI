# HlidoTrustTool

## Description

`HlidoTrustTool` lets a CrewAI agent vet another AI agent against its independent
[Hlido](https://hlido.eu) review **before delegating to it**. Hlido publishes
independent, evidence-backed trust scores for AI agents (independent — it doesn't
take money to rank anyone). The tool returns a PASS/FAIL gate plus the agent's
score, tier, what it fails at, red flags, and an evidence URL.

Agents with no Hlido review, or with recorded red flags, **fail the gate**
(fail-closed).

## Installation

```shell
pip install 'crewai[tools]' hlido-trust
```

No API key is required for trust checks.

## Example

```python
from crewai import Agent
from crewai_tools import HlidoTrustTool

tool = HlidoTrustTool()

router = Agent(
    role="Delegation Router",
    goal="Only delegate to agents that pass an independent Hlido trust check.",
    backstory="You refuse to rely on any agent Hlido hasn't vetted to STEADY tier or above.",
    tools=[tool],
)
```

The agent calls the tool with an agent `slug` (e.g. `"aider"`) and an optional
`min_score` (default `70`):

```
[PASS @ min_score=70.0] Aider — VITAL (92/100), confidence high. Evidence: https://hlido.eu/reviews/aider/
```

## Arguments

- `slug` (str, required): the Hlido agent slug to vet, e.g. `"aider"` or `"crewai"`.
- `min_score` (float, optional, default `70.0`): minimum acceptable Hlido score
  (0–100). `70` = STEADY tier or above.

## Environment variables

- `HLIDO_API_KEY` (optional): a `hlk_live_*` key from <https://hlido.eu/api/>,
  needed only for top-k recommendations — not for trust checks.
