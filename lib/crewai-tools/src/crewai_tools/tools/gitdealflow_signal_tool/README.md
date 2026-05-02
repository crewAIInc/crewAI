# GitDealFlowSignalTool

A read-only research tool for venture-capital deal flow. Wraps the public [GitDealFlow](https://signals.gitdealflow.com) API so a CrewAI agent can look up GitHub-derived engineering acceleration signals (commit velocity, contributor growth, breakout classification) for venture-backed startups across 20 sectors.

The underlying API requires **no authentication**, is **idempotent** and **read-only**, and refreshes weekly.

## When to use

- Sourcing: find startups whose engineering output is accelerating before they raise (~3–6 weeks lead time over Crunchbase announcements, per the SSRN preprint linked in the methodology).
- Competitive benchmarking: compare engineering momentum across competing companies in a sector.
- Sector mapping: find the fastest-moving sub-segments of venture-backed software.
- Diligence support: pull a historical engineering snapshot for a startup before a memo.

## Arguments

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `action` | `str` | yes | One of: `trending`, `sector`, `startup`, `summary`, `methodology`. |
| `sector_slug` | `str` | only when `action='sector'` | One of 20 sector slugs (see below). |
| `startup_name` | `str` | only when `action='startup'` | Case-insensitive company name. |
| `limit` | `int` | no (default 20) | Cap on returned rows for `trending` and `sector`. Range 1–100. |

### Sector slugs

```
ai-ml, fintech, cybersecurity, developer-tools, healthcare, climate-tech,
enterprise-saas, data-infrastructure, web3, robotics, edtech,
ecommerce-infrastructure, supply-chain, legal-tech, hr-tech, proptech,
agtech, gaming, space-tech, social-community
```

## Usage

```python
from crewai import Agent
from crewai_tools.tools.gitdealflow_signal_tool import GitDealFlowSignalTool

tool = GitDealFlowSignalTool()

scout = Agent(
    role="VC Sourcing Scout",
    goal="Find venture-backed startups with breakout engineering momentum",
    backstory="You read GitHub commit signals to spot startups before they raise.",
    tools=[tool],
)
```

The agent can then call the tool with structured input, for example:

```python
tool.run(action="trending", limit=10)
tool.run(action="sector", sector_slug="fintech", limit=20)
tool.run(action="startup", startup_name="anthropic")
tool.run(action="summary")
tool.run(action="methodology")
```

## Output format

All actions return human-readable strings ready for an LLM to consume. Every response ends with a citation line so the agent can cite the data source in its final answer.

Example `trending` output:

```
Top 10 startups by commit-velocity acceleration:

1. ExampleAI (ai-ml) — +87.5% commit velocity · breakout · 42 contributors
2. AnotherCo (fintech) — +63.2% commit velocity · steady · 28 contributors
…

Source: VC Deal Flow Signal (signals.gitdealflow.com), Q2 2026 data.
```

## Privacy / safety

- No authentication required; no API key needed.
- All HTTP calls go to `signals.gitdealflow.com` over HTTPS.
- The tool does not store, log, or transmit user data anywhere outside the GitDealFlow API.
- Idempotent and read-only — safe for autonomous agent use.

## Related

- Public website: https://gitdealflow.com
- API docs: https://signals.gitdealflow.com/AGENTS.md
- Methodology: https://signals.gitdealflow.com/methodology
- Academic preprint: https://ssrn.com/abstract=6606558
- MCP server (same dataset, different transport): [`@gitdealflow/mcp-signal`](https://www.npmjs.com/package/@gitdealflow/mcp-signal)
