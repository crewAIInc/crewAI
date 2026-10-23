# Mekong x402 Tools

Native CrewAI tools for the [Mekong x402 fleet](https://api.mekongtradeintel.com/tools/) —
37 pay-per-call APIs on Base (USDC). **No signup, no API key.** Your wallet is the credential.

## Tools

| Tool | What it does | Price |
|---|---|---|
| `MekongTaskRouterTool` | Describe a task in plain language → top-3 endpoints + curls | **FREE** |
| `MekongWebSearchTool` | Live web search, deduped + relevance-scored | $0.01 |
| `MekongTokenPriceTool` | Token price / mcap / liquidity (chain+address exact) | $0.001 |
| `MekongAIVerdictTool` | Token safety verdict SAFE/RISKY/SCAM + reasoning | $0.01 |
| `MekongEnrichTool` | Company enrichment from a domain | $0.02 |
| `MekongSTTTool` | Speech-to-text from an audio URL | $0.05 |
| `MekongBriefTool` | Morning briefing: news + trends + sentiment | $0.05 |

## Usage

```python
from crewai import Agent
from crewai_tools import MekongTaskRouterTool, MekongWebSearchTool

agent = Agent(
    role="Research Analyst",
    goal="Brief me with live data",
    tools=[MekongTaskRouterTool(), MekongWebSearchTool()],
    backstory="You use pay-per-call tools. No API keys needed.",
)
```

No environment variables required — there are no API keys. Every endpoint has a free
daily trial quota per IP; paid calls raise `MekongPaymentRequired` with exact x402 pay
instructions once the trial is exhausted (never a silent charge).

## Verify independently

https://api.mekongtradeintel.com/tools/verify — one-command script that checks paywall
shape, CDP preflight and on-chain settlement from the outside.
