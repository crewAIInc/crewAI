# MadeOnSol Tools

## Description

Tools for Solana trading intelligence via the [MadeOnSol API](https://madeonsol.com/api-docs):

- **MadeOnSolKolFeedTool** — latest trades from 2,000+ tracked Solana KOL (influencer) wallets.
- **MadeOnSolKolLeaderboardTool** — KOL performance rankings by realized PnL and win rate.
- **MadeOnSolKolCoordinationTool** — convergence signals: tokens multiple KOLs are buying in the same window.
- **MadeOnSolDeployerAlertsTool** — Pump.fun launch alerts enriched with the deployer's historical track record.
- **MadeOnSolTokenRiskTool** — token risk assessment (deployer reputation, sniper/bundler concentration, holder distribution).

## Installation

```shell
uv add crewai-tools madeonsol-x402
```

## Authentication

Either of:

- `MADEONSOL_API_KEY` — an `msk_` API key. A free tier (200 requests/day) is available at [madeonsol.com/pricing](https://madeonsol.com/pricing); free-tier live feeds are delayed 5 minutes, paid tiers are real-time.
- `SVM_PRIVATE_KEY` — a Solana private key for keyless [x402](https://madeonsol.com/x402) pay-per-call micropayments in USDC ($0.005–$0.02 per request, always real-time). No signup needed.

## Example

```python
from crewai import Agent
from crewai_tools import (
    MadeOnSolKolFeedTool,
    MadeOnSolKolCoordinationTool,
    MadeOnSolTokenRiskTool,
)

analyst = Agent(
    role="Solana Market Analyst",
    goal="Find tokens that profitable KOLs are converging on and assess their risk",
    backstory="An on-chain analyst tracking smart-money wallets on Solana.",
    tools=[
        MadeOnSolKolFeedTool(),
        MadeOnSolKolCoordinationTool(),
        MadeOnSolTokenRiskTool(),
    ],
)
```
