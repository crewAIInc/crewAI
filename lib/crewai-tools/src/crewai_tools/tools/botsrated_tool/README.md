# BotsRated Tools

## Description

[BotsRated](https://botsrated.com) (beta) probes about 4,800 x402 seller hosts on Base without paying
them. The hosts come from the CDP Bazaar, Agentic.Market, 402index and x402-list. Each probe records the
HTTP status, whether the 402 challenge was valid, latency, and the quoted price and payTo. Each probe is a
signed receipt.

These tools let an agent check a seller before it pays.

- `BotsRatedPreflightTool`: free, no API key, no wallet. If you pass the `pay_to` of the 402 option the agent
  is about to pay, `pay_to_check` is `matches_observed` only when BotsRated's own unpaid probe saw that payTo
  on that network. `listing_only` means only the seller's directory listing has it; `not_seen` means neither.
- `BotsRatedCheckTool`: $0.001 USDC on Base per call; the first 100 per day are free (`charged: false`, budget untouched). It is opt-in and needs a budget and a GET that can pay
  x402. It reads the quote unpaid first (no redirects), accepts it only if **every** payment option is exact
  Base USDC to BotsRated at $0.001 or less, and stops at the budget. A paid call that fails keeps its
  reservation (the payment may already have settled), so the budget is a hard ceiling.

## Installation

```shell
pip install 'crewai[tools]'
# for the paid check only:
pip install 'x402[evm,requests]'
```

## Example

```python
from crewai import Agent
from crewai_tools import BotsRatedPreflightTool

buyer = Agent(
    role="Purchasing agent",
    goal="Only pay x402 sellers whose payTo matches what BotsRated observed",
    backstory="You call BotsRated Preflight before every payment.",
    tools=[BotsRatedPreflightTool()],
)
```

Opt-in paid check:

```python
from x402 import x402ClientSync
from x402.http.clients import x402_requests
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact import ExactEvmScheme
from crewai_tools import BotsRatedCheckTool
from crewai_tools.tools.botsrated_tool import botsrated_payment_policy

client = x402ClientSync()
# Base mainnet only, and the signer may only pick BotsRated's own check terms.
client.register("eip155:8453", ExactEvmScheme(signer=EthAccountSigner(account)))
client.register_policy(botsrated_payment_policy)
check = BotsRatedCheckTool(
    budget_usd="0.02", paying_get=x402_requests(client).get
)  # at most $0.02 in total
```

Use a dedicated client (or wallet) for this tool: the policy rejects every payment that is not BotsRated's
check, so it should not be shared with clients that pay other sellers.

## Notes

`covered: false` means BotsRated has no data for the host. It says nothing about the seller. Results are
measurements, not endorsements.
