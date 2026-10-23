# BotsRated Tools

## Description

[BotsRated](https://botsrated.com) (beta) probes about 4,800 x402 seller hosts on Base without paying
them. The hosts come from the CDP Bazaar, Agentic.Market, 402index and x402-list. Each probe records the
HTTP status, whether the 402 challenge was valid, latency, and the quoted price and payTo. Each probe is a
signed receipt.

These tools let an agent check a seller before it pays.

- `BotsRatedPreflightTool`: free, no API key, no wallet. If you pass the `pay_to` of the 402 option the agent
  is about to pay, `pay_to_check` reports whether BotsRated has seen that payTo for the host.
- `BotsRatedCheckTool`: $0.001 USDC on Base per call; the first 100 per day are free (`charged: false`, budget untouched). It is opt-in and needs a budget and a GET that can pay
  x402. It reads the quote unpaid first, accepts only exact Base USDC to BotsRated at $0.001 or less, and
  stops at the budget.

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

client = x402ClientSync()
client.register("eip155:*", ExactEvmScheme(signer=EthAccountSigner(account)))
check = BotsRatedCheckTool(
    budget_usd="0.02", paying_get=x402_requests(client).get
)  # at most $0.02 in total
```

## Notes

`covered: false` means BotsRated has no data for the host. It says nothing about the seller. Results are
measurements, not endorsements.
