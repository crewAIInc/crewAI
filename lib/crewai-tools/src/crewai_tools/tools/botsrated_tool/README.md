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
- `BotsRatedCheckTool`: $0.001 USDC on Base per call; the first 100 per day are free (`charged: false`, budget
  untouched). It is opt-in and needs a budget and an `x402ClientSync`. It reads the quote unpaid (no
  redirects) and accepts it only if **every** payment option is exact Base USDC to BotsRated at $0.001 or less.
  It then signs that validated quote with your client and sends the signature itself, so no other payment can
  be signed. Once signed, the amount stays counted even if the call fails (it may have settled), so the budget
  is a hard ceiling.

## Installation

```shell
pip install 'crewai[tools]'
# for the paid check only:
pip install 'x402[evm]'
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
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact import ExactEvmScheme
from crewai_tools import BotsRatedCheckTool

client = x402ClientSync()
client.register("eip155:8453", ExactEvmScheme(signer=EthAccountSigner(account)))
check = BotsRatedCheckTool(
    budget_usd="0.02", x402_client=client
)  # at most $0.02 in total
```

## Notes

`covered: false` means BotsRated has no data for the host. It says nothing about the seller. Results are
measurements, not endorsements.
