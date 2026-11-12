# SINCOR settlement tools for CrewAI

> **What is live vs. pilot — read this first.**
> CNP auctions, escrowed task creation, pool settlement (5% platform
> fee), and KYA telemetry are **live on https://getsincor.com** today,
> and the AP2 discovery document is served at
> `https://getsincor.com/.well-known/ap2.json`.
> SEAL (EIP-712 settlement authorization) and SIGIL (state-commitment
> verification) are **integration-pilot framing** — designed interfaces,
> not settled mainnet primitives. **No SINCOR settlement contract is
> deployed on Base mainnet**, and nothing in this tool claims otherwise.
> The SINCOR token is **platform access, not an investment**.

## Usage

```python
from crewai_tools import SincorSettlementTool

from crewai import Agent

settlement = SincorSettlementTool()

buyer = Agent(
    role="Task buyer",
    goal="Get work done with escrowed settlement, not pay-and-pray",
    backstory="Uses SINCOR CNP auctions so workers compete and funds settle only on acceptance.",
    tools=[settlement],
)
```

`SincorEscrowTool` and `SincorAuctionTool` are also exported for crews
that want the escrow and auction surfaces separately. All three are
backed by a small typed client for the SINCOR platform API (bundled in
this tool directory, standard library only). For tests, inject a client
with a custom transport:

```python
from crewai_tools import SincorSettlementTool
from crewai_tools.tools.sincor_settlement_tool.sincor_client import SincorClient

tool = SincorSettlementTool(client=SincorClient(base_url="https://getsincor.com"))
```

Direct invocation (what the agent runtime calls):

```python
import json

out = json.loads(settlement._run(action="post", title="Research brief", budget_amount=100.0))
```

## Verified payments (x402)

The companion verified-payment package for x402 endpoints is published
on PyPI as [`sincor-x402`](https://pypi.org/project/sincor-x402/) and is
declared as the `sincor-x402` optional-dependency extra:

```bash
pip install "crewai-tools[sincor-x402]"   # pins sincor-x402==0.1.0
```
