# SoChain Tools

## Description

Tools that read Bitcoin, Litecoin, Dogecoin and Zcash blockchain data through the [SoChain API v3](https://sochain.com/docs): address balances and summaries, address transactions, transactions, blocks, and network state with fee rates.

| Tool | What it returns |
|------|-----------------|
| `SoChainAddressBalanceTool` | Confirmed balance and pending change of an address |
| `SoChainAddressSummaryTool` | Balances, total received, transaction counts, first and last activity |
| `SoChainAddressTransactionsTool` | An address's transactions, ten per page, newest first |
| `SoChainTransactionTool` | One transaction with its inputs, outputs and fee |
| `SoChainBlockTool` | One block by height or hash (transaction IDs on request) |
| `SoChainNetworkInfoTool` | Newest block, hashrate, mempool and projected fee rates |

Every tool takes a `network`: `BTC`, `LTC`, `DOGE`, `ZEC`, or a testnet: `BTCTEST` (Bitcoin testnet4), `LTCTEST`, `DOGETEST`, `ZECTEST`. Amounts are decimal strings in coin units, as the API returns them.

## Installation

```shell
pip install 'crewai[tools]'
```

## API key

The tools work without a key, within SoChain's keyless daily limit per IP address. For higher limits, create a free account and key at [sochain.com/dashboard](https://sochain.com/dashboard) and either set it in the environment:

```shell
export SOCHAIN_API_KEY="your-key"
```

or pass it to a tool: `SoChainAddressSummaryTool(api_key="your-key")`. The key is sent as the `API-KEY` header. Plans are listed at [sochain.com/pricing](https://sochain.com/pricing).

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import (
    SoChainAddressSummaryTool,
    SoChainAddressTransactionsTool,
    SoChainNetworkInfoTool,
    SoChainTransactionTool,
)

analyst = Agent(
    role="Blockchain analyst",
    goal="Answer questions about addresses and transactions",
    backstory="You read on-chain data and report exact amounts.",
    tools=[
        SoChainAddressSummaryTool(),
        SoChainAddressTransactionsTool(),
        SoChainTransactionTool(),
        SoChainNetworkInfoTool(),
    ],
)

task = Task(
    description=(
        "Summarize the DOGE address DGUzATWpGdehtujWNyM464LB9Zx7Ch1Z67: its "
        "balance, how many transactions it has, and its latest transaction."
    ),
    expected_output="A short summary with exact amounts.",
    agent=analyst,
)

print(Crew(agents=[analyst], tasks=[task]).kickoff())
```

Calling a tool directly:

```python
from crewai_tools import SoChainBlockTool

print(SoChainBlockTool().run(network="BTC", block="840000"))
```

## Errors

API failures come back as a short message for the agent, for example `SoChain API error (HTTP 404): Unable to fetch address balance. Invalid address?`. When the API answers 429 with a `Retry-After` of up to 10 seconds, the tool waits and retries once.
