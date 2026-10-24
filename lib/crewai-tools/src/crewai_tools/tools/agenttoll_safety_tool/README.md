# AgentTollSafetyTool Documentation

## Description

This tool checks whether a Base (Coinbase's L2) token contract is a honeypot or a rug
before you trade it or recommend it. It calls
[AgentToll](https://agenttoll.app)'s `/api/base/safety` endpoint, which runs a
simulated buy **and** sell, checks buy/sell tax, owner privileges, holder
concentration, liquidity risk, and the deployer's own history — a token shipped from a
wallet with a handful of transactions and dust is the shape most rugs share.

There is no API key and no subscription. Every call is paid for inline, in USDC on
Base, via the [x402 protocol](https://x402.org) (HTTP 402) — the agent's wallet pays
$0.003 per call, and only once the response comes back successfully.

## Installation

```shell
uv add crewai-tools --extra x402
# or
pip install 'crewai[tools]' 'x402[requests,evm]'
```

## Example

```python
from crewai_tools import AgentTollSafetyTool

# EVM_PRIVATE_KEY must be set to a Base wallet holding a little USDC
tool = AgentTollSafetyTool()

result = tool.run(address="0x940181a94a35a4569e4529a3cdfb74e38fd98631")
```

## Steps to Get Started

1. **Package installation**: install the `x402` extra as shown above.
2. **Wallet funding**: the wallet behind `EVM_PRIVATE_KEY` needs a small amount of USDC
   on Base mainnet (each call costs $0.003). To test without real funds, pass
   `base_url="http://localhost:4021"` when constructing the tool to point at a
   self-hosted AgentToll instance running on Base Sepolia, and fund the wallet with
   free testnet USDC from [faucet.circle.com](https://faucet.circle.com).
3. **Environment configuration**: `EVM_PRIVATE_KEY=0x...`

## Arguments

| Argument | Type | Description |
|---|---|---|
| `address` | `str` | Base token contract address to check, e.g. `0x1234...` |

## Conclusion

`AgentTollSafetyTool` gives an agent a real safety verdict on a Base token — clear,
caution, high-risk, or insufficient-data — without an account, an API key, or a
subscription: the agent pays for exactly the checks it runs, and nothing when a call
fails.
