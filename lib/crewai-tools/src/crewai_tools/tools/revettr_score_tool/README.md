# RevettrScoreTool

Score counterparties 0-100 before sending payments in agentic commerce.

Covers domain intelligence, IP reputation, wallet history, and sanctions screening (OFAC/EU/UN). Uses the [Revettr](https://revettr.com) x402-native API -- no API keys needed, just a funded wallet.

## Dependencies

```bash
pip install revettr
# For x402 auto-payment:
pip install revettr[x402]
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REVETTR_WALLET_KEY` | Optional | EVM private key for x402 auto-payment ($0.01 USDC per score on Base) |

## Usage

```python
from crewai_tools import RevettrScoreTool

tool = RevettrScoreTool()

# Score by domain
result = tool.run(domain="uniswap.org")

# Score by wallet address
result = tool.run(wallet_address="0x1234...", chain="base")

# Score by company name (sanctions screening)
result = tool.run(company_name="Acme Corp")

# Combine multiple signals for higher confidence
result = tool.run(
    domain="merchant.com",
    wallet_address="0xabc...",
    company_name="Merchant Inc",
)
```
