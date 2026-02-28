# Nory x402 Payment Tools

Enable CrewAI agents to make payments using the x402 HTTP protocol.

## Description

These tools allow AI agents to handle HTTP 402 "Payment Required" responses programmatically. When an agent encounters a paywall, it can:

1. Get payment requirements (amount, networks, wallet address)
2. Verify a signed transaction
3. Settle the payment on-chain (~400ms)
4. Retry the original request

## Installation

```bash
pip install 'crewai[tools]'
```

## Tools

| Tool | Description |
|------|-------------|
| `NoryPaymentRequirementsTool` | Get payment requirements for a resource |
| `NoryVerifyPaymentTool` | Verify a signed payment before settlement |
| `NorySettlePaymentTool` | Submit payment to blockchain (~400ms) |
| `NoryTransactionLookupTool` | Look up transaction status |
| `NoryHealthCheckTool` | Check service health |

## Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import NoryPaymentRequirementsTool, NorySettlePaymentTool

# Create tools
payment_tools = [
    NoryPaymentRequirementsTool(),
    NorySettlePaymentTool(),
]

# Create an agent with payment capabilities
payment_agent = Agent(
    role="Payment Handler",
    goal="Handle payment requests for premium APIs",
    backstory="Expert at processing x402 payments",
    tools=payment_tools,
)

# Create a task
task = Task(
    description="Check what payment is needed for /api/premium/data at $0.10",
    agent=payment_agent,
)

# Run
crew = Crew(agents=[payment_agent], tasks=[task])
result = crew.kickoff()
```

## Supported Networks

- **Solana** (mainnet/devnet) - sub-400ms settlement
- **Base**
- **Polygon**
- **Arbitrum**
- **Optimism**
- **Avalanche**
- **Sei**
- **IoTeX**

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NORY_API_KEY` | No | API key for authenticated requests |

## Links

- [Nory API](https://noryx402.com)
- [GitHub](https://github.com/TheMemeBanker/x402-pay)
- [OpenAPI Spec](https://noryx402.com/openapi.yaml)
