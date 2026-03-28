# Oris

[Oris](https://useoris.xyz) provides AI agent payment infrastructure: KYA identity verification,
programmable spending policies, and real-time sanctions screening before every on-chain payment.

## Installation

```bash
pip install crewai-oris
```

Get an API key at [useoris.xyz](https://useoris.xyz).

## Available tools

| Tool | Description |
|---|---|
| `OrisPaymentTool` | Execute a compliant stablecoin payment |
| `OrisCheckBalanceTool` | Retrieve wallet balances |
| `OrisGetSpendingTool` | Get payment and spending history |
| `OrisGetTierInfoTool` | Get KYA compliance tier and limits |

## Setup

Set credentials as environment variables:

```bash
export ORIS_API_KEY="oris_sk_live_..."
export ORIS_API_SECRET="your_api_secret"
export ORIS_AGENT_ID="your_agent_uuid"
```

Tools read from these variables by default. Credentials can also be passed directly to each
tool constructor.

## Example

```python
from crewai import Agent, Task, Crew
from oris_crewai import OrisPaymentTool, OrisCheckBalanceTool

finance_agent = Agent(
    role="Finance Agent",
    goal="Execute compliant payments on behalf of the crew",
    backstory=(
        "You handle all financial transactions with built-in KYA compliance "
        "and programmable spending policy enforcement."
    ),
    tools=[OrisPaymentTool(), OrisCheckBalanceTool()],
    verbose=True,
)

payment_task = Task(
    description="Pay 25 USDC to 0xRecipient... for the vendor invoice.",
    expected_output="Payment confirmation with transaction hash",
    agent=finance_agent,
)

crew = Crew(agents=[finance_agent], tasks=[payment_task])
result = crew.kickoff()
print(result)
```

## Payment pipeline

Each payment passes through four stages before execution:

1. **KYA identity check** — agent registration verified (p95 &lt; 5ms)
2. **Policy evaluation** — per-transaction limits and daily caps checked (p95 &lt; 10ms)
3. **Sanctions screening** — OFAC, EU, and UN consolidated lists (p95 &lt; 100ms)
4. **On-chain execution** — ERC-4337 UserOp on Base, Polygon, Arbitrum, and more

If any stage fails or is unreachable, the task returns a structured error string.

## Documentation

Full API reference: [docs.useoris.xyz](https://docs.useoris.xyz)
