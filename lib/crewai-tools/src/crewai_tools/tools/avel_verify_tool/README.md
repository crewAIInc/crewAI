# AvelVerifyTool

The **AvelVerifyTool** checks whether the agent or merchant about to receive a payment is a real,
identified party with a reasonable reputation — before the payment goes out. It's complementary to
payment-execution tools (batch payments, escrow, etc.), not a replacement for them: this tool never
moves money, the payment itself stays on whatever rail is normally used (x402, card, bank transfer).

Backed by [AVEL](https://github.com/Ensi81/Avel), a verification layer that scores agents on
reputation derived from real recorded transaction outcomes, never a self-reported value.

Requires the `agent-interchange` package: `uv add agent-interchange`.

## Arguments

| Argument   | Type    | Required | Description                                                    |
| ---------- | ------- | -------- | ---------------------------------------------------------------|
| `payee_id` | `str`   | ✅        | Identifier of the agent or merchant about to receive payment.  |
| `amount`   | `float` | ✅        | Transaction amount, in `currency` units.                       |
| `currency` | `str`   | ✅        | Currency of `amount`, e.g. `USDC` or `EUR`.                     |
| `nonce`    | `str`   | ❌        | Optional nonce binding this check to one specific transaction. |

## Initialization Parameters

| Parameter      | Type  | Default                       | Description                                                                       |
| -------------- | ----- | ------------------------------| ----------------------------------------------------------------------------------|
| `private_key`  | `str` | `AVEL_PRIVATE_KEY` env var    | EVM private key of the paying agent's identity. Signs every check, never exposed to the model. |
| `base_url`     | `str` | `https://aisrail.fly.dev`     | Base URL of the AVEL instance to verify against.                                  |

## Usage Example

```python
from crewai import Agent
from crewai_tools import AvelVerifyTool

# private_key can also come from the AVEL_PRIVATE_KEY env var
avel_verify = AvelVerifyTool(private_key="0x...")

agent = Agent(
    role="Purchasing Agent",
    goal="Only pay counterparties that check out",
    backstory="An agent that checks before it pays.",
    tools=[avel_verify],
)
```

```python
avel_verify.run(payee_id="merchant-42", amount=25.0, currency="EUR")
# 'APPROVED. Reasons: Payer: no prior history. Payer reputation: {...}.'
```

One tool instance represents one payer identity — construct a separate `AvelVerifyTool` per agent
identity that needs its own reputation checked.
