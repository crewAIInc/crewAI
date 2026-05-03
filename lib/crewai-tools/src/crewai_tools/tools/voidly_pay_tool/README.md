# VoidlyPayTool Documentation

## Description

`VoidlyPayTool` lets a CrewAI agent call any HTTP endpoint that speaks the
[x402 protocol](https://x402.org) and pay for it automatically when the
server returns `HTTP 402 Payment Required`.

It wraps the official [`voidly-pay-crewai`](https://pypi.org/project/voidly-pay-crewai/)
PyPI package; if that package is installed the tool delegates to its richer
`VoidlyPayFetchTool`. Otherwise it falls back to a self-contained x402
round-trip using [`voidly-pay`](https://pypi.org/project/voidly-pay/)
directly so the import never breaks.

Settles in <200ms via a Sourcify-verified USDC vault on Base mainnet
([`0xb592...1c12`](https://repo.sourcify.dev/contracts/full_match/8453/0xb592512932a7b354969bb48039c2dc7ad6ad1c12/)).

## Installation

```shell
pip install voidly-pay voidly-pay-crewai
pip install 'crewai[tools]'
```

Then either set environment variables:

```shell
export VOIDLY_PAY_DID="did:voidly:..."
export VOIDLY_PAY_SECRET="..."  # base64 secret key
```

…or visit <https://voidly.ai/pay/claim> to mint a DID + claim the
free 10-credit faucet in the browser. The page shows you the secret to
copy into the env vars.

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import VoidlyPayTool

researcher = Agent(
    role="Research Analyst",
    goal="Pay for premium APIs without managing API keys.",
    backstory="Holds the team's wallet. Verifies before settling.",
    tools=[VoidlyPayTool()],
)

task = Task(
    description=(
        "Use voidly_pay_fetch to call "
        "https://api.voidly.ai/v1/pay/wiki?topic=Anthropic and summarise the "
        "result in two sentences."
    ),
    agent=researcher,
    expected_output="Two-sentence summary of the wiki entry.",
)

crew = Crew(agents=[researcher], tasks=[task])
crew.kickoff()
```

## Args

- `url` (str, required) — endpoint URL.
- `method` (str, default `GET`) — HTTP method.
- `body` (Any, optional) — JSON-serialisable body for POST/PUT.
- `max_amount_credits` (float, default `0.05`) — guardrail on a single
  auto-pay (1 credit = $0.001 USDC).

## How it works

1. The tool issues the request. If the server responds 200, it returns the
   body unchanged.
2. On 402, it parses the [voidly-credit accept option](https://api.voidly.ai/v1/pay/manifest.json):
   `recipient_did`, `amount_micro`, `quote_id`.
3. If the asking price exceeds `max_amount_credits`, the tool refuses
   without spending.
4. Otherwise, it Ed25519-signs a `pay()` transfer to the recipient DID.
5. Re-issues the call with `?quote_id=<id>` and returns the resolved
   response (with `transfer_id` for receipt tracking).

## Discoverability

- Marketplace of paid endpoints: <https://api.voidly.ai/v1/pay/marketplace>
- List your own service: <https://voidly.ai/pay/list-your-service>
- Manifest: <https://api.voidly.ai/v1/pay/manifest.json>
- Public proof of reserves: <https://voidly.ai/pay/proof>
