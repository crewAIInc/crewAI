# SkimReaderTool

## Description

[Skim](https://skim402.com) is the x402-native clean reader API for AI agents.
Give it any URL and it returns clean, agent-ready Markdown — nav, ads, and
boilerplate stripped — plus structured metadata (title, byline, published date,
language, excerpt).

`SkimReaderTool` is pay-per-call over the [x402](https://x402.org) protocol:
each read costs **$0.002 in USDC on Base**, paid automatically by a wallet you
control. There is no signup and there are no API keys — your wallet is your
identity. The private key never leaves your machine; it only signs an EIP-3009
USDC authorization locally.

## Installation

Install the x402 client (with EVM support) alongside `crewai[tools]`:

```
pip install "x402[evm]" 'crewai[tools]'
```

Fund a dedicated Base wallet with a small amount of USDC (about $1 covers ~500
reads) and expose its private key to the tool:

```
export SKIM_WALLET_PRIVATE_KEY=0xYOUR_BASE_WALLET_PRIVATE_KEY
```

Use a fresh wallet, never your personal one. Step-by-step wallet setup:
<https://skim402.com/wallet>.

## Example

```python
from crewai_tools import SkimReaderTool

tool = SkimReaderTool()  # reads SKIM_WALLET_PRIVATE_KEY from the environment
tool.run(url="https://en.wikipedia.org/wiki/HTTP_402")
```

Drop it into any agent's tool list:

```python
from crewai import Agent
from crewai_tools import SkimReaderTool

researcher = Agent(
    role="Research Analyst",
    goal="Read and summarize web articles accurately",
    backstory="You turn messy web pages into clean, citable notes.",
    tools=[SkimReaderTool()],
)
```

## Arguments

- `private_key`: Optional. Hex private key (with or without `0x`) for the Base
  wallet that pays for reads. Defaults to the `SKIM_WALLET_PRIVATE_KEY`
  environment variable. Use a dedicated wallet, never your personal one.
- `base_url`: Optional. Skim API base URL. Defaults to `https://skim402.com`.
- `max_price_usd`: Optional. Hard per-call price cap in USD. The wallet refuses
  to sign for anything above this. Defaults to `0.01` (Skim is `$0.002`/call).
- `include_metadata`: Optional. When `True` (default), prepend a YAML
  frontmatter block of the page metadata to the returned Markdown.
- `timeout`: Optional. Per-request timeout in seconds. Defaults to `60`.
