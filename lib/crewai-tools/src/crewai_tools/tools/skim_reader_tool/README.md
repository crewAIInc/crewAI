# SkimReaderTool

## Description

[Skim](https://skim402.com) is an x402-native clean reader API for AI agents.
Give it a URL and it returns clean, agent-ready Markdown plus structured
metadata (title, byline, published date, language, excerpt) — nav, ads, and
boilerplate stripped out.

`SkimReaderTool` lets a CrewAI agent read any web page (articles, docs, blog
posts, GitHub READMEs, research papers) as Markdown. Reads are paid per call
over the [x402 protocol](https://x402.org) — $0.002 in USDC on Base — using a
wallet you control. There are no API keys and no signup: the wallet is the
identity, and payment happens automatically on the HTTP 402 handshake.

## Installation

Install the tool with the `x402` extra, which pulls the x402 client with EVM
support:

```shell
pip install 'crewai[tools]'
pip install 'crewai-tools[x402]'
```

## Requirements

- A Base wallet private key, funded with a small amount of USDC on Base, exposed
  as the `SKIM_WALLET_PRIVATE_KEY` environment variable (or passed via
  `private_key=`). Use a dedicated wallet, never your personal one. The key is
  used only to sign payment authorizations locally and never leaves your machine.

## Example

```python
from crewai_tools import SkimReaderTool

# Reads SKIM_WALLET_PRIVATE_KEY from the environment.
tool = SkimReaderTool()

markdown = tool.run(url="https://en.wikipedia.org/wiki/HTTP_402")
print(markdown)
```

Or wire it into an agent:

```python
from crewai import Agent
from crewai_tools import SkimReaderTool

researcher = Agent(
    role="Researcher",
    goal="Read and summarize web pages",
    backstory="An analyst who reads primary sources before drawing conclusions.",
    tools=[SkimReaderTool()],
)
```

## Arguments

- `private_key` (`SecretStr`, optional): Hex private key for the paying Base
  wallet (with or without `0x`). Falls back to `SKIM_WALLET_PRIVATE_KEY`.
- `base_url` (`str`, optional): Skim API base URL. Defaults to
  `https://skim402.com`.
- `max_price_usd` (`float`, optional): Hard per-call price cap in USD. The wallet
  refuses to sign for anything above this. Defaults to `0.01` (Skim is `$0.002`).
- `include_metadata` (`bool`, optional): When `True` (default), prepend a YAML
  frontmatter block of the page metadata to the returned Markdown.
- `timeout` (`float`, optional): Per-request timeout in seconds. Defaults to `60`.
