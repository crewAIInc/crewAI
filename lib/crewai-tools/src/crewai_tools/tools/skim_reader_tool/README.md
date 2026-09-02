# SkimReaderTool

**Give your CrewAI agents the ability to read any URL — clean Markdown, ~4x smaller than raw HTML. No ads, no nav, no boilerplate.**


`SkimReaderTool` gives CrewAI agents a clean web reader backed by [Skim](https://skim402.com). It fetches a URL as agent-ready Markdown plus structured metadata. Output is ~4x smaller than raw HTML, so agents spend fewer tokens and process pages faster.

**Two ways to pay:** card plan with API key (free tier: 1,000 credits/month — [skim402.com/pricing](https://skim402.com/pricing)) or x402 wallet pay-per-call ($0.002 USDC on Base, no account needed).

> **See it before you wire it:** [try Skim free in your browser](https://freeskims.skim402.com) — 10 free skims a day, no wallet, no signup. Paste a URL, see exactly what your agent gets back.

---

## Install

```bash
pip install 'crewai[tools]'
```

---

## Quickstart (60 seconds)

### 1. Get a free API key

Go to **[skim402.com/pricing](https://skim402.com/pricing)**, sign up for the free plan (1,000 credits/month), and copy your `sk402_...` key.

### 2. Set the env var

```bash
export SKIM_API_KEY=sk402_your_key_here
```

### 3. Use it

```python
from crewai_tools import SkimReaderTool

reader = SkimReaderTool()  # reads SKIM_API_KEY from the environment

markdown = reader.run(url="https://en.wikipedia.org/wiki/HTTP_402")
print(markdown)
```

---

## Alternative: pay per call with a crypto wallet

If you prefer x402 wallet pay-per-call instead of a card plan, install the optional wallet dependencies and set a dedicated wallet key:

```bash
pip install 'crewai-tools[x402]'
export SKIM_WALLET_PRIVATE_KEY=0xYOUR_BASE_WALLET_PRIVATE_KEY
```

Fund a dedicated Base wallet with a small USDC balance. Each read costs $0.002 on Base. Full setup guide: **<https://skim402.com/wallet>**.

> **Use a fresh wallet, not your personal one.** This wallet's private key signs payment authorizations on your machine — treat it like a hot wallet for paying $0.002 tolls, not a savings account.

---

## Use it in a crew

`SkimReaderTool` is a standard CrewAI `BaseTool`, so it drops straight into any agent's tool list:

```python
from crewai import Agent, Task, Crew
from crewai_tools import SkimReaderTool

researcher = Agent(
    role="Research Analyst",
    goal="Read and summarize web articles accurately",
    backstory="You turn messy web pages into clean, citable notes.",
    tools=[SkimReaderTool()],
)

task = Task(
    description="Read https://en.wikipedia.org/wiki/HTTP_402 and summarize it in 5 bullet points.",
    expected_output="A 5-bullet summary.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
print(crew.kickoff())
```

The agent decides when to call the Skim Web Reader, Skim returns clean Markdown (~4x smaller than raw HTML), and the model gets a faster, cheaper read.

---

## Output shape

`SkimReaderTool` returns Markdown with a YAML frontmatter block of the page metadata:

```markdown
---
title: Example article
byline: Jane Doe
publishedAt: 2025-01-15
lang: en
excerpt: A short summary...
---

# Example article

The cleaned article body in Markdown...
```

Set `include_metadata=False` to get just the Markdown body.

---

## Configuration

`SkimReaderTool` takes the following parameters:

| Parameter          | Default                    | Notes                                                                                                                           |
| ------------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `api_key`          | `$SKIM_API_KEY`            | Card-plan API key (`sk402_...`). Get one free at skim402.com/pricing. Takes priority over `private_key`.                      |
| `private_key`      | `$SKIM_WALLET_PRIVATE_KEY` | Wallet lane only. Hex private key for the Base wallet. Ignored when `api_key` is set.                                         |
| `base_url`         | `https://skim402.com`      | Override the API base URL. For self-hosting or local development.                                                              |
| `max_price_usd`    | `0.01`                     | Wallet lane only. Hard cap on per-call price in USD. Skim is `$0.002`/call.                                                   |
| `include_metadata` | `True`                     | Prepend a YAML frontmatter block of page metadata to the returned Markdown.                                                    |
| `timeout`          | `60`                       | Per-request timeout in seconds.                                                                                                |

```python
# Card key (recommended)
reader = SkimReaderTool(api_key="sk402_...")

# Or wallet
reader = SkimReaderTool(
    private_key="0x...",
    max_price_usd=0.005,
    include_metadata=False,
)
```

---

## Security

- **No outbound telemetry from this package.** `SkimReaderTool` only talks to `skim402.com` (or whatever you set as `base_url`). No analytics, no error reporting, no phone-home.
- **Wallet lane:** the private key only signs payment authorizations locally — it never leaves your machine.

---

## Links

- **Skim website** — <https://skim402.com>
- **Pricing & free key** — <https://skim402.com/pricing>
- **Wallet setup guide** — <https://skim402.com/wallet>
- **API docs** — <https://skim402.com/docs>
- **GitHub** — <https://github.com/JessieJanie/skim402>

---

