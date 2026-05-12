# Coinbase Agentic Wallet Tool

## Description

`CoinbaseAgenticWalletTool` exposes the Coinbase Agentic Wallet MCP server to CrewAI agents. Lets agents discover and pay for HTTP APIs autonomously via the x402 protocol, using a Coinbase-managed embedded wallet -- no API keys, no manual onramp, no seed phrases.

The wrapper starts the local MCP bundle installed by `npx @coinbase/payments-mcp` through CrewAI's `MCPServerAdapter` and returns the MCP tools as CrewAI-compatible tools.

## Installation

Install CrewAI tools with MCP support:

```shell
pip install "crewai-tools[mcp]"
```

Install Node.js so `npx` can run the Coinbase installer and `node` can launch the local MCP bundle.

Run the Coinbase installer once:

```shell
npx @coinbase/payments-mcp install --client other
```

The installer creates the stdio server bundle at `~/.payments-mcp/bundle.js`.

## Usage

```python
from crewai import Agent, Crew, Task
from crewai_tools import CoinbaseAgenticWalletTool

with CoinbaseAgenticWalletTool() as coinbase_tools:
    agent = Agent(
        role="Payments Researcher",
        goal="Find paid APIs and use them when they are useful",
        backstory=(
            "You can search the x402 bazaar, inspect payment requirements, "
            "and pay for HTTP APIs in USDC."
        ),
        tools=coinbase_tools,
    )

    task = Task(
        description="Find a paid weather API and get today's forecast for San Francisco.",
        expected_output="A concise forecast with the paid API response summarized.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()
```

You can also manage the lifecycle manually:

```python
wallet = CoinbaseAgenticWalletTool(connect_timeout=90)
try:
    agent = Agent(
        role="Payments Researcher",
        goal="Use x402 APIs",
        backstory="You can discover and pay for x402 APIs.",
        tools=wallet.tools,
    )
finally:
    wallet.stop()
```

## Authentication

On first use, the MCP server opens a companion window for email and OTP sign-in. This creates a Coinbase-managed embedded wallet, so you do not need to manage a seed phrase. Fund the wallet through the built-in Coinbase Onramp.

## Resources

- [Coinbase Agentic Wallet MCP docs](https://docs.cdp.coinbase.com/agentic-wallet/mcp/welcome)
- [x402 protocol docs](https://docs.cdp.coinbase.com/x402/welcome)
