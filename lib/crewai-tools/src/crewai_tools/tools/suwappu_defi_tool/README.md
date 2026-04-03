# SuwappuDefiTool Documentation

## Description

A suite of DeFi tools for cross-chain token operations via the [Suwappu](https://suwappu.com) DEX aggregator API. Enables AI agents to check token prices, get swap quotes, view portfolio balances, and discover supported chains and tokens across 15+ blockchain networks including Ethereum, Base, Arbitrum, Solana, and more.

## Installation

```shell
pip install 'crewai[tools]' suwappu
```

## Tools

| Tool | Description |
|------|-------------|
| `SuwappuGetPricesTool` | Get current USD price and 24h change for any token |
| `SuwappuGetQuoteTool` | Get a swap quote with price impact, route, gas, and fees |
| `SuwappuGetPortfolioTool` | Check wallet token balances across chains |
| `SuwappuListChainsTool` | List all supported blockchain networks |
| `SuwappuListTokensTool` | List available tokens on a specific chain |

## Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import SuwappuGetPricesTool, SuwappuGetQuoteTool, SuwappuGetPortfolioTool

# Initialize tools
price_tool = SuwappuGetPricesTool()
quote_tool = SuwappuGetQuoteTool()
portfolio_tool = SuwappuGetPortfolioTool()

# Create an agent with DeFi capabilities
defi_agent = Agent(
    role="DeFi Analyst",
    goal="Analyze token prices and find optimal swap routes",
    tools=[price_tool, quote_tool, portfolio_tool],
)

# Example task
task = Task(
    description="Check the current price of ETH on Base and get a quote to swap 0.5 ETH to USDC",
    agent=defi_agent,
)

crew = Crew(agents=[defi_agent], tasks=[task])
result = crew.kickoff()
```

## Setup

1. Install dependencies: `pip install 'crewai[tools]' suwappu`
2. Get a Suwappu API key at [suwappu.com](https://suwappu.com)
3. Set the environment variable: `export SUWAPPU_API_KEY=your_key_here`
