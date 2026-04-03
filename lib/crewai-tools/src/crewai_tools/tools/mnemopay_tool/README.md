# MnemoPay Tool

## Description

[MnemoPay](https://github.com/Jemiiah/mnemopay-sdk) gives AI agents persistent cognitive memory and micropayment capabilities via MCP (Model Context Protocol).

**Memory tools** let agents remember, recall, forget, reinforce, and consolidate information across sessions. Memory decays naturally unless reinforced, mimicking human cognition.

**Payment tools** let agents charge for work delivered (escrow), settle transactions, issue refunds, and check balances. A built-in reputation system tracks agent trustworthiness.

## Installation

1. Install the `crewai[tools]` package:

```shell
pip install 'crewai[tools]'
```

2. Install the MnemoPay SDK (Node.js required):

```shell
npm install -g @mnemopay/sdk
```

Or use the standalone PyPI package:

```shell
pip install mnemopay-crewai
```

## Usage

### Quick start with all tools

```python
from crewai import Agent
from crewai_tools import mnemopay_tools

agent = Agent(
    role="Research Assistant",
    goal="Help users with research and remember preferences",
    tools=mnemopay_tools(),
)
```

### Individual tools

```python
from crewai import Agent
from crewai_tools import (
    MnemoPayRememberTool,
    MnemoPayRecallTool,
    MnemoPayChargeTool,
)

agent = Agent(
    role="Paid Research Assistant",
    goal="Research topics, remember findings, and charge for work",
    tools=[
        MnemoPayRememberTool(),
        MnemoPayRecallTool(),
        MnemoPayChargeTool(),
    ],
)
```

### Connect to a remote MnemoPay server

```python
from crewai_tools import mnemopay_tools

tools = mnemopay_tools(server_url="https://mnemopay-mcp.fly.dev")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `MnemoPayRememberTool` | Store a memory that persists across sessions |
| `MnemoPayRecallTool` | Recall memories with optional semantic search |
| `MnemoPayForgetTool` | Permanently delete a memory by ID |
| `MnemoPayReinforceTool` | Boost a memory's importance score |
| `MnemoPayConsolidateTool` | Prune stale memories below decay threshold |
| `MnemoPayChargeTool` | Create an escrow charge for work delivered |
| `MnemoPaySettleTool` | Finalize a pending escrow transaction |
| `MnemoPayRefundTool` | Refund a transaction (docks reputation) |
| `MnemoPayBalanceTool` | Check wallet balance and reputation |
| `MnemoPayProfileTool` | Full agent stats |
| `MnemoPayHistoryTool` | Transaction history |
| `MnemoPayLogsTool` | Immutable audit trail |
