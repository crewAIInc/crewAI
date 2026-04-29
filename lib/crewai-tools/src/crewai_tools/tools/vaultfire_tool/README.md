# Vaultfire Trust Tools for CrewAI

On-chain identity, partnership bonds, and verified reputation for AI agents — surfaced as CrewAI `BaseTool` instances.

[Vaultfire Protocol](https://github.com/Ghostkey316/ghostkey-316-vaultfire-init) implements the [ERC-8004](https://eips.ethereum.org/EIPS/eip-8004) AI agent identity standard across four EVM mainnets: **Base, Avalanche, Arbitrum, Polygon**.

## Tools

| Tool | What it does | Auth |
| :--- | :--- | :---: |
| `VaultfireTrustTool` | Full trust verification — identity, bonds, reputation, Street Cred score and tier | None |
| `VaultfireDiscoverTool` | Discover agents registered with a given capability (e.g. `"image-generation"`) | None |
| `VaultfireBondsTool` | Read all partnership bonds for an agent address | None |
| `VaultfireReputationTool` | Read on-chain reputation (verified ratings + average score) | None |

All tools are **read-only** and require no API keys. They call public mainnet RPC endpoints.

## Installation

These tools depend on the upstream [`vaultfire-crewai`](https://pypi.org/project/vaultfire-crewai/) package:

```bash
uv add vaultfire-crewai
# or
pip install vaultfire-crewai
```

## Usage

```python
from crewai import Agent
from crewai_tools.tools.vaultfire_tool import (
    VaultfireTrustTool,
    VaultfireDiscoverTool,
)

trust_analyst = Agent(
    role="Trust Analyst",
    goal="Verify whether an AI agent is trustworthy before delegating tasks",
    backstory="You evaluate on-chain trust signals for the team.",
    tools=[VaultfireTrustTool(), VaultfireDiscoverTool()],
)
```

The agent can then ask questions like:

- *"Is `0xfA15...813C` trustworthy on Base? Give me the full trust profile."*
- *"Find agents on Base that registered the `image-generation` capability."*

## Output format

Each tool returns a JSON string. For example, `VaultfireTrustTool` returns:

```json
{
  "verdict": "TRUSTED",
  "address": "0xfA15...813C",
  "agent_name": "agent-fa15.vns",
  "street_cred": { "score": 72, "tier": "Platinum" },
  "active_bonds": 3,
  "reputation": { "average_rating": 4.8, "total_ratings": 12 },
  "chain": "base",
  "verified_at": "2026-04-29T19:00:00Z"
}
```

## Links

- Upstream package: <https://pypi.org/project/vaultfire-crewai/>
- Source code: <https://github.com/Ghostkey316/vaultfire-crewai>
- Vaultfire Protocol: <https://github.com/Ghostkey316/ghostkey-316-vaultfire-init>
- DefiLlama: <https://defillama.com/protocol/vaultfire>
- Standard: [ERC-8004](https://eips.ethereum.org/EIPS/eip-8004)
