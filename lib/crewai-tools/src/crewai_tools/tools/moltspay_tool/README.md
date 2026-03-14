# MoltsPay Tool

Pay for AI services using USDC (gasless) via the x402 protocol.

## Description

MoltsPayTool enables CrewAI agents to autonomously pay for and use AI services from other agents or providers. It uses the x402 protocol for HTTP-native payments - no gas fees, pay-for-success model.

## Installation

```bash
pip install moltspay
```

## Setup

1. Initialize a MoltsPay wallet:
```bash
npx moltspay init --chain base
```

2. Fund your wallet with USDC on Base network

## Usage

```python
from crewai_tools import MoltsPayTool

# Initialize the tool
moltspay = MoltsPayTool()

# Use in an agent
agent = Agent(
    role="Content Creator",
    goal="Generate engaging video content",
    tools=[moltspay]
)

# The agent can now pay for services
# Example: Generate a video ($0.99 USDC)
result = moltspay.run(
    provider_url="https://juai8.com/zen7",
    service_id="text-to-video", 
    prompt="A cat dancing on a rainbow"
)
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| provider_url | str | Yes | Service provider URL |
| service_id | str | Yes | Service identifier (e.g., "text-to-video") |
| prompt | str | No | Text prompt for the service |
| image_path | str | No | Path to image file for image-based services |

## Links

- [MoltsPay Documentation](https://docs.moltspay.com)
- [x402 Protocol](https://www.x402.org)
- [Available Services](https://moltspay.com/services)
