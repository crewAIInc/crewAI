# KrakenAgents - QRI Trading Organization

AI-powered trading agents for Kraken Spot and Futures trading using CrewAI.

## Features

- 74 autonomous trading agents
- Kraken Spot API integration (54 tools)
- Kraken Futures API integration (43 tools)
- Local LLM support (LM Studio, Ollama)

## Installation

```bash
uv pip install -e krakenagents
```

## Usage

```bash
# Check configuration
python -m krakenagents --config

# List all agents
python -m krakenagents --list

# Run minimal test
python -m krakenagents --minimal

# Run specific desk
python -m krakenagents --desk spot
python -m krakenagents --desk futures

# Run full organization
python -m krakenagents
```

## Configuration

Copy `.env.example` to `.env` and configure your API keys and endpoints.
