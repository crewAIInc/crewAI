# TruthBearVerifyTool

Verify official facts from government and institutional sources with Bitcoin-anchored proof.

## Description

TruthBearVerifyTool lets CrewAI agents verify facts against 180+ official data signals (FRED, USGS, SEC EDGAR, NOAA, EPA, and more). Each record includes:

- The **official source URL** linking to the primary data source
- A **SHA-256 record_hash** anchored to Bitcoin via daily Merkle tree + OpenTimestamps
- A **freshness stamp** indicating when the data was last verified

Screening-level factual grounding, not decision-grade advice.

## Installation

No additional dependencies required beyond `requests` (already included in crewai-tools).

## Usage

```python
from crewai_tools import TruthBearVerifyTool

tool = TruthBearVerifyTool()

# Verify a fact
result = tool.run("US unemployment rate")
print(result)
# Value: 4.2%
# Source: https://fred.stlouisfed.org/series/UNRATE
# Record Hash: sha256:7a3f...d91e
# Freshness: 2024-12-01T00:00:00Z
```

## Environment Variables

None required. The free preview endpoint is used by default.

For paid tiers with 3-source cross-validation, payment settles automatically per call via x402 (USDC on Base) — no API key, no signup.

## Links

- [API](https://api.truthbear.co)
- [Website](https://truthbear.co)
- [MCP Server](https://www.npmjs.com/package/mcp-gauge-x402)
- [Source](https://github.com/CHANGCHINFU/mcp-gauge)
