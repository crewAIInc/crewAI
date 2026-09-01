# TruthBearVerifyTool

Free preview of official-data fact checking for CrewAI agents.

## Description

TruthBearVerifyTool queries the free `/trust/preview` endpoint to retrieve a single-source summary from 180+ official data signals (FRED, USGS, SEC EDGAR, NOAA, EPA, and more). Each response includes:

- The **matched signal** name
- The **official value**
- The **source URL** linking to the primary data source
- A **SHA-256 record_hash** for the record
- A **freshness stamp** indicating when the data was last verified

This is a screening-level preview — not decision-grade verification. No API key or signup required.

## Installation

No additional dependencies required beyond `requests` (already included in crewai-tools).

## Usage

```python
from crewai_tools import TruthBearVerifyTool

tool = TruthBearVerifyTool()

# Look up a fact
result = tool.run("US unemployment rate")
print(result)
# Signal: UNRATE
# Value: 4.2%
# Source: https://fred.stlouisfed.org/series/UNRATE
# Record Hash: sha256:7a3f...d91e
# Freshness: 2024-12-01T00:00:00Z
```

## Environment Variables

None required. The free preview endpoint is used by default.

## Links

- [API](https://api.truthbear.co)
- [Website](https://truthbear.co)
- [MCP Server](https://www.npmjs.com/package/mcp-gauge-x402)
- [Source](https://github.com/CHANGCHINFU/mcp-gauge)
