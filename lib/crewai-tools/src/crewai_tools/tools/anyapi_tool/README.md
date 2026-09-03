# AnyAPI Tools Documentation

## Description

AnyAPI is a unified marketplace for scraping and data APIs: any API, one wallet, USD, no
subscriptions. Reach hundreds of third-party APIs (social media, search results, web data)
through one key and one normalized interface; pay per request in real dollars; failed calls
are never charged - AnyAPI fails over across providers automatically under one price.

These three tools give a CrewAI agent the whole loop, and they are meant to be used in this
order:

- **AnyApiSearchTool**: search the catalog for an API that returns the data you need
- **AnyApiDescribeTool**: fetch one endpoint's input schema, output schema and USD price
- **AnyApiRunTool**: execute an endpoint with a JSON input and get normalized output back

Every AnyAPI input schema is strict, so an agent must read the schema with
`AnyApiDescribeTool` before its first `AnyApiRunTool` call on a slug. An invented field name
fails the call.

## Installation

```shell
pip install crewai[tools] getanyapi
```

or

```shell
uv add crewai-tools --extra getanyapi
```

## Examples

### Search the catalog

```python
from crewai_tools import AnyApiSearchTool

tool = AnyApiSearchTool()
result = tool.run(query="instagram profile", platform="instagram", limit=5)
```

### Read an endpoint's schema and price

```python
from crewai_tools import AnyApiDescribeTool

tool = AnyApiDescribeTool()
result = tool.run(slug="instagram.profile")
```

### Run an endpoint

```python
from crewai_tools import AnyApiRunTool

tool = AnyApiRunTool()
result = tool.run(slug="instagram.profile", input={"handle": "nasa"})
```

### Give an agent the full loop

```python
from crewai import Agent
from crewai_tools import AnyApiDescribeTool, AnyApiRunTool, AnyApiSearchTool

researcher = Agent(
    role="Data researcher",
    goal="Answer questions with live third-party data",
    backstory="Finds the right API, reads its schema, then calls it.",
    tools=[AnyApiSearchTool(), AnyApiDescribeTool(), AnyApiRunTool()],
)
```

## Steps to Get Started

1. **Package Installation**: install `crewai[tools]` and the `getanyapi` extra.

2. **API Key Acquisition**: create a key at `https://getanyapi.com/dashboard`. New accounts
   start with free trial credit, so the first calls cost nothing.

3. **Environment Configuration**:
   ```bash
   export ANYAPI_API_KEY="aa_live_..."
   ```
   Or pass it directly: `AnyApiSearchTool(api_key="aa_live_...")`.

4. **Tool Order**: search for a slug, describe it to get the strict input schema and the
   USD price, then run it. `AnyApiRunTool` reports what the call cost as `costUsd`.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `ANYAPI_API_KEY` | Yes | AnyAPI key from `https://getanyapi.com/dashboard`. Can be passed as the `api_key` argument instead. |

## Conclusion

AnyAPI gives AI agents one API key and one USD wallet to call hundreds of scraping and data
APIs - Instagram, TikTok, YouTube, Reddit, Facebook, Google search results, general web
scraping, and more - through a single normalized interface. Every API exposes a normalized
input/output JSON Schema, so an agent discovers, inspects, and runs any of them with the same
loop. Pricing is real-dollar pay-per-request with no subscriptions; failed calls are never
charged because AnyAPI fails over across upstream providers automatically under one price
reservation. The provider is always reported as AnyAPI.
