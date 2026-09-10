# FXMacroDataTool

## Description

Official-source macroeconomic, FX and central-bank data across 18 currencies, from
[FXMacroData](https://fxmacrodata.com).

The point of this tool is aggregation. Answering "what did US core inflation print at, and
when is the next release" otherwise means knowing which of eighteen publishers to call and
how each one formats its data. One `dataset="latest"` call returns the most recent print of
every indicator for an economy, and every observation carries the instant it was published,
so an agent can reason about what was knowable at a point in time rather than only about now.

## Installation

```shell
pip install 'crewai[tools]'
```

No API key is required for USD data. A key widens the history window (anonymous access
returns the most recent 90 days) and unlocks the other seventeen currencies plus FX rates,
rate differentials, COT positioning and commodities:

```shell
export FXMACRODATA_API_KEY=your-key
```

## Example

```python
from crewai import Agent
from crewai_tools import FXMacroDataTool

tool = FXMacroDataTool()

agent = Agent(
    role="Macroeconomic Analyst",
    goal="Explain the current state of an economy and what is due next",
    backstory="You read official statistical releases and central-bank publications.",
    tools=[tool],
)
```

## Datasets

| `dataset` | Arguments | Returns |
| --- | --- | --- |
| `catalogue` | `currency` | Every indicator slug published for a currency, with units and coverage |
| `latest` | `currency` | The newest print of every indicator, in one request |
| `history` | `currency`, `indicator`, `start_date`, `end_date`, `limit` | One indicator's published history |
| `calendar` | `currency`, `limit` | Upcoming scheduled releases with publication times |
| `press_releases` | `currency`, `limit` | Official central-bank headlines |
| `fx_rate` | `base`, `quote`, `limit` | Official reference exchange rates |
| `rate_differential` | `base`, `quote`, `limit` | Policy rate differential, the first look at carry |
| `cot` | `currency`, `limit` | CFTC Commitment of Traders positioning |
| `commodities` | — | Latest tracked commodity prices |
| `market_sessions` | — | Which FX sessions are open now |
| `risk_sentiment` | — | Cross-asset risk sentiment reading |

Call `dataset="catalogue"` first when the indicator slug is not already known; it is the
discovery step that makes the rest of the surface usable.

## Notes

- The API key is sent as an `X-API-Key` header rather than a query parameter, so it does not
  end up in proxy or server access logs.
- A `401` or `403` returns a message explaining that a key is required, rather than surfacing
  as an outage, so the agent can fall back to USD rather than retrying blindly.
- `limit` is capped at 100 by the API.
