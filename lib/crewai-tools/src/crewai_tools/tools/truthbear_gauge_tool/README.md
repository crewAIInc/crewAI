# 🐻 Truth Bear GAUGE Tools

Three tools for querying **Truth Bear GAUGE**, a pay-per-call source of official-record signals.
Sources are official government records and named registries — USGS, NOAA, NWS, EPA, SEC EDGAR,
openFDA, EIA, BLS, FAA, US CBP, USTR, USITC and the Federal Register, plus IMF PortWatch and
UNFCCC feeds.

Two of the three tools are free and need no API key, no account and no signup. The third is paid
over [x402](https://x402.org) and **returns the payment challenge rather than settling it** — no
key material or funds are touched by this package.

---

## Description

Each record is stated against the **official threshold ladder published by the source agency**
rather than a scale invented by the vendor, and carries a **same-season percentile** so an agent
can tell a routine reading from an unusual one. Every paid record also ships a canonical sha256
`record_hash` that can be recomputed offline from the delivered payload, so an agent does not have
to trust the endpoint it just paid.

The service is **descriptive only**: no forecast, no recommendation, no adjudication.

| Tool | Cost | Purpose |
| ---- | ---- | ------- |
| `TruthBearCoverageTool` | free | Does a signal line exist, how many objects it covers, how fresh those readings are, and whether the source agency is on schedule or overdue |
| `TruthBearCatalogTool` | free | Which entity ids are valid for a signal line — `signal_id` and `entity` must be used as a pair |
| `TruthBearRecordTool` | paid (x402) | Requests one official record; returns the HTTP 402 payment challenge untouched |

---

## Arguments

### `TruthBearCoverageTool`

| Argument | Type | Required | Description |
| -------- | ---- | -------- | ----------- |
| `signal_id` | `str` | ❌ | Signal id in `genus.species` form, e.g. `"hydrology.river-level"`. Omit for a summary across all 185 signal lines. |

### `TruthBearCatalogTool`

| Argument | Type | Required | Description |
| -------- | ---- | -------- | ----------- |
| `signal_id` | `str` | ⚠️ | Filter to one signal line, e.g. `"hydrology.river-level"`. |
| `industry` | `str` | ⚠️ | Filter to one industry, e.g. `"hydrology"`, `"airquality"`, `"macro"`. |

⚠️ **One of the two is required.** The unfiltered catalog response is about 1.5 MB, which would
flood an agent's context window, so a call with neither filter is rejected rather than served.

### `TruthBearRecordTool`

| Argument | Type | Required | Description |
| -------- | ---- | -------- | ----------- |
| `signal_id` | `str` | ✅ | Signal id, e.g. `"hydrology.river-level"`. |
| `entity` | `str` | ✅ | Monitored object id **paired with** `signal_id`, e.g. `"06893000"`. Get valid pairs from `TruthBearCatalogTool`. |

---

## Installation

No extra installation step. The tools use the Python standard library (`urllib`, `json`) plus
`pydantic`, which `crewai-tools` already depends on.

```python
from crewai_tools import (
    TruthBearCoverageTool,
    TruthBearCatalogTool,
    TruthBearRecordTool,
)
```

---

## Behaviour worth knowing

- **A miss is not an error.** An unknown `signal_id` comes back as HTTP 200 with an empty
  `signals` list; `TruthBearCoverageTool` converts that to an explicit `{"found": false}` so an
  agent does not mistake an empty answer for a failed call.
- **402 is a price quote, not a failure.** `TruthBearRecordTool` hands the challenge back intact —
  network, asset, amount and recipient — so an x402-capable wallet elsewhere in your stack can
  settle it and retry the same URL.
- **Charging is bound to HTTP 200.** A paid query with no data answers 422 and is not billed.
- **Freshness is explicit.** Coverage reports `fresh` / `recent` / `stale` counts and an
  `update_status` of `on_schedule`, `overdue` or `not_published`, measured against the cadence the
  source agency itself declares.
