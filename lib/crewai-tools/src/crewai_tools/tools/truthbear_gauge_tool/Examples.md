# 🐻 Truth Bear GAUGE Tools — Usage Examples

### 🔧 Tool Initialization

```python
from crewai_tools import (
    TruthBearCoverageTool,
    TruthBearCatalogTool,
    TruthBearRecordTool,
)
```

---

### Example 1: Check a signal line before spending anything

```python
tool = TruthBearCoverageTool()
print(tool.run(signal_id="hydrology.river-level"))
```

```json
{
  "found": true,
  "totals": {
    "signal_ids": 1,
    "industries": 1,
    "distinct_entities": 10,
    "freshness": { "fresh": 10, "recent": 0, "stale": 0 }
  },
  "signals": [
    {
      "signal_id": "hydrology.river-level",
      "industry": "hydrology",
      "entities_count": 10,
      "freshness_counts": { "fresh": 10, "recent": 0, "stale": 0 },
      "update_status": "on_schedule"
    }
  ]
}
```

---

### Example 2: An unknown signal is reported, not raised

```python
tool = TruthBearCoverageTool()
print(tool.run(signal_id="not.a.real.signal"))
```

```json
{
  "found": false,
  "signal_id": "not.a.real.signal",
  "hint": "List valid signal ids with the Truth Bear GAUGE catalog tool."
}
```

---

### Example 3: Find a valid entity for a signal line

`signal_id` and `entity` must be used as a pair, so look the entity up first.

```python
tool = TruthBearCatalogTool()
print(tool.run(signal_id="hydrology.river-level"))
```

Returns the 10 monitored gauges for that line with their human-readable names, for example
`06893000` — *Missouri River at Kansas City, MO*.

Calling it with no filter is rejected on purpose:

```python
tool.run()          # ValueError: Provide either signal_id or industry -
                    # the unfiltered catalog is ~1.5 MB.
```

---

### Example 4: Request a paid record — the tool never pays

```python
tool = TruthBearRecordTool()
print(tool.run(signal_id="hydrology.river-level", entity="06893000"))
```

```json
{
  "payment_required": true,
  "note": "Settle this x402 challenge with your own wallet, then retry the same URL.",
  "challenge": {
    "x402Version": 1,
    "accepts": [
      {
        "scheme": "exact",
        "network": "base",
        "maxAmountRequired": "10000",
        "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "payTo": "0x…",
        "resource": "https://aeml-x402.zeabur.app/gauge"
      }
    ]
  }
}
```

Settle that challenge with your own x402 client and retry the same URL to receive the record.

---

### Example 5: Inside a Crew

```python
from crewai import Agent, Crew, Task
from crewai_tools import TruthBearCoverageTool, TruthBearCatalogTool

analyst = Agent(
    role="Environmental screening analyst",
    goal="Report whether a river gauge reading is routine or unusual, citing the official source",
    backstory="You never state a conclusion the official record does not support.",
    tools=[TruthBearCoverageTool(), TruthBearCatalogTool()],
)

task = Task(
    description=(
        "Check whether the hydrology.river-level line is covered and current, then list the "
        "gauges available on it."
    ),
    expected_output="Coverage status, freshness counts, and the list of monitored gauges.",
    agent=analyst,
)

Crew(agents=[analyst], tasks=[task]).kickoff()
```
