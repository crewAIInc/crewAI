# EurostatTool

## Description

The **EurostatTool** fetches live, official European Union statistics from
[Eurostat](https://ec.europa.eu/eurostat), the statistical office of the EU.
It queries Eurostat's public dissemination API directly, so an agent gets
real published figures (GDP, unemployment, inflation, population, ...)
instead of guessing from training data.

No API key or registration is required — the API is fully public.

Eurostat dataset codes are terse and non-obvious (e.g. `une_rt_m` for the
monthly unemployment rate). This tool exposes a small set of common
indicators under friendly names via `indicator`; use `dataset_code` with any
raw code from the [Eurostat data browser](https://ec.europa.eu/eurostat/databrowser)
for anything else.

Eurostat responds in JSON-stat format, which encodes values as a flat,
index-addressed array rather than a list of records. This tool unpacks that
into plain `dimension=label` records before returning them, which is far
easier for a model to read directly than the raw JSON-stat shape.

## Arguments

| Argument       | Type            | Required | Description                                                                                   |
| -------------- | --------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `indicator`    | `str`           | ❌       | One of `unemployment_rate`, `inflation_rate`, `gdp`, `population`. Provide this or `dataset_code`. |
| `dataset_code` | `str`           | ❌       | A raw Eurostat dataset code, e.g. `nama_10_gdp`. Provide this or `indicator`.                    |
| `geo`          | `str`           | ❌       | Eurostat geo code, e.g. `DE` (Germany), `FR` (France), `EU27_2020` (the EU).                     |
| `since`        | `str`           | ❌       | Only return observations from this period onward, e.g. `2020` or `2024-01`.                     |
| `filters`      | `Dict[str,str]` | ❌       | Extra raw Eurostat dimension filters, e.g. `{"sex": "T", "unit": "PC_ACT"}`.                     |

## Usage Examples

### Tool Initialization

```python
from crewai_tools import EurostatTool

tool = EurostatTool()
```

### Example 1: A well-known indicator by friendly name

```python
result = tool._run(indicator="unemployment_rate", geo="DE")
print(result)
```

### Example 2: Inflation since a given period

```python
result = tool._run(indicator="inflation_rate", geo="FR", since="2026-01")
print(result)
```

### Example 3: Any raw Eurostat dataset code

```python
result = tool._run(
    dataset_code="demo_pjan",
    geo="NL",
    since="2020",
    filters={"sex": "T", "age": "TOTAL"},
)
print(result)
```

See `Examples.md` for a full `Agent`/`Task`/`Crew` example.
