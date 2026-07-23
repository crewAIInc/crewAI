# OptionsAhoyTool

## Description

`OptionsAhoyTool` computes the tax outcome of common equity-compensation decisions
using the [OptionsAhoy](https://optionsahoy.com) public REST API. The API is keyless
and runs each calculation against the federal tax code and all fifty states plus the
District of Columbia.

A single tool exposes seven calculators through a `calculator` selector and an
`inputs` object:

| `calculator`       | What it computes |
| ------------------ | ---------------- |
| `amt-iso`          | Optimizes a multi-year incentive stock option (ISO) exercise schedule under the alternative minimum tax (AMT). |
| `nso`              | Tax and after-tax proceeds of exercising non-qualified stock options (NSOs), holding versus selling. |
| `rsu-sell-vs-hold` | Selling vested restricted stock units (RSUs) at vest versus holding them, on an after-tax, risk-adjusted basis. |
| `concentration`    | Analyzes a concentrated single-stock position and the after-tax cost of diversifying it. |
| `protective-put`   | Prices a protective put hedge at a given downside protection level and tenor. |
| `qsbs`             | Checks qualified small business stock (QSBS) eligibility and the resulting capital-gains exclusion. |
| `equity-funding`   | Plans which equity lots to sell, and when, to fund a cash goal by a target date at the least after-tax cost. |

No API key is read, stored, or sent. Field names in `inputs` match the published
schema at `https://optionsahoy.com/openapi.json` exactly.

## Installation

The tool ships with `crewai-tools` and uses `requests`, which is already a core
dependency. No extra install is required.

```shell
pip install crewai[tools]
```

## Example

```python
from crewai_tools import OptionsAhoyTool

tool = OptionsAhoyTool()

result = tool.run(
    calculator="qsbs",
    inputs={
        "acquisitionDate": "2018-01-01",
        "saleDate": "2026-02-01",
        "entityType": "us-c-corp",
        "acquisitionMethod": "original-issuance",
        "assetCategory": "under-50m",
        "industry": "tech-software",
        "activeBusiness": "yes",
        "adjustedBasis": 10000,
        "expectedGain": 2000000,
        "stateCode": "CA",
        "ordinaryIncome": 250000,
        "filingStatus": "single",
    },
)
```

The tool returns the calculator's result as a JSON string.

## Arguments

- `calculator` (required): one of `amt-iso`, `nso`, `rsu-sell-vs-hold`,
  `concentration`, `protective-put`, `qsbs`, `equity-funding`.
- `inputs` (required): a JSON object of inputs for the chosen calculator. Money values
  are plain numbers, dates are ISO strings (`YYYY-MM-DD`), and US state codes are two
  letters.

The tool can be constructed with a custom `base_url` or `timeout` if needed:

```python
tool = OptionsAhoyTool(timeout=60)
```
