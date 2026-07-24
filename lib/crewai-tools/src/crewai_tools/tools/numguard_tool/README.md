# NumGuardTool

Verify a number before an agent asserts it. `NumGuardTool` routes a numeric claim to the statistical check that
most directly tests it and returns whether it survives — so a crew can gate a backtest Sharpe, an A/B accuracy
gap, a cherry-picked subset win, or an LLM-judge preference before reporting it as real.

Backed by the open-source [`numguard`](https://github.com/ipezygj/numguard) library (a Deflated Sharpe Ratio for
backtests plus eval-integrity statistics).

## Installation

```shell
pip install numguard
```

## Supported claims

| `kind` | Checks | Example `params` |
|---|---|---|
| `backtest` | Deflated Sharpe Ratio (multiple-testing + finite-sample) | `{"sr": 0.12, "T": 250, "n_trials": 100}` |
| `model_gap` | Is an accuracy gap above the detectable effect? | `{"n": 2000, "p1": 0.85, "p2": 0.80}` |
| `subset_win` | Does a subset win survive multiple-testing correction? | `{"p": 0.03, "n_tests": 20}` |
| `judge_bias` | Is an LLM-judge preference real or noise/position bias? | `{"wins": 68, "n": 100}` |

## Example

```python
from crewai_tools import NumGuardTool

tool = NumGuardTool()

# an overfit backtest — the best of 100 configs
print(tool.run(kind="backtest", params={"sr": 0.12, "T": 250, "n_trials": 100}))
# {"survives": false, "verdict": "... does NOT survive deflation ...", "detail": {...}}

# a genuine edge — one hypothesis, long sample
print(tool.run(kind="backtest", params={"sr": 0.15, "T": 1000, "n_trials": 1}))
# {"survives": true, "verdict": "... SURVIVES deflation ...", "detail": {...}}
```

Give the tool to an agent so it can check a number instead of asserting it:

```python
from crewai import Agent
from crewai_tools import NumGuardTool

analyst = Agent(
    role="Quant Analyst",
    goal="Only report backtest results that survive verification",
    tools=[NumGuardTool()],
)
```

The tool returns a JSON string with `survives` (bool), a human-readable `verdict`, and the full `detail`.
