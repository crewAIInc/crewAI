# Jev Decision Tool

## Description

`JevDecisionTool` asks TypeSafe's [Jev](https://docs.typesafe.ai/introduction) model one
Choice question about a state and returns the selected label as structured data. It is
synchronous and calls the official [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python/).

## Installation

```shell
uv add 'crewai-tools[typesafe-sdk]'
```

Set `TYPESAFE_API_KEY`. The SDK also reads `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL`.

## Example

```python
from crewai.tools import ToolFailure
from crewai_tools import JevDecisionTool

tool = JevDecisionTool()  # or JevDecisionTool(model="jev-latest", client=my_client)

result = tool.run(
    state="I was charged twice and want a refund.",
    question="Which team should handle this request?",
    choices={
        "billing": "Payments, invoices and refunds",
        "technical": "Product bugs and integration problems",
        "support": None,  # no description: the label speaks for itself
    },
)

if isinstance(result, ToolFailure):
    print(result.message)
else:
    print(result["choice"], result["confidence"])
```

## Arguments

- `state`: text, a JSON object, or an array for Jev to evaluate.
- `question`: the non-empty question or instruction to answer about the state.
- `choices`: labels mapped to a description of when each applies, or `None`.

Pass them by keyword: `run(...)` validates keyword arguments and raises `ValueError`
before anything is sent.

## Result

A direct call returns this dict; an agent receives the same fields as JSON.

| Key | Value |
| :-- | :-- |
| `choice` | The selected label, always one of the supplied `choices` |
| `confidence` | TypeSafe's [confidence](https://docs.typesafe.ai/confidence) in the selection, exactly as reported |
| `probabilities` | The probabilities TypeSafe reports, keyed by label |
| `model`, `usage` | The model that answered and its token counts |
| `request_id` | The TypeSafe request id, or `None` when the response has none |

A failed SDK call, a missing answer, an empty `choices` mapping, or a label outside
`choices` returns a
[`ToolFailure`](https://docs.crewai.com/concepts/tools#reporting-tool-failures)
instead of a result, so a failed call is never mistaken for a decision.
