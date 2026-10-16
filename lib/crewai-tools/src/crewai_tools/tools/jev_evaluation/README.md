# Jev Evaluation

Offline evaluation of TypeSafe Jev routing (Choice) and guardrail (Noul) decisions against labeled records. It needs no SDK or credentials unless you opt in to live collection.

## Dataset

The dataset is a `.jsonl` file (one object per line; blank lines are skipped) or a `.json` array, read as UTF-8, with a leading BOM allowed. Malformed JSON, unknown fields, and missing or duplicate ids raise `ValueError` with the line or index. Save these two lines as `examples.jsonl` to try it:

```jsonl
{"id": "r1", "kind": "route", "state": "Refund for order 1234", "expected_route": "billing", "decision": {"choice": "billing", "confidence": 0.91, "probabilities": {"billing": 0.95, "tech": 0.05}}, "metadata": {"latency_ms": 212.0, "usage": {"input_tokens": 40, "output_tokens": 1}}}
{"id": "g1", "kind": "guardrail", "state": "Paris is the capital of France.", "expected": "accept", "decision": {"noul": 0.97}}
```

Route records have `expected_route`. Guardrail records have `expected` (`"accept"` or `"reject"`). Both can have an `error` (a recorded failure) and `metadata` (`model`, `usage`, `latency_ms`).

```python
from crewai_tools.tools.jev_evaluation import evaluate, load_dataset

report = evaluate(load_dataset("examples.jsonl"), min_confidence=0.7, noul_threshold=0.8)
print(report.model_dump_json(indent=2))
```

## Classification and metrics

A record **fails** if it has a recorded `error`, no `decision`, or a malformed decision (a missing field, or a value that isn't a finite number in [0, 1]). A route also fails if its `choice` isn't a known route. Known routes default to every `expected_route` in the dataset, or can be passed as `routes=`. Failed records are never counted as correct or accepted.

- **Routes:** `confidence >= min_confidence` is decided (correct or incorrect); anything lower is abstained.
- **Guardrails:** `noul >= noul_threshold` is accepted; anything lower is rejected.

Both thresholds are inclusive. Choice confidence and Noul probability are never converted into each other.

| Metric | Definition |
|---|---|
| `routing.accuracy` / `coverage` / `abstention_rate` | correct / decided / abstained ÷ all route records |
| `guardrail.false_accept_rate` | accepted ÷ decided records expected to be rejected |
| `guardrail.false_reject_rate` | rejected ÷ decided records expected to be accepted |
| `usage`, `latency` | token sums and mean `latency_ms`, over only the records that report them |

A rate is `null` when its denominator is 0. `report.outcomes` gives each record's outcome and reason.

## Live collection (opt-in)

`collect_live_decisions` asks Jev about every record that has no `decision` or `error`, and times each successful call. You pass in a client created with your own credentials, and you're responsible for closing it. Client errors are stored on the record, which then counts as failed.

```python
from crewai_tools.tools.jev_evaluation import collect_live_decisions, evaluate, load_dataset
from typesafe_sdk import TypeSafeClient

with TypeSafeClient(api_key="...") as client:
    records = collect_live_decisions(
        load_dataset("unlabeled.jsonl"),
        client=client,
        route_instructions="Which team should handle this ticket?",
        route_criteria={"billing": "Payments and invoices", "tech": "Bugs and outages"},
        guardrail_instructions="Is this statement true?",
    )
report = evaluate(records, min_confidence=0.7, noul_threshold=0.8, routes={"billing", "tech"})
```
