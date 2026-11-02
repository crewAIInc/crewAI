# ruff: noqa: T201
"""Offline evaluation of Jev routing (Choice) and guardrail (Noul) decisions.

Usage:
    uv run python scripts/jev_evaluation_runner.py examples.jsonl \\
        --min-confidence 0.7 --noul-threshold 0.8 [--route billing --route tech]

The dataset is a .jsonl file (one record per line; blank lines are skipped) or
a .json array, read as UTF-8 with an optional BOM. Malformed JSON, unknown
fields, and missing or duplicate ids are reported with the line or index:

    {"id": "r1", "kind": "route", "state": "Refund for order 1234", "expected_route": "billing", "decision": {"choice": "billing", "confidence": 0.91, "probabilities": {"billing": 0.95, "tech": 0.05}}}
    {"id": "g1", "kind": "guardrail", "state": "Paris is the capital of France.", "expected": "accept", "decision": {"noul": 0.97}}

Route records have expected_route; guardrail records have expected ("accept"
or "reject"). Both can have an error (a recorded failure) and metadata (model,
usage, latency_ms). The report is printed as JSON.

A record fails if it has a recorded error, no decision, or a malformed decision
(a missing field, or a value that isn't a finite number in [0, 1]); a route
also fails if its choice isn't a known route (every expected_route in the
dataset, or the --route values). Failed records are never counted as correct or
accepted. Routes with confidence >= min_confidence are decided (correct or
incorrect), lower ones abstain; guardrails with noul >= noul_threshold are
accepted, lower ones rejected. Choice confidence and Noul probability are never
converted into each other.

- routing accuracy / coverage / abstention_rate: correct / decided / abstained
  over all route records.
- guardrail false_accept_rate / false_reject_rate: accepted / rejected over
  decided records expected to be rejected / accepted.
- usage, latency: token sums and mean latency_ms over only the records that
  report them.

A rate is null when its denominator is 0. These are decision-quality metrics,
not a replay of a runtime guardrail: a guardrail that rejects when the answer
is missing or the call fails would count those cases as rejections, while here
they are failed and left out of the false-accept and false-reject rates.

Live collection is opt-in and needs your own TypeSafe client, which you close:

    with TypeSafeClient(api_key="...") as client:
        records = collect_live_decisions(
            load_dataset("unlabeled.jsonl"),
            client=client,
            route_instructions="Which team should handle this ticket?",
            route_criteria={"billing": "Payments", "tech": "Bugs and outages"},
            guardrail_instructions="Is this statement true?",
        )

It times each call, stores client errors on the record (which then fails), and
fails an answer whose type doesn't match the question. The type is not stored,
so recorded decisions are checked by their fields only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)


Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_THRESHOLD: TypeAdapter[float] = TypeAdapter(
    Probability, config=ConfigDict(strict=True)
)

ROUTE_QUESTION = "route"
GUARDRAIL_QUESTION = "guardrail"


class ChoiceDecision(BaseModel):
    """A Jev Choice answer, mirroring the SDK's ``ChoiceAnswer`` fields."""

    model_config = ConfigDict(strict=True, extra="ignore")

    choice: str
    confidence: Probability
    probabilities: dict[str, Probability]


class NoulDecision(BaseModel):
    """A Jev Noul answer: the probability that the question is true."""

    model_config = ConfigDict(strict=True, extra="ignore")

    noul: Probability


class TokenUsage(BaseModel):
    """Token counts as reported by the SDK; ``None`` means not reported."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionMetadata(BaseModel):
    """Request metadata kept only when it was actually recorded."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


class _BaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    state: Any
    decision: dict[str, Any] | None = None
    error: str | None = None
    metadata: DecisionMetadata | None = None


class RouteRecord(_BaseRecord):
    """A labeled routing example with an optional recorded Choice decision."""

    kind: Literal["route"]
    expected_route: str = Field(min_length=1)


class GuardrailRecord(_BaseRecord):
    """A labeled guardrail example with an optional recorded Noul decision."""

    kind: Literal["guardrail"]
    expected: Literal["accept", "reject"]


EvaluationRecord = Annotated[RouteRecord | GuardrailRecord, Field(discriminator="kind")]
_RECORD_ADAPTER: TypeAdapter[RouteRecord | GuardrailRecord] = TypeAdapter(
    EvaluationRecord
)

Outcome = Literal["correct", "incorrect", "abstained", "accepted", "rejected", "failed"]


class RecordOutcome(BaseModel):
    """How a single record was classified, and why."""

    id: str
    kind: Literal["route", "guardrail"]
    outcome: Outcome
    reason: str


class RoutingReport(BaseModel):
    """Routing counts and rates; every rate is over all routing records."""

    total: int
    decided: int
    abstained: int
    failed: int
    correct: int
    accuracy: float | None
    coverage: float | None
    abstention_rate: float | None


class GuardrailReport(BaseModel):
    """Guardrail counts; error rates are over decided records only."""

    total: int
    decided: int
    failed: int
    accepted: int
    rejected: int
    false_accepts: int
    false_accept_denominator: int
    false_accept_rate: float | None
    false_rejects: int
    false_reject_denominator: int
    false_reject_rate: float | None


class UsageSummary(BaseModel):
    """Token totals over the records that reported each count."""

    input_tokens: int
    input_token_records: int
    output_tokens: int
    output_token_records: int


class LatencySummary(BaseModel):
    """Mean latency over records with a measured latency."""

    count: int
    mean_ms: float | None


class EvaluationReport(BaseModel):
    """Full evaluation result."""

    routing: RoutingReport
    guardrail: GuardrailReport
    usage: UsageSummary
    latency: LatencySummary
    outcomes: list[RecordOutcome]


class SystemOneClient(Protocol):
    """The part of the TypeSafe client used for live collection."""

    def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> Any:
        """Answer the named questions about ``state``."""
        ...


def load_dataset(path: str | Path) -> list[RouteRecord | GuardrailRecord]:
    """Load and validate records from a ``.jsonl`` or ``.json`` file.

    Args:
        path: A ``.jsonl`` file (one record per line; blank lines are skipped)
            or a ``.json`` file containing an array of records.

    Returns:
        The validated records, in file order.

    Raises:
        ValueError: If the file or any record is malformed.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        items = _parse_jsonl(text)
    elif path.suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError("A .json dataset must be an array of records")
        items = [(f"index {i}", item) for i, item in enumerate(data)]
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix!r}")

    records: list[RouteRecord | GuardrailRecord] = []
    seen: set[str] = set()
    for location, item in items:
        try:
            record = _RECORD_ADAPTER.validate_python(item)
        except ValidationError as e:
            raise ValueError(f"Invalid record at {location}: {e}") from e
        if record.id in seen:
            raise ValueError(f"Duplicate record id {record.id!r} at {location}")
        seen.add(record.id)
        records.append(record)
    return records


def evaluate(
    records: Sequence[RouteRecord | GuardrailRecord],
    *,
    min_confidence: float,
    noul_threshold: float,
    routes: Collection[str] | None = None,
) -> EvaluationReport:
    """Classify every record and compute routing and guardrail metrics.

    Both thresholds are inclusive: a Choice confidence equal to
    ``min_confidence`` is decided, and a Noul probability equal to
    ``noul_threshold`` is accepted.

    Args:
        records: Records to evaluate.
        min_confidence: Minimum Choice confidence for a route to be decided.
        noul_threshold: Minimum Noul probability for an output to be accepted.
        routes: Known route labels. Defaults to every ``expected_route``.

    Returns:
        Counts, rates, usage, latency, and a per-record outcome with its reason.

    Raises:
        ValueError: If a threshold is not a finite number in [0, 1].
    """
    _validate_threshold("min_confidence", min_confidence)
    _validate_threshold("noul_threshold", noul_threshold)
    route_records = [r for r in records if isinstance(r, RouteRecord)]
    known_routes = (
        set(routes) if routes is not None else {r.expected_route for r in route_records}
    )

    outcomes: list[RecordOutcome] = []
    false_accepts = false_rejects = fa_denominator = fr_denominator = 0
    for record in records:
        if isinstance(record, RouteRecord):
            outcome, reason = _classify_route(record, known_routes, min_confidence)
        else:
            outcome, reason = _classify_guardrail(record, noul_threshold)
            if outcome != "failed" and record.expected == "reject":
                fa_denominator += 1
                if outcome == "accepted":
                    false_accepts += 1
            elif outcome != "failed":
                fr_denominator += 1
                if outcome == "rejected":
                    false_rejects += 1
        outcomes.append(
            RecordOutcome(
                id=record.id, kind=record.kind, outcome=outcome, reason=reason
            )
        )

    route_counts = Counter(o.outcome for o in outcomes if o.kind == "route")
    guard_counts = Counter(o.outcome for o in outcomes if o.kind == "guardrail")
    route_total = len(route_records)
    route_decided = route_counts["correct"] + route_counts["incorrect"]
    guard_total = len(records) - route_total
    return EvaluationReport(
        routing=RoutingReport(
            total=route_total,
            decided=route_decided,
            abstained=route_counts["abstained"],
            failed=route_counts["failed"],
            correct=route_counts["correct"],
            accuracy=_rate(route_counts["correct"], route_total),
            coverage=_rate(route_decided, route_total),
            abstention_rate=_rate(route_counts["abstained"], route_total),
        ),
        guardrail=GuardrailReport(
            total=guard_total,
            decided=guard_total - guard_counts["failed"],
            failed=guard_counts["failed"],
            accepted=guard_counts["accepted"],
            rejected=guard_counts["rejected"],
            false_accepts=false_accepts,
            false_accept_denominator=fa_denominator,
            false_accept_rate=_rate(false_accepts, fa_denominator),
            false_rejects=false_rejects,
            false_reject_denominator=fr_denominator,
            false_reject_rate=_rate(false_rejects, fr_denominator),
        ),
        usage=_summarize_usage(records),
        latency=_summarize_latency(records),
        outcomes=outcomes,
    )


def collect_live_decisions(
    records: Sequence[RouteRecord | GuardrailRecord],
    *,
    client: SystemOneClient,
    route_instructions: str,
    route_criteria: Mapping[str, str],
    guardrail_instructions: str,
    model: str | None = None,
) -> list[RouteRecord | GuardrailRecord]:
    """Ask Jev for each record that has no recorded decision or error.

    Opt-in: the caller supplies a client built with their own credentials,
    and remains responsible for closing it. Client errors are recorded on the
    record (which then evaluates as failed) instead of stopping the run.

    Args:
        records: Records to complete; those with a decision or error are kept as is.
        client: A TypeSafe client, or any object with a compatible ``system_one``.
        route_instructions: Instructions for the Choice question on route records.
        route_criteria: Route labels mapped to their descriptions.
        guardrail_instructions: Instructions for the Noul question on guardrail records.
        model: Optional Jev model name passed to the client.

    Returns:
        The records with decisions, latency, and usage filled in where collected.

    Raises:
        ValueError: If instructions are blank or ``route_criteria`` is empty.
    """
    if not route_instructions.strip() or not guardrail_instructions.strip():
        raise ValueError("route_instructions and guardrail_instructions are required")
    if not route_criteria:
        raise ValueError("route_criteria must not be empty")

    collected: list[RouteRecord | GuardrailRecord] = []
    for record in records:
        if record.decision is not None or record.error is not None:
            collected.append(record)
            continue
        if isinstance(record, RouteRecord):
            name = ROUTE_QUESTION
            question: dict[str, Any] = {
                "type": "choice",
                "instructions": route_instructions,
                "criteria": dict(route_criteria),
            }
        else:
            name = GUARDRAIL_QUESTION
            question = {"type": "noul", "instructions": guardrail_instructions}
        collected.append(_query(record, client, name, question, model))
    return collected


def _query(
    record: RouteRecord | GuardrailRecord,
    client: SystemOneClient,
    name: str,
    question: dict[str, Any],
    model: str | None,
) -> RouteRecord | GuardrailRecord:
    start = time.perf_counter()
    try:
        response = client.system_one(record.state, {name: question}, model=model)
    except Exception as e:
        return record.model_copy(update={"error": f"{type(e).__name__}: {e}"})
    latency_ms = (time.perf_counter() - start) * 1000

    usage = getattr(response, "usage", None)
    response_model = getattr(response, "model", None)
    metadata = DecisionMetadata(
        model=response_model if isinstance(response_model, str) else None,
        latency_ms=latency_ms,
    )
    if usage is not None:
        try:
            metadata.usage = TokenUsage.model_validate(usage, from_attributes=True)
        except ValidationError as e:
            first = e.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "usage"
            return record.model_copy(
                update={
                    "error": f"invalid usage: {location}: {first['msg']}",
                    "metadata": metadata,
                }
            )
    answers = getattr(response, "answers", None)
    answer = answers.get(name) if isinstance(answers, Mapping) else None
    if answer is None:
        return record.model_copy(
            update={"error": f"no answer returned for {name!r}", "metadata": metadata}
        )
    answer_type = getattr(answer, "type", None)
    if answer_type is not None and answer_type != question["type"]:
        return record.model_copy(
            update={
                "error": f"expected a {question['type']!r} answer for {name!r},"
                f" got {answer_type!r}",
                "metadata": metadata,
            }
        )
    if isinstance(record, RouteRecord):
        probabilities = getattr(answer, "probabilities", None)
        if probabilities is None:
            probabilities = {}
        elif isinstance(probabilities, Mapping):
            probabilities = dict(probabilities)
        decision = {
            "choice": getattr(answer, "choice", None),
            "confidence": getattr(answer, "confidence", None),
            "probabilities": probabilities,
        }
    else:
        decision = {"noul": getattr(answer, "noul", None)}
    return record.model_copy(update={"decision": decision, "metadata": metadata})


def _parse_decision(
    record: RouteRecord | GuardrailRecord,
    model: type[ChoiceDecision] | type[NoulDecision],
) -> tuple[ChoiceDecision | NoulDecision | None, str]:
    """Return the validated decision, or ``None`` and the failure reason."""
    if record.error is not None:
        return None, f"recorded error: {record.error}"
    if record.decision is None:
        return None, "missing decision"
    try:
        return model.model_validate(record.decision), ""
    except ValidationError as e:
        first = e.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "decision"
        return None, f"invalid {model.__name__}: {location}: {first['msg']}"


def _classify_route(
    record: RouteRecord, known_routes: set[str], min_confidence: float
) -> tuple[Outcome, str]:
    decision, reason = _parse_decision(record, ChoiceDecision)
    if not isinstance(decision, ChoiceDecision):
        return "failed", reason
    if decision.choice not in known_routes:
        return "failed", f"choice {decision.choice!r} is not a known route"
    if decision.confidence < min_confidence:
        return (
            "abstained",
            f"confidence {decision.confidence} below min_confidence {min_confidence}",
        )
    outcome: Outcome = (
        "correct" if decision.choice == record.expected_route else "incorrect"
    )
    return (
        outcome,
        f"confidence {decision.confidence} meets min_confidence {min_confidence}",
    )


def _classify_guardrail(
    record: GuardrailRecord, noul_threshold: float
) -> tuple[Outcome, str]:
    decision, reason = _parse_decision(record, NoulDecision)
    if not isinstance(decision, NoulDecision):
        return "failed", reason
    if decision.noul >= noul_threshold:
        return "accepted", f"noul {decision.noul} meets threshold {noul_threshold}"
    return "rejected", f"noul {decision.noul} below threshold {noul_threshold}"


def _summarize_usage(records: Sequence[RouteRecord | GuardrailRecord]) -> UsageSummary:
    usages = [r.metadata.usage for r in records if r.metadata and r.metadata.usage]
    inputs = [u.input_tokens for u in usages if u.input_tokens is not None]
    outputs = [u.output_tokens for u in usages if u.output_tokens is not None]
    return UsageSummary(
        input_tokens=sum(inputs),
        input_token_records=len(inputs),
        output_tokens=sum(outputs),
        output_token_records=len(outputs),
    )


def _summarize_latency(
    records: Sequence[RouteRecord | GuardrailRecord],
) -> LatencySummary:
    values = [
        r.metadata.latency_ms
        for r in records
        if r.metadata and r.metadata.latency_ms is not None
    ]
    return LatencySummary(
        count=len(values), mean_ms=sum(values) / len(values) if values else None
    )


def _parse_jsonl(text: str) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            items.append((f"line {number}", json.loads(line)))
        except ValueError as e:
            raise ValueError(f"Invalid JSONL at line {number}: {e}") from e
    return items


def _validate_threshold(name: str, value: float) -> None:
    try:
        _THRESHOLD.validate_python(value)
    except ValidationError as e:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value!r}"
        ) from e


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a dataset file and print the report as JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="a .jsonl or .json file")
    parser.add_argument("--min-confidence", type=float, required=True)
    parser.add_argument("--noul-threshold", type=float, required=True)
    parser.add_argument(
        "--route",
        action="append",
        dest="routes",
        help="a known route; repeat for each (default: every expected_route)",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            load_dataset(args.dataset),
            min_confidence=args.min_confidence,
            noul_threshold=args.noul_threshold,
            routes=args.routes,
        )
    except (OSError, ValueError) as e:
        parser.error(str(e))
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
