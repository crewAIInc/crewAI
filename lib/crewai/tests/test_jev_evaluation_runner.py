"""Offline checks for the Jev routing and guardrail evaluation runner."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_runner():
    path = Path(__file__).resolve().parents[3] / "scripts" / "jev_evaluation_runner.py"
    spec = importlib.util.spec_from_file_location("jev_evaluation_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
GuardrailRecord = runner.GuardrailRecord
RouteRecord = runner.RouteRecord
collect_live_decisions = runner.collect_live_decisions
evaluate = runner.evaluate
load_dataset = runner.load_dataset


EXAMPLE = [
    {
        "id": "r1",
        "kind": "route",
        "state": "Refund for order 1234",
        "expected_route": "billing",
        "decision": {
            "choice": "billing",
            "confidence": 0.91,
            "probabilities": {"billing": 0.95, "tech": 0.05},
        },
        "metadata": {
            "latency_ms": 212.0,
            "usage": {"input_tokens": 40, "output_tokens": 1},
        },
    },
    {
        "id": "r2",
        "kind": "route",
        "state": "App crashes on login",
        "expected_route": "tech",
        "decision": {
            "choice": "tech",
            "confidence": 0.62,
            "probabilities": {"billing": 0.19, "tech": 0.81},
        },
    },
    {
        "id": "r3",
        "kind": "route",
        "state": "Change my plan",
        "expected_route": "billing",
        "decision": {
            "choice": "tech",
            "confidence": 0.88,
            "probabilities": {"billing": 0.06, "tech": 0.94},
        },
    },
    {
        "id": "r4",
        "kind": "route",
        "state": "Where is my invoice?",
        "expected_route": "billing",
        "decision": None,
        "error": "TypeSafeAPITimeoutError: request timed out",
    },
    {
        "id": "g1",
        "kind": "guardrail",
        "state": "Paris is the capital of France.",
        "expected": "accept",
        "decision": {"noul": 0.97},
    },
    {
        "id": "g2",
        "kind": "guardrail",
        "state": "The moon is made of cheese.",
        "expected": "reject",
        "decision": {"noul": 0.8},
    },
    {
        "id": "g3",
        "kind": "guardrail",
        "state": "Water boils at 100 C at sea level.",
        "expected": "accept",
        "decision": {"noul": 0.79},
    },
    {
        "id": "g4",
        "kind": "guardrail",
        "state": "2 + 2 = 5",
        "expected": "reject",
        "decision": {"noul": 1.4},
    },
]


def write_jsonl(tmp_path, records, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def route(decision, expected="billing", **extra):
    return RouteRecord(
        id="r",
        kind="route",
        state="s",
        expected_route=expected,
        decision=decision,
        **extra,
    )


def guardrail(decision, expected="accept", **extra):
    return GuardrailRecord(
        id="g",
        kind="guardrail",
        state="s",
        expected=expected,
        decision=decision,
        **extra,
    )


def outcome_of(record, **thresholds):
    kwargs = {
        "min_confidence": 0.7,
        "noul_threshold": 0.8,
        "routes": {"billing", "tech"},
    } | thresholds
    return evaluate([record], **kwargs).outcomes[0]


# Metric denominators


def test_example_dataset_report(tmp_path):
    report = evaluate(
        load_dataset(write_jsonl(tmp_path, EXAMPLE)),
        min_confidence=0.7,
        noul_threshold=0.8,
    )

    assert report.routing.model_dump() == {
        "total": 4,
        "decided": 2,
        "abstained": 1,
        "failed": 1,
        "correct": 1,
        "accuracy": 0.25,
        "coverage": 0.5,
        "abstention_rate": 0.25,
    }
    assert report.guardrail.model_dump() == {
        "total": 4,
        "decided": 3,
        "failed": 1,
        "accepted": 2,
        "rejected": 1,
        "false_accepts": 1,
        "false_accept_denominator": 1,
        "false_accept_rate": 1.0,
        "false_rejects": 1,
        "false_reject_denominator": 2,
        "false_reject_rate": 0.5,
    }
    assert report.usage.model_dump() == {
        "input_tokens": 40,
        "input_token_records": 1,
        "output_tokens": 1,
        "output_token_records": 1,
    }
    assert report.latency.model_dump() == {"count": 1, "mean_ms": 212.0}
    assert [o.outcome for o in report.outcomes] == [
        "correct",
        "abstained",
        "incorrect",
        "failed",
        "accepted",
        "accepted",
        "rejected",
        "failed",
    ]


def test_failed_guardrails_are_excluded_from_false_rate_denominators():
    records = [
        guardrail(None, expected="reject"),
        guardrail({"noul": 0.9}, expected="accept"),
    ]
    report = evaluate(records, min_confidence=0.5, noul_threshold=0.8)

    assert report.guardrail.failed == 1
    assert report.guardrail.false_accept_denominator == 0
    assert report.guardrail.false_accept_rate is None
    assert report.guardrail.false_reject_denominator == 1
    assert report.guardrail.false_reject_rate == 0.0


def test_empty_dataset_reports_none_rates():
    report = evaluate([], min_confidence=0.5, noul_threshold=0.5)

    assert report.routing.accuracy is None
    assert report.routing.coverage is None
    assert report.guardrail.false_reject_rate is None
    assert report.latency.mean_ms is None


# Threshold boundaries


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.69, "abstained"), (0.7, "correct"), (0.71, "correct")],
)
def test_choice_threshold_is_inclusive(confidence, expected):
    record = route({"choice": "billing", "confidence": confidence, "probabilities": {}})

    assert outcome_of(record, min_confidence=0.7).outcome == expected


@pytest.mark.parametrize(
    ("noul", "threshold", "expected"),
    [
        (0.79, 0.8, "rejected"),
        (0.8, 0.8, "accepted"),
        (0.81, 0.8, "accepted"),
        (0.0, 0.0, "accepted"),
        (1.0, 1.0, "accepted"),
    ],
)
def test_noul_threshold_is_inclusive(noul, threshold, expected):
    assert (
        outcome_of(guardrail({"noul": noul}), noul_threshold=threshold).outcome
        == expected
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_thresholds_raise(bad):
    with pytest.raises(ValueError, match="noul_threshold"):
        evaluate([], min_confidence=0.5, noul_threshold=bad)
    with pytest.raises(ValueError, match="min_confidence"):
        evaluate([], min_confidence=bad, noul_threshold=0.5)


# Missing and invalid decisions


@pytest.mark.parametrize(
    ("decision", "extra", "reason"),
    [
        (None, {}, "missing decision"),
        (
            {"choice": "billing", "confidence": 0.9, "probabilities": {}},
            {"error": "boom"},
            "recorded error: boom",
        ),
        (
            {"choice": "sales", "confidence": 0.9, "probabilities": {}},
            {},
            "not a known route",
        ),
        (
            {"choice": "billing", "confidence": 1.2, "probabilities": {}},
            {},
            "invalid ChoiceDecision: confidence",
        ),
        (
            {"choice": "billing", "confidence": "0.9", "probabilities": {}},
            {},
            "invalid ChoiceDecision: confidence",
        ),
        (
            {"choice": "billing", "probabilities": {}},
            {},
            "invalid ChoiceDecision: confidence",
        ),
        (
            {"choice": "billing", "confidence": 0.9, "probabilities": {"billing": 1.5}},
            {},
            "invalid ChoiceDecision: probabilities.billing",
        ),
    ],
)
def test_unusable_route_decisions_fail(decision, extra, reason):
    result = outcome_of(route(decision, **extra))

    assert result.outcome == "failed"
    assert reason in result.reason


@pytest.mark.parametrize(
    "decision",
    [
        None,
        {"noul": -0.1},
        {"noul": 1.4},
        {"noul": True},
        {"noul": "0.9"},
        {"confidence": 0.9},
    ],
)
def test_unusable_guardrail_decisions_fail_and_never_accept(decision):
    assert outcome_of(guardrail(decision, expected="reject")).outcome == "failed"


def test_recorded_guardrail_error_fails_even_with_a_passing_decision():
    result = outcome_of(guardrail({"noul": 0.99}, error="boom"))

    assert (result.outcome, result.reason) == ("failed", "recorded error: boom")


def test_route_confidence_is_never_used_as_noul():
    record = guardrail(
        {"choice": "accept", "confidence": 0.99, "probabilities": {"accept": 1.0}}
    )

    assert outcome_of(record).outcome == "failed"


# Invalid inputs


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"id": "a"', ""], "line 1"),
        (['["not", "an", "object"]'], "Invalid record at line 1"),
        (
            ['{"kind": "route", "state": "s", "expected_route": "b"}'],
            "Invalid record at line 1",
        ),
        (['{"id": "a", "kind": "other", "state": "s"}'], "Invalid record at line 1"),
        (
            ['{"id": "a", "kind": "guardrail", "state": "s", "expected": "maybe"}'],
            "Invalid record at line 1",
        ),
        (
            [json.dumps(EXAMPLE[0]), json.dumps(EXAMPLE[0])],
            "Duplicate record id 'r1' at line 2",
        ),
        (
            [
                '{"id": "a", "kind": "route", "state": "s", "expected_route": "b", "notes": "x"}'
            ],
            "Invalid record at line 1",
        ),
        (
            ['{"id": "a", "kind": "route", "expected_route": "b"}'],
            "Invalid record at line 1",
        ),
        (
            [
                '{"id": "a", "kind": "route", "state": "s", "expected_route": "b", "metadata": {"latency_ms": -1}}'
            ],
            "Invalid record at line 1",
        ),
    ],
)
def test_malformed_jsonl_raises_with_location(tmp_path, lines, message):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(path)


@pytest.mark.parametrize("name", ["data.json", "data.jsonl"])
def test_json_and_jsonl_with_bom_and_blank_lines_are_accepted(tmp_path, name):
    content = (
        json.dumps(EXAMPLE)
        if name.endswith(".json")
        else "\n\n".join(json.dumps(r) for r in EXAMPLE)
    )
    (tmp_path / name).write_text(content, encoding="utf-8-sig")

    assert [r.id for r in load_dataset(tmp_path / name)] == [r["id"] for r in EXAMPLE]


def test_jsonl_strings_may_contain_unicode_line_separators(tmp_path):
    record = {"id": "a", "kind": "route", "state": "a\u2028b\x85c", "expected_route": "b"}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    assert [r.state for r in load_dataset(path)] == ["a\u2028b\x85c"]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("data.json", "{}", "must be an array"),
        ("data.json", "[1,", "Invalid JSON"),
        ("data.json", '[{"id": "a"}]', "Invalid record at index 0"),
        ("data.csv", "", "Unsupported"),
    ],
)
def test_wrong_json_shape_or_extension_raises(tmp_path, name, content, message):
    (tmp_path / name).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset(tmp_path / name)


# Command line

CLI_THRESHOLDS = ["--min-confidence", "0.7", "--noul-threshold", "0.8"]


def test_main_prints_the_report_as_json(tmp_path, capsys):
    path = write_jsonl(tmp_path, EXAMPLE)

    assert runner.main([str(path), *CLI_THRESHOLDS]) == 0
    expected = evaluate(load_dataset(path), min_confidence=0.7, noul_threshold=0.8)
    assert json.loads(capsys.readouterr().out) == expected.model_dump(mode="json")


def test_main_reports_invalid_input_as_a_usage_error(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        runner.main([str(path), *CLI_THRESHOLDS])

    assert exit_info.value.code == 2
    assert "Invalid JSONL at line 1" in capsys.readouterr().err


# Usage and latency


def test_missing_usage_and_latency_are_not_counted_as_zero():
    records = [
        guardrail(
            {"noul": 0.9}, metadata={"usage": {"input_tokens": 10}, "latency_ms": 100.0}
        ),
        guardrail({"noul": 0.9}, metadata={"usage": {"output_tokens": 2}}),
        guardrail({"noul": 0.9}),
    ]
    report = evaluate(records, min_confidence=0.5, noul_threshold=0.5)

    assert report.usage.model_dump() == {
        "input_tokens": 10,
        "input_token_records": 1,
        "output_tokens": 2,
        "output_token_records": 1,
    }
    assert report.latency.model_dump() == {"count": 1, "mean_ms": 100.0}


# Live collection with a fake client


class FakeClient:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.closed = False

    def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


LIVE_KWARGS = {
    "route_instructions": "Pick a team",
    "route_criteria": {"billing": "Payments", "tech": "Bugs"},
    "guardrail_instructions": "Is it true?",
}


def test_live_collection_builds_questions_and_records_metadata(monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(
        runner, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    choice = SimpleNamespace(
        choice="tech", confidence=0.8, probabilities={"billing": 0.1, "tech": 0.9}
    )
    client = FakeClient(
        SimpleNamespace(
            model="jev-1",
            usage=SimpleNamespace(input_tokens=5, output_tokens=1),
            answers={"route": choice},
        ),
        SimpleNamespace(
            model="jev-1", usage=None, answers={"guardrail": SimpleNamespace(noul=0.9)}
        ),
    )
    records = collect_live_decisions(
        [route(None), guardrail(None)], client=client, model="jev", **LIVE_KWARGS
    )

    assert client.calls[0] == (
        "s",
        {
            "route": {
                "type": "choice",
                "instructions": "Pick a team",
                "criteria": {"billing": "Payments", "tech": "Bugs"},
            }
        },
        "jev",
    )
    assert client.calls[1] == (
        "s",
        {"guardrail": {"type": "noul", "instructions": "Is it true?"}},
        "jev",
    )
    assert records[0].decision == {
        "choice": "tech",
        "confidence": 0.8,
        "probabilities": {"billing": 0.1, "tech": 0.9},
    }
    assert records[0].metadata.model_dump() == {
        "model": "jev-1",
        "usage": {"input_tokens": 5, "output_tokens": 1},
        "latency_ms": 250.0,
    }
    assert records[1].decision == {"noul": 0.9}
    assert records[1].metadata.latency_ms == 500.0
    assert not client.closed


def test_live_client_errors_become_failed_records():
    client = FakeClient(RuntimeError("timeout"), SimpleNamespace(answers={}))
    records = collect_live_decisions(
        [guardrail(None), guardrail(None)], client=client, **LIVE_KWARGS
    )
    report = evaluate(records, min_confidence=0.5, noul_threshold=0.5)

    assert records[0].error == "RuntimeError: timeout"
    assert records[0].metadata is None
    assert records[1].error == "no answer returned for 'guardrail'"
    assert [o.outcome for o in report.outcomes] == ["failed", "failed"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (SimpleNamespace(answers=["guardrail"]), "no answer returned for 'guardrail'"),
        (
            SimpleNamespace(
                usage=SimpleNamespace(input_tokens=-1, output_tokens=1),
                answers={"guardrail": SimpleNamespace(noul=0.9)},
            ),
            "invalid usage: input_tokens: Input should be greater than or equal to 0",
        ),
    ],
)
def test_malformed_live_responses_become_failed_records(response, error):
    [record] = collect_live_decisions(
        [guardrail(None)], client=FakeClient(response), **LIVE_KWARGS
    )

    assert record.error == error
    assert record.decision is None
    assert record.metadata.latency_ms is not None


@pytest.mark.parametrize(
    ("record", "answers", "error"),
    [
        (
            guardrail(None),
            {
                "guardrail": SimpleNamespace(
                    type="choice", choice="tech", confidence=0.9, probabilities={}
                )
            },
            "expected a 'noul' answer for 'guardrail', got 'choice'",
        ),
        (
            guardrail(None),
            {"guardrail": SimpleNamespace(type="choice", choice="tech", noul=0.9)},
            "expected a 'noul' answer for 'guardrail', got 'choice'",
        ),
        (
            route(None),
            {"route": SimpleNamespace(type="noul", noul=0.9)},
            "expected a 'choice' answer for 'route', got 'noul'",
        ),
    ],
)
def test_live_answer_of_the_wrong_type_fails(record, answers, error):
    client = FakeClient(SimpleNamespace(answers=answers))
    [record] = collect_live_decisions([record], client=client, **LIVE_KWARGS)

    assert record.error == error
    assert record.decision is None
    assert outcome_of(record).outcome == "failed"


@pytest.mark.parametrize("probabilities", [42, 0, False, "", []])
def test_malformed_probabilities_fail_without_stopping_collection(probabilities):
    bad = SimpleNamespace(choice="tech", confidence=0.9, probabilities=probabilities)
    good = SimpleNamespace(choice="tech", confidence=0.9, probabilities={"tech": 0.9})
    client = FakeClient(
        SimpleNamespace(answers={"route": bad}),
        SimpleNamespace(answers={"route": good}),
    )
    records = collect_live_decisions(
        [route(None, expected="tech"), route(None, expected="tech")],
        client=client,
        **LIVE_KWARGS,
    )

    assert [outcome_of(r).outcome for r in records] == ["failed", "correct"]
    assert "probabilities" in outcome_of(records[0]).reason


def test_live_answer_without_probabilities_or_usage():
    answer = SimpleNamespace(choice="tech", confidence=0.8, probabilities=None)
    client = FakeClient(SimpleNamespace(answers={"route": answer}))
    [record] = collect_live_decisions([route(None)], client=client, **LIVE_KWARGS)

    assert record.decision["probabilities"] == {}
    assert record.metadata.usage is None
    assert record.metadata.model is None


@pytest.mark.parametrize(
    "override",
    [
        {"route_instructions": " "},
        {"guardrail_instructions": ""},
        {"route_criteria": {}},
    ],
)
def test_live_collection_requires_questions(override):
    with pytest.raises(ValueError):
        collect_live_decisions([], client=FakeClient(), **(LIVE_KWARGS | override))


def test_live_collection_skips_recorded_decisions():
    client = FakeClient()
    records = [guardrail({"noul": 0.9}), guardrail(None, error="earlier failure")]

    assert collect_live_decisions(records, client=client, **LIVE_KWARGS) == records
    assert client.calls == []
