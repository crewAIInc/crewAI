"""Tests for the shared runtime-input prompting used by flows and crews."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import click
import pytest

from crewai_cli.input_prompt import (
    closest_name,
    parse_inputs_json,
    prompt_for_inputs,
)


def test_parse_inputs_json_returns_none_for_none():
    assert parse_inputs_json(None) is None


def test_parse_inputs_json_parses_object():
    assert parse_inputs_json('{"topic": "AI"}') == {"topic": "AI"}


def test_parse_inputs_json_rejects_invalid_json(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_inputs_json("not json")

    assert exc_info.value.code == 1
    assert "Invalid --inputs JSON" in capsys.readouterr().err


def test_parse_inputs_json_rejects_non_object(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_inputs_json("[1, 2, 3]")

    assert exc_info.value.code == 1
    assert "expected an object" in capsys.readouterr().err


def test_closest_name_suggests_near_miss():
    assert (
        closest_name("prospect_emai", ["prospect_email", "topic"]) == "prospect_email"
    )


def test_closest_name_returns_none_when_nothing_close():
    assert closest_name("zzzzz", ["prospect_email", "topic"]) is None


def test_prompt_for_inputs_uses_describe_and_coerce(monkeypatch, capsys):
    seen: list[str] = []

    def fake_prompt(text: str, **kwargs: object) -> str:
        seen.append(text)
        return "42"

    monkeypatch.setattr("crewai_cli.input_prompt.click.prompt", fake_prompt)

    result = prompt_for_inputs(
        ["count"],
        title="Flow inputs",
        subtitle="This flow needs the following to run.",
        describe=lambda name: f"How many {name}?",
        coerce=lambda name, raw: int(raw),
    )

    captured = capsys.readouterr()
    assert result == {"count": 42}
    assert any("count" in text for text in seen)
    # Header, subtitle, and description hint all render on stderr.
    assert "Flow inputs" in captured.err
    assert "How many count?" in captured.err


def test_prompt_for_inputs_keeps_raw_string_without_coerce(monkeypatch):
    monkeypatch.setattr(
        "crewai_cli.input_prompt.click.prompt", lambda text, **kwargs: "AI"
    )

    result = prompt_for_inputs(
        ["topic"],
        title="Crew inputs",
        subtitle="This crew needs the following to run.",
    )

    assert result == {"topic": "AI"}


@pytest.mark.parametrize(
    ("names", "input_text", "expected"),
    [
        (["topic"], "AI\n", {"topic": "AI"}),
        (["count", "topic"], "42\n研究\n", {"count": 42, "topic": "研究"}),
        (["topic"], "", None),
    ],
    ids=["single-input", "multiple-inputs", "abort"],
)
def test_prompt_for_inputs_keeps_prompt_chrome_on_stderr(
    names: list[str], input_text: str, expected: dict[str, str | int] | None
) -> None:
    """Real Click prompts must leave stdout available for serialized results."""
    script = (
        "import json\n"
        "import click\n"
        "from crewai_cli.input_prompt import prompt_for_inputs\n"
        f"names = {names!r}\n"
        "try:\n"
        "    result = prompt_for_inputs(\n"
        "        names, title='Runtime inputs', subtitle='Required values',\n"
        "        describe=lambda name: f'Value for {name}',\n"
        "        coerce=lambda name, raw: int(raw) if name == 'count' else raw,\n"
        "    )\n"
        "except click.Abort:\n"
        "    result = None\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == json.dumps(expected, ensure_ascii=False) + "\n"
    # Click passes styled prompt text to input() unstripped, so escape codes reach a
    # piped stderr. Unstyle until the prompts stop emitting them off a terminal.
    stderr = click.unstyle(result.stderr)
    assert "Runtime inputs" in stderr
    assert "Required values" in stderr
    for name in names:
        assert f"Value for {name}" in stderr
        assert f"{name} > " in stderr
