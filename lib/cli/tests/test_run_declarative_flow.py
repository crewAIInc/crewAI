from __future__ import annotations

import os
from pathlib import Path

import pytest

import crewai_cli.input_prompt as input_prompt_module
import crewai_cli.run_declarative_flow as run_declarative_flow_module


FLOW_YAML = """\
schema: crewai.flow/v1
name: TestFlow
config:
  suppress_flow_events: true
methods:
  begin:
    start: true
    do:
      call: expression
      expr: state.topic
"""


def test_run_declarative_flow_reads_definition_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")

    run_declarative_flow_module.run_declarative_flow(
        str(definition_path), '{"topic":"AI"}'
    )

    assert capsys.readouterr().out == "AI\n"


def test_run_declarative_flow_rejects_non_object_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(
            str(definition_path), '["not", "an", "object"]'
        )

    assert "Invalid --inputs JSON: expected an object." in capsys.readouterr().err


def test_run_declarative_flow_reports_missing_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow("missing-flow.yaml")

    assert (
        "Invalid --definition path: missing-flow.yaml does not exist."
        in capsys.readouterr().err
    )


def test_run_declarative_flow_reports_empty_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(" \n", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(str(definition_path))

    assert "Flow declaration file is empty" in capsys.readouterr().err


@pytest.mark.parametrize(
    "contents, expected_error",
    [
        ("[]\n", "Flow declaration must contain a mapping"),
        ("schema: crewai.flow/v1\nmethods: {}\n", "Field required"),
    ],
)
def test_load_declarative_flow_reports_invalid_declarations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
    expected_error: str,
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        run_declarative_flow_module.load_declarative_flow(str(definition_path))

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert f"Unable to read --definition path {definition_path}:" in stderr
    assert expected_error in stderr


def test_run_declarative_flow_in_project_env_uses_uv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(
        run_declarative_flow_module,
        "build_env_with_all_tool_credentials",
        lambda: {"EXISTING": "value"},
    )
    monkeypatch.setattr(
        run_declarative_flow_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append((command, kwargs)),
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env("flow.yaml")

    assert subprocess_calls == [
        (
            ["uv", "run", "crewai", "run"],
            {
                "capture_output": False,
                "text": True,
                "check": True,
                "env": {"EXISTING": "value"},
            },
        )
    ]


def test_run_declarative_flow_in_process_inside_uv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "flow.yaml").write_text(FLOW_YAML, encoding="utf-8")

    run_declarative_flow_module.run_declarative_flow_in_project_env(
        "flow.yaml", '{"topic":"AI"}'
    )

    assert capsys.readouterr().out == "AI\n"


def test_run_declarative_flow_in_project_env_forwards_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess_calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(
        run_declarative_flow_module,
        "build_env_with_all_tool_credentials",
        lambda: {},
    )
    monkeypatch.setattr(
        run_declarative_flow_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append(command),
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env(
        "flow.yaml", '{"topic":"AI"}'
    )

    # --inputs is forwarded to the in-env run instead of being rejected.
    assert subprocess_calls == [
        ["uv", "run", "crewai", "run", "--inputs", '{"topic":"AI"}']
    ]


# ── Schema-driven inputs: prompt, validate, override ────────────────

REQUIRED_FLOW_YAML = """\
schema: crewai.flow/v1
name: RequiredInputFlow
config:
  suppress_flow_events: true
state:
  type: json_schema
  json_schema:
    type: object
    properties:
      prospect_email:
        type: string
        description: Email address of the prospect to research
    required: [prospect_email]
methods:
  begin:
    start: true
    do:
      call: expression
      expr: state.prospect_email
"""

DEFAULTS_FLOW_YAML = """\
schema: crewai.flow/v1
name: DefaultsFlow
config:
  suppress_flow_events: true
state:
  type: json_schema
  json_schema:
    type: object
    properties:
      topic: {type: string}
      audience: {type: string}
    required: [topic, audience]
  default:
    topic: AI
methods:
  begin:
    start: true
    do:
      call: expression
      expr: state.audience
"""

TYPED_FLOW_YAML = """\
schema: crewai.flow/v1
name: TypedFlow
config:
  suppress_flow_events: true
state:
  type: json_schema
  json_schema:
    type: object
    properties:
      count: {type: integer}
    required: [count]
methods:
  begin:
    start: true
    do:
      call: expression
      expr: state.count
"""


def _write(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "flow.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_inputs_flag_satisfies_required_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, REQUIRED_FLOW_YAML)

    run_declarative_flow_module.run_declarative_flow(
        str(path), '{"prospect_email":"a@b.com"}'
    )

    assert capsys.readouterr().out == "a@b.com\n"


def test_missing_required_reports_pointed_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_declarative_flow_module, "_is_interactive", lambda: False)
    path = _write(tmp_path, REQUIRED_FLOW_YAML)

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(str(path))

    assert (
        "Missing required input 'prospect_email' — "
        "Email address of the prospect to research" in capsys.readouterr().err
    )


def test_prompts_for_missing_required_when_interactive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, REQUIRED_FLOW_YAML)
    monkeypatch.setattr(run_declarative_flow_module, "_is_interactive", lambda: True)
    prompted: list[str] = []

    def fake_prompt(text: str, **kwargs: object) -> str:
        prompted.append(text)
        return "typed@example.com"

    monkeypatch.setattr(input_prompt_module.click, "prompt", fake_prompt)

    run_declarative_flow_module.run_declarative_flow(str(path))

    assert capsys.readouterr().out == "typed@example.com\n"
    assert any("prospect_email" in text for text in prompted)


def test_defaults_satisfy_required_and_are_not_prompted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_declarative_flow_module, "_is_interactive", lambda: False)
    path = _write(tmp_path, DEFAULTS_FLOW_YAML)

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(str(path))

    err = capsys.readouterr().err
    # topic has a state default -> satisfied; only audience is missing.
    assert "Missing required input 'audience'" in err
    assert "'topic'" not in err


def test_warns_on_unknown_input_with_suggestion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, REQUIRED_FLOW_YAML)

    run_declarative_flow_module.run_declarative_flow(
        str(path), '{"prospect_email":"a@b.com","prospect_emai":"typo"}'
    )

    captured = capsys.readouterr()
    assert captured.out == "a@b.com\n"
    assert "Ignoring unknown input 'prospect_emai'" in captured.err
    assert "Did you mean 'prospect_email'?" in captured.err


def test_validates_input_types_before_kickoff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, TYPED_FLOW_YAML)

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(str(path), '{"count":"nope"}')

    assert "Invalid input 'count'" in capsys.readouterr().err


def test_reserved_id_input_is_forwarded_not_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `id` is a reserved kickoff key (persistence restore); it must pass through
    # instead of being flagged as an unknown key and dropped.
    path = _write(tmp_path, REQUIRED_FLOW_YAML)

    run_declarative_flow_module.run_declarative_flow(
        str(path), '{"id":"run-123","prospect_email":"a@b.com"}'
    )

    captured = capsys.readouterr()
    assert captured.out == "a@b.com\n"
    assert "Ignoring unknown input 'id'" not in captured.err


def test_run_declarative_flow_loads_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Flow projects must pick up the project's .env, like crew projects do,
    # overriding any pre-existing value.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DECL_FLOW_ENV_PROBE", "old")
    (tmp_path / ".env").write_text("DECL_FLOW_ENV_PROBE=from_dotenv\n", encoding="utf-8")
    path = _write(tmp_path, REQUIRED_FLOW_YAML)

    run_declarative_flow_module.run_declarative_flow(
        str(path), '{"prospect_email":"a@b.com"}'
    )

    assert os.environ["DECL_FLOW_ENV_PROBE"] == "from_dotenv"


def test_id_only_input_skips_required_validation(tmp_path: Path) -> None:
    # Resume via `crewai run --inputs '{"id":"..."}'` must not be blocked by the
    # required-field check: kickoff hydrates required state from persistence.
    path = _write(tmp_path, REQUIRED_FLOW_YAML)
    flow = run_declarative_flow_module.load_declarative_flow(str(path))

    resolved = run_declarative_flow_module._resolve_flow_inputs(flow, {"id": "run-123"})

    assert resolved == {"id": "run-123"}


def test_id_restore_still_drops_unknown_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A persistence restore (`id` present) still filters typo keys so they don't
    # reach kickoff and trip strict (extra="forbid") state models — it only
    # skips the required-field prompt/validation, not the unknown-key warning.
    path = _write(tmp_path, REQUIRED_FLOW_YAML)
    flow = run_declarative_flow_module.load_declarative_flow(str(path))

    resolved = run_declarative_flow_module._resolve_flow_inputs(
        flow, {"id": "run-123", "prospect_emai": "typo"}
    )

    captured = capsys.readouterr()
    assert resolved == {"id": "run-123"}  # id kept, typo dropped
    assert "Ignoring unknown input 'prospect_emai'" in captured.err
    assert "Ignoring unknown input 'id'" not in captured.err
