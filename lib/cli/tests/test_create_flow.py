from __future__ import annotations

from collections.abc import Iterator
import importlib
import io
from pathlib import Path
import sys
from types import ModuleType

import click
from click.testing import CliRunner, Result
import pytest
from pytest import MonkeyPatch
import tomli

from crewai_cli import create_json_crew as json_crew
from crewai_cli.cli import crewai
import crewai_cli.create_flow as create_flow_module
from crewai_cli.utils import get_project_id
from crewai_cli.version import get_crewai_dependency_range


create_flow = create_flow_module.create_flow


@pytest.fixture(autouse=True)
def _stdin_is_not_a_terminal(monkeypatch: MonkeyPatch) -> None:
    """Run `crewai create flow` without a terminal unless a test asks for one, so
    `pytest -s` at a terminal does not open the model picker and wait for keys."""
    monkeypatch.setattr("sys.stdin", io.StringIO())


def test_create_flow_declarative_project_can_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow", declarative=True)

    project_root = tmp_path / "research_flow"
    assert project_root.is_dir()

    pyproject = tomli.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["name"] == "research_flow"
    assert pyproject["project"]["requires-python"]
    assert pyproject["project"]["dependencies"]
    assert (project_root / pyproject["tool"]["crewai"]["definition"]).is_file()
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "CrewAI Flow declaration" in agents_md
    assert "schema: crewai.flow/v1" in agents_md
    assert "do not assemble the string with CEL `+`" in agents_md
    assert "Agent prompt template. Insert Flow values with `${...}`" in agents_md
    assert "Runtime inputs passed to the Crew" in agents_md
    assert "call: expression" in agents_md
    assert "call: tool" not in agents_md
    assert "call: script" not in agents_md
    assert "call: each" not in agents_md
    assert "human_feedback" not in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    cursor_md = (project_root / "CURSOR.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in cursor_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()

    monkeypatch.chdir(project_root)
    result = CliRunner().invoke(crewai, ["run"], env={"UV_RUN_RECURSION_DEPTH": "1"})

    assert result.exit_code == 0
    assert "Running the Flow" not in result.output
    assert "AI agents" in result.output


@pytest.mark.parametrize("declarative", [False, True])
def test_create_flow_depends_on_crewai_with_tools_only(
    tmp_path: Path, monkeypatch: MonkeyPatch, declarative: bool
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow", declarative=declarative)

    pyproject = tomli.loads(
        (tmp_path / "research_flow" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["dependencies"][0] == (
        f"crewai[tools]{get_crewai_dependency_range()}"
    )


@pytest.mark.parametrize("declarative", [False, True])
def test_create_flow_returns_the_folder_it_created(
    tmp_path: Path, monkeypatch: MonkeyPatch, declarative: bool
):
    monkeypatch.chdir(tmp_path)

    project = create_flow("News v2.0", declarative=declarative)

    # A flow folder keeps the ".", which a crew folder would drop.
    assert project == Path("news_v2.0")
    assert (tmp_path / "news_v2.0" / "pyproject.toml").is_file()


@pytest.mark.parametrize("declarative", [False, True])
def test_create_flow_returns_none_when_the_folder_already_exists(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    declarative: bool,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "research_flow").mkdir()

    assert create_flow("Research Flow", declarative=declarative) is None
    assert "Error: Folder research_flow already exists." in capsys.readouterr().out
    assert list((tmp_path / "research_flow").iterdir()) == []


SONNET = "anthropic/claude-sonnet-5-5"
LUNA = "openai/gpt-6-luna"


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _at_a_terminal(monkeypatch: MonkeyPatch) -> None:
    """Run `crewai create flow` as a person at a terminal: stdin and stdout are
    terminals and CREWAI_DMN is unset. Call it in the test body: pytest gives the
    test a different captured stdout than its fixtures see."""
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr("sys.stdin", _Terminal())
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


def _output(capsys: pytest.CaptureFixture[str]) -> str:
    """What was printed, without the colors click adds at a terminal."""
    return click.unstyle(capsys.readouterr().out)


def _pick_model(monkeypatch: MonkeyPatch, model: str) -> None:
    monkeypatch.setattr(json_crew, "_select_model", lambda: model)


def _answer_key_prompts(
    monkeypatch: MonkeyPatch, answers: list[str]
) -> list[dict[str, object]]:
    """Answer API-key prompts in order and record each prompt's text and options."""
    calls: list[dict[str, object]] = []
    replies = iter(answers)

    def fake_prompt(text: str, **kwargs: object) -> str:
        calls.append({"text": click.unstyle(text), **kwargs})
        return next(replies)

    monkeypatch.setattr(json_crew.click, "prompt", fake_prompt)
    return calls


def _forbid_prompts(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        json_crew, "_select_model", lambda: pytest.fail("asked for a model")
    )
    monkeypatch.setattr(
        json_crew.click,
        "prompt",
        lambda *_args, **_kwargs: pytest.fail("asked for an API key"),
    )


def _crewai_dependency(project_root: Path) -> str:
    pyproject = tomli.loads((project_root / "pyproject.toml").read_text())
    return str(pyproject["project"]["dependencies"][0])


def _assert_next_steps(output: str, folder_name: str, key_line: str | None) -> None:
    steps = output[output.index("Next steps:") :]
    assert f"    cd {folder_name}\n" in steps
    assert "  Run your flow:\n    crewai run\n" in steps
    if key_line is None:
        assert "in .env before `crewai run`" not in steps
    else:
        assert f"  {key_line}\n" in steps
        assert steps.index(f"cd {folder_name}") < steps.index(key_line)
        assert steps.index(key_line) < steps.index("Run your flow:")


def test_create_flow_at_a_terminal_saves_the_picked_model_and_key(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _pick_model(monkeypatch, SONNET)
    calls = _answer_key_prompts(monkeypatch, ["sk-ant-test"])

    create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == (
        f"MODEL={SONNET}\nANTHROPIC_API_KEY=sk-ant-test\n"
    )
    assert [call["text"] for call in calls] == [
        "  Enter your ANTHROPIC API key (press Enter to skip)"
    ]
    assert [call["hide_input"] for call in calls] == [True]
    assert _crewai_dependency(project_root) == (
        f"crewai[tools,anthropic]{get_crewai_dependency_range()}"
    )
    output = _output(capsys)
    assert "  API keys and model saved to .env file\n" in output
    _assert_next_steps(output, "research_flow", key_line=None)


def test_create_flow_at_a_terminal_blank_key_prints_the_line_to_add(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _pick_model(monkeypatch, LUNA)
    _answer_key_prompts(monkeypatch, [""])

    create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == f"MODEL={LUNA}\n"
    assert _crewai_dependency(project_root) == (
        f"crewai[tools]{get_crewai_dependency_range()}"
    )
    output = _output(capsys)
    assert (
        f"  API key not saved. Add this line to {Path('research_flow') / '.env'} "
        "before `crewai run`:\n    OPENAI_API_KEY=<your key>\n"
    ) in output
    assert "saved to .env file" not in output
    _assert_next_steps(output, "research_flow", key_line=None)


def test_create_flow_provider_preselects_the_model_and_still_asks_for_the_key(
    tmp_path: Path, monkeypatch: MonkeyPatch, no_provider_keys: None
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        json_crew, "_select_model", lambda: pytest.fail("--provider picks the model")
    )
    calls = _answer_key_prompts(monkeypatch, ["sk-ant-test"])

    create_flow("Research Flow", provider="anthropic")

    assert [call["text"] for call in calls] == [
        "  Enter your ANTHROPIC API key (press Enter to skip)"
    ]
    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == (
        f"MODEL={SONNET}\nANTHROPIC_API_KEY=sk-ant-test\n"
    )
    assert _crewai_dependency(project_root) == (
        f"crewai[tools,anthropic]{get_crewai_dependency_range()}"
    )


def test_create_flow_skip_provider_asks_nothing_at_a_terminal(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", skip_provider=True)

    project_root = tmp_path / "research_flow"
    assert not (project_root / ".env").exists()
    assert _crewai_dependency(project_root) == (
        f"crewai[tools]{get_crewai_dependency_range()}"
    )
    _assert_next_steps(
        _output(capsys),
        "research_flow",
        key_line="Set OPENAI_API_KEY in .env before `crewai run`",
    )


@pytest.mark.parametrize(
    ("stdin_is_terminal", "stdout_is_terminal", "dmn"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
    ids=["no-terminal", "piped-stdin", "piped-stdout", "crewai-dmn"],
)
def test_create_flow_without_a_terminal_asks_nothing_and_writes_no_placeholder(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
    stdin_is_terminal: bool,
    stdout_is_terminal: bool,
    dmn: bool,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.stdin", _Terminal() if stdin_is_terminal else io.StringIO()
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_is_terminal)
    if dmn:
        monkeypatch.setenv("CREWAI_DMN", "true")
    else:
        monkeypatch.delenv("CREWAI_DMN", raising=False)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow")

    assert not (tmp_path / "research_flow" / ".env").exists()
    _assert_next_steps(
        _output(capsys),
        "research_flow",
        key_line="Set OPENAI_API_KEY in .env before `crewai run`",
    )


def test_crewai_create_flow_without_a_terminal_exits_zero_with_next_steps(
    tmp_path: Path, monkeypatch: MonkeyPatch, no_provider_keys: None
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    _forbid_prompts(monkeypatch)

    result = CliRunner().invoke(crewai, ["create", "flow", "Research Flow"])

    assert result.exit_code == 0, result.output
    assert "Flow Research Flow created successfully!" in result.output
    _assert_next_steps(
        result.output,
        "research_flow",
        key_line="Set OPENAI_API_KEY in .env before `crewai run`",
    )
    assert not (tmp_path / "research_flow" / ".env").exists()


def test_create_flow_declarative_asks_nothing_at_a_terminal(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", declarative=True)

    project_root = tmp_path / "research_flow"
    assert not (project_root / ".env").exists()
    assert _crewai_dependency(project_root) == (
        f"crewai[tools]{get_crewai_dependency_range()}"
    )
    # Its starter runs without a key, so the next steps name none.
    _assert_next_steps(_output(capsys), "research_flow", key_line=None)


@pytest.mark.parametrize(
    ("declarative", "key_line"),
    [
        (False, "Set ANTHROPIC_API_KEY in .env before `crewai run`"),
        (True, None),
    ],
    ids=["python", "declarative"],
)
def test_create_flow_provider_without_a_terminal_saves_only_the_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
    declarative: bool,
    key_line: str | None,
):
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", declarative=declarative, provider="anthropic")

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == f"MODEL={SONNET}\n"
    assert _crewai_dependency(project_root) == (
        f"crewai[tools,anthropic]{get_crewai_dependency_range()}"
    )
    _assert_next_steps(_output(capsys), "research_flow", key_line=key_line)


@pytest.mark.parametrize(
    ("provider", "extras", "key_line"),
    [
        (
            "claude/claude-sonnet-5",
            "tools,anthropic",
            "Set ANTHROPIC_API_KEY in .env before `crewai run`",
        ),
        (
            "bedrock/anthropic.claude-sonnet-5",
            "tools,bedrock",
            "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env before `crewai run`",
        ),
    ],
    ids=["aliased-prefix", "unlisted-model"],
)
def test_create_flow_provider_model_names_the_key_of_the_provider_it_runs_on(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
    provider: str,
    extras: str,
    key_line: str,
):
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", provider=provider)

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == f"MODEL={provider}\n"
    assert _crewai_dependency(project_root) == (
        f"crewai[{extras}]{get_crewai_dependency_range()}"
    )
    _assert_next_steps(_output(capsys), "research_flow", key_line=key_line)


def test_create_flow_aliased_provider_at_a_terminal_asks_for_its_key(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = _answer_key_prompts(monkeypatch, ["sk-ant-test"])

    create_flow("Research Flow", provider="claude/claude-sonnet-5")

    assert [call["text"] for call in calls] == [
        "  Enter your ANTHROPIC API key (press Enter to skip)"
    ]
    assert (tmp_path / "research_flow" / ".env").read_text() == (
        "MODEL=claude/claude-sonnet-5\nANTHROPIC_API_KEY=sk-ant-test\n"
    )
    assert "  API keys and model saved to .env file\n" in _output(capsys)


@pytest.mark.parametrize(
    ("provider", "model", "extras"),
    [
        ("bedrock", "bedrock/global.anthropic.claude-sonnet-5-5", "tools,bedrock"),
        ("azure", "azure/gpt-6-luna", "tools,azure-ai-inference"),
    ],
)
def test_create_flow_provider_uses_its_first_listed_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    no_provider_keys: None,
    provider: str,
    model: str,
    extras: str,
):
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", provider=provider)

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == f"MODEL={model}\n"
    assert _crewai_dependency(project_root) == (
        f"crewai[{extras}]{get_crewai_dependency_range()}"
    )


@pytest.mark.parametrize("at_a_terminal", [False, True], ids=["no-terminal", "terminal"])
@pytest.mark.parametrize("provider", ["mistral", "unknown"])
def test_create_flow_rejects_a_provider_with_no_listed_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
    at_a_terminal: bool,
):
    if at_a_terminal:
        _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    with pytest.raises(
        click.BadParameter,
        match=(
            f"^'{provider}' has no default model. Pass one of openai, anthropic, "
            "gemini, groq, ollama, bedrock, azure, nvidia_nim, huggingface, "
            "cerebras, deepinfra, sambanova, watson, or a provider/model string.$"
        ),
    ):
        create_flow("Research Flow", provider=provider)

    assert list(tmp_path.iterdir()) == []
    assert "Creating flow" not in _output(capsys)


def test_crewai_create_flow_with_a_provider_with_no_listed_model_exits_two(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    result = CliRunner().invoke(
        crewai, ["create", "flow", "Research Flow", "--provider", "mistral"]
    )

    assert result.exit_code == 2, result.output
    assert (
        "Invalid value for '--provider': 'mistral' has no default model."
    ) in result.output
    assert not (tmp_path / "research_flow").exists()


def test_create_flow_ctrl_c_at_the_key_prompt_leaves_a_complete_project(
    tmp_path: Path, monkeypatch: MonkeyPatch, no_provider_keys: None
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _pick_model(monkeypatch, SONNET)

    def ctrl_c(*_args: object, **_kwargs: object) -> str:
        raise click.Abort()

    monkeypatch.setattr(json_crew.click, "prompt", ctrl_c)
    spans: list[tuple[str, ...]] = []

    class _Telemetry:
        def flow_creation_span(self, flow_name: str) -> None:
            spans.append(("flow_creation", flow_name))

        def project_created_span(self, kind: str, project_id: str | None) -> None:
            spans.append(("project_created", kind, str(project_id)))

    monkeypatch.setattr(create_flow_module, "Telemetry", _Telemetry)

    with pytest.raises(click.Abort):
        create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    assert (project_root / ".env").read_text() == f"MODEL={SONNET}\n"
    project_id = get_project_id(project_root / "pyproject.toml")
    assert project_id
    assert spans == [
        ("flow_creation", "ResearchFlow"),
        ("project_created", "flow", project_id),
    ]
    assert (project_root / ".git").is_dir()


def test_create_flow_with_a_keyless_provider_names_no_key(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    monkeypatch.chdir(tmp_path)
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow", provider="ollama/llama3.3")

    env_file = tmp_path / "research_flow" / ".env"
    assert env_file.read_text() == "MODEL=ollama/llama3.3\n"
    _assert_next_steps(_output(capsys), "research_flow", key_line=None)


def test_create_flow_next_steps_skip_a_key_already_exported(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_provider_keys: None,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow")

    _assert_next_steps(_output(capsys), "research_flow", key_line=None)


def test_create_flow_into_an_existing_folder_asks_nothing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _at_a_terminal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "research_flow").mkdir()
    _forbid_prompts(monkeypatch)

    create_flow("Research Flow")

    assert "Error: Folder research_flow already exists." in _output(capsys)
    assert list((tmp_path / "research_flow").iterdir()) == []


def test_create_flow_scaffolds_assistant_instructions(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_md.index("## Working in this project") < agents_md.index("## Reference")
    assert "Never turn off tracing or telemetry" in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()


def test_scaffolded_flow_agents_md_tells_assistants_to_evaluate_with_crewai_eval(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")
    create_flow("Declared Flow", declarative=True)

    python_flow = (tmp_path / "research_flow" / "AGENTS.md").read_text(encoding="utf-8")
    assert "is done when a traced run passes `crewai eval`" in python_flow
    assert "prints a markdown brief of what failed and the change to make" in python_flow
    assert "`crewai eval --models " in python_flow
    # The declarative flow ships its own AGENTS.md.
    declared = (tmp_path / "declared_flow" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Run `crewai eval` after a change" in declared
    assert 'crewai eval --models "provider/model,…"' in declared
    for agents_md in (python_flow, declared):
        assert "{{name}}" not in agents_md
        assert "crewai-eval:" not in agents_md


class _FakeCrewOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


class _FakeCrew:
    def kickoff(self, inputs: dict[str, str]) -> _FakeCrewOutput:
        return _FakeCrewOutput(f"# Post on {inputs['topic']}")


class _FakeContentCrew:
    def crew(self) -> _FakeCrew:
        return _FakeCrew()


@pytest.fixture
def scaffolded_flow_main(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> Iterator[ModuleType]:
    """The `main` module of a freshly scaffolded Python flow, with its crew
    replaced by a fake so no model is called."""
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")
    monkeypatch.syspath_prepend(str(tmp_path / "research_flow" / "src"))
    modules_before = set(sys.modules)
    try:
        main = importlib.import_module("research_flow.main")
        monkeypatch.setattr(main, "ContentCrew", _FakeContentCrew)
        yield main
    finally:
        for name in set(sys.modules) - modules_before:
            if name == "research_flow" or name.startswith("research_flow."):
                del sys.modules[name]


def test_scaffolded_flow_returns_the_post_from_its_last_step(
    scaffolded_flow_main: ModuleType, tmp_path: Path
) -> None:
    result = scaffolded_flow_main.ContentFlow().kickoff()

    assert result == "# Post on AI Agents"
    assert (tmp_path / "output" / "post.md").read_text() == "# Post on AI Agents"


def test_scaffolded_flow_entry_points_return_none_so_uv_scripts_exit_zero(
    scaffolded_flow_main: ModuleType, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # `kickoff` and `run_with_trigger` are console scripts; their wrapper calls
    # sys.exit(func()), so any non-None return fails `crewai run` and
    # `crewai triggers run` after a successful run.
    monkeypatch.setattr(sys, "argv", ["kickoff"])
    assert scaffolded_flow_main.kickoff() is None

    monkeypatch.setattr(sys, "argv", ["run_with_trigger", '{"topic": "Rust"}'])
    assert scaffolded_flow_main.run_with_trigger() is None
    assert (tmp_path / "output" / "post.md").read_text() == "# Post on Rust"


def test_scaffolded_flow_run_with_trigger_rejects_a_missing_payload(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_with_trigger"])

    with pytest.raises(Exception, match="No trigger payload provided"):
        scaffolded_flow_main.run_with_trigger()


def _written_post() -> str:
    return Path("output/post.md").read_text(encoding="utf-8")


def _run_in_project_env(
    tmp_path: Path, monkeypatch: MonkeyPatch, *args: str
) -> Result:
    """`crewai run` from the scaffolded project, as `uv run crewai run` runs it."""
    monkeypatch.chdir(tmp_path / "research_flow")
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    return CliRunner().invoke(crewai, ["run", *args])


def test_crewai_run_inputs_kick_off_the_scaffolded_flow(
    scaffolded_flow_main: ModuleType, tmp_path: Path, monkeypatch: MonkeyPatch
):
    result = _run_in_project_env(
        tmp_path, monkeypatch, "--inputs", '{"topic": "Quantum"}'
    )

    assert result.exit_code == 0, result.output
    assert _written_post() == "# Post on Quantum"


def test_crewai_run_inputs_reach_a_flow_whose_kickoff_ignores_them(
    scaffolded_flow_main: ModuleType, tmp_path: Path, monkeypatch: MonkeyPatch
):
    """A project whose `kickoff()` was written before `--inputs` reached Python
    flows: the flow class is kicked off directly, as AMP does, so the inputs
    still arrive instead of the run silently using defaults."""
    monkeypatch.setattr(
        scaffolded_flow_main,
        "kickoff",
        lambda: scaffolded_flow_main.ContentFlow().kickoff(),
    )

    result = _run_in_project_env(
        tmp_path, monkeypatch, "--inputs", '{"topic": "Quantum"}'
    )

    assert result.exit_code == 0, result.output
    assert _written_post() == "# Post on Quantum"


def test_scaffolded_flow_kickoff_without_inputs_uses_default_topic(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["kickoff"])

    scaffolded_flow_main.kickoff()

    assert _written_post() == "# Post on AI Agents"


def test_scaffolded_flow_start_step_keeps_inputs_passed_to_flow_kickoff(
    scaffolded_flow_main: ModuleType,
):
    # AMP calls `YourFlow().kickoff(inputs=...)` directly, not main.py's kickoff().
    scaffolded_flow_main.ContentFlow().kickoff(inputs={"topic": "Robots"})

    assert _written_post() == "# Post on Robots"


def test_scaffolded_flow_run_with_trigger_uses_payload_topic(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["run_with_trigger", '{"topic": "Rockets"}'])

    scaffolded_flow_main.run_with_trigger()

    assert _written_post() == "# Post on Rockets"


def test_scaffolded_flow_run_with_trigger_without_topic_keeps_default(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["run_with_trigger", '{"source": "slack"}'])

    scaffolded_flow_main.run_with_trigger()

    assert _written_post() == "# Post on AI Agents"


def test_scaffolded_flow_docs_tell_how_to_pass_inputs(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Demo Flow")

    agents_md = (tmp_path / "demo_flow" / "AGENTS.md").read_text(encoding="utf-8")
    readme = (tmp_path / "demo_flow" / "README.md").read_text(encoding="utf-8")
    assert "`crewai run --inputs '{\"topic\":\"...\"}'`" in agents_md
    assert "crewai run --inputs '{\"topic\": \"Quantum Computing\"}'" in readme
    assert "uv run kickoff '" not in agents_md + readme
