from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
import sys
from types import ModuleType

from click.testing import CliRunner, Result
import pytest
from pytest import MonkeyPatch
import tomli

from crewai_cli.cli import crewai
from crewai_cli.create_flow import create_flow


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
