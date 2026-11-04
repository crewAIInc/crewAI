"""`crewai run` asks for an API key the project's models need and nothing sets.

At a terminal it saves a pasted key to the project's `.env` and hands it to the
run; elsewhere it prints the line to add. It never stops the run.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import click
from click.testing import CliRunner, Result
import pytest

from crewai_cli.cli import crewai
from crewai_cli.constants import ENV_VARS
import crewai_cli.run_crew as run_crew_module


CREW_PYPROJECT = '[project]\nname = "demo"\n\n[tool.crewai]\ntype = "crew"\n'
JSON_CREW_PYPROJECT = '[project]\nname = "demo"\n\n[tool.crewai]\ntype = "crew"\ndefinition = "crew.jsonc"\n'
FLOW_PYPROJECT = '[project]\nname = "demo"\n\n[tool.crewai]\ntype = "flow"\n'
DECLARATIVE_FLOW_PYPROJECT = '[project]\nname = "demo"\n\n[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n'
PROMPT = "{} isn't set. Paste it to save it to .env (Enter to skip)"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def warning(project: Path, name: str) -> str:
    return (
        f"{name} is not set. Add {name}=<your key> to {project / '.env'}, "
        "then run `crewai run` again."
    )


def crew_project(folder: Path) -> None:
    write(folder / "pyproject.toml", CREW_PYPROJECT)


def flow_project(folder: Path) -> None:
    write(folder / "pyproject.toml", FLOW_PYPROJECT)


def json_crew_project(folder: Path, llm: str = "openai/gpt-5.6-luna") -> None:
    write(folder / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(folder / "crew.jsonc", '{"agents": ["researcher"]}')
    write(folder / "agents" / "researcher.jsonc", f'{{"llm": "{llm}"}}')
    # Installed already, so the run starts its child without installing first.
    write(folder / "uv.lock", "")
    (folder / ".venv").mkdir()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project folder to run in, outside any `uv run`, with no key or model set."""
    folder = tmp_path / "demo"
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    names = {
        str(details["key_name"]).upper()
        for entries in ENV_VARS.values()
        for details in entries
        if "key_name" in details
    }
    names |= {
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "MODEL",
        "MODEL_NAME",
        "OPENAI_MODEL_NAME",
        run_crew_module.RUN_COUNTED_ENV,
    }
    for name in names:
        # Set first, so teardown restores what the machine had even after the
        # command under test exports a key.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return folder


@pytest.fixture
def children(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, str]]]:
    """Every subprocess the run starts, each succeeding, with its environment."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, dict(kwargs.get("env") or {})))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(f"prompted: {args}")

    monkeypatch.setattr(click, "prompt", fail)


def at_a_terminal(
    monkeypatch: pytest.MonkeyPatch, *answers: str
) -> list[tuple[str, dict[str, Any]]]:
    """A person at a terminal who gives ANSWERS, in order; returns the prompts shown.

    Call it in the test itself: pytest swaps `sys.stdout` after fixtures run."""
    prompts: list[tuple[str, dict[str, Any]]] = []
    queued = list(answers)

    def fake_prompt(text: str, **kwargs: Any) -> str:
        prompts.append((text, kwargs))
        return queued.pop(0)

    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(click, "prompt", fake_prompt)
    return prompts


def crewai_run(*args: str) -> Result:
    return CliRunner().invoke(crewai, ["run", *args])


# At a terminal


@pytest.mark.parametrize(
    "setup",
    [crew_project, flow_project, json_crew_project],
    ids=["crew", "flow", "json_crew"],
)
def test_a_pasted_key_is_saved_and_handed_to_the_run(
    project, children, monkeypatch, capsys, setup
) -> None:
    setup(project)
    write(
        project / ".env",
        "# keys for this crew\nOPENAI_API_KEY=YOUR_API_KEY\n\nDEMO_SETTING=from-dotenv\n",
    )
    prompts = at_a_terminal(monkeypatch, "  sk-pasted  ")

    run_crew_module.run_crew()

    [(text, options)] = prompts
    assert text == PROMPT.format("OPENAI_API_KEY")
    assert options == {"hide_input": True, "default": "", "show_default": False}
    assert (project / ".env").read_text() == (
        "# keys for this crew\nOPENAI_API_KEY=sk-pasted\n\nDEMO_SETTING=from-dotenv\n"
    )
    assert f"Saved OPENAI_API_KEY to {project / '.env'}" in capsys.readouterr().out
    [(_, env)] = children
    assert env["OPENAI_API_KEY"] == "sk-pasted"
    # The key alone: the rest of `.env` is still the run's to load.
    assert "DEMO_SETTING" not in env


def test_a_pasted_key_creates_the_env_file(project, children, monkeypatch) -> None:
    crew_project(project)
    at_a_terminal(monkeypatch, "sk-pasted")

    run_crew_module.run_crew()

    assert (project / ".env").read_text() == "OPENAI_API_KEY=sk-pasted\n"
    [(_, env)] = children
    assert env["OPENAI_API_KEY"] == "sk-pasted"


def test_enter_skips_the_key_and_the_run_goes_on(
    project, children, monkeypatch, capsys
) -> None:
    crew_project(project)
    write(project / ".env", "OPENAI_API_KEY=YOUR_API_KEY\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    assert len(prompts) == 1
    assert (project / ".env").read_text() == "OPENAI_API_KEY=YOUR_API_KEY\n"
    assert "Saved" not in capsys.readouterr().out
    [(command, env)] = children
    assert command == ["uv", "run", "run_crew"]
    assert "OPENAI_API_KEY" not in env


def test_each_providers_key_is_asked_for_once(project, children, monkeypatch) -> None:
    json_crew_project(project, llm="anthropic/claude-sonnet-5")
    write(project / "agents" / "writer.jsonc", '{"llm": "anthropic/claude-haiku-4-5"}')
    write(project / "agents" / "editor.jsonc", '{"llm": "gpt-5.6-luna"}')
    prompts = at_a_terminal(monkeypatch, "sk-openai", "")

    run_crew_module.run_crew()

    assert [text for text, _ in prompts] == [
        PROMPT.format("OPENAI_API_KEY"),
        PROMPT.format("ANTHROPIC_API_KEY"),
    ]
    assert (project / ".env").read_text() == "OPENAI_API_KEY=sk-openai\n"
    [(_, env)] = children
    assert env["OPENAI_API_KEY"] == "sk-openai"
    assert "ANTHROPIC_API_KEY" not in env


def test_a_terminal_whose_output_is_captured_gets_the_line_instead(
    project, children, monkeypatch, no_prompt, capsys
) -> None:
    """`crewai eval` sends a run's output to a log when nobody is watching it."""
    crew_project(project)
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    run_crew_module.run_crew()

    assert warning(project, "OPENAI_API_KEY") in capsys.readouterr().err
    assert children


# Without a terminal


def test_without_a_terminal_the_line_to_add_is_printed_and_the_run_goes_on(
    project, children, no_prompt
) -> None:
    crew_project(project)

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert warning(project, "OPENAI_API_KEY") in result.output
    assert not (project / ".env").exists()
    [(command, env)] = children
    assert command == ["uv", "run", "run_crew"]
    assert "OPENAI_API_KEY" not in env


@pytest.mark.parametrize("value", ["", "YOUR_API_KEY"])
@pytest.mark.parametrize("where", ["environment", ".env"])
def test_an_empty_key_or_the_placeholder_counts_as_missing(
    project, children, no_prompt, monkeypatch, value, where
) -> None:
    crew_project(project)
    if where == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", value)
    else:
        write(project / ".env", f"OPENAI_API_KEY={value}\n")

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output


# When nothing is missing, nothing changes


def test_a_key_in_the_environment_asks_nothing(
    project, children, no_prompt, monkeypatch
) -> None:
    crew_project(project)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY" not in result.output
    assert children


def test_a_key_in_the_env_file_asks_nothing_and_stays_unloaded(
    project, children, no_prompt
) -> None:
    crew_project(project)
    write(project / ".env", "OPENAI_API_KEY=sk-dotenv\n")

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY" not in result.output
    [(_, env)] = children
    assert "OPENAI_API_KEY" not in env


def test_a_gemini_key_under_googles_name_counts(
    project, children, no_prompt, monkeypatch
) -> None:
    crew_project(project)
    write(project / ".env", "MODEL=gemini/gemini-3.8-flash\nGOOGLE_API_KEY=g-key\n")

    result = crewai_run()

    assert "GEMINI_API_KEY" not in result.output

    write(project / ".env", "MODEL=gemini/gemini-3.8-flash\n")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")

    result = crewai_run()

    assert "GEMINI_API_KEY" not in result.output


@pytest.mark.parametrize(
    "model",
    [
        "ollama/llama3.3",
        "azure/gpt-4o",
        "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "watson/ibm/granite-3-8b-instruct",
        "my_llm",
        "mystery/model-1",
    ],
)
def test_a_model_without_a_single_known_key_asks_nothing(
    project, children, no_prompt, model
) -> None:
    crew_project(project)
    write(project / ".env", f"MODEL={model}\n")

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output
    assert children


def test_a_re_run_inside_the_project_env_asks_nothing(
    project, children, no_prompt, monkeypatch
) -> None:
    crew_project(project)
    monkeypatch.setenv(run_crew_module.RUN_COUNTED_ENV, "1")

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output


def test_a_declarative_flow_is_not_checked(project, children, no_prompt) -> None:
    write(project / "pyproject.toml", DECLARATIVE_FLOW_PYPROJECT)
    write(project / "flow.yaml", "schema: crewai.flow/v1\n")

    result = crewai_run()

    assert "is not set" not in result.output


# Which models a project uses


def test_a_json_crew_reads_its_agents_and_crew_jsonc_past_comments(
    project, children, no_prompt
) -> None:
    json_crew_project(project)
    write(
        project / "crew.jsonc",
        """{
  // "manager_llm": "gemini/gemini-3.8-flash",
  "agents": ["researcher"],
  "manager_llm": "anthropic/claude-sonnet-5", /* "function_calling_llm": "groq/x" */
}
""",
    )
    write(
        project / "agents" / "researcher.jsonc",
        """{
  "backstory": "Reads https://example.com // and /* this */ is text",
  // Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-6"
  // "llm": "gemini/gemini-3.8-flash",
  "llm": "openai/gpt-5.6-luna",
  /*
  "function_calling_llm": "groq/llama-3.3-70b-versatile",
  */
  "tools": [],
}
""",
    )

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert warning(project, "OPENAI_API_KEY") in result.output
    assert warning(project, "ANTHROPIC_API_KEY") in result.output
    assert "GEMINI_API_KEY" not in result.output
    assert "GROQ_API_KEY" not in result.output


def test_a_json_crew_file_that_does_not_parse_is_left_to_the_run(
    project, children, no_prompt
) -> None:
    json_crew_project(project)
    write(project / "agents" / "broken.jsonc", '{"llm": "anthropic/claude-sonnet-5"')

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert warning(project, "OPENAI_API_KEY") in result.output
    assert "ANTHROPIC_API_KEY" not in result.output
    assert children


def test_a_json_crew_without_any_llm_asks_nothing(project, children, no_prompt) -> None:
    json_crew_project(project)
    write(
        project / "agents" / "researcher.jsonc",
        '{"llm": {"model": "llama3", "provider": "ollama"}}',
    )

    result = crewai_run()

    assert "is not set" not in result.output


def test_a_python_crew_reads_model_from_the_env_file(
    project, children, no_prompt
) -> None:
    crew_project(project)
    write(project / ".env", "MODEL=anthropic/claude-sonnet-5\n")

    result = crewai_run()

    assert warning(project, "ANTHROPIC_API_KEY") in result.output
    assert "OPENAI_API_KEY" not in result.output


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("gpt-5.6-luna", "OPENAI_API_KEY"),
        ("claude-sonnet-5", "ANTHROPIC_API_KEY"),
        ("gemini/gemini-3.8-flash", "GEMINI_API_KEY"),
        ("groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
        ("cerebras/llama3.1-8b", "CEREBRAS_API_KEY"),
    ],
)
def test_each_single_key_provider_names_its_key(
    project, children, no_prompt, model, key
) -> None:
    crew_project(project)
    write(project / ".env", f"MODEL={model}\n")

    result = crewai_run()

    assert result.output.strip() == warning(project, key)


def test_the_environments_model_wins_over_the_env_file(
    project, children, no_prompt, monkeypatch
) -> None:
    crew_project(project)
    write(project / ".env", "MODEL=anthropic/claude-sonnet-5\n")
    monkeypatch.setenv("MODEL", "groq/llama-3.3-70b-versatile")

    result = crewai_run()

    assert warning(project, "GROQ_API_KEY") in result.output
    assert "ANTHROPIC_API_KEY" not in result.output


def test_without_a_model_crewais_default_model_is_checked(
    project, children, no_prompt
) -> None:
    flow_project(project)

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output


def test_model_name_is_read_when_model_is_not_set(project, children, no_prompt) -> None:
    crew_project(project)
    write(project / ".env", "MODEL_NAME=gemini/gemini-3.8-flash\n")

    result = crewai_run()

    assert warning(project, "GEMINI_API_KEY") in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_a_python_crew_reads_the_llm_its_agents_yaml_sets(
    project, children, no_prompt
) -> None:
    flow_project(project)
    write(
        project / "src" / "demo" / "crews" / "poem_crew" / "config" / "agents.yaml",
        """researcher:
  role: >
    Researcher
  llm: groq/llama-3.3-70b-versatile # fast
writer:
  role: Writer
  llm: "anthropic/claude-sonnet-5"
""",
    )

    result = crewai_run()

    assert warning(project, "GROQ_API_KEY") in result.output
    assert warning(project, "ANTHROPIC_API_KEY") in result.output
    # Every agent names its model, so crewAI's default is not in play.
    assert "OPENAI_API_KEY" not in result.output


# Importable where the JSON crew runs


@pytest.fixture
def without_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports as in an environment with neither crewai nor crewai_core, with
    `crewai_cli.api_keys` not imported yet; all of it restored afterwards."""
    import crewai_cli

    for name in list(sys.modules):
        if name.split(".")[0] in ("crewai", "crewai_core"):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai_core", None)
    for name in ("api_keys", "constants"):
        monkeypatch.delitem(sys.modules, f"crewai_cli.{name}", raising=False)
        monkeypatch.setattr(crewai_cli, name, None, raising=False)


def test_the_module_imports_without_crewai_or_crewai_core(without_crewai) -> None:
    with pytest.raises(ImportError, match="crewai_core"):
        importlib.import_module("crewai_core.project")

    api_keys = importlib.import_module("crewai_cli.api_keys")

    assert callable(api_keys.ask_for_missing_api_keys)
