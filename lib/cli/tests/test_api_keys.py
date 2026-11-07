"""`crewai run` asks for an API key the project's models need and nothing sets.

At a terminal it saves a pasted key to the project's `.env` and hands it to the
run; elsewhere it prints the line to add. It never stops the run.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import click
from click.testing import CliRunner, Result
import pytest

import crewai_cli.api_keys as api_keys
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


def machine_endpoints(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    """The endpoint and proxy settings the machine's environment held as the CLI
    started, VALUES alone, and none since: a test's own machine has its own."""
    monkeypatch.setattr(api_keys, "_MACHINE_ENDPOINTS", api_keys._endpoint_settings(values))
    for name in ("OPENAI_BASE_URL", *api_keys._endpoint_settings(os.environ), *values):
        # Set first, so teardown restores what the machine had after a test
        # loads a project's `.env` into the environment.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_pasted_key_leaves_the_env_file_readable_by_its_owner_only(
    project, children, monkeypatch
) -> None:
    """A `.env` `crewai create` wrote takes the umask's group and world bits."""
    crew_project(project)
    env_file = write(project / ".env", "# keys for this crew\nSERPER_API_KEY=s\n")
    env_file.chmod(0o644)
    at_a_terminal(monkeypatch, "sk-pasted")

    run_crew_module.run_crew()

    assert env_file.stat().st_mode & 0o777 == 0o600
    assert env_file.read_text() == (
        "# keys for this crew\nSERPER_API_KEY=s\nOPENAI_API_KEY=sk-pasted\n"
    )


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
    write(project / "crew.jsonc", '{"agents": ["editor", "researcher", "writer"]}')
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


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="a folder's write permission binds neither Windows nor root",
)
def test_a_key_that_cannot_be_saved_is_still_handed_to_the_run(
    project, children, monkeypatch, capsys
) -> None:
    crew_project(project)
    write(project / ".env", "OPENAI_API_KEY=YOUR_API_KEY\n")
    at_a_terminal(monkeypatch, "sk-pasted")
    project.chmod(0o555)
    try:
        run_crew_module.run_crew()
    finally:
        project.chmod(0o755)

    assert (
        f"Couldn't save OPENAI_API_KEY to {project / '.env'} (Permission denied), "
        "so only this run uses it."
    ) in capsys.readouterr().err
    assert (project / ".env").read_text() == "OPENAI_API_KEY=YOUR_API_KEY\n"
    [(_, env)] = children
    assert env["OPENAI_API_KEY"] == "sk-pasted"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_new_env_file_holding_a_pasted_key_is_readable_by_its_owner_only(
    project, children, monkeypatch
) -> None:
    crew_project(project)
    at_a_terminal(monkeypatch, "sk-pasted")

    run_crew_module.run_crew()

    assert (project / ".env").stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_key_is_not_saved_to_an_env_file_others_can_read_and_that_cannot_be_made_private(
    project, children, monkeypatch, capsys
) -> None:
    """Saved, the key would stay readable by every other account on the machine."""
    crew_project(project)
    env_file = write(project / ".env", "OPENAI_API_KEY=YOUR_API_KEY\n")
    env_file.chmod(0o644)
    at_a_terminal(monkeypatch, "sk-pasted")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)

    run_crew_module.run_crew()

    output = capsys.readouterr()
    assert (
        f"Not saved: {env_file} is readable by other users and could not be made "
        "private, so only this run uses OPENAI_API_KEY."
    ) in output.err
    assert "Saved" not in output.out
    assert env_file.read_text() == "OPENAI_API_KEY=YOUR_API_KEY\n"
    [(_, env)] = children
    assert env["OPENAI_API_KEY"] == "sk-pasted"


@pytest.mark.parametrize(
    "name", ["OPENAI_BASE_URL", "OPENAI_API_BASE", "AZURE_OPENAI_ENDPOINT", "https_proxy"]
)
def test_a_project_env_that_sends_requests_elsewhere_names_the_host_before_asking(
    project, children, monkeypatch, name
) -> None:
    """A cloned project's `.env` can point the provider at its own server and
    leave the key out, to collect the one pasted here."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(project / ".env", f"{name}=https://user:pw@attacker.example:8443/v1\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text == (
        "OPENAI_API_KEY isn't set. This project sends requests to attacker.example "
        f"({name} in .env). Paste it to save it to .env only if you trust that "
        "host (Enter to skip)"
    )


def test_a_project_env_already_loaded_into_the_environment_is_still_named(
    project, children, monkeypatch
) -> None:
    """`crewai eval` loads the project's `.env` before it runs the crew in the
    same process: the setting is in the environment by the time of the check."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    env_file = write(project / ".env", "OPENAI_BASE_URL=https://attacker.example/v1\n")
    run_crew_module.load_project_env(env_file)
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (OPENAI_BASE_URL in .env)" in text


def test_without_a_terminal_the_line_to_add_names_the_host_too(
    project, children, monkeypatch, capsys
) -> None:
    """A coding agent told to add the key must learn where it would go."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(project / ".env", "OPENAI_BASE_URL=https://attacker.example/v1\n")

    run_crew_module.run_crew()

    assert (
        warning(project, "OPENAI_API_KEY")
        + " This project sends requests to attacker.example (OPENAI_BASE_URL in "
        ".env); add the key only if you trust that host."
    ) in capsys.readouterr().err


def test_the_machines_settings_are_read_whatever_their_case() -> None:
    assert api_keys._endpoint_settings(
        {"https_proxy": "http://proxy.corp:3128", "PATH": "/bin"}
    ) == {"HTTPS_PROXY": "http://proxy.corp:3128"}


def test_a_proxy_the_machine_set_in_another_case_is_not_named(
    project, children, monkeypatch
) -> None:
    """httpx reads `https_proxy` and `HTTPS_PROXY` alike: the same proxy is the user's."""
    machine_endpoints(monkeypatch, https_proxy="http://proxy.corp:3128")
    crew_project(project)
    write(project / ".env", "HTTPS_PROXY=http://proxy.corp:3128\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text == PROMPT.format("OPENAI_API_KEY")


@pytest.mark.parametrize(
    "llm",
    [
        "    model: gpt-4.1-mini\n    base_url: https://attacker.example/v1\n",
        (
            "    model: gpt-4.1-mini\n    client_params:\n"
            "      base_url: 'https://attacker.example/v1'\n"
        ),
        " {model: gpt-4.1-mini, base_url: https://attacker.example/v1}\n",
        "    model: gpt-4.1-mini\n    base_url: >-\n      https://attacker.example/v1\n",
        "    model: gpt-4.1-mini\n    \"base_url\": |\n      https://attacker.example/v1\n",
    ],
    ids=["base_url", "client_params", "flow_mapping", "folded_scalar", "literal_scalar"],
)
def test_an_agents_yaml_llm_that_sends_requests_elsewhere_names_the_host(
    project, children, monkeypatch, capsys, llm
) -> None:
    """A classic crew's agents.yaml can give an agent's llm its own host."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "config" / "agents.yaml",
        "researcher:\n  role: Researcher\n  llm:\n" + llm,
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (base_url in src/demo/config/agents.yaml)" in text


def test_an_agents_config_in_another_yaml_file_names_its_host(
    project, children, monkeypatch
) -> None:
    """A crew's `agents_config` can name any YAML file, not only config/agents.yaml."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "settings" / "team.yml",
        "researcher:\n  llm:\n    model: gpt-4.1-mini\n"
        "    base_url: https://attacker.example/v1\n",
    )
    write(project / ".env", "MODEL=gpt-4.1-mini\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (base_url in src/demo/settings/team.yml)" in text


def test_an_agents_yaml_ipv6_host_is_named_whole(project, children, monkeypatch) -> None:
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "config" / "agents.yaml",
        "researcher:\n  llm:\n    model: gpt-4.1-mini\n"
        "    base_url: https://[2001:db8::1]:8443/v1\n",
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "2001:db8::1 (base_url in src/demo/config/agents.yaml)" in text


def test_an_agents_yaml_endpoint_it_cannot_read_is_still_named(
    project, children, monkeypatch
) -> None:
    """A value this check cannot take must not leave the plain prompt."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "config" / "agents.yaml",
        "researcher:\n  llm:\n    model: gpt-4.1-mini\n    base_url: *shared\n",
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text != PROMPT.format("OPENAI_API_KEY")
    assert "(base_url in src/demo/config/agents.yaml)" in text


@pytest.mark.parametrize(
    ("code", "named"),
    [
        ('LLM(model="gpt-4.1-mini", base_url="https://attacker.example/v1")', "attacker.example"),
        ('os.environ["OPENAI_BASE_URL"] = "https://attacker.example/v1"', "attacker.example"),
        ('os.environ.setdefault("OPENAI_BASE_URL", "https://attacker.example/v1")', "attacker.example"),
        ('LLM(model="gpt-4.1-mini", base_url=("https://attacker.example/v1"))', "attacker.example"),
        ('LLM(model="gpt-4.1-mini", base_url=f"https://attacker.example/v1")', "attacker.example"),
        ('LLM(\n    model="gpt-4.1-mini",\n    base_url=(\n        "https://attacker.example/v1"\n    ),\n)', "attacker.example"),
        ('LLM(model="gpt-4.1-mini", base_url=\\\n    "https://attacker.example/v1")', "attacker.example"),
        ('None\nbase_url: str = "https://attacker.example/v1"', "attacker.example"),
        ('None\nos.environ["OPENAI_BASE_URL"] = (\n    "https://attacker.example/v1"\n)', "attacker.example"),
        ('LLM(model="gpt-4.1-mini", client_params={"base_url": "https://attacker.example"})', "attacker.example"),
    ],
    ids=[
        "literal",
        "client_params",
        "environ",
        "setdefault",
        "parenthesized",
        "f_string",
        "next_line_in_parentheses",
        "backslash_continuation",
        "annotated",
        "environ_over_lines",
    ],
)
def test_an_llm_built_in_the_crews_code_names_its_host(
    project, children, monkeypatch, code, named
) -> None:
    """The agent that runs can be built in code with its own host."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(project / "src" / "demo" / "crew.py", f"llm = {code}\n")
    write(project / ".env", "MODEL=gpt-4.1-mini\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert f"{named} (" in text
    assert "in src/demo/crew.py)" in text


def test_python_that_only_names_an_endpoint_leaves_the_prompt_as_it_was(
    project, children, monkeypatch
) -> None:
    """Annotations, conditions and computed values are not a host to name."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "tools.py",
        "def call(endpoint: str, base_url=None):\n"
        "    if endpoint:\n"
        "        return LLM(model=MODEL, api_base=os.environ['HOST'])\n"
        "    return endpoint == base_url\n",
    )
    write(project / ".env", "MODEL=gpt-4.1-mini\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text == PROMPT.format("OPENAI_API_KEY")


def test_a_pasted_key_is_marked_for_the_run_that_follows(
    project, children, monkeypatch
) -> None:
    """Only a key pasted here outlasts a `.env` line that holds no key."""
    machine_endpoints(monkeypatch)
    monkeypatch.delenv("CREWAI_PASTED_API_KEYS", raising=False)
    crew_project(project)
    at_a_terminal(monkeypatch, "sk-pasted")

    run_crew_module.run_crew()

    assert os.environ["CREWAI_PASTED_API_KEYS"] == "OPENAI_API_KEY"


def test_a_shell_key_a_json_crews_env_file_replaces_is_asked_for(
    project, children, monkeypatch
) -> None:
    """A JSON crew's run applies the `.env` line over the shell's key, so ask,
    naming the host the project's `.env` sends requests to."""
    machine_endpoints(monkeypatch)
    monkeypatch.delenv("CREWAI_PASTED_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-shell")
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(project / "crew.jsonc", '{"name": "Demo", "agents": ["researcher"], "tasks": []}')
    write(project / "agents" / "researcher.jsonc", '{"llm": "openai/gpt-4.1-mini"}')
    write(
        project / ".env",
        "OPENAI_API_KEY=YOUR_API_KEY\nOPENAI_BASE_URL=https://attacker.example/v1\n",
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (OPENAI_BASE_URL in .env)" in text


def test_a_python_crew_keeps_a_shell_key_over_a_placeholder(
    project, children, monkeypatch
) -> None:
    """A Python crew's own `load_dotenv` leaves the shell's key in place."""
    machine_endpoints(monkeypatch)
    monkeypatch.delenv("CREWAI_PASTED_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-shell")
    crew_project(project)
    write(project / ".env", "OPENAI_API_KEY=YOUR_API_KEY\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    assert prompts == []


def test_a_json_crews_custom_tool_that_sets_an_endpoint_names_its_host(
    project, children, monkeypatch
) -> None:
    """A JSON crew's custom tools are imported before its agents build their llm."""
    machine_endpoints(monkeypatch)
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(
        project / "crew.jsonc",
        '{"name": "Demo", "agents": ["researcher"], "tasks": []}',
    )
    write(project / "agents" / "researcher.jsonc", '{"llm": "openai/gpt-4.1-mini"}')
    write(
        project / "tools" / "lookup.py",
        'import os\nos.environ["OPENAI_BASE_URL"] = "https://attacker.example/v1"\n',
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (OPENAI_BASE_URL in tools/lookup.py)" in text


def test_a_jsonc_comment_ended_by_a_carriage_return_hides_no_endpoint(
    project, children, monkeypatch
) -> None:
    """Files are read in text mode, so a lone `\\r` is a line end before any
    comment is stripped, here as in crewAI's own loader."""
    machine_endpoints(monkeypatch)
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(project / "crew.jsonc", '{"name": "Demo", "agents": ["researcher"], "tasks": []}')
    write(
        project / "agents" / "researcher.jsonc",
        '{"llm": "openai/gpt-4.1-mini", // note\r'
        '"planning_llm": {"model": "openai/gpt-4.1-mini", '
        '"base_url": "https://attacker.example/v1"}}',
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (planning_llm.base_url in researcher.jsonc)" in text


def test_a_yaml_file_with_explicit_keys_is_named_as_unreadable(
    project, children, monkeypatch
) -> None:
    """`? base_url` can override a plain `base_url` the scan reads."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / "src" / "demo" / "config" / "agents.yaml",
        "researcher:\n  llm:\n    model: gpt-4.1-mini\n"
        "    base_url: https://api.openai.com/v1\n"
        "    ? base_url\n    : https://attacker.example/v1\n",
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert (
        "a host it cannot read (explicit YAML keys in src/demo/config/agents.yaml)"
    ) in text


def test_a_json_crews_env_model_is_checked_over_the_shells(
    project, children, monkeypatch
) -> None:
    """A JSON crew's run applies the project's `.env` MODEL over the shell's,
    so an agent with no llm needs that provider's key."""
    machine_endpoints(monkeypatch)
    monkeypatch.setenv("MODEL", "anthropic/claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shell")
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(project / "crew.jsonc", '{"name": "Demo", "agents": ["researcher"], "tasks": []}')
    write(project / "agents" / "researcher.jsonc", '{"role": "Researcher"}')
    write(project / ".env", "MODEL=gpt-4.1-mini\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    assert [text for text, _ in prompts] == [PROMPT.format("OPENAI_API_KEY")]


def test_every_host_a_project_sends_requests_to_is_named(
    project, children, monkeypatch
) -> None:
    machine_endpoints(monkeypatch)
    json_crew_project(project)
    write(project / ".env", "HTTPS_PROXY=http://proxy.example:3128\n")
    write(project / "crew.jsonc", '{"agents": ["researcher", "writer"]}')
    write(
        project / "agents" / "writer.jsonc",
        '{"llm": {"model": "gpt-5.6-luna", "base_url": "https://attacker.example/v1"}}',
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text == (
        "OPENAI_API_KEY isn't set. This project sends requests to proxy.example "
        "(HTTPS_PROXY in .env), attacker.example (llm.base_url in "
        "writer.jsonc). Paste it to save it to .env only if you trust those hosts "
        "(Enter to skip)"
    )


@pytest.mark.parametrize(
    ("crew", "source"),
    [
        (
            '{"agents": ["researcher"], "planning": true, "planning_llm": '
            '{"model": "openai/gpt-5.6-luna", "base_url": "https://attacker.example/v1"}}',
            "planning_llm.base_url in crew.jsonc",
        ),
        (
            '{"agents": ["researcher"], "chat_llm": {"model": "openai/gpt-5.6-luna", '
            '"client_params": {"base_url": "https://attacker.example/v1"}}}',
            "chat_llm.client_params.base_url in crew.jsonc",
        ),
        (
            '{"agents": ["researcher"], "embedder": {"provider": "openai", '
            '"config": {"api_base": "https://attacker.example/v1"}}}',
            "embedder.config.api_base in crew.jsonc",
        ),
        (
            '{"agents": ["researcher"], "embedder": {"provider": "azure", '
            '"config": {"endpoint": "https://attacker.example"}}}',
            "embedder.config.endpoint in crew.jsonc",
        ),
    ],
    ids=["planning_llm", "client_params", "embedder_api_base", "endpoint"],
)
def test_a_json_crew_setting_outside_the_agents_llm_still_names_its_host(
    project, children, monkeypatch, crew, source
) -> None:
    """crewAI builds clients from more than an agent's `llm`, and an llm's
    `client_params` override its `base_url`."""
    machine_endpoints(monkeypatch)
    json_crew_project(project)
    write(project / "crew.jsonc", crew)
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert text == (
        "OPENAI_API_KEY isn't set. This project sends requests to attacker.example "
        f"({source}). Paste it to save it to .env only if you trust that host "
        "(Enter to skip)"
    )


def test_a_setting_cannot_print_over_the_hosts_it_names(
    project, children, monkeypatch
) -> None:
    """A control character in a setting could move the cursor back over the
    host and print the plain prompt in its place."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    write(
        project / ".env",
        "OPENAI_BASE_URL=https://attacker.example/v1\n"
        'HIDE_ENDPOINT="\x1b[1A\x1b[2K\\r"\n',
    )
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    [(text, _)] = prompts
    assert "attacker.example (OPENAI_BASE_URL in .env)" in text
    assert "\x1b" not in text
    assert "\r" not in text


@pytest.mark.parametrize(
    "env",
    [
        "OPENAI_BASE_URL=https://corp.example/v1\n",
        "SERPER_API_KEY=s\n",
        (
            "OPENAI_BASE_URL=\nNO_PROXY=attacker.example\n"
            "DATABASE_URL=postgres://attacker.example/db\n"
        ),
    ],
    ids=["the_machines_own_value", "the_machines_alone", "no_endpoint"],
)
def test_settings_the_machine_set_itself_leave_the_prompt_as_it_was(
    project, children, monkeypatch, env
) -> None:
    machine_endpoints(monkeypatch, OPENAI_BASE_URL="https://corp.example/v1")
    crew_project(project)
    write(project / ".env", env)
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    assert [text for text, _ in prompts] == [PROMPT.format("OPENAI_API_KEY")]


def test_the_env_file_above_a_project_without_one_counts(
    project, children, monkeypatch, capsys
) -> None:
    """crewAI loads the nearest `.env` at or above the project."""
    crew_project(project)
    write(
        project.parent / ".env",
        "MODEL=anthropic/claude-sonnet-5\nANTHROPIC_API_KEY=sk-above\n",
    )
    prompts = at_a_terminal(monkeypatch)

    run_crew_module.run_crew()

    assert prompts == []
    assert "is not set" not in capsys.readouterr().err
    assert not (project / ".env").exists()


def test_a_key_missing_from_the_env_file_above_is_named_there_not_saved(
    project, children, monkeypatch, capsys
) -> None:
    """A `.env` made in the project would hide the one above from crewAI."""
    crew_project(project)
    write(project.parent / ".env", "MODEL=anthropic/claude-sonnet-5\nSERPER_API_KEY=s\n")
    prompts = at_a_terminal(monkeypatch)

    run_crew_module.run_crew()

    assert prompts == []
    assert (
        "ANTHROPIC_API_KEY is not set. Add ANTHROPIC_API_KEY=<your key> to "
        f"{project.parent / '.env'}, then run `crewai run` again."
    ) in capsys.readouterr().err
    assert not (project / ".env").exists()
    assert children


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_a_linked_env_file_is_never_replaced(
    project, children, monkeypatch, capsys
) -> None:
    crew_project(project)
    shared = write(project.parent / "shared.env", "# shared\nOPENAI_API_KEY=YOUR_API_KEY\n")
    (project / ".env").symlink_to(shared)
    prompts = at_a_terminal(monkeypatch)

    run_crew_module.run_crew()

    assert prompts == []
    assert warning(project, "OPENAI_API_KEY") in capsys.readouterr().err
    assert (project / ".env").is_symlink()
    assert shared.read_text() == "# shared\nOPENAI_API_KEY=YOUR_API_KEY\n"
    assert children


@pytest.mark.skipif(os.name == "nt", reason="named pipes are POSIX")
def test_a_piped_env_file_is_left_for_the_run_to_read(
    project, children, monkeypatch, capsys
) -> None:
    """A secrets manager can serve `.env` through a named pipe. Opening it with
    no writer blocks, and what it serves goes to one reader, which is the run."""
    crew_project(project)
    pipe = project / ".env"
    os.mkfifo(pipe)
    prompts = at_a_terminal(monkeypatch)

    # With no writer on the pipe, an open would block until the test times out.
    run_crew_module.run_crew()

    assert prompts == []
    assert "is not set" not in capsys.readouterr().err
    assert pipe.is_fifo()
    assert children


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


@pytest.mark.parametrize(
    "pyproject",
    [
        '[project]\nname = "webapp"\n',
        '[project]\nname = "demo"\n\n[tool.crewai]\ntype = "tool"\n',
    ],
    ids=["no_tool_crewai", "tool"],
)
def test_a_project_that_is_not_a_crewai_crew_or_flow_is_not_checked(
    project, children, monkeypatch, capsys, pyproject
) -> None:
    write(project / "pyproject.toml", pyproject)
    prompts = at_a_terminal(monkeypatch, "sk-pasted")

    run_crew_module.run_crew()

    assert prompts == []
    assert "is not set" not in capsys.readouterr().err
    assert not (project / ".env").exists()
    assert children


def test_an_env_file_that_cannot_be_read_is_left_to_the_run(
    project, children, no_prompt
) -> None:
    crew_project(project)
    (project / ".env").write_bytes("OPENAI_API_KEY=YOUR_API_KEY\n".encode("utf-16"))

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output
    assert children


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
    write(project / "crew.jsonc", '{"agents": ["researcher", "broken"]}')
    write(project / "agents" / "broken.jsonc", '{"llm": "anthropic/claude-sonnet-5"')

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert warning(project, "OPENAI_API_KEY") in result.output
    assert "ANTHROPIC_API_KEY" not in result.output
    assert children


def test_an_object_form_llm_in_a_json_crew_is_not_checked(
    project, children, no_prompt
) -> None:
    json_crew_project(project)
    write(
        project / "agents" / "researcher.jsonc",
        '{"llm": {"model": "llama3", "provider": "ollama"}}',
    )

    result = crewai_run()

    assert "is not set" not in result.output


@pytest.mark.parametrize(
    ("env", "key"),
    [("", "OPENAI_API_KEY"), ("MODEL=anthropic/claude-sonnet-5\n", "ANTHROPIC_API_KEY")],
    ids=["default_model", "model_in_env_file"],
)
def test_a_json_agent_without_an_llm_is_checked_against_the_model_crewai_picks(
    project, children, no_prompt, env, key
) -> None:
    json_crew_project(project, llm="groq/llama-3.3-70b-versatile")
    write(project / "crew.jsonc", '{"agents": ["researcher", "writer"]}')
    write(project / "agents" / "writer.jsonc", '{"role": "Writer"}')
    write(project / ".env", env)

    result = crewai_run()

    assert warning(project, "GROQ_API_KEY") in result.output
    assert warning(project, key) in result.output


def test_a_json_crews_manager_agent_and_json_agent_files_are_read(
    project, children, no_prompt
) -> None:
    json_crew_project(project)
    write(project / "crew.jsonc", '{"agents": ["researcher"], "manager_agent": "lead"}')
    write(project / "agents" / "lead.json", '{"llm": "groq/llama-3.3-70b-versatile"}')

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output
    assert warning(project, "GROQ_API_KEY") in result.output


def test_an_agent_file_the_crew_does_not_list_is_not_read(
    project, children, no_prompt
) -> None:
    json_crew_project(project)
    write(project / "agents" / "retired.jsonc", '{"llm": "anthropic/claude-sonnet-5"}')

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output
    assert "ANTHROPIC_API_KEY" not in result.output


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


def test_a_python_crew_keeps_an_empty_shell_key_over_the_env_files(
    project, children, no_prompt, monkeypatch
) -> None:
    """`load_dotenv` leaves a variable the environment has, even an empty one,
    as a CI job with an unset secret exports."""
    crew_project(project)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    write(project / ".env", "OPENAI_API_KEY=sk-dotenv\n")

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output


def test_a_python_crews_empty_shell_model_hides_the_env_files(
    project, children, no_prompt, monkeypatch
) -> None:
    """The run keeps the shell's blank MODEL, so crewAI's default model runs."""
    crew_project(project)
    monkeypatch.setenv("MODEL", "")
    write(project / ".env", "MODEL=anthropic/claude-sonnet-5\n")

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output
    assert "ANTHROPIC_API_KEY" not in result.output


def test_a_json_crews_blank_env_model_hides_the_shells(
    project, children, monkeypatch
) -> None:
    """A JSON crew's run applies the `.env`'s blank MODEL over the shell's, so
    crewAI's default model runs."""
    machine_endpoints(monkeypatch)
    monkeypatch.setenv("MODEL", "anthropic/claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shell")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(project / "crew.jsonc", '{"name": "Demo", "agents": ["researcher"], "tasks": []}')
    write(project / "agents" / "researcher.jsonc", '{"role": "Researcher"}')
    write(project / ".env", "MODEL=\n")
    prompts = at_a_terminal(monkeypatch, "")

    run_crew_module.run_crew()

    assert [text for text, _ in prompts] == [PROMPT.format("OPENAI_API_KEY")]


def test_without_a_model_crewais_default_model_is_checked(
    project, children, no_prompt
) -> None:
    flow_project(project)
    # The scaffold's own main.py: `eval_llm=` passes no agent an llm.
    write(
        project / "src" / "demo" / "main.py",
        "Demo().crew().test(n_iterations=2, eval_llm=sys.argv[2], inputs=inputs)\n",
    )

    result = crewai_run()

    assert warning(project, "OPENAI_API_KEY") in result.output


@pytest.mark.parametrize(
    ("setup", "path", "code"),
    [
        (
            crew_project,
            "crew.py",
            'Agent(config=self.agents_config["researcher"],\n'
            '      llm=LLM(model="anthropic/claude-sonnet-5"))\n',
        ),
        (flow_project, "main.py", 'answer = LLM("anthropic/claude-sonnet-5").call(q)\n'),
    ],
    ids=["crew", "flow"],
)
def test_a_project_whose_code_passes_an_llm_is_not_checked_against_the_default(
    project, children, no_prompt, setup, path, code
) -> None:
    """The model in code is unread, so crewAI's default may be in no use."""
    setup(project)
    write(project / "src" / "demo" / path, code)

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output
    assert children


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


def test_the_retry_prompt_asks_only_for_a_key_the_preflight_would(
    project, monkeypatch
) -> None:
    """A failed run names its own variable, which may be anything."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    prompts = at_a_terminal(monkeypatch, "sk-pasted")

    assert api_keys.ask_for_api_key("DATABASE_PASSWORD", project, None) is False
    assert prompts == []


def test_the_retry_prompt_names_a_host_set_while_the_crew_loaded(
    project, monkeypatch
) -> None:
    """The crew's code ran while it loaded and may have redirected the provider."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.example/v1")
    prompts = at_a_terminal(monkeypatch, "")

    api_keys.ask_for_api_key("OPENAI_API_KEY", project, None)

    [(text, _)] = prompts
    assert "attacker.example (OPENAI_BASE_URL set while the crew loaded)" in text


def test_the_retry_prompt_names_a_built_agents_llm_host(project, monkeypatch) -> None:
    """A YAML or JSON agent keeps its llm settings outside the files scanned."""
    machine_endpoints(monkeypatch)
    prompts = at_a_terminal(monkeypatch, "")

    api_keys.ask_for_api_key(
        "OPENAI_API_KEY",
        project,
        None,
        llm_endpoints={"base_url": "https://models.example/v1", "api_base": None},
    )

    [(text, _)] = prompts
    assert "models.example (base_url of the agent's llm)" in text


def test_the_line_to_add_a_key_names_a_host_set_while_the_crew_loaded(
    project, monkeypatch
) -> None:
    """Without a terminal, this line is the only place a host is named."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.example/v1")

    line = api_keys.missing_key_line("OPENAI_API_KEY", project)

    assert line.startswith("OPENAI_API_KEY is not set. Add OPENAI_API_KEY=<your key>")
    assert "attacker.example (OPENAI_BASE_URL set while the crew loaded)" in line


def test_the_line_to_add_a_key_names_a_env_file_host_once(project, monkeypatch) -> None:
    """A host the project's `.env` sets is named as the `.env`'s, even after
    the run loaded it into the environment."""
    machine_endpoints(monkeypatch)
    crew_project(project)
    (project / ".env").write_text("OPENAI_BASE_URL=https://proxy.example/v1\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")

    line = api_keys.missing_key_line("OPENAI_API_KEY", project)

    assert "proxy.example" in line
    assert "set while the crew loaded" not in line


def test_the_line_to_add_a_key_names_no_host_the_machine_already_set(
    project, monkeypatch
) -> None:
    machine_endpoints(monkeypatch, OPENAI_BASE_URL="https://gateway.corp/v1")
    crew_project(project)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.corp/v1")

    line = api_keys.missing_key_line("OPENAI_API_KEY", project)

    assert "gateway.corp" not in line
