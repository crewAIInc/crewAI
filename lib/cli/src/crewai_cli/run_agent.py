from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
from typing import Any

import click

from crewai_cli.api_keys import (
    PLACEHOLDER_API_KEY,
    _missing_keys,
    ask_for_api_key,
    missing_key_line,
)
from crewai_cli.constants import ENV_VARS, required_key_names
from crewai_cli.input_prompt import is_interactive, parse_inputs_json
from crewai_cli.run_crew import (
    _CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV,
    _CREWAI_RUNNER_SOURCE_DIR_ENV,
    _find_local_crewai_source_dir,
    _install_project_dependencies_if_needed,
    _project_run_command,
    _project_runner_code,
    load_project_env,
    record_run_stopped,
    stops_run_as,
)
from crewai_cli.utils import build_env_with_all_tool_credentials


_CREWAI_AGENT_INPUTS_ENV = "CREWAI_AGENT_INPUTS"
_AGENT_RUNNER_CODE = _project_runner_code(
    "run_agent",
    """
try:
    module.run_agent(inputs=os.environ["CREWAI_AGENT_INPUTS"] or None)
except module.click.ClickException as exc:
    exc.show()
    raise SystemExit(exc.exit_code)
except module.click.Abort:
    # Ctrl-C or Ctrl-D at the API key prompt, which counted the stop.
    module.click.echo("Aborted!", err=True)
    raise SystemExit(module._ABORTED_EXIT_CODE)
""",
)
# The project-env run exits with this when it was aborted at a prompt, so the
# launching process does not count it again as a failed script.
_ABORTED_EXIT_CODE = 130


def _agent_messages(inputs: str | None) -> Any:
    provided = parse_inputs_json(inputs) or {}
    if set(provided) - {"messages"}:
        raise click.UsageError("Agent inputs only support 'messages'")
    messages = provided.get("messages")
    if isinstance(messages, str) and messages.strip():
        return messages
    if (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(message, dict)
            and message.get("role") in ("system", "user", "assistant", "tool")
            and "content" in message
            and (
                message.get("content") is None
                or isinstance(message.get("content"), str)
                or (
                    isinstance(message.get("content"), list)
                    and all(isinstance(part, dict) for part in message["content"])
                )
            )
            for message in messages
        )
    ):
        return messages
    raise click.UsageError(
        'Agent projects require --inputs {"messages":"Your request"} '
        "or a non-empty messages history"
    )


def run_agent_in_project_env(inputs: str | None) -> None:
    """Run a standalone Agent using its project environment."""
    with stops_run_as("invalid_inputs"):
        if inputs is not None:
            _agent_messages(inputs)
    if inputs is None and not is_interactive():
        with stops_run_as("no_terminal"):
            raise click.UsageError(
                "Agent chat needs an interactive terminal; use --inputs for a single kickoff"
            )
    with stops_run_as("install_failed"):
        _install_project_dependencies_if_needed()
    command = _project_run_command(_AGENT_RUNNER_CODE)
    env = build_env_with_all_tool_credentials()
    env[_CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(Path(__file__).resolve().parent)
    env[_CREWAI_AGENT_INPUTS_ENV] = inputs or ""
    if local_source := _find_local_crewai_source_dir():
        env[_CREWAI_RUNNER_SOURCE_DIR_ENV] = str(local_source)
    try:
        subprocess.run(command, check=True, env=env)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        if exc.returncode != _ABORTED_EXIT_CODE:
            record_run_stopped("script_failed")
        raise SystemExit(exc.returncode) from exc
    except OSError as exc:
        record_run_stopped("launch_failed")
        raise click.ClickException(f"Could not launch agent: {exc}") from exc


def _ask_for_missing_api_key(agent: Any) -> None:
    """Ask for the API key the Agent's model needs before its first turn.

    Providers build their clients lazily, so a missing key would otherwise
    surface only when the first turn fails.
    """
    llm = getattr(agent, "llm", None)
    if llm is None or getattr(llm, "api_key", None) not in (
        None,
        "",
        PLACEHOLDER_API_KEY,
    ):
        return
    model = f"{getattr(llm, 'provider', '')}/{getattr(llm, 'model', '')}"
    for name in _missing_keys([model], {}, env_file_wins=False):
        # Ctrl-C or Ctrl-D at the prompt stops the run there.
        with stops_run_as("missing_api_key"):
            pasted = ask_for_api_key(
                name,
                Path.cwd(),
                None,
                llm_endpoints={
                    key: getattr(llm, key, None)
                    for key in ("base_url", "api_base", "endpoint")
                },
            )
        if not pasted:
            record_run_stopped("missing_api_key")
            raise click.ClickException(missing_key_line(name, Path.cwd()))


def _drop_blank_api_keys() -> None:
    """Unset provider API keys that are blank.

    A blank key fails while the model is built, before the loaded Agent can be
    checked and the key asked for. Unset, the provider waits for its first call.
    """
    for provider in ENV_VARS:
        for name in required_key_names(provider):
            if name in os.environ and not os.environ[name].strip():
                del os.environ[name]


def run_agent(inputs: str | None = None) -> None:
    """Load and execute a standalone Agent in the current interpreter."""
    if inputs is None and not is_interactive():
        raise click.UsageError(
            "Agent chat needs an interactive terminal; use --inputs for a single kickoff"
        )
    messages: Any = _agent_messages(inputs) if inputs is not None else None
    load_project_env(Path.cwd() / ".env")
    try:
        from crewai.project import load_agent_project

        # After the import: crewai loads the project's .env again as it imports.
        _drop_blank_api_keys()
        agent = load_agent_project()
        _ask_for_missing_api_key(agent)
        if inputs is None:
            agent.chat(tui=True)
            return
        result = asyncio.run(agent.kickoff_async(messages=messages))
    except (click.ClickException, click.Abort):
        raise
    except Exception as exc:
        raise click.ClickException(f"Agent run failed: {exc}") from exc
    click.echo(result.raw)
