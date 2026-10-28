from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
from typing import Any

import click

from crewai_cli.input_prompt import parse_inputs_json
from crewai_cli.run_crew import (
    _CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV,
    _CREWAI_RUNNER_SOURCE_DIR_ENV,
    _find_local_crewai_source_dir,
    _install_project_dependencies_if_needed,
    _project_run_command,
    _project_runner_code,
    load_project_env,
)
from crewai_cli.utils import build_env_with_all_tool_credentials


_CREWAI_AGENT_INPUTS_ENV = "CREWAI_AGENT_INPUTS"
_AGENT_RUNNER_CODE = _project_runner_code(
    "run_agent",
    """
try:
    module.run_agent(inputs=os.environ["CREWAI_AGENT_INPUTS"])
except module.click.ClickException as exc:
    exc.show()
    raise SystemExit(exc.exit_code)
""",
)


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
    _agent_messages(inputs)
    _install_project_dependencies_if_needed()
    command = _project_run_command(_AGENT_RUNNER_CODE)
    env = build_env_with_all_tool_credentials()
    env[_CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(Path(__file__).resolve().parent)
    env[_CREWAI_AGENT_INPUTS_ENV] = inputs or "{}"
    if local_source := _find_local_crewai_source_dir():
        env[_CREWAI_RUNNER_SOURCE_DIR_ENV] = str(local_source)
    try:
        subprocess.run(command, check=True, env=env)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def run_agent(inputs: str) -> None:
    """Load and execute a standalone Agent in the current interpreter."""
    messages = _agent_messages(inputs)
    load_project_env(Path.cwd() / ".env")
    try:
        from crewai.project import load_agent_project

        agent = load_agent_project()
        result = asyncio.run(agent.kickoff_async(messages=messages))
    except Exception as exc:
        raise click.ClickException(f"Agent run failed: {exc}") from exc
    click.echo(result.raw)
