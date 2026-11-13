"""Choose and set up tools for a standalone agent project.

Reuses the JSON crew onboarding tool picker and CrewAI Platform integration
setup, writing the result into the agent's JSON or YAML definition.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import click
from crewai_core.platform_apps import PLATFORM_APP_TOOLS
from crewai_core.project import ProjectDefinitionError, resolve_agent_project_entrypoint
from packaging.requirements import InvalidRequirement, Requirement
import tomli

from crewai_cli.create_json_crew import (
    _select_platform_actions,
    _select_tools,
    _setup_platform_auth,
    _success,
)
from crewai_cli.input_prompt import is_interactive
from crewai_cli.utils import load_env_vars, write_env_file


def select_agent_tools() -> list[str]:
    """Open the onboarding tool picker and return the chosen tool references."""
    return _select_platform_actions(_select_tools())


def setup_platform_tools(tools: list[str]) -> str | None:
    """Validate CrewAI Platform apps among ``tools`` and return the AMP token."""
    return _setup_platform_auth([{"tools": tools}])


def add_tools_to_agent_project(
    project_root: Path, tools: list[str], platform_token: str | None = None
) -> list[str]:
    """Append ``tools`` to the agent definition and return the newly added ones.

    Saves ``platform_token`` to the project's ``.env`` and adds the
    ``crewai[tools]`` extra when a selected tool needs it.
    """
    definition = agent_definition_path(project_root)
    data = _read_definition(definition)
    existing = data.get("tools") or []
    if not isinstance(existing, list):
        raise click.ClickException(f"{definition.name}: 'tools' must be a list.")
    added = [tool for tool in dict.fromkeys(tools) if tool not in existing]
    if added:
        data["tools"] = [*existing, *added]
        _write_definition(definition, data)
    if platform_token:
        save_platform_token(project_root, platform_token)
    if any(not tool.startswith("custom:") for tool in added):
        _require_tools_extra(project_root / "pyproject.toml")
    return added


def add_tools(project_root: Path, tools: list[str]) -> None:
    """Add tools to an existing agent project, opening the picker when none given.

    Runs CrewAI Platform setup for newly added applications before the
    definition changes, so a cancelled sign-in leaves it untouched.
    """
    agent_definition_path(project_root)
    if tools:
        validate_tool_references(tools)
    else:
        if not is_interactive():
            raise click.UsageError(
                "Name the tools to add, e.g. `crewai tool add SerperDevTool "
                "platform:gmail`, or run in an interactive terminal to pick them."
            )
        tools = select_agent_tools()

    existing = agent_definition_tools(project_root)
    new_tools = [tool for tool in dict.fromkeys(tools) if tool not in existing]
    if not new_tools:
        _success("No new tools to add", dim=True)
        return
    token = setup_platform_tools(new_tools)
    added = add_tools_to_agent_project(project_root, new_tools, token)
    _success(f"Added {', '.join(added)}")


def setup_tools(project_root: Path) -> None:
    """Validate the agent definition's CrewAI Platform apps and save the token."""
    tools = agent_definition_tools(project_root)
    if not any(tool.startswith("platform:") for tool in tools):
        _success("No CrewAI Platform applications to set up", dim=True)
        return
    token = setup_platform_tools(tools)
    if token:
        save_platform_token(project_root, token)


def agent_definition_tools(project_root: Path) -> list[str]:
    """Return the string tool references in the agent definition."""
    data = _read_definition(agent_definition_path(project_root))
    tools = data.get("tools") or []
    if not isinstance(tools, list):
        raise click.ClickException("The agent definition's 'tools' must be a list.")
    return [tool for tool in tools if isinstance(tool, str)]


def save_platform_token(project_root: Path, token: str) -> None:
    """Persist the Enterprise Action Auth Token in the project's ``.env``."""
    os.environ["CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN"] = token
    env_vars = load_env_vars(project_root)
    env_vars["CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN"] = token
    write_env_file(project_root, env_vars)
    _success("Enterprise Action Auth Token saved to .env")


def validate_tool_references(tools: list[str]) -> None:
    """Reject empty references and unknown CrewAI Platform applications."""
    for tool in tools:
        if not tool.strip():
            raise click.BadParameter("Tool references cannot be empty.")
        if not tool.startswith("platform:"):
            continue
        app, _, action = tool.removeprefix("platform:").partition("/")
        if app not in PLATFORM_APP_TOOLS:
            raise click.BadParameter(f"Unknown CrewAI Platform application '{app}'.")
        if action and action not in {item.slug for item in PLATFORM_APP_TOOLS[app]}:
            raise click.BadParameter(
                f"Unknown action '{action}' for CrewAI Platform application '{app}'."
            )


def agent_definition_path(project_root: Path) -> Path:
    """Return the project's JSON or YAML agent definition."""
    try:
        definition = resolve_agent_project_entrypoint(project_root)
    except ProjectDefinitionError as error:
        raise click.ClickException(
            f"{error} Run this command from a standalone agent project."
        ) from error
    if definition.suffix.lower() == ".py":
        raise click.ClickException(
            f"{definition.name} is a Python agent; add tools to `Agent(tools=[...])` "
            "in code. This command edits JSON and YAML agent definitions."
        )
    return definition


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def _read_definition(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if _is_yaml(path):
        yaml = _import_yaml()
        parse, parse_error = yaml.safe_load, yaml.YAMLError
    else:
        parse, parse_error = json.loads, ValueError
    try:
        data = parse(text)
    except parse_error as error:
        raise click.ClickException(
            f"Cannot parse {path.name}: {error}. Comments are not supported here; "
            "edit its 'tools' list by hand."
        ) from error
    if not isinstance(data, dict):
        raise click.ClickException(f"{path.name} must contain an object.")
    return data


def _write_definition(path: Path, data: dict[str, Any]) -> None:
    if not _is_yaml(path):
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return

    # Reading the definition already checked that PyYAML is installed.
    import yaml

    class _Dumper(yaml.SafeDumper):
        # Indent list items under their key, as the scaffold writes them.
        def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
            return super().increase_indent(flow, False)

    def _str(dumper: Any, value: str) -> Any:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _Dumper.add_representer(str, _str)
    path.write_text(
        yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as error:
        raise click.ClickException(
            "YAML agent definitions require PyYAML. Install it with `uv add pyyaml`."
        ) from error
    return yaml


def _require_tools_extra(pyproject: Path) -> None:
    """Add the ``tools`` extra to the project's ``crewai`` dependency."""
    if not pyproject.is_file():
        return
    text = pyproject.read_text(encoding="utf-8")
    dependencies = tomli.loads(text).get("project", {}).get("dependencies", [])
    for dependency in dependencies:
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement:
            continue
        if requirement.name != "crewai":
            continue
        if "tools" in requirement.extras:
            return
        extras = ",".join(sorted({"tools", *requirement.extras}))
        with_tools = re.sub(
            r"^(\s*crewai)\s*(\[[^\]]*\])?", rf"\1[{extras}]", dependency, count=1
        )
        for quote in ('"', "'"):
            text = text.replace(
                f"{quote}{dependency}{quote}", f"{quote}{with_tools}{quote}", 1
            )
        pyproject.write_text(text, encoding="utf-8")
        click.secho(
            f"  Updated pyproject.toml to {with_tools}; run `crewai install`.",
            fg="cyan",
        )
        return
