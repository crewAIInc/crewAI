from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


def run_flow_definition(definition: str, inputs: str | None = None) -> None:
    """Run a flow from a Flow Definition YAML/JSON string or file path."""
    try:
        from crewai.flow.flow import Flow
        from crewai.flow.flow_definition import FlowDefinition
    except ImportError as exc:
        click.echo(
            "Running flows from definitions requires the full crewai package.",
            err=True,
        )
        raise SystemExit(1) from exc

    parsed_inputs = _parse_inputs(inputs)
    definition_source = _read_definition_source(definition)

    try:
        flow_definition = _parse_flow_definition(FlowDefinition, definition_source)
        flow = Flow.from_definition(flow_definition)
        result = flow.kickoff(inputs=parsed_inputs)
    except Exception as exc:
        click.echo(
            f"An error occurred while running the flow definition: {exc}", err=True
        )
        raise SystemExit(1) from exc

    click.echo(_format_result(result))


def _parse_inputs(inputs: str | None) -> dict[str, Any] | None:
    if inputs is None:
        return None

    try:
        parsed = json.loads(inputs)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid --inputs JSON: {exc}", err=True)
        raise SystemExit(1) from exc

    if not isinstance(parsed, dict):
        click.echo("Invalid --inputs JSON: expected an object.", err=True)
        raise SystemExit(1)

    return parsed


def _read_definition_source(definition: str) -> str:
    path = Path(definition).expanduser()
    try:
        is_file = path.is_file()
    except OSError as exc:
        if _looks_like_inline_definition(definition):
            return definition
        click.echo(f"Invalid --definition path: {definition} ({exc})", err=True)
        raise SystemExit(1) from exc

    if is_file:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            click.echo(
                f"Unable to read --definition path {path}: {exc}",
                err=True,
            )
            raise SystemExit(1) from exc

    try:
        if path.exists():
            click.echo(
                f"Invalid --definition path: {definition} is not a file.", err=True
            )
            raise SystemExit(1)
    except OSError as exc:
        click.echo(f"Invalid --definition path: {definition} ({exc})", err=True)
        raise SystemExit(1) from exc

    return definition


def _looks_like_inline_definition(definition: str) -> bool:
    stripped = definition.lstrip()
    return "\n" in definition or stripped.startswith(("{", "---")) or ":" in stripped


def _parse_flow_definition(flow_definition_cls: type[Any], source: str) -> Any:
    if _looks_like_json(source):
        return flow_definition_cls.from_json(source)

    return flow_definition_cls.from_yaml(source)


def _looks_like_json(source: str) -> bool:
    stripped = source.lstrip()
    return stripped.startswith("{")


def _format_result(result: Any) -> str:
    raw_result = getattr(result, "raw", result)
    if isinstance(raw_result, str):
        return raw_result

    try:
        return json.dumps(raw_result, default=str)
    except TypeError:
        return str(raw_result)
