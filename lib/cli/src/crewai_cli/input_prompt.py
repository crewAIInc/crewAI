"""Shared interactive prompting for runtime inputs (flows and crews).

``crewai run`` asks the user for values that were not provided up front — for a
declarative flow (derived from its state schema) and for a declarative (JSON)
crew (derived from the ``{placeholder}`` references in its agents and tasks).
Both paths go through this module so the experience is identical: the same
header, the same per-field prompt styling, and prompt chrome on stderr so
stdout carries only the run's result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import difflib
import json
import sys
from typing import Any

import click

from crewai_cli.utils import enable_prompt_line_editing, is_dmn_mode_enabled


def parse_inputs_json(inputs: str | None) -> dict[str, Any] | None:
    """Parse a ``--inputs`` JSON object, exiting with a pointed error if invalid."""
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


def closest_name(key: str, candidates: Iterable[str]) -> str | None:
    """Nearest candidate name to a likely typo, if one is close enough."""
    matches = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.7)
    return matches[0] if matches else None


def is_interactive() -> bool:
    """Prompt only in an interactive terminal, never in non-interactive mode."""
    return not is_dmn_mode_enabled() and sys.stdin.isatty()


def prompt_for_inputs(
    names: list[str],
    *,
    title: str,
    subtitle: str,
    describe: Callable[[str], str | None] | None = None,
    coerce: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Prompt for each name and return ``{name: value}``.

    ``describe(name)`` returns an optional hint shown dim above the prompt (used
    by flows to surface a field's schema description). ``coerce(name, raw)``
    converts the typed string to the stored value (used by flows to coerce to
    the field's JSON-schema type); by default the raw string is kept as-is.

    Prompt chrome is written to stderr so stdout carries only the run result.
    """
    enable_prompt_line_editing()
    click.echo(err=True)
    click.secho(f"  {title}", fg="cyan", bold=True, err=True)
    click.secho(f"  {subtitle}", dim=True, err=True)

    collected: dict[str, Any] = {}
    for name in names:
        if describe is not None and (hint := describe(name)):
            click.secho(f"  {hint}", dim=True, err=True)
        raw = click.prompt(
            click.style(f"  {name}", fg="cyan"),
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )
        collected[name] = coerce(name, raw) if coerce is not None else raw
    return collected
