"""Ask for the API keys a project's models need before `crewai run` starts them.

`run_crew.py` imports this module, and a JSON crew's runner loads `run_crew.py`
inside the project's environment, whose crewai and crewai_core can be older
than this CLI. So this module imports neither: only the standard library, click
and python-dotenv.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal

import click
from dotenv import dotenv_values, set_key

from crewai_cli.constants import (
    DEFAULT_LLM_MODEL,
    provider_for_model,
    required_key_names,
)


# What `crewai run` did about a missing key: saved one pasted at the prompt,
# skipped every prompt, or, without a terminal, printed how to add it.
ApiKeyOutcome = Literal["saved", "skipped", "warned"]

# crewAI treats this value, which `crewai create flow` used to write, as no key.
_PLACEHOLDER = "YOUR_API_KEY"
# crewAI's Gemini provider takes GOOGLE_API_KEY too, and with
# GOOGLE_CLOUD_PROJECT it can use Vertex AI credentials instead of a key.
_ALTERNATIVE_KEYS: dict[str, tuple[str, ...]] = {
    "GEMINI_API_KEY": ("GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"),
}
# Where crewAI reads the model of an agent that sets none, in its order
# (crewai/utilities/llm_utils.py); DEFAULT_LLM_MODEL when none is set.
_MODEL_ENV = ("MODEL", "MODEL_NAME", "OPENAI_MODEL_NAME")
_JSON_LLM_FIELDS = ("llm", "manager_llm", "function_calling_llm")
_JSONC_STRING_OR_COMMENT = re.compile(
    r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', re.DOTALL
)
_JSONC_STRING_OR_TRAILING_COMMA = re.compile(r'("(?:\\.|[^"\\])*")|,(?=\s*[}\]])')
_YAML_LLM = re.compile(r"""^[ \t]+llm:[ \t]*["']?([^"'#\s]+)""", re.MULTILINE)


def ask_for_missing_api_keys(
    project_dir: Path, json_crew: Path | None
) -> ApiKeyOutcome | None:
    """Ask for each API key the project's models need and nothing sets.

    The models are the model strings of the JSON crew ``json_crew`` and its
    ``agents/`` files or, for a Python crew or flow, ``MODEL`` and the ``llm:``
    values of its ``config/agents.yaml`` files. A key counts as set when the
    environment or the project's ``.env`` holds a value other than the
    ``YOUR_API_KEY`` placeholder. Only providers that take a single key are
    checked.

    At a terminal a pasted key is saved to ``.env`` and exported to this
    process, so the run started next inherits it; elsewhere each missing key is
    named with the line to add. Either way the run goes on: the models found
    here are a guess, and a run this check doubts may still work.

    Returns:
        What happened, or ``None`` when no key was missing.
    """
    env_file = project_dir / ".env"
    saved = dotenv_values(env_file) if env_file.is_file() else {}
    models = (
        _json_crew_models(json_crew)
        if json_crew
        else _python_models(project_dir, saved)
    )
    missing = _missing_keys(models, saved)
    if not missing:
        return None

    if not _can_prompt():
        for name in missing:
            click.secho(
                f"{name} is not set. Add {name}=<your key> to {env_file}, "
                "then run `crewai run` again.",
                fg="yellow",
                err=True,
            )
        return "warned"

    outcome: ApiKeyOutcome = "skipped"
    for name in missing:
        value = click.prompt(
            f"{name} isn't set. Paste it to save it to .env (Enter to skip)",
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if value:
            set_key(env_file, name, value, quote_mode="never")
            os.environ[name] = value
            click.echo(f"Saved {name} to {env_file}")
            outcome = "saved"
    return outcome


def _can_prompt() -> bool:
    # input_prompt imports crewai_core through crewai_cli.utils.
    from crewai_cli.input_prompt import is_interactive

    # stdout too: `crewai eval` sends a run's output to a log when nobody watches.
    return is_interactive() and sys.stdout.isatty()


def _json_crew_models(crew_file: Path) -> list[str]:
    files = [crew_file]
    agents_dir = crew_file.parent / "agents"
    if agents_dir.is_dir():
        files += sorted(
            path for path in agents_dir.iterdir() if path.suffix in (".jsonc", ".json")
        )
    models: list[str] = []
    for path in files:
        try:
            definition = _load_jsonc(path)
        except (OSError, ValueError):
            # The run's own loader reports a file it cannot read.
            continue
        if isinstance(definition, dict):
            models += [
                value
                for field in _JSON_LLM_FIELDS
                if isinstance(value := definition.get(field), str)
            ]
    return models


def _load_jsonc(path: Path) -> object:
    """Parse a ``.jsonc`` file as crewAI does: comments and trailing commas allowed."""
    text = _JSONC_STRING_OR_COMMENT.sub(
        lambda match: match.group(1) or "", path.read_text(encoding="utf-8")
    )
    return json.loads(
        _JSONC_STRING_OR_TRAILING_COMMA.sub(lambda match: match.group(1) or "", text)
    )


def _python_models(project_dir: Path, saved: Mapping[str, str | None]) -> list[str]:
    models = [
        model
        for path in sorted((project_dir / "src").glob("**/config/agents.yaml"))
        for model in _YAML_LLM.findall(
            path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    configured = next(
        (
            value
            for name in _MODEL_ENV
            if (value := os.environ.get(name) or saved.get(name))
        ),
        None,
    )
    if configured:
        models.append(configured)
    # With no model named anywhere, the agents run on crewAI's default.
    return models or [DEFAULT_LLM_MODEL]


def _missing_keys(models: Iterable[str], saved: Mapping[str, str | None]) -> list[str]:
    missing: list[str] = []
    for model in models:
        provider = provider_for_model(model)
        names = required_key_names(provider) if provider else []
        if len(names) != 1 or names[0] in missing:
            continue
        name = names[0]
        if not any(
            _is_set(key, saved) for key in (name, *_ALTERNATIVE_KEYS.get(name, ()))
        ):
            missing.append(name)
    return missing


def _is_set(name: str, saved: Mapping[str, str | None]) -> bool:
    return any(
        value and value.strip() not in ("", _PLACEHOLDER)
        for value in (os.environ.get(name), saved.get(name))
    )
