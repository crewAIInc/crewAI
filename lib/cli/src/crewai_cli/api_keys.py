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


# What `crewai run` did about a missing key: took one pasted at the prompt,
# skipped every prompt, or printed the line to add.
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
# An llm passed in Python code, whose model this check cannot read.
_CODE_LLM = re.compile(r"\bllm\s*=|\bLLM\(")


def ask_for_missing_api_keys(
    project_dir: Path, json_crew: Path | None
) -> ApiKeyOutcome | None:
    """Ask for each API key the project's models need and nothing sets.

    The models are the model strings of the JSON crew ``json_crew`` and its
    agents, plus the one crewAI picks for an agent that names none. For a
    Python crew or flow they are ``MODEL`` and the ``llm:`` values of its
    ``config/agents.yaml`` files, or crewAI's default model when none is set
    and its code passes no llm. A key counts as set when the environment or the
    ``.env`` crewAI loads (the project's own, or else the nearest one above
    it) holds a value other than the ``YOUR_API_KEY`` placeholder. Only
    providers that take a single key are checked.

    At a terminal a pasted key is saved to the project's ``.env`` and exported
    to this process, so the run started next inherits it. Elsewhere, and when
    saving would replace a link or a pipe or hide a ``.env`` above the
    project, each missing key is named with the line to add. Either way the
    run goes on: the models found here are a guess, and a run this check
    doubts may still work.

    Returns:
        What happened, or ``None`` when no key was missing.
    """
    env_file = _env_file(project_dir)
    try:
        saved = dotenv_values(env_file) if env_file.exists() else {}
    except (OSError, ValueError):
        # crewAI fails on a `.env` it cannot read as well, and says why.
        return None
    models = (
        _json_crew_models(json_crew, saved)
        if json_crew
        else _python_models(project_dir, saved)
    )
    missing = _missing_keys(models, saved)
    if not missing:
        return None

    # Only a regular `.env` of the project's own is rewritten, or a new one
    # made: a new one would hide the `.env` above it, and a rewrite turns a
    # link, or a pipe a secrets manager serves, into a plain file.
    savable = (
        env_file.parent == project_dir
        and not env_file.is_symlink()
        and (env_file.is_file() or not env_file.exists())
    )
    if not savable or not _can_prompt():
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
        if not value:
            continue
        os.environ[name] = value
        outcome = "saved"
        try:
            set_key(env_file, name, value, quote_mode="never")
        except OSError as error:
            # The run has the key either way; only the next run will not.
            click.secho(
                f"Couldn't save {name} to {env_file} ({error.strerror or error}), "
                "so only this run uses it.",
                fg="yellow",
                err=True,
            )
        else:
            click.echo(f"Saved {name} to {env_file}")
    return outcome


def _can_prompt() -> bool:
    # input_prompt imports crewai_core through crewai_cli.utils.
    from crewai_cli.input_prompt import is_interactive

    # stdout too: `crewai eval` sends a run's output to a log when nobody watches.
    return is_interactive() and sys.stdout.isatty()


def _env_file(project_dir: Path) -> Path:
    """The `.env` crewAI loads for the project: the nearest one at or above
    it, as python-dotenv's `find_dotenv` finds it from the project's venv."""
    return next(
        (
            path
            for folder in (project_dir, *project_dir.parents)
            if (path := folder / ".env").exists()
        ),
        project_dir / ".env",
    )


def _json_crew_models(crew_file: Path, saved: Mapping[str, str | None]) -> list[str]:
    crew = _load_jsonc(crew_file)
    listed = crew.get("agents")
    names = [*(listed if isinstance(listed, list) else []), crew.get("manager_agent")]
    agents = [
        _load_jsonc(path)
        for name in names
        if isinstance(name, str) and (path := _agent_file(crew_file.parent, name))
    ]
    models = [
        value
        for definition in (crew, *agents)
        for field in _JSON_LLM_FIELDS
        if isinstance(value := definition.get(field), str)
    ]
    # An agent that names no llm runs on the model crewAI picks for it.
    if any(agent and agent.get("llm") is None for agent in agents):
        models.append(_configured_model(saved) or DEFAULT_LLM_MODEL)
    return models


def _agent_file(crew_dir: Path, name: str) -> Path | None:
    # crewAI's find_json_project_file: `<name>.jsonc`, else `<name>.json`.
    return next(
        (
            path
            for suffix in (".jsonc", ".json")
            if (path := crew_dir / "agents" / f"{name}{suffix}").exists()
        ),
        None,
    )


def _load_jsonc(path: Path) -> dict[str, object]:
    """Parse a ``.jsonc`` file as crewAI does: comments and trailing commas
    allowed. Empty when it cannot be, which the run's own loader reports."""
    try:
        text = _JSONC_STRING_OR_COMMENT.sub(
            lambda match: match.group(1) or "", path.read_text(encoding="utf-8")
        )
        definition = json.loads(
            _JSONC_STRING_OR_TRAILING_COMMA.sub(
                lambda match: match.group(1) or "", text
            )
        )
    except (OSError, ValueError):
        return {}
    return definition if isinstance(definition, dict) else {}


def _python_models(project_dir: Path, saved: Mapping[str, str | None]) -> list[str]:
    source = project_dir / "src"
    models = [
        model
        for path in sorted(source.glob("**/config/agents.yaml"))
        for model in _YAML_LLM.findall(_text(path))
    ]
    configured = _configured_model(saved)
    if configured:
        models.append(configured)
    # With no model named anywhere, agents run on crewAI's default, unless the
    # project's code passes them an llm this check cannot read.
    if not models and not any(
        _CODE_LLM.search(_text(path)) for path in source.glob("**/*.py")
    ):
        models.append(DEFAULT_LLM_MODEL)
    return models


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""  # the run's own loader reports a file it cannot read


def _configured_model(saved: Mapping[str, str | None]) -> str | None:
    return next(
        (
            value
            for name in _MODEL_ENV
            if (value := os.environ.get(name) or saved.get(name))
        ),
        None,
    )


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
