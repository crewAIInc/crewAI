"""Ask for the API keys a project's models need before `crewai run` starts them.

`run_crew.py` imports this module, and a JSON crew's runner loads `run_crew.py`
inside the project's environment, whose crewai and crewai_core can be older
than this CLI. So this module imports neither: only the standard library, click
and python-dotenv.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal
from urllib.parse import urlsplit

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
PLACEHOLDER_API_KEY = "YOUR_API_KEY"
# The keys pasted at a prompt in this run, comma-separated, so the run that
# follows keeps them over a `.env` that still holds no key. Only this process
# sets it: `run_crew.load_project_env` never takes it from a `.env`.
PASTED_KEYS_ENV = "CREWAI_PASTED_API_KEYS"
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
# Any `key: value` of a crew's YAML config, at any depth and in any style: a
# block mapping, a flow mapping `{base_url: ...}`, or a block scalar `>-`/`|`
# whose value is on the next line. PyYAML is not a CLI dependency. A value it
# cannot take is still reported, as unreadable, rather than skipped.
_YAML_SUFFIXES = (".yaml", ".yml")
_YAML_SETTING = re.compile(
    r"""(?:^|[\s{,])["']?([A-Za-z_][\w-]*)["']?[ \t]*:"""
    r"""(?:[ \t]*[>|][-+0-9]*[ \t]*(?:\#[^\n]*)?\n)?[ \t]*["']?([^"'\s,{}#]*)""",
    re.MULTILINE,
)
# An endpoint a crew's Python code sets to a string literal: a keyword such as
# `base_url="..."`, a dict entry such as `"base_url": "..."` (`client_params`),
# or an environment variable, `os.environ["OPENAI_BASE_URL"] = "..."` or
# `setdefault`/`putenv("OPENAI_BASE_URL", "...")`. Code that computes the value
# is not read: it runs with the key and could send it anywhere regardless.
_CODE_SETTING = re.compile(
    r"""\b([A-Za-z_]\w*)[ \t]*=(?!=)[ \t]*[rRuU]?["']([^"'\n]*)["']"""
    r"""|["']([A-Za-z_]\w*)["'][ \t]*(?::|\][ \t]*=|,)[ \t]*[rRuU]?["']([^"'\n]*)["']"""
)
# An llm passed in Python code, whose model this check cannot read.
_CODE_LLM = re.compile(r"\bllm\s*=|\bLLM\(")
# Settings that send a model's requests to another host: the provider SDKs'
# `*_BASE_URL`, LiteLLM's `*_API_BASE`, Azure's `*_ENDPOINT`, crewAI's own
# `BASE_URL` and `API_BASE` (crewai/utilities/llm_utils.py), and a proxy.
_ENDPOINT_NAMES = frozenset(
    {"BASE_URL", "API_BASE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
)
_ENDPOINT_SUFFIXES = ("_BASE_URL", "_API_BASE", "_ENDPOINT")


def _sets_endpoint(name: str) -> bool:
    """Whether setting ``name`` sends requests to another host: a base URL,
    an API base, an endpoint or a proxy."""
    upper = name.upper()
    return upper in _ENDPOINT_NAMES or upper.endswith(_ENDPOINT_SUFFIXES)


# Those settings as this process started: the user's own. Every command loads a
# project's `.env` with `run_crew.load_project_env`, and importing `run_crew`
# imports this module first, so none of the `.env` is here yet.
def _endpoint_settings(environ: Mapping[str, str]) -> dict[str, str]:
    """ENVIRON's endpoint and proxy settings, keyed by upper-case name."""
    return {
        name.upper(): value for name, value in environ.items() if _sets_endpoint(name)
    }


_MACHINE_ENDPOINTS = _endpoint_settings(os.environ)


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
    to this process, so the run started next inherits it. When the project's
    ``.env`` or JSON crew sends model requests to another host, through a
    ``*_BASE_URL``, ``*_API_BASE`` or ``*_ENDPOINT`` setting or a proxy that the
    machine's own environment does not set, or a ``base_url``, ``api_base``
    or ``endpoint`` anywhere in the JSON crew's files, the prompt names each
    host first. A key is only saved to a
    ``.env`` its owner alone can read: when one that others can read cannot be
    made private, the key is used for this run only. Elsewhere, and when
    saving would replace a link or hide a ``.env`` above the project, each
    missing key is named with the line to add. A ``.env`` that is not a
    regular file, such as a pipe, is left for the run and not checked. Either
    way the run goes on: the models found here are a guess, and a run this
    check doubts may still work.

    Returns:
        What happened, or ``None`` when no key was missing.
    """
    env_file = _env_file(project_dir)
    if env_file.exists() and not env_file.is_file():
        # A pipe a secrets manager serves `.env` through blocks an open that has
        # no writer, and hands what it serves to one reader, which must be the run.
        return None
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

    # Only the project's own `.env` is rewritten, or a new one made: a new one
    # would hide the `.env` above it, and a rewrite turns a link into a plain file.
    savable = env_file.parent == project_dir and not env_file.is_symlink()
    endpoints = _project_endpoints(saved, json_crew, project_dir)
    if not savable or not _can_prompt():
        for name in missing:
            click.secho(
                f"{name} is not set. Add {name}=<your key> to {env_file}, "
                f"then run `crewai run` again.{_endpoint_warning(endpoints)}",
                fg="yellow",
                err=True,
            )
        return "warned"

    outcome: ApiKeyOutcome = "skipped"
    for name in missing:
        value = click.prompt(
            _key_prompt(name, endpoints),
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if not value:
            continue
        os.environ[name] = value
        _mark_pasted(name)
        outcome = "saved"
        try:
            if not _keep_private(env_file):
                click.secho(
                    f"Not saved: {env_file} is readable by other users and could "
                    f"not be made private, so only this run uses {name}.",
                    fg="yellow",
                    err=True,
                )
                continue
            set_key(env_file, name, value, quote_mode="never")
        except OSError as error:
            # The run has the key either way; only the next run will not.
            click.secho(
                f"Couldn't save {name} to {env_file} ({error.strerror or error}), "
                "so only this run uses it.",
                fg="yellow",
                err=True,
            )
            continue
        click.echo(f"Saved {name} to {env_file}")
    return outcome


def _endpoint_warning(endpoints: list[str]) -> str:
    if not endpoints:
        return ""
    hosts = "that host" if len(endpoints) == 1 else "those hosts"
    return (
        f" This project sends requests to {', '.join(endpoints)}; "
        f"add the key only if you trust {hosts}."
    )


def _mark_pasted(name: str) -> None:
    pasted = set(filter(None, os.environ.get(PASTED_KEYS_ENV, "").split(",")))
    os.environ[PASTED_KEYS_ENV] = ",".join(sorted(pasted | {name}))


def _key_prompt(name: str, endpoints: list[str]) -> str:
    if not endpoints:
        return f"{name} isn't set. Paste it to save it to .env (Enter to skip)"
    hosts = "that host" if len(endpoints) == 1 else "those hosts"
    return (
        f"{name} isn't set. This project sends requests to {', '.join(endpoints)}. "
        f"Paste it to save it to .env only if you trust {hosts} (Enter to skip)"
    )


def _project_endpoints(
    saved: Mapping[str, str | None], json_crew: Path | None, project_dir: Path
) -> list[str]:
    """Where the project's `.env`, JSON crew or agents.yaml send model requests
    that the machine's own environment does not, each as `host (setting)`."""
    found = [
        (value, f"{name} in .env")
        for name, value in saved.items()
        if value
        and _sets_endpoint(name)
        # httpx reads `https_proxy` and `HTTPS_PROXY` alike.
        and value != _MACHINE_ENDPOINTS.get(name.upper())
    ]
    if json_crew:
        found += [
            (value, f"{where} in {path.name}")
            for path, definition in _json_crew_files(json_crew)
            for value, where in _json_endpoints(definition, "")
        ]
    else:
        found += [
            (value, f"{key} in {path.relative_to(project_dir)}")
            # Any YAML: a crew's `agents_config` can name a file other than
            # config/agents.yaml.
            for path in sorted(
                p
                for p in (project_dir / "src").rglob("*")
                if p.suffix in _YAML_SUFFIXES
            )
            for key, value in _YAML_SETTING.findall(_text(path))
            if _sets_endpoint(key) or key.lower() == "endpoint"
        ]
        found += [
            (value, f"{key} in {path.relative_to(project_dir)}")
            for path in sorted((project_dir / "src").glob("**/*.py"))
            for keyword, literal, entry, entry_literal in _CODE_SETTING.findall(
                _text(path)
            )
            for key, value in ((keyword or entry, literal or entry_literal),)
            if _sets_endpoint(key) or key.lower() == "endpoint"
        ]
    # Escaped: a control character the project put in a setting could move the
    # cursor and print over the host.
    return list(
        dict.fromkeys(
            f"{_host(value) if value else 'a host it cannot read'} ({source})".encode(
                "unicode_escape"
            ).decode("ascii")
            for value, source in found
        )
    )


def _json_endpoints(value: object, where: str) -> Iterator[tuple[str, str]]:
    """Each endpoint a JSON crew file sets, at any depth, with its dotted path.

    Every object is read, not only the llm fields: crewAI also builds a client
    from `planning_llm`, `chat_llm`, an agent's `planning_config`, an embedder's
    `config`, and an llm's `client_params`, which override its `base_url`.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            # `endpoint` is how Azure's provider takes its host.
            if isinstance(item, str) and (
                _sets_endpoint(key) or key.lower() == "endpoint"
            ):
                if item:
                    yield item, path
            else:
                yield from _json_endpoints(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _json_endpoints(item, f"{where}.{index}")


def _host(url: str) -> str:
    try:
        # urlsplit reads no host from `host:port` or `host/path`.
        host = urlsplit(url if "//" in url else f"//{url}").hostname
    except ValueError:  # an unclosed `[`, as of an IPv6 address
        host = None
    return host or url


def _keep_private(env_file: Path) -> bool:
    """Make the `.env` a key is about to be written to readable by its owner
    only, and say whether it is.

    One that `crewai create` wrote, or the user made, takes the umask's group
    and world read bits, and python-dotenv keeps a file's mode when it rewrites
    it. One it creates is private already: it moves its own temporary file there.
    """
    if not env_file.exists() or not env_file.stat().st_mode & 0o077:
        return True
    try:
        os.chmod(env_file, 0o600)
    except OSError:
        return False
    return True


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
    crew, *agents = [definition for _, definition in _json_crew_files(crew_file)]
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


def _json_crew_files(crew_file: Path) -> list[tuple[Path, dict[str, object]]]:
    """The JSON crew's file, then each agent file it names, with what they hold."""
    crew = _load_jsonc(crew_file)
    listed = crew.get("agents")
    names = [*(listed if isinstance(listed, list) else []), crew.get("manager_agent")]
    return [
        (crew_file, crew),
        *(
            (path, _load_jsonc(path))
            for name in names
            if isinstance(name, str) and (path := _agent_file(crew_file.parent, name))
        ),
    ]


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
    """Whether the run will have NAME once `run_crew.load_project_env` applies
    the `.env`: its line wins, unless that line holds no key and NAME was
    pasted at a prompt in this run."""
    line = saved.get(name)
    pasted = name in os.environ.get(PASTED_KEYS_ENV, "").split(",")
    if line is None or (pasted and line.strip() in ("", PLACEHOLDER_API_KEY)):
        value = os.environ.get(name)
    else:
        value = line
    return value is not None and value.strip() not in ("", PLACEHOLDER_API_KEY)
