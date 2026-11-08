"""Ask for the API keys a project's models need before `crewai run` starts them,
or once a JSON crew's run has failed without one.

`run_crew.py` imports this module, and a JSON crew's runner loads `run_crew.py`
inside the project's environment, whose crewai and crewai_core can be older
than this CLI. So this module imports neither: only the standard library, click
and python-dotenv.
"""

from __future__ import annotations

import ast
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
    ENV_VARS,
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
_YAML_EXPLICIT_KEY = re.compile(r"^[ \t-]*\?(?:[ \t]|$)", re.MULTILINE)
_YAML_SETTING = re.compile(
    r"""(?:^|[\s{,])["']?([A-Za-z_][\w-]*)["']?[ \t]*:"""
    r"""(?:[ \t]*[>|][-+0-9]*[ \t]*(?:\#[^\n]*)?\n)?[ \t]*["']?([^"'\s,{}#]*)""",
    re.MULTILINE,
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
    missing = _missing_keys(models, saved, env_file_wins=json_crew is not None)
    if not missing:
        return None

    if not _can_ask(project_dir, env_file):
        for name in missing:
            click.secho(
                missing_key_line(name, project_dir, json_crew), fg="yellow", err=True
            )
        return "warned"

    endpoints = _project_endpoints(saved, json_crew, project_dir)
    outcome: ApiKeyOutcome = "skipped"
    for name in missing:
        if _ask_and_save(name, env_file, endpoints):
            outcome = "saved"
    return outcome


def ask_for_api_key(
    name: str,
    project_dir: Path,
    json_crew: Path | None,
    *,
    llm_endpoints: Mapping[str, str | None] | None = None,
) -> bool:
    """Ask for ``name``, an API key a run failed without, and save it.

    The key is asked for, exported and saved as
    :func:`ask_for_missing_api_keys` does, with the hosts the project's
    ``.env`` and the JSON crew ``json_crew`` send requests to named first, and
    not asked for where that function would only print the line to add.
    ``llm_endpoints`` are the endpoint settings of a model that is already
    built, such as a standalone agent's, named too unless the machine or the
    ``.env`` already set them.

    Returns:
        Whether a key was pasted.
    """
    env_file = _env_file(project_dir)
    if not _askable(name) or not _can_ask(project_dir, env_file):
        return False
    saved = dotenv_values(env_file) if env_file.exists() else {}
    endpoints = _run_endpoints(saved, json_crew, project_dir)
    known = {*_MACHINE_ENDPOINTS.values(), *saved.values()}
    endpoints += _named_hosts(
        (value, f"{key} of the agent's llm")
        for key, value in (llm_endpoints or {}).items()
        if value and value not in known
    )
    return _ask_and_save(name, env_file, list(dict.fromkeys(endpoints)))


# crewAI's providers name the key they miss `<PROVIDER>_API_KEY`.
_PROVIDER_KEY_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_API_KEY")


def _askable(name: str) -> bool:
    """Whether NAME is a provider API key the CLI may ask for. A failed run
    names its own variable, and project code can raise that error too."""
    if _PROVIDER_KEY_NAME.fullmatch(name):
        return True
    single = (required_key_names(provider) for provider in ENV_VARS)
    return name in {names[0] for names in single if len(names) == 1} or any(
        name in names for names in _ALTERNATIVE_KEYS.values()
    )


def missing_key_line(
    name: str, project_dir: Path, json_crew: Path | None = None
) -> str:
    """The line that tells how to set ``name`` for the project's runs, naming
    the hosts the project's ``.env`` and JSON crew ``json_crew`` send requests to."""
    env_file = _env_file(project_dir)
    try:
        # Not a pipe: what a secrets manager serves through one is the run's.
        saved = dotenv_values(env_file) if env_file.is_file() else {}
    except (OSError, ValueError):
        saved = {}
    return (
        f"{name} is not set. Add {name}=<your key> to {env_file}, "
        "then run `crewai run` again."
        f"{_endpoint_warning(_run_endpoints(saved, json_crew, project_dir))}"
    )


def _run_endpoints(
    saved: Mapping[str, str | None], json_crew: Path | None, project_dir: Path
) -> list[str]:
    """The hosts :func:`_project_endpoints` names, and those the crew's own code
    pointed a provider at through the environment while it loaded."""
    from_env_file = _endpoint_settings(
        {key: value for key, value in saved.items() if value}
    )
    loaded = [
        f"{_host(value)} ({key} set while the crew loaded)".encode(
            "unicode_escape"
        ).decode("ascii")
        for key, value in _endpoint_settings(os.environ).items()
        if value and value not in (_MACHINE_ENDPOINTS.get(key), from_env_file.get(key))
    ]
    return _project_endpoints(saved, json_crew, project_dir) + loaded


def _can_ask(project_dir: Path, env_file: Path) -> bool:
    # Only a regular `.env` of the project's own is rewritten, or a new one
    # made: a new one would hide the `.env` above it, and a rewrite turns a
    # link, or a pipe a secrets manager serves, into a plain file. The
    # preflight never gets here with a pipe; a run that failed without a key can.
    savable = (
        env_file.parent == project_dir
        and not env_file.is_symlink()
        and (env_file.is_file() or not env_file.exists())
    )
    return savable and _can_prompt()


def _ask_and_save(name: str, env_file: Path, endpoints: list[str]) -> bool:
    value = click.prompt(
        _key_prompt(name, endpoints),
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if not value:
        return False
    os.environ[name] = value
    _mark_pasted(name)
    try:
        if not _keep_private(env_file):
            click.secho(
                f"Not saved: {env_file} is readable by other users and could "
                f"not be made private, so only this run uses {name}.",
                fg="yellow",
                err=True,
            )
            return True
        set_key(env_file, name, value, quote_mode="never")
    except OSError as error:
        # The run has the key either way; only the next run will not.
        click.secho(
            f"Couldn't save {name} to {env_file} ({error.strerror or error}), "
            "so only this run uses it.",
            fg="yellow",
            err=True,
        )
        return True
    click.echo(f"Saved {name} to {env_file}")
    return True


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
    # Python the run imports: a crew's or flow's sources, or a JSON crew's
    # custom tools, which load before its agents build their llm.
    code_root = project_dir / ("tools" if json_crew else "src")
    found += [
        (value, f"{name} in {path.relative_to(project_dir)}")
        for path in sorted(code_root.rglob("*.py"))
        for name, value in _code_endpoints(_text(path))
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
        # YAML's explicit `? key` / `: value` form can override what the scan
        # reads, and PyYAML is not a CLI dependency: name the file instead.
        found += [
            ("", f"explicit YAML keys in {path.relative_to(project_dir)}")
            for path in sorted(
                p
                for p in (project_dir / "src").rglob("*")
                if p.suffix in _YAML_SUFFIXES
            )
            if _YAML_EXPLICIT_KEY.search(_text(path))
        ]
    return _named_hosts(found)


def _named_hosts(found: Iterable[tuple[str, str]]) -> list[str]:
    """Each (value, source) as `host (source)`, once each."""
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


def _code_endpoints(source: str) -> Iterator[tuple[str, str]]:
    """Each endpoint Python source sets to a constant string, as (name, value):
    a keyword argument (`LLM(base_url=...)`), a dict entry (`client_params`),
    an assignment, `os.environ[...] = ...` included, or `setdefault`/`putenv`.

    Parsed, not matched, so line breaks, parentheses and annotations do not
    hide one. A computed value is not read: that code runs with the key and
    could send it anywhere regardless.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return  # crewAI cannot import it either
    for node in ast.walk(tree):
        pairs: list[tuple[str | None, ast.expr | None]] = []
        if isinstance(node, ast.keyword):
            pairs.append((node.arg, node.value))
        elif isinstance(node, ast.Dict):
            pairs += [
                (_constant(key), value)
                for key, value in zip(node.keys, node.values, strict=True)
            ]
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            pairs += [(_target_name(target), node.value) for target in targets]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("setdefault", "putenv")
            and len(node.args) >= 2
        ):
            pairs.append((_constant(node.args[0]), node.args[1]))
        for name, value in pairs:
            literal = _constant(value)
            if (
                name
                and literal is not None
                and (_sets_endpoint(name) or name.lower() == "endpoint")
            ):
                yield name, literal


def _constant(node: ast.expr | None) -> str | None:
    """NODE's value when it is a string constant, or an f-string with no field."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(part, ast.Constant) for part in node.values
    ):
        return "".join(
            str(part.value) for part in node.values if isinstance(part, ast.Constant)
        )
    return None


def _target_name(target: ast.expr) -> str | None:
    """The name an assignment sets: `x`, `obj.x`, or `mapping["x"]`."""
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        return _constant(target.slice)
    return None


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
        models.append(_configured_model(saved, env_file_wins=True) or DEFAULT_LLM_MODEL)
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
    configured = _configured_model(saved, env_file_wins=False)
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


def _configured_model(
    saved: Mapping[str, str | None], env_file_wins: bool
) -> str | None:
    """The model `MODEL` and its aliases name for the run. A JSON crew's run
    applies the `.env` over the environment (`run_crew.load_project_env`); a
    Python crew's own `load_dotenv` leaves the environment's value."""

    def configured(name: str) -> str | None:
        first, second = (saved, os.environ) if env_file_wins else (os.environ, saved)
        # A blank value still wins over the other source; crewAI then reads the
        # next name.
        value = first.get(name)
        return value if value is not None else second.get(name)

    return next((value for name in _MODEL_ENV if (value := configured(name))), None)


def _missing_keys(
    models: Iterable[str], saved: Mapping[str, str | None], env_file_wins: bool
) -> list[str]:
    missing: list[str] = []
    for model in models:
        provider = provider_for_model(model)
        names = required_key_names(provider) if provider else []
        if len(names) != 1 or names[0] in missing:
            continue
        name = names[0]
        if not any(
            _is_set(key, saved, env_file_wins)
            for key in (name, *_ALTERNATIVE_KEYS.get(name, ()))
        ):
            missing.append(name)
    return missing


def _is_set(name: str, saved: Mapping[str, str | None], env_file_wins: bool) -> bool:
    """Whether the run will have NAME. A JSON crew's run applies the `.env`
    (`run_crew.load_project_env`): its line wins, unless that line holds no key
    and NAME was pasted at a prompt in this run. A Python crew's own
    `load_dotenv` leaves a value the environment already has."""
    line = saved.get(name)
    pasted = name in os.environ.get(PASTED_KEYS_ENV, "").split(",")
    no_key = line is not None and line.strip() in ("", PLACEHOLDER_API_KEY)
    value: str | None
    if not env_file_wins and name in os.environ:
        value = os.environ[name]
    elif line is None or (pasted and no_key):
        value = os.environ.get(name)
    else:
        value = line
    return value is not None and value.strip() not in ("", PLACEHOLDER_API_KEY)
