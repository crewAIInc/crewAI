"""Loader utilities for JSON/JSONC agent, crew, task, and tool definitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError


logger = logging.getLogger(__name__)


class JSONProjectError(ValueError):
    """User-facing error raised while loading JSON-first crew projects."""


class JSONProjectValidationError(JSONProjectError):
    """Aggregates validation errors found without executing a JSON project."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


_AGENT_RUNTIME_FIELDS = {
    "id",
    "crew",
    "cache_handler",
    "tools_handler",
    "tools_results",
    "knowledge",
    "knowledge_storage",
    "adapted_agent",
    "agent_knowledge_context",
    "crew_knowledge_context",
    "knowledge_search_query",
    "execution_context",
    "checkpoint_kickoff_event_id",
}

_TASK_RUNTIME_FIELDS = {
    "id",
    "used_tools",
    "tools_errors",
    "delegations",
    "output",
    "processed_by_agents",
    "retry_count",
    "start_time",
    "end_time",
    "checkpoint_original_description",
    "checkpoint_original_expected_output",
}

_CREW_RUNTIME_FIELDS = {
    "id",
    "usage_metrics",
    "task_execution_output_json_files",
    "execution_logs",
    "token_usage",
    "execution_context",
    "checkpoint_inputs",
    "checkpoint_train",
    "checkpoint_kickoff_event_id",
}


JSON_PROJECT_EXTENSIONS = (".jsonc", ".json")


@dataclass(frozen=True)
class JSONAgentDefinition:
    """Parsed JSON agent definition and constructor kwargs."""

    name: str
    path: Path
    definition: dict[str, Any]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class JSONCrewProject:
    """Parsed JSON crew project used by runtime loading and validation."""

    crew_path: Path
    agents_dir: Path
    definition: dict[str, Any]
    agent_names: list[str]
    agents: dict[str, JSONAgentDefinition]
    task_definitions: list[dict[str, Any]]


def find_json_project_file(directory: str | Path, stem: str) -> Path | None:
    """Return ``stem.jsonc`` or ``stem.json``, preferring JSONC."""
    root = Path(directory)
    for ext in JSON_PROJECT_EXTENSIONS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_crew_json_file(project_root: str | Path = ".") -> Path | None:
    """Find the JSON crew definition in a project root."""
    return find_json_project_file(project_root, "crew")


def strip_jsonc_comments(text: str) -> str:
    """Strip JSONC comments and trailing commas while preserving string values."""
    without_comments = _strip_jsonc_comments(text)
    return _strip_trailing_commas(without_comments)


def parse_jsonc(text: str, source: str | Path = "<string>") -> Any:
    """Parse JSON/JSONC text into Python data with path-aware error messages."""
    source_label = str(source)
    try:
        return json.loads(strip_jsonc_comments(text))
    except json.JSONDecodeError as exc:
        raise JSONProjectError(
            f"{source_label}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc


def load_jsonc_file(source: str | Path) -> Any:
    """Load a JSON or JSONC file."""
    path = Path(source)
    return parse_jsonc(path.read_text(encoding="utf-8"), source=path)


def load_agent(source: str | Path) -> Any:
    """Load an existing ``Agent`` from a ``.json`` / ``.jsonc`` definition file."""
    from crewai import Agent

    path = Path(source)
    defn = _expect_object(load_jsonc_file(path), path)
    agent_kwargs = _agent_kwargs_from_definition(defn, path)

    try:
        return Agent(**agent_kwargs)
    except ValidationError as exc:
        raise JSONProjectError(_format_validation_error(path, exc)) from exc
    except Exception as exc:
        raise JSONProjectError(f"{path}: failed to load agent: {exc}") from exc


def validate_crew_project(
    source: str | Path,
    agents_dir: Path | None = None,
) -> JSONCrewProject:
    """Validate JSON crew structure without kicking off the crew."""
    return load_json_crew_project(source, agents_dir=agents_dir, collect_errors=True)


def load_json_crew_project(
    source: str | Path,
    agents_dir: Path | None = None,
    *,
    collect_errors: bool = False,
) -> JSONCrewProject:
    """Parse and structurally validate a JSON crew project.

    When ``collect_errors`` is true, all discoverable structural errors are
    returned as a single ``JSONProjectValidationError`` for deploy validation.
    Runtime loading keeps the previous fail-fast behavior where possible.
    """
    crew_path = Path(source)
    if agents_dir is None:
        agents_dir = crew_path.parent / "agents"

    errors: list[str] = []

    def fail(message: str, exc_type: type[Exception] = JSONProjectError) -> None:
        if collect_errors:
            errors.append(message)
            return
        raise exc_type(message)

    def fail_many(messages: list[str]) -> None:
        if not messages:
            return
        if collect_errors:
            errors.extend(messages)
            return
        raise JSONProjectValidationError(messages)

    try:
        defn = _expect_object(load_jsonc_file(crew_path), crew_path)
    except Exception as exc:
        if collect_errors:
            raise JSONProjectValidationError([str(exc)]) from exc
        raise

    fail_many(
        _field_errors(
            defn,
            _crew_allowed_fields(),
            _CREW_RUNTIME_FIELDS,
            crew_path,
            {"inputs"},
        )
    )

    agent_names = defn.get("agents", [])
    if not isinstance(agent_names, list) or not agent_names:
        fail(f"{crew_path}: 'agents' must be a non-empty list")
        agent_names = []

    agents_dir = Path(agents_dir)
    agent_definitions: dict[str, JSONAgentDefinition] = {}
    for agent_name in agent_names:
        if not isinstance(agent_name, str) or not agent_name:
            fail(f"{crew_path}: each agent reference must be a non-empty string")
            continue
        agent_file = find_json_project_file(agents_dir, agent_name)
        if agent_file is None:
            message = (
                f"Agent definition for '{agent_name}' not found in {agents_dir} "
                f"(tried {agent_name}.jsonc and {agent_name}.json)"
            )
            if collect_errors:
                errors.append(
                    f"{crew_path}: agent '{agent_name}' not found in {agents_dir} "
                    f"(tried {agent_name}.jsonc and {agent_name}.json)"
                )
            else:
                raise FileNotFoundError(message)
            continue
        try:
            agent_defn = _expect_object(load_jsonc_file(agent_file), agent_file)
            agent_kwargs = _agent_kwargs_from_definition(agent_defn, agent_file)
        except Exception as exc:
            if collect_errors:
                errors.append(str(exc))
                continue
            raise
        agent_definitions[agent_name] = JSONAgentDefinition(
            name=agent_name,
            path=agent_file,
            definition=agent_defn,
            kwargs=agent_kwargs,
        )

    task_defs = defn.get("tasks", [])
    if not isinstance(task_defs, list) or not task_defs:
        fail(f"{crew_path}: 'tasks' must be a non-empty list")
        task_defs = []

    known_tasks: set[str] = set()
    known_agents = {name for name in agent_names if isinstance(name, str)}
    for index, task_defn in enumerate(task_defs):
        task_path = f"{crew_path}: tasks[{index}]"
        if not isinstance(task_defn, dict):
            fail(f"{task_path} must be an object")
            continue
        fail_many(
            _field_errors(
                task_defn,
                _task_allowed_fields(),
                _TASK_RUNTIME_FIELDS,
                task_path,
            )
        )
        missing_required = [
            f"{task_path} missing required field '{required}'"
            for required in ("description", "expected_output")
            if required not in task_defn
        ]
        fail_many(missing_required)

        agent_ref = task_defn.get("agent")
        if agent_ref is not None and agent_ref not in known_agents:
            fail(
                f"{task_path} references agent '{agent_ref}' which is not in the crew agents list"
            )

        context_names = task_defn.get("context")
        if context_names is not None:
            if not isinstance(context_names, list):
                fail(f"{task_path} field 'context' must be a list of task names")
            else:
                fail_many(
                    [
                        f"{task_path} has context reference '{ctx_name}' but that task "
                        "has not been defined yet"
                        for ctx_name in context_names
                        if ctx_name not in known_tasks
                    ]
                )

        task_name = task_defn.get("name")
        if isinstance(task_name, str) and task_name:
            known_tasks.add(task_name)

    if errors:
        raise JSONProjectValidationError(errors)

    return JSONCrewProject(
        crew_path=crew_path,
        agents_dir=agents_dir,
        definition=defn,
        agent_names=list(agent_names),
        agents=agent_definitions,
        task_definitions=task_defs,
    )


def _strip_jsonc_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    escape = False

    while i < len(text):
        char = text[i]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        next_char = text[i + 1] if i + 1 < len(text) else ""
        if char == "/" and next_char == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            closed = False
            while i < len(text) - 1:
                if text[i] == "\n":
                    result.append("\n")
                if text[i] == "*" and text[i + 1] == "/":
                    i += 2
                    closed = True
                    break
                i += 1
            if not closed:
                raise JSONProjectError("unterminated block comment in JSONC input")
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    escape = False

    while i < len(text):
        char = text[i]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue

        result.append(char)
        i += 1

    return "".join(result)


def _expect_object(value: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JSONProjectError(f"{source}: expected a JSON object")
    return value


def _agent_kwargs_from_definition(
    defn: dict[str, Any], path: Path | str
) -> dict[str, Any]:
    errors = _field_errors(
        defn,
        _agent_allowed_fields(),
        _AGENT_RUNTIME_FIELDS,
        path,
        {"settings"},
    )
    for required in ("role", "goal", "backstory"):
        if required not in defn:
            errors.append(f"{path}: missing required field '{required}'")

    settings = defn.get("settings", {})
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        errors.append(f"{path}: 'settings' must be an object when provided")
        settings = {}
    else:
        errors.extend(
            _field_errors(
                settings,
                _agent_allowed_fields(),
                _AGENT_RUNTIME_FIELDS,
                f"{path}: settings",
            )
        )

    if errors:
        raise JSONProjectValidationError(errors)

    agent_kwargs = {
        key: value for key, value in defn.items() if key in _agent_allowed_fields()
    }
    agent_kwargs.update(settings)
    _resolve_tool_fields(agent_kwargs)
    return agent_kwargs


def _task_kwargs_from_definition(
    task_defn: dict[str, Any],
    agents_map: dict[str, Any],
    task_name_map: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    errors = _field_errors(
        task_defn,
        _task_allowed_fields(),
        _TASK_RUNTIME_FIELDS,
        source,
    )
    if errors:
        raise JSONProjectValidationError(errors)

    task_kwargs = {
        key: value for key, value in task_defn.items() if key in _task_allowed_fields()
    }

    agent_ref = task_kwargs.get("agent")
    if agent_ref is not None and isinstance(agent_ref, str):
        if agent_ref not in agents_map:
            raise JSONProjectError(
                f"{source} references agent '{agent_ref}' which is not in the crew agents list"
            )
        task_kwargs["agent"] = agents_map[agent_ref]

    context_names = task_kwargs.get("context")
    if context_names:
        context_tasks: list[Any] = []
        for ctx_name in context_names:
            if ctx_name not in task_name_map:
                raise JSONProjectError(
                    f"{source} has context reference '{ctx_name}' but that task "
                    "has not been defined yet"
                )
            context_tasks.append(task_name_map[ctx_name])
        task_kwargs["context"] = context_tasks

    _resolve_tool_fields(task_kwargs)
    return task_kwargs


def _crew_kwargs_from_definition(
    defn: dict[str, Any],
    agents: list[Any],
    tasks: list[Any],
    agents_map: dict[str, Any],
    source: Path | str,
) -> dict[str, Any]:
    errors = _field_errors(
        defn,
        _crew_allowed_fields(),
        _CREW_RUNTIME_FIELDS,
        source,
        {"inputs"},
    )
    if errors:
        raise JSONProjectValidationError(errors)

    crew_kwargs = {
        key: value for key, value in defn.items() if key in _crew_allowed_fields()
    }
    crew_kwargs["agents"] = agents
    crew_kwargs["tasks"] = tasks

    manager_agent = crew_kwargs.get("manager_agent")
    if isinstance(manager_agent, str):
        if manager_agent not in agents_map:
            raise JSONProjectError(
                f"{source}: manager_agent '{manager_agent}' is not in the crew agents list"
            )
        crew_kwargs["manager_agent"] = agents_map[manager_agent]

    return crew_kwargs


def _resolve_tool_fields(kwargs: dict[str, Any]) -> None:
    tools = kwargs.get("tools")
    if tools is not None:
        kwargs["tools"] = _resolve_tools(tools)


def _field_errors(
    data: dict[str, Any],
    allowed_fields: set[str],
    runtime_fields: set[str],
    source: str | Path,
    extra_allowed: set[str] | None = None,
) -> list[str]:
    extra_allowed = extra_allowed or set()
    keys = set(data)
    runtime = sorted(keys & runtime_fields)
    unknown = sorted(keys - allowed_fields - runtime_fields - extra_allowed)

    errors: list[str] = []
    if runtime:
        errors.append(
            f"{source}: runtime-only field(s) are not supported in JSON config: "
            + ", ".join(runtime)
        )
    if unknown:
        errors.append(f"{source}: unsupported field(s): " + ", ".join(unknown))
    return errors


def _agent_allowed_fields() -> set[str]:
    from crewai import Agent

    return set(Agent.model_fields) - _AGENT_RUNTIME_FIELDS


def _task_allowed_fields() -> set[str]:
    from crewai import Task

    return set(Task.model_fields) - _TASK_RUNTIME_FIELDS


def _crew_allowed_fields() -> set[str]:
    from crewai import Crew

    return set(Crew.model_fields) - _CREW_RUNTIME_FIELDS


def _format_validation_error(path: str | Path, exc: ValidationError) -> str:
    return f"{path}: validation failed: {exc}"


def _resolve_tools(tool_defs: list[Any]) -> list[Any]:
    """Resolve tool specs into tool instances or serialized BaseTool dicts.

    Strings keep the existing shorthand behavior. Dicts are passed through so
    ``BaseTool``'s Pydantic validator can hydrate serialized ``tool_type`` data.
    """
    if not isinstance(tool_defs, list):
        raise JSONProjectError("'tools' must be a list")

    tools: list[Any] = []
    for tool_def in tool_defs:
        if isinstance(tool_def, dict):
            tools.append(tool_def)
            continue
        if not isinstance(tool_def, str):
            raise JSONProjectError(
                f"Tool definitions must be strings or objects, got {type(tool_def).__name__}"
            )
        if not tool_def:
            continue
        if tool_def.startswith("custom:"):
            tools.append(_resolve_custom_tool(tool_def[7:]))
            continue
        try:
            tool_cls = _find_tool_class(tool_def)
        except Exception as e:
            raise JSONProjectError(f"Failed to resolve tool '{tool_def}': {e}") from e
        if tool_cls is None:
            raise JSONProjectError(
                f"Unknown tool '{tool_def}'. Tool names must match a class from "
                f"the 'crewai_tools' package (e.g. 'SerperDevTool') or use the "
                f"'custom:<name>' prefix for a tool defined in tools/<name>.py."
            )
        try:
            tools.append(tool_cls())
        except Exception as e:
            raise JSONProjectError(
                f"Failed to initialize tool '{tool_def}': {e}"
            ) from e
    return tools


_tool_class_cache: dict[str, type | None] = {}


def _find_tool_class(name: str) -> type | None:
    """Look up a tool class by name from the ``crewai_tools`` package."""
    if name in _tool_class_cache:
        return _tool_class_cache[name]

    candidates = [name]
    if not name.endswith("Tool"):
        candidates.append(name + "Tool")
    snake_pascal = "".join(word.capitalize() for word in name.split("_")) + "Tool"
    if snake_pascal not in candidates:
        candidates.append(snake_pascal)

    for class_name in candidates:
        cls = _try_import_tool(class_name)
        if cls is not None:
            _tool_class_cache[name] = cls
            return cls

    _tool_class_cache[name] = None
    return None


def _try_import_tool(class_name: str) -> type | None:
    """Attempt to import a single tool class without loading all of crewai_tools."""
    import re as _re

    base = (
        class_name.removesuffix("Tool") if class_name.endswith("Tool") else class_name
    )
    snake = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", base).lower()
    tool_snake = snake + "_tool" if not snake.endswith("_tool") else snake

    module_paths = [
        f"crewai_tools.tools.{tool_snake}.{tool_snake}",
        f"crewai_tools.tools.{tool_snake}",
    ]

    for mod_path in module_paths:
        cls = _import_tool_class(mod_path, class_name)
        if cls is not None:
            return cls

    try:
        import crewai_tools

        return getattr(crewai_tools, class_name, None)
    except ImportError:
        return None


def _import_tool_class(mod_path: str, class_name: str) -> type | None:
    try:
        import importlib

        mod = importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError):
        return None
    return getattr(mod, class_name, None)


def _resolve_custom_tool(tool_name: str) -> Any:
    """Resolve a custom tool from the project's ``tools/`` directory.

    Note: ``custom:<name>`` tools execute ``tools/<name>.py`` as local Python
    code at load time — JSON configs referencing them are no longer pure data.
    Only run JSON crew projects from sources you trust.
    """
    tools_dir = Path.cwd() / "tools"
    tool_file = tools_dir / f"{tool_name}.py"
    if not tool_file.exists():
        raise JSONProjectError(
            f"Custom tool 'custom:{tool_name}' not found: expected {tool_file}. "
            f"Create the file with a BaseTool subclass, or remove the tool from "
            f"your crew JSON."
        )
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"custom_tools.{tool_name}", tool_file
        )
        if spec is None or spec.loader is None:
            raise JSONProjectError(
                f"Could not load custom tool 'custom:{tool_name}' from {tool_file}"
            )
        logger.debug("Executing custom tool module: %s", tool_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from crewai.tools.base_tool import BaseTool

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseTool)
                and attr is not BaseTool
            ):
                # Concrete subclasses supply name/description defaults that
                # BaseTool's signature requires.
                tool_cls: type[Any] = attr
                return tool_cls()
        raise JSONProjectError(
            f"No BaseTool subclass found in {tool_file}. Custom tools must "
            f"define a class inheriting from crewai.tools.BaseTool."
        )
    except JSONProjectError:
        raise
    except Exception as e:
        raise JSONProjectError(
            f"Failed to load custom tool 'custom:{tool_name}' from {tool_file}: {e}"
        ) from e
