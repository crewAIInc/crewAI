"""Loader utilities for JSON/JSONC agent, crew, task, and tool definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, cast

from pydantic import BaseModel, ValidationError


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
PYTHON_REF_KEY = "python"

_AGENT_TYPE_ALIASES = {
    "agent",
    "Agent",
    "crewai.Agent",
    "crewai.agent.core.Agent",
}
_TASK_TYPE_ALIASES = {
    "task",
    "Task",
    "crewai.Task",
    "crewai.task.Task",
}
_CONDITIONAL_TASK_TYPE_ALIASES = {
    "conditional",
    "conditional_task",
    "ConditionalTask",
    "crewai.tasks.conditional_task.ConditionalTask",
}
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

_AGENT_CALLABLE_FIELDS = {"guardrail", "step_callback"}
_AGENT_CALLABLE_LIST_FIELDS = {"callbacks"}
_TASK_CALLABLE_FIELDS = {"callback", "condition", "guardrail"}
_TASK_CALLABLE_LIST_FIELDS = {"guardrails"}
_TASK_MODEL_CLASS_FIELDS = {"output_json", "output_pydantic", "response_model"}
_CREW_CALLABLE_FIELDS = {"step_callback", "task_callback"}
_CREW_CALLABLE_LIST_FIELDS = {"before_kickoff_callbacks", "after_kickoff_callbacks"}
_AGENT_OBJECT_REF_FIELDS = {
    "agent_executor",
    "checkpoint",
    "embedder",
    "function_calling_llm",
    "i18n",
    "knowledge",
    "knowledge_config",
    "knowledge_sources",
    "knowledge_storage",
    "llm",
    "memory",
    "planning_config",
    "security_config",
    "skills",
}
_TASK_OBJECT_REF_FIELDS = {"security_config"}
_CREW_OBJECT_REF_FIELDS = {
    "chat_llm",
    "checkpoint",
    "embedder",
    "function_calling_llm",
    "knowledge",
    "knowledge_sources",
    "manager_agent",
    "manager_llm",
    "memory",
    "planning_llm",
    "security_config",
    "skills",
}


@dataclass(frozen=True)
class JSONAgentDefinition:
    """Parsed JSON agent definition and constructor kwargs."""

    name: str
    path: Path
    definition: dict[str, Any]
    kwargs: dict[str, Any]
    agent_class: type[Any]


@dataclass(frozen=True)
class JSONCrewProject:
    """Parsed JSON crew project used by runtime loading and validation."""

    crew_path: Path
    agents_dir: Path
    definition: dict[str, Any]
    agent_names: list[str]
    agents: dict[str, JSONAgentDefinition]
    task_definitions: list[dict[str, Any]]


_AgentDefinitionSource = tuple[dict[str, Any], str | Path]
_AgentDefinitionLoader = Callable[[str], _AgentDefinitionSource | None]


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
    path = Path(source)
    defn = _expect_object(load_jsonc_file(path), path)
    root = path.parent.parent if path.parent.name == "agents" else Path.cwd()
    agent_class = _agent_class_from_definition(defn, f"{path}: type")
    agent_kwargs = _agent_kwargs_from_definition(
        defn,
        path,
        agent_class=agent_class,
        project_root=root,
    )

    try:
        return agent_class(**agent_kwargs)
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
    """Load and structurally validate a JSON crew project from files.

    When ``collect_errors`` is true, all discoverable structural errors are
    returned as a single ``JSONProjectValidationError`` for deploy validation.
    Runtime loading keeps the previous fail-fast behavior where possible.
    """
    crew_path = Path(source)
    if agents_dir is None:
        agents_dir = crew_path.parent / "agents"
    agents_dir = Path(agents_dir)

    def load_agent_definition_source(agent_name: str) -> _AgentDefinitionSource | None:
        agent_file = find_json_project_file(agents_dir, agent_name)
        if agent_file is None:
            return None
        return _expect_object(load_jsonc_file(agent_file), agent_file), agent_file

    try:
        defn = _expect_object(load_jsonc_file(crew_path), crew_path)
    except Exception as exc:
        if collect_errors:
            raise JSONProjectValidationError([str(exc)]) from exc
        raise

    return _load_json_crew_project_definition(
        defn,
        source=crew_path,
        agents_dir=agents_dir,
        project_root=crew_path.parent,
        load_agent_definition_source=load_agent_definition_source,
        missing_agent_hint=(
            f"not found in {agents_dir} "
            f"(tried {{agent_name}}.jsonc and {{agent_name}}.json)"
        ),
        collect_errors=collect_errors,
    )


def _load_json_crew_project_definition(
    defn: dict[str, Any],
    *,
    source: str | Path,
    agents_dir: str | Path,
    project_root: Path,
    load_agent_definition_source: _AgentDefinitionLoader,
    missing_agent_hint: str | None,
    collect_errors: bool,
) -> JSONCrewProject:
    """Structurally validate a parsed JSON crew project definition."""
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

    fail_many(
        _field_errors(
            defn,
            _crew_allowed_fields(),
            _CREW_RUNTIME_FIELDS,
            source,
            {"inputs"},
        )
    )
    fail_many(_python_reference_definition_errors(defn, source))

    agent_names = defn.get("agents", [])
    if not isinstance(agent_names, list) or not agent_names:
        fail(f"{source}: 'agents' must be a non-empty list")
        agent_names = []

    agent_definitions: dict[str, JSONAgentDefinition] = {}

    def load_agent_definition(agent_name: str) -> None:
        if not isinstance(agent_name, str) or not agent_name:
            fail(f"{source}: each agent reference must be a non-empty string")
            return
        if agent_name in agent_definitions:
            return
        try:
            loaded_agent = load_agent_definition_source(agent_name)
            if loaded_agent is None:
                hint = (
                    missing_agent_hint.format(agent_name=agent_name)
                    if missing_agent_hint is not None
                    else "not found in provided agent definitions"
                )
                message = f"Agent definition for '{agent_name}' {hint}"
                if collect_errors:
                    errors.append(f"{source}: agent '{agent_name}' {hint}")
                else:
                    raise FileNotFoundError(message)
                return
            agent_defn, agent_source = loaded_agent
            agent_class = _agent_class_from_definition(
                agent_defn,
                f"{agent_source}: type",
                resolve_python_refs=not collect_errors,
            )
            agent_kwargs = _agent_kwargs_from_definition(
                agent_defn,
                agent_source,
                agent_class=agent_class,
                # Validation must never execute project code (custom tools).
                resolve_tools=not collect_errors,
                resolve_python_refs=not collect_errors,
                project_root=project_root,
            )
        except Exception as exc:
            if collect_errors:
                errors.append(str(exc))
                return
            raise
        agent_definitions[agent_name] = JSONAgentDefinition(
            name=agent_name,
            path=Path(str(agent_source)),
            definition=agent_defn,
            kwargs=agent_kwargs,
            agent_class=agent_class,
        )

    for agent_name in agent_names:
        load_agent_definition(agent_name)

    manager_agent = defn.get("manager_agent")
    if manager_agent is not None:
        if isinstance(manager_agent, str) and manager_agent:
            load_agent_definition(manager_agent)
        elif _is_python_ref(manager_agent):
            pass
        else:
            fail(
                f"{source}: 'manager_agent' must be an agent definition name "
                f'or a {{"{PYTHON_REF_KEY}": "module.agent"}} reference'
            )

    known_agents = set(agent_definitions)

    task_defs = defn.get("tasks", [])
    if not isinstance(task_defs, list) or not task_defs:
        fail(f"{source}: 'tasks' must be a non-empty list")
        task_defs = []

    known_tasks: set[str] = set()
    for index, task_defn in enumerate(task_defs):
        task_path = f"{source}: tasks[{index}]"
        if not isinstance(task_defn, dict):
            fail(f"{task_path} must be an object")
            continue
        fail_many(
            _task_definition_errors(
                task_defn,
                task_path,
                resolve_python_refs=not collect_errors,
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
                f"{task_path} references agent '{agent_ref}' which does not match "
                "a loaded agent definition"
            )

        fail_many(
            _tool_definition_errors(task_defn.get("tools"), task_path, project_root)
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
        crew_path=Path(str(source)),
        agents_dir=Path(str(agents_dir)),
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


def _is_python_ref(value: Any) -> bool:
    return isinstance(value, dict) and PYTHON_REF_KEY in value


def _python_ref_errors(value: Any, source: str | Path) -> list[str]:
    if not isinstance(value, dict):
        return [
            f"{source}: Python reference must be an object like "
            f'{{"{PYTHON_REF_KEY}": "module.attribute"}}'
        ]
    if set(value) != {PYTHON_REF_KEY}:
        return [
            f"{source}: Python reference objects must only contain '{PYTHON_REF_KEY}'"
        ]
    path = value.get(PYTHON_REF_KEY)
    if not isinstance(path, str) or not path.strip():
        return [f"{source}: Python reference '{PYTHON_REF_KEY}' must be a string"]
    if "." not in path:
        return [
            f"{source}: Python reference '{path}' must be a dotted import path "
            "like 'module.attribute'"
        ]
    return []


def _python_ref_path(value: Any, source: str | Path) -> str:
    errors = _python_ref_errors(value, source)
    if errors:
        raise JSONProjectValidationError(errors)
    path = cast(str, value[PYTHON_REF_KEY])
    return path.strip()


def _resolve_python_ref(
    value: Any,
    source: str | Path,
    *,
    expected: str,
) -> Any:
    from crewai.utilities.import_utils import import_and_validate_definition

    path = _python_ref_path(value, source)
    try:
        resolved = import_and_validate_definition(path)
    except Exception as exc:
        raise JSONProjectError(f"{source}: failed to import '{path}': {exc}") from exc

    if expected == "any":
        return resolved
    if expected == "callable" and not callable(resolved):
        raise JSONProjectError(f"{source}: Python reference '{path}' is not callable")
    if expected == "class" and not isinstance(resolved, type):
        raise JSONProjectError(f"{source}: Python reference '{path}' is not a class")
    return resolved


def _resolve_python_class(
    value: Any,
    source: str | Path,
    *,
    base_class: type[Any] | None = None,
) -> type[Any]:
    cls = cast(type[Any], _resolve_python_ref(value, source, expected="class"))
    if base_class is not None and not issubclass(cls, base_class):
        raise JSONProjectError(
            f"{source}: Python reference '{_python_ref_path(value, source)}' "
            f"must be a subclass of {base_class.__module__}.{base_class.__name__}"
        )
    return cls


def _agent_class_from_definition(
    defn: dict[str, Any],
    source: str | Path,
    *,
    resolve_python_refs: bool = True,
) -> type[Any]:
    from crewai import Agent

    agent_class = cast(type[Any], Agent)
    type_value = defn.get("type")
    if type_value is None:
        return agent_class
    if isinstance(type_value, str) and type_value in _AGENT_TYPE_ALIASES:
        return agent_class
    if _is_python_ref(type_value):
        if not resolve_python_refs:
            errors = _python_ref_errors(type_value, source)
            if errors:
                raise JSONProjectValidationError(errors)
            return agent_class
        from crewai.agents.agent_builder.base_agent import BaseAgent

        return _resolve_python_class(type_value, source, base_class=BaseAgent)
    if isinstance(type_value, str):
        raise JSONProjectError(
            f"{source}: unsupported agent type '{type_value}'. Use 'Agent' or "
            f'{{"{PYTHON_REF_KEY}": "module.CustomAgent"}}.'
        )
    raise JSONProjectValidationError(_python_ref_errors(type_value, source))


def _task_class_from_definition(
    defn: dict[str, Any],
    source: str | Path,
    *,
    resolve_python_refs: bool = True,
) -> type[Any]:
    from crewai import Task

    task_class = cast(type[Any], Task)
    type_value = defn.get("type")
    if type_value is None:
        return task_class
    if isinstance(type_value, str) and type_value in _TASK_TYPE_ALIASES:
        return task_class
    if isinstance(type_value, str) and type_value in _CONDITIONAL_TASK_TYPE_ALIASES:
        from crewai.tasks.conditional_task import ConditionalTask

        return cast(type[Any], ConditionalTask)
    if _is_python_ref(type_value):
        if not resolve_python_refs:
            errors = _python_ref_errors(type_value, source)
            if errors:
                raise JSONProjectValidationError(errors)
            return task_class
        return _resolve_python_class(type_value, source, base_class=task_class)
    if isinstance(type_value, str):
        raise JSONProjectError(
            f"{source}: unsupported task type '{type_value}'. Use 'Task', "
            f"'ConditionalTask', or "
            f'{{"{PYTHON_REF_KEY}": "module.CustomTask"}}.'
        )
    raise JSONProjectValidationError(_python_ref_errors(type_value, source))


def _model_fields_for(model_cls: type[Any], source: str | Path) -> set[str]:
    fields = getattr(model_cls, "model_fields", None)
    if not isinstance(fields, dict):
        raise JSONProjectError(
            f"{source}: {model_cls.__module__}.{model_cls.__name__} must be a "
            "Pydantic model class"
        )
    return set(fields)


def _definition_has_python_type(defn: dict[str, Any]) -> bool:
    return _is_python_ref(defn.get("type"))


def _agent_kwargs_from_definition(
    defn: dict[str, Any],
    path: Path | str,
    *,
    agent_class: type[Any] | None = None,
    resolve_tools: bool = True,
    resolve_python_refs: bool = True,
    project_root: Path | None = None,
) -> dict[str, Any]:
    agent_class = agent_class or _agent_class_from_definition(
        defn,
        f"{path}: type",
        resolve_python_refs=resolve_python_refs,
    )
    allowed_fields = _agent_allowed_fields(agent_class)
    extra_allowed = {"settings", "type"}
    skip_unknown = _definition_has_python_type(defn) and not resolve_python_refs
    errors = _field_errors(
        defn,
        allowed_fields,
        _AGENT_RUNTIME_FIELDS,
        path,
        extra_allowed,
        skip_unknown=skip_unknown,
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
                allowed_fields,
                _AGENT_RUNTIME_FIELDS,
                f"{path}: settings",
                skip_unknown=skip_unknown,
            )
        )
    errors.extend(_python_reference_definition_errors(defn, path))
    if isinstance(settings, dict):
        errors.extend(
            _python_reference_definition_errors(settings, f"{path}: settings")
        )

    if errors:
        raise JSONProjectValidationError(errors)

    agent_kwargs = {key: value for key, value in defn.items() if key in allowed_fields}
    agent_kwargs.update(settings)
    if resolve_tools:
        _resolve_tool_fields(agent_kwargs, project_root=project_root)
        _resolve_agent_python_refs(agent_kwargs, path)
    else:
        # Validation/deploy mode: check tool declarations structurally without
        # importing or instantiating anything — custom:<name> tools execute
        # project Python on resolution, which must not happen here.
        tool_errors = _tool_definition_errors(
            agent_kwargs.get("tools"), path, project_root
        )
        if tool_errors:
            raise JSONProjectValidationError(tool_errors)
    return agent_kwargs


def _task_kwargs_from_definition(
    task_defn: dict[str, Any],
    agents_map: dict[str, Any],
    task_name_map: dict[str, Any],
    source: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    task_class = _task_class_from_definition(task_defn, f"{source}: type")
    allowed_fields = _task_allowed_fields(task_class)
    errors = _field_errors(
        task_defn,
        allowed_fields,
        _TASK_RUNTIME_FIELDS,
        source,
        {"type"},
    )
    if errors:
        raise JSONProjectValidationError(errors)

    task_kwargs = {
        key: value for key, value in task_defn.items() if key in allowed_fields
    }

    agent_ref = task_kwargs.get("agent")
    if agent_ref is not None and isinstance(agent_ref, str):
        if agent_ref not in agents_map:
            raise JSONProjectError(
                f"{source} references agent '{agent_ref}' which does not match "
                "a loaded agent definition"
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

    _resolve_tool_fields(task_kwargs, project_root=project_root)
    _resolve_task_python_refs(task_kwargs, source)
    if "input_files" in task_kwargs:
        task_kwargs["input_files"] = _normalize_input_files(
            task_kwargs["input_files"],
            source,
            project_root,
        )
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
                f"{source}: manager_agent '{manager_agent}' does not match an agent definition"
            )
        crew_kwargs["manager_agent"] = agents_map[manager_agent]

    _resolve_crew_python_refs(crew_kwargs, source)

    return crew_kwargs


def _resolve_tool_fields(
    kwargs: dict[str, Any], project_root: Path | None = None
) -> None:
    tools = kwargs.get("tools")
    if tools is not None:
        kwargs["tools"] = _resolve_tools(tools, project_root=project_root)
    if "mcps" in kwargs:
        kwargs["mcps"] = _resolve_mcp_python_refs(kwargs["mcps"])


def _field_errors(
    data: dict[str, Any],
    allowed_fields: set[str],
    runtime_fields: set[str],
    source: str | Path,
    extra_allowed: set[str] | None = None,
    *,
    skip_unknown: bool = False,
) -> list[str]:
    extra_allowed = extra_allowed or set()
    keys = set(data)
    runtime = sorted(keys & runtime_fields)
    unknown = (
        []
        if skip_unknown
        else sorted(keys - allowed_fields - runtime_fields - extra_allowed)
    )

    errors: list[str] = []
    if runtime:
        errors.append(
            f"{source}: runtime-only field(s) are not supported in JSON config: "
            + ", ".join(runtime)
        )
    if unknown:
        errors.append(f"{source}: unsupported field(s): " + ", ".join(unknown))
    return errors


def _agent_allowed_fields(agent_class: type[Any] | None = None) -> set[str]:
    from crewai import Agent

    return _model_fields_for(agent_class or Agent, "agent type") - _AGENT_RUNTIME_FIELDS


def _task_allowed_fields(task_class: type[Any] | None = None) -> set[str]:
    from crewai import Task

    return _model_fields_for(task_class or Task, "task type") - _TASK_RUNTIME_FIELDS


def _crew_allowed_fields() -> set[str]:
    from crewai import Crew

    return set(Crew.model_fields) - _CREW_RUNTIME_FIELDS


def _task_definition_errors(
    task_defn: dict[str, Any],
    source: str | Path,
    *,
    resolve_python_refs: bool,
) -> list[str]:
    skip_unknown = _definition_has_python_type(task_defn) and not resolve_python_refs
    try:
        task_class = _task_class_from_definition(
            task_defn,
            f"{source}: type",
            resolve_python_refs=resolve_python_refs,
        )
    except JSONProjectValidationError as exc:
        return exc.errors
    except JSONProjectError as exc:
        return [str(exc)]

    errors = _field_errors(
        task_defn,
        _task_allowed_fields(task_class),
        _TASK_RUNTIME_FIELDS,
        source,
        {"type"},
        skip_unknown=skip_unknown,
    )
    errors.extend(_python_reference_definition_errors(task_defn, source))
    return errors


def _python_reference_definition_errors(
    defn: dict[str, Any],
    source: str | Path,
) -> list[str]:
    errors: list[str] = []
    for field in (
        _AGENT_CALLABLE_FIELDS
        | _AGENT_CALLABLE_LIST_FIELDS
        | _TASK_CALLABLE_FIELDS
        | _TASK_CALLABLE_LIST_FIELDS
        | _TASK_MODEL_CLASS_FIELDS
        | _CREW_CALLABLE_FIELDS
        | _CREW_CALLABLE_LIST_FIELDS
        | {"converter_cls", "executor_class"}
    ):
        if field not in defn:
            continue
        errors.extend(_python_reference_value_errors(defn[field], f"{source}: {field}"))

    for field in (
        _AGENT_OBJECT_REF_FIELDS | _TASK_OBJECT_REF_FIELDS | _CREW_OBJECT_REF_FIELDS
    ):
        if field not in defn:
            continue
        errors.extend(
            _python_reference_value_errors_recursive(defn[field], f"{source}: {field}")
        )

    errors.extend(
        _embedder_python_ref_errors(defn.get("embedder"), f"{source}: embedder")
    )
    errors.extend(_a2a_python_ref_errors(defn.get("a2a"), f"{source}: a2a"))
    errors.extend(_mcp_python_ref_errors(defn.get("mcps"), f"{source}: mcps"))

    type_value = defn.get("type")
    if _is_python_ref(type_value):
        errors.extend(_python_ref_errors(type_value, f"{source}: type"))
    return errors


def _python_reference_value_errors(value: Any, source: str | Path) -> list[str]:
    errors: list[str] = []
    if _is_python_ref(value):
        return _python_ref_errors(value, source)
    if isinstance(value, list):
        for index, item in enumerate(value):
            if _is_python_ref(item):
                errors.extend(_python_ref_errors(item, f"{source}[{index}]"))
    return errors


def _python_reference_value_errors_recursive(
    value: Any, source: str | Path
) -> list[str]:
    if _is_python_ref(value):
        return _python_ref_errors(value, source)
    errors: list[str] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(
                _python_reference_value_errors_recursive(item, f"{source}[{index}]")
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(
                _python_reference_value_errors_recursive(item, f"{source}.{key}")
            )
    return errors


def _embedder_python_ref_errors(value: Any, source: str | Path) -> list[str]:
    if not isinstance(value, dict):
        return []
    config = value.get("config")
    if not isinstance(config, dict):
        return []
    embedding_callable = config.get("embedding_callable")
    if _is_python_ref(embedding_callable):
        return _python_ref_errors(
            embedding_callable, f"{source}.config.embedding_callable"
        )
    return []


def _a2a_python_ref_errors(value: Any, source: str | Path) -> list[str]:
    configs = value if isinstance(value, list) else [value]
    errors: list[str] = []
    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            continue
        response_model = config.get("response_model")
        if _is_python_ref(response_model):
            errors.extend(
                _python_ref_errors(response_model, f"{source}[{index}].response_model")
            )
    return errors


def _mcp_python_ref_errors(value: Any, source: str | Path) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for index, config in enumerate(value):
        if not isinstance(config, dict):
            continue
        tool_filter = config.get("tool_filter")
        if _is_python_ref(tool_filter):
            errors.extend(
                _python_ref_errors(tool_filter, f"{source}[{index}].tool_filter")
            )
        elif isinstance(tool_filter, dict) and tool_filter.get("type") == "static":
            for key in ("allowed_tool_names", "blocked_tool_names"):
                names = tool_filter.get(key)
                if names is not None and not _is_string_list(names):
                    errors.append(
                        f"{source}[{index}].tool_filter.{key} must be a list of strings"
                    )
    return errors


def _resolve_agent_python_refs(kwargs: dict[str, Any], source: str | Path) -> None:
    _resolve_callable_fields(
        kwargs,
        source,
        scalar_fields=_AGENT_CALLABLE_FIELDS,
        list_fields=_AGENT_CALLABLE_LIST_FIELDS,
    )
    if _is_python_ref(kwargs.get("executor_class")):
        kwargs["executor_class"] = _resolve_python_class(
            kwargs["executor_class"], f"{source}: executor_class"
        )
    if "embedder" in kwargs:
        kwargs["embedder"] = _resolve_embedder_python_refs(kwargs["embedder"], source)
    if "a2a" in kwargs:
        kwargs["a2a"] = _resolve_a2a_python_refs(kwargs["a2a"], source)
    _resolve_object_reference_fields(kwargs, source, _AGENT_OBJECT_REF_FIELDS)


def _resolve_task_python_refs(kwargs: dict[str, Any], source: str | Path) -> None:
    _resolve_callable_fields(
        kwargs,
        source,
        scalar_fields=_TASK_CALLABLE_FIELDS,
        list_fields=_TASK_CALLABLE_LIST_FIELDS,
    )
    for field in _TASK_MODEL_CLASS_FIELDS:
        if _is_python_ref(kwargs.get(field)):
            kwargs[field] = _resolve_model_class(kwargs[field], f"{source}: {field}")
    if _is_python_ref(kwargs.get("converter_cls")):
        from crewai.utilities.converter import Converter

        kwargs["converter_cls"] = _resolve_python_class(
            kwargs["converter_cls"],
            f"{source}: converter_cls",
            base_class=Converter,
        )
    elif isinstance(kwargs.get("converter_cls"), str):
        raise JSONProjectError(
            f"{source}: converter_cls must use "
            f'{{"{PYTHON_REF_KEY}": "module.ConverterSubclass"}}'
        )
    _resolve_object_reference_fields(kwargs, source, _TASK_OBJECT_REF_FIELDS)


def _resolve_crew_python_refs(kwargs: dict[str, Any], source: str | Path) -> None:
    _resolve_callable_fields(
        kwargs,
        source,
        scalar_fields=_CREW_CALLABLE_FIELDS,
        list_fields=_CREW_CALLABLE_LIST_FIELDS,
    )
    if "embedder" in kwargs:
        kwargs["embedder"] = _resolve_embedder_python_refs(kwargs["embedder"], source)
    _resolve_object_reference_fields(kwargs, source, _CREW_OBJECT_REF_FIELDS)


def _resolve_object_reference_fields(
    kwargs: dict[str, Any],
    source: str | Path,
    fields: set[str],
) -> None:
    for field in fields:
        if field not in kwargs:
            continue
        kwargs[field] = _resolve_python_refs_recursively(
            kwargs[field], f"{source}: {field}"
        )


def _resolve_python_refs_recursively(value: Any, source: str | Path) -> Any:
    if _is_python_ref(value):
        return _resolve_python_ref(value, source, expected="any")
    if isinstance(value, list):
        return [
            _resolve_python_refs_recursively(item, f"{source}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _resolve_python_refs_recursively(item, f"{source}.{key}")
            for key, item in value.items()
        }
    return value


def _resolve_callable_fields(
    kwargs: dict[str, Any],
    source: str | Path,
    *,
    scalar_fields: set[str],
    list_fields: set[str],
) -> None:
    for field in scalar_fields:
        if _is_python_ref(kwargs.get(field)):
            kwargs[field] = _resolve_python_ref(
                kwargs[field],
                f"{source}: {field}",
                expected="callable",
            )
    for field in list_fields:
        value = kwargs.get(field)
        if not isinstance(value, list):
            continue
        kwargs[field] = [
            _resolve_python_ref(
                item, f"{source}: {field}[{index}]", expected="callable"
            )
            if _is_python_ref(item)
            else item
            for index, item in enumerate(value)
        ]


def _resolve_model_class(value: Any, source: str | Path) -> type[BaseModel]:
    return _resolve_python_class(value, source, base_class=BaseModel)


def _resolve_embedder_python_refs(value: Any, source: str | Path) -> Any:
    if not isinstance(value, dict):
        return value
    config = value.get("config")
    if not isinstance(config, dict):
        return value
    embedding_callable = config.get("embedding_callable")
    if not _is_python_ref(embedding_callable):
        return value

    from crewai.rag.embeddings.providers.custom.embedding_callable import (
        CustomEmbeddingFunction,
    )

    normalized = dict(value)
    normalized_config = dict(config)
    normalized_config["embedding_callable"] = _resolve_python_class(
        embedding_callable,
        f"{source}: embedder.config.embedding_callable",
        base_class=CustomEmbeddingFunction,
    )
    normalized["config"] = normalized_config
    return normalized


def _resolve_a2a_python_refs(value: Any, source: str | Path) -> Any:
    if isinstance(value, list):
        return [
            _resolve_a2a_python_refs(item, f"{source}: a2a[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    response_model = value.get("response_model")
    if response_model is None:
        return value

    normalized = dict(value)
    if _is_python_ref(response_model):
        normalized["response_model"] = _resolve_model_class(
            response_model,
            f"{source}: a2a.response_model",
        )
    elif isinstance(response_model, dict):
        from crewai.utilities.pydantic_schema_utils import create_model_from_schema

        normalized["response_model"] = create_model_from_schema(response_model)
    return normalized


def _resolve_mcp_python_refs(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [
        _resolve_mcp_config_python_refs(config, index)
        if isinstance(config, dict)
        else config
        for index, config in enumerate(value)
    ]


def _resolve_mcp_config_python_refs(
    config: dict[str, Any], index: int
) -> dict[str, Any]:
    tool_filter = config.get("tool_filter")
    if tool_filter is None:
        return config
    normalized = dict(config)
    if _is_python_ref(tool_filter):
        normalized["tool_filter"] = _resolve_python_ref(
            tool_filter,
            f"mcps[{index}].tool_filter",
            expected="callable",
        )
    elif isinstance(tool_filter, dict) and tool_filter.get("type") == "static":
        from crewai.mcp.filters import create_static_tool_filter

        allowed_tool_names = tool_filter.get("allowed_tool_names")
        blocked_tool_names = tool_filter.get("blocked_tool_names")
        if allowed_tool_names is not None and not _is_string_list(allowed_tool_names):
            raise JSONProjectValidationError(
                [
                    f"mcps[{index}].tool_filter.allowed_tool_names must be a list of strings"
                ]
            )
        if blocked_tool_names is not None and not _is_string_list(blocked_tool_names):
            raise JSONProjectValidationError(
                [
                    f"mcps[{index}].tool_filter.blocked_tool_names must be a list of strings"
                ]
            )
        normalized["tool_filter"] = create_static_tool_filter(
            allowed_tool_names=allowed_tool_names,
            blocked_tool_names=blocked_tool_names,
        )
    return normalized


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _normalize_input_files(
    value: Any,
    source: str | Path,
    project_root: Path | None,
) -> Any:
    if value is None:
        return value
    if not isinstance(value, dict):
        raise JSONProjectValidationError(
            [f"{source}: input_files must be an object mapping names to file specs"]
        )

    normalized: dict[str, Any] = {}
    for name, file_spec in value.items():
        if isinstance(file_spec, str):
            normalized[name] = {
                "source": _resolve_project_path(file_spec, project_root)
            }
            continue
        if isinstance(file_spec, dict):
            normalized_spec = dict(file_spec)
            for field in ("source", "path"):
                field_value = normalized_spec.get(field)
                if isinstance(field_value, str):
                    normalized_spec[field] = _resolve_project_path(
                        field_value, project_root
                    )
            normalized[name] = normalized_spec
            continue
        normalized[name] = file_spec
    return normalized


def _resolve_project_path(value: str, project_root: Path | None) -> str:
    if not value or _URI_RE.match(value):
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str(((project_root or Path.cwd()) / path).resolve())


def _format_validation_error(path: str | Path, exc: ValidationError) -> str:
    return f"{path}: validation failed: {exc}"


def _resolve_tools(tool_defs: list[Any], project_root: Path | None = None) -> list[Any]:
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
            tools.append(_resolve_custom_tool(tool_def[7:], project_root=project_root))
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


_CUSTOM_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _custom_tool_file(tool_name: str, project_root: Path | None) -> Path:
    """Return the validated path of a custom tool inside ``tools/``.

    Rejects names that aren't plain identifiers and (belt-and-suspenders)
    any resolved path that escapes the project's ``tools/`` directory, so
    ``custom:../evil`` or absolute-path style names cannot execute code
    outside the project.
    """
    if not _CUSTOM_TOOL_NAME_RE.fullmatch(tool_name):
        raise JSONProjectError(
            f"Invalid custom tool name 'custom:{tool_name}': names must match "
            f"[A-Za-z_][A-Za-z0-9_]* and resolve to tools/<name>.py inside "
            f"the project."
        )
    tools_dir = ((project_root or Path.cwd()) / "tools").resolve()
    tool_file = (tools_dir / f"{tool_name}.py").resolve()
    try:
        tool_file.relative_to(tools_dir)
    except ValueError:
        raise JSONProjectError(
            f"Custom tool 'custom:{tool_name}' resolves outside the project's "
            f"tools/ directory."
        ) from None
    return tool_file


def _tool_definition_errors(
    tool_defs: Any, source: Path | str, project_root: Path | None
) -> list[str]:
    """Structurally validate tool declarations WITHOUT importing anything.

    Used by validation/deploy paths where executing project code (which
    ``custom:`` resolution does) would be unsafe. Library tool names are not
    resolved here either — that requires importing crewai_tools modules and
    would falsely fail when optional dependencies are absent in the
    validation environment.
    """
    if tool_defs is None:
        return []
    if not isinstance(tool_defs, list):
        return [f"{source}: 'tools' must be a list"]
    errors: list[str] = []
    for tool_def in tool_defs:
        if isinstance(tool_def, dict):
            continue
        if not isinstance(tool_def, str):
            errors.append(
                f"{source}: tool definitions must be strings or objects, "
                f"got {type(tool_def).__name__}"
            )
            continue
        if not tool_def.startswith("custom:"):
            continue
        try:
            tool_file = _custom_tool_file(tool_def[7:], project_root)
        except JSONProjectError as exc:
            errors.append(f"{source}: {exc}")
            continue
        if not tool_file.exists():
            errors.append(
                f"{source}: custom tool '{tool_def}' not found: expected "
                f"{tool_file}. Create the file with a BaseTool subclass, or "
                f"remove the tool from your crew JSON."
            )
    return errors


def _resolve_custom_tool(tool_name: str, project_root: Path | None = None) -> Any:
    """Resolve a custom tool from the project's ``tools/`` directory.

    Note: ``custom:<name>`` tools execute ``tools/<name>.py`` as local Python
    code at load time — JSON configs referencing them are no longer pure data.
    Only run JSON crew projects from sources you trust. Validation paths must
    use ``_tool_definition_errors`` instead, which never executes anything.
    """
    tool_file = _custom_tool_file(tool_name, project_root)
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
