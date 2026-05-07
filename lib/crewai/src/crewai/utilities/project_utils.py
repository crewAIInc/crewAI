"""Project utility functions for discovering crews, flows, and tools."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import inspect
from inspect import getmro, isclass, isfunction, ismethod
import os
from pathlib import Path
import sys
import types
from typing import Any, cast, get_type_hints

from crewai_core.project import (
    get_project_description as get_project_description,
    get_project_name as get_project_name,
    get_project_version as get_project_version,
    parse_toml as parse_toml,
    read_toml as read_toml,
)
from crewai_core.tool_credentials import (
    build_env_with_all_tool_credentials as build_env_with_all_tool_credentials,
    build_env_with_tool_repository_credentials as build_env_with_tool_repository_credentials,
)
from rich.console import Console

from crewai.crew import Crew
from crewai.flow import Flow


__all__ = [
    "build_env_with_all_tool_credentials",
    "build_env_with_tool_repository_credentials",
    "extract_available_exports",
    "extract_tools_metadata",
    "fetch_crews",
    "get_crew_instance",
    "get_crews",
    "get_flow_instance",
    "get_flows",
    "get_project_description",
    "get_project_name",
    "get_project_version",
    "is_valid_tool",
    "parse_toml",
    "read_toml",
]


console = Console()


def get_crews(crew_path: str = "crew.py", require: bool = False) -> list[Crew]:
    """Get the crew instances from a file."""
    crew_instances = []
    try:
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        src_dir = os.path.join(current_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        search_paths = [".", "src"] if os.path.isdir("src") else ["."]

        for search_path in search_paths:
            for root, _, files in os.walk(search_path):
                if crew_path in files and "cli/templates" not in root:
                    crew_os_path = os.path.join(root, crew_path)
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "crew_module", crew_os_path
                        )
                        if not spec or not spec.loader:
                            continue

                        module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = module

                        try:
                            spec.loader.exec_module(module)

                            for attr_name in dir(module):
                                module_attr = getattr(module, attr_name)
                                try:
                                    crew_instances.extend(fetch_crews(module_attr))
                                except Exception as e:
                                    console.print(
                                        f"Error processing attribute {attr_name}: {e}",
                                        style="bold red",
                                    )
                                    continue

                            if crew_instances:
                                break

                        except Exception as exec_error:
                            console.print(
                                f"Error executing module: {exec_error}",
                                style="bold red",
                            )

                    except (ImportError, AttributeError) as e:
                        if require:
                            console.print(
                                f"Error importing crew from {crew_path}: {e!s}",
                                style="bold red",
                            )
                        continue

            if crew_instances:
                break

        if require and not crew_instances:
            console.print("No valid Crew instance found in crew.py", style="bold red")
            raise SystemExit

    except Exception as e:
        if require:
            console.print(
                f"Unexpected error while loading crew: {e!s}", style="bold red"
            )
            raise SystemExit from e
    return crew_instances


def get_crew_instance(module_attr: Any) -> Crew | None:
    """Get a Crew instance from a module attribute."""
    if (
        callable(module_attr)
        and hasattr(module_attr, "is_crew_class")
        and module_attr.is_crew_class
    ):
        return cast(Crew, module_attr().crew())
    try:
        if (ismethod(module_attr) or isfunction(module_attr)) and get_type_hints(
            module_attr
        ).get("return") is Crew:
            return cast(Crew, module_attr())
    except Exception:
        return None

    if isinstance(module_attr, Crew):
        return module_attr
    return None


def fetch_crews(module_attr: Any) -> list[Crew]:
    """Fetch crew instances from a module attribute."""
    crew_instances: list[Crew] = []

    if crew_instance := get_crew_instance(module_attr):
        crew_instances.append(crew_instance)

    if isinstance(module_attr, type) and issubclass(module_attr, Flow):
        instance = module_attr()
        for attr_name in dir(instance):
            attr = getattr(instance, attr_name)
            if crew_instance := get_crew_instance(attr):
                crew_instances.append(crew_instance)
    return crew_instances


def get_flow_instance(module_attr: Any) -> Flow[Any] | None:
    """Check if a module attribute is a user-defined Flow subclass and return an instance.

    Args:
        module_attr: An attribute from a loaded module.

    Returns:
        A Flow instance if the attribute is a valid user-defined Flow subclass,
        None otherwise.
    """
    if (
        isinstance(module_attr, type)
        and issubclass(module_attr, Flow)
        and module_attr is not Flow
    ):
        try:
            return module_attr()
        except Exception:
            return None
    return None


_SKIP_DIRS = frozenset(
    {".venv", "venv", ".git", "__pycache__", "node_modules", ".tox", ".nox"}
)


def get_flows(flow_path: str = "main.py") -> list[Flow[Any]]:
    """Get the flow instances from project files.

    Walks the project directory looking for files matching ``flow_path``
    (default ``main.py``), loads each module, and extracts Flow subclass
    instances.

    Args:
        flow_path: Filename to search for (default ``main.py``).

    Returns:
        A list of discovered Flow instances.
    """
    flow_instances: list[Flow[Any]] = []
    try:
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        src_dir = os.path.join(current_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        search_paths = [".", "src"] if os.path.isdir("src") else ["."]

        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [
                    d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")
                ]
                if flow_path in files and "cli/templates" not in root:
                    file_os_path = os.path.join(root, flow_path)
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "flow_module", file_os_path
                        )
                        if not spec or not spec.loader:
                            continue

                        module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = module

                        try:
                            spec.loader.exec_module(module)

                            for attr_name in dir(module):
                                module_attr = getattr(module, attr_name)
                                try:
                                    if flow_instance := get_flow_instance(module_attr):
                                        flow_instances.append(flow_instance)
                                except Exception:  # noqa: S112
                                    continue

                            if flow_instances:
                                break

                        except Exception:  # noqa: S112
                            continue

                    except (ImportError, AttributeError):
                        continue

            if flow_instances:
                break

    except Exception as e:
        import logging

        logging.getLogger(__name__).debug(
            f"Could not load tool repository credentials: {e}"
        )

    return flow_instances


def is_valid_tool(obj: Any) -> bool:
    """Check if an object is a valid CrewAI tool."""
    from crewai.tools.base_tool import Tool

    if isclass(obj):
        try:
            return any(base.__name__ == "BaseTool" for base in getmro(obj))
        except (TypeError, AttributeError):
            return False

    return isinstance(obj, Tool)


def extract_available_exports(dir_path: str = "src") -> list[dict[str, Any]]:
    """Extract available tool classes from the project's __init__.py files.

    Only includes classes that inherit from BaseTool or functions decorated with @tool.

    Returns:
        A list of valid tool class names or ["BaseTool"] if none found.
    """
    try:
        init_files = Path(dir_path).glob("**/__init__.py")
        available_exports = []

        for init_file in init_files:
            tools = _load_tools_from_init(init_file)
            available_exports.extend(tools)

        if not available_exports:
            _print_no_tools_warning()
            raise SystemExit(1)

        return available_exports

    except Exception as e:
        console.print(f"[red]Error: Could not extract tool classes: {e!s}[/red]")
        console.print(
            "Please ensure your project contains valid tools (classes inheriting from BaseTool or functions with @tool decorator)."
        )
        raise SystemExit(1) from e


@contextmanager
def _load_module_from_file(
    init_file: Path, module_name: str | None = None
) -> Generator[types.ModuleType | None, None, None]:
    """
    Context manager for loading a module from file with automatic cleanup.

    Yields the loaded module or None if loading fails.
    """
    if module_name is None:
        module_name = (
            f"temp_module_{hashlib.sha256(str(init_file).encode()).hexdigest()[:8]}"
        )

    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if not spec or not spec.loader:
        yield None
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _load_tools_from_init(init_file: Path) -> list[dict[str, Any]]:
    """Load and validate tools from a given __init__.py file."""
    try:
        with _load_module_from_file(init_file) as module:
            if module is None:
                return []

            if not hasattr(module, "__all__"):
                console.print(
                    f"Warning: No __all__ defined in {init_file}",
                    style="bold yellow",
                )
                raise SystemExit(1)

            return [
                {"name": name}
                for name in module.__all__
                if hasattr(module, name) and is_valid_tool(getattr(module, name))
            ]
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Warning: Could not load {init_file}: {e!s}[/red]")
        raise SystemExit(1) from e


def _print_no_tools_warning() -> None:
    """Display warning and usage instructions if no tools were found."""
    console.print(
        "\n[bold yellow]Warning: No valid tools were exposed in your __init__.py file![/bold yellow]"
    )
    console.print(
        "Your __init__.py file must contain all classes that inherit from [bold]BaseTool[/bold] "
        "or functions decorated with [bold]@tool[/bold]."
    )
    console.print(
        "\nExample:\n[dim]# In your __init__.py file[/dim]\n"
        "[green]__all__ = ['YourTool', 'your_tool_function'][/green]\n\n"
        "[dim]# In your tool.py file[/dim]\n"
        "[green]from crewai.tools import BaseTool, tool\n\n"
        "# Tool class example\n"
        "class YourTool(BaseTool):\n"
        '    name = "your_tool"\n'
        '    description = "Your tool description"\n'
        "    # ... rest of implementation\n\n"
        "# Decorated function example\n"
        "@tool\n"
        "def your_tool_function(text: str) -> str:\n"
        '    """Your tool description"""\n'
        "    # ... implementation\n"
        "    return result\n"
    )


def extract_tools_metadata(dir_path: str = "src") -> list[dict[str, Any]]:
    """
    Extract rich metadata from tool classes in the project.

    Returns a list of tool metadata dictionaries containing:
    - name: Class name
    - humanized_name: From name field default
    - description: From description field default
    - run_params_schema: JSON Schema for _run() params (from args_schema)
    - init_params_schema: JSON Schema for __init__ params (filtered)
    - env_vars: List of environment variable dicts
    """
    tools_metadata: list[dict[str, Any]] = []

    for init_file in Path(dir_path).glob("**/__init__.py"):
        tools = _extract_tool_metadata_from_init(init_file)
        tools_metadata.extend(tools)

    return tools_metadata


def _extract_tool_metadata_from_init(init_file: Path) -> list[dict[str, Any]]:
    """
    Load module from init file and extract metadata from valid tool classes.
    """
    from crewai.tools.base_tool import BaseTool

    try:
        with _load_module_from_file(init_file) as module:
            if module is None:
                return []

            exported_names = getattr(module, "__all__", None)
            if not exported_names:
                return []

            tools_metadata = []
            for name in exported_names:
                obj = getattr(module, name, None)
                if obj is None or not (
                    inspect.isclass(obj) and issubclass(obj, BaseTool)
                ):
                    continue
                if tool_info := _extract_single_tool_metadata(obj):
                    tools_metadata.append(tool_info)

            return tools_metadata
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not extract metadata from {init_file}: {e}[/yellow]"
        )
        return []


def _extract_single_tool_metadata(tool_class: type) -> dict[str, Any] | None:
    """
    Extract metadata from a single tool class.
    """
    try:
        core_schema = cast(Any, tool_class).__pydantic_core_schema__
        if not core_schema:
            return None

        schema = _unwrap_schema(core_schema)
        fields = schema.get("schema", {}).get("fields", {})

        try:
            file_path = inspect.getfile(tool_class)
            relative_path = Path(file_path).relative_to(Path.cwd())
            module_path = relative_path.with_suffix("")
            if module_path.parts[0] == "src":
                module_path = Path(*module_path.parts[1:])
            if module_path.name == "__init__":
                module_path = module_path.parent
            module = ".".join(module_path.parts)
        except (TypeError, ValueError):
            module = tool_class.__module__

        return {
            "name": tool_class.__name__,
            "module": module,
            "humanized_name": _extract_field_default(
                fields.get("name"), fallback=tool_class.__name__
            ),
            "description": str(
                _extract_field_default(fields.get("description"))
            ).strip(),
            "run_params_schema": _extract_run_params_schema(fields.get("args_schema")),
            "init_params_schema": _extract_init_params_schema(tool_class),
            "env_vars": _extract_env_vars(fields.get("env_vars")),
        }

    except Exception:
        return None


def _unwrap_schema(schema: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """
    Unwrap nested schema structures to get to the actual schema definition.
    """
    result: dict[str, Any] = dict(schema)
    while (
        result.get("type")
        in {"function-after", "function-before", "function-wrap", "default"}
        and "schema" in result
    ):
        result = dict(result["schema"])
    if result.get("type") == "definitions" and "schema" in result:
        result = dict(result["schema"])
    return result


def _extract_field_default(
    field: dict[str, Any] | None, fallback: str | list[Any] = ""
) -> str | list[Any] | int:
    """
    Extract the default value from a field schema.
    """
    if not field:
        return fallback

    schema = field.get("schema", {})
    default = schema.get("default")
    return default if isinstance(default, (list, str, int)) else fallback


@lru_cache(maxsize=1)
def _get_schema_generator() -> type:
    """Get a SchemaGenerator that omits non-serializable defaults."""
    from pydantic.json_schema import GenerateJsonSchema
    from pydantic_core import PydanticOmit

    class SchemaGenerator(GenerateJsonSchema):
        def handle_invalid_for_json_schema(
            self, schema: Any, error_info: Any
        ) -> dict[str, Any]:
            raise PydanticOmit

    return SchemaGenerator


def _extract_run_params_schema(
    args_schema_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Extract JSON Schema for the tool's run parameters from args_schema field.
    """
    from pydantic import BaseModel

    if not args_schema_field:
        return {}

    args_schema_class = args_schema_field.get("schema", {}).get("default")
    if not (
        inspect.isclass(args_schema_class) and issubclass(args_schema_class, BaseModel)
    ):
        return {}

    try:
        return args_schema_class.model_json_schema(
            schema_generator=_get_schema_generator()
        )
    except Exception:
        return {}


_IGNORED_INIT_PARAMS = frozenset(
    {
        "name",
        "description",
        "env_vars",
        "args_schema",
        "description_updated",
        "cache_function",
        "result_as_answer",
        "max_usage_count",
        "current_usage_count",
        "package_dependencies",
    }
)


def _extract_init_params_schema(tool_class: type) -> dict[str, Any]:
    """
    Extract JSON Schema for the tool's __init__ parameters, filtering out base fields.
    """
    try:
        json_schema: dict[str, Any] = cast(Any, tool_class).model_json_schema(
            schema_generator=_get_schema_generator(), mode="serialization"
        )
        filtered_properties = {
            key: value
            for key, value in json_schema.get("properties", {}).items()
            if key not in _IGNORED_INIT_PARAMS
        }
        json_schema["properties"] = filtered_properties
        if "required" in json_schema:
            json_schema["required"] = [
                key for key in json_schema["required"] if key in filtered_properties
            ]
        return json_schema
    except Exception:
        return {}


def _extract_env_vars(env_vars_field: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Extract environment variable definitions from env_vars field.
    """
    from crewai.tools.base_tool import EnvVar

    if not env_vars_field:
        return []

    schema = env_vars_field.get("schema", {})
    default = schema.get("default")
    if default is None:
        default_factory = schema.get("default_factory")
        if callable(default_factory):
            try:
                default = default_factory()
            except Exception:
                default = []

    if not isinstance(default, list):
        return []

    return [
        {
            "name": env_var.name,
            "description": env_var.description,
            "required": env_var.required,
            "default": env_var.default,
        }
        for env_var in default
        if isinstance(env_var, EnvVar)
    ]
