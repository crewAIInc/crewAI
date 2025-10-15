"""Base metaclass for creating crew classes with configuration and function management."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

from crewai.tools import BaseTool

if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput
    from crewai.project.wrappers import CrewInstance

load_dotenv()


class CrewBaseMeta(type):
    """Metaclass that adds crew functionality to classes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """Create crew class with configuration and method injection.

        Args:
            name: Class name.
            bases: Base classes.
            namespace: Class namespace dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            New crew class with injected methods and attributes.
        """
        cls = super().__new__(mcs, name, bases, namespace)

        cls.is_crew_class = True  # type: ignore[attr-defined]
        cls._crew_name = name  # type: ignore[attr-defined]

        try:
            cls.base_directory = Path(inspect.getfile(cls)).parent  # type: ignore[attr-defined]
        except (TypeError, OSError):
            cls.base_directory = Path.cwd()  # type: ignore[attr-defined]

        cls.original_agents_config_path = getattr(  # type: ignore[attr-defined]
            cls, "agents_config", "config/agents.yaml"
        )
        cls.original_tasks_config_path = getattr(  # type: ignore[attr-defined]
            cls, "tasks_config", "config/tasks.yaml"
        )

        cls.mcp_server_params = getattr(cls, "mcp_server_params", None)  # type: ignore[attr-defined]
        cls.mcp_connect_timeout = getattr(cls, "mcp_connect_timeout", 30)  # type: ignore[attr-defined]

        cls._close_mcp_server = _close_mcp_server  # type: ignore[attr-defined]
        cls.get_mcp_tools = get_mcp_tools  # type: ignore[attr-defined]
        cls._load_config = _load_config  # type: ignore[attr-defined]
        cls.load_configurations = load_configurations  # type: ignore[attr-defined]
        cls.load_yaml = staticmethod(load_yaml)  # type: ignore[attr-defined]
        cls.map_all_agent_variables = map_all_agent_variables  # type: ignore[attr-defined]
        cls._map_agent_variables = _map_agent_variables  # type: ignore[attr-defined]
        cls.map_all_task_variables = map_all_task_variables  # type: ignore[attr-defined]
        cls._map_task_variables = _map_task_variables  # type: ignore[attr-defined]

        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> CrewInstance:
        """Intercept instance creation to initialize crew functionality.

        Args:
            *args: Positional arguments for instance creation.
            **kwargs: Keyword arguments for instance creation.

        Returns:
            Initialized crew instance.
        """
        instance: CrewInstance = super().__call__(*args, **kwargs)
        CrewBaseMeta._initialize_crew_instance(instance, cls)
        return instance

    @staticmethod
    def _initialize_crew_instance(instance: CrewInstance, cls: type) -> None:
        """Initialize crew instance attributes and load configurations.

        Args:
            instance: Crew instance to initialize.
            cls: Crew class type.
        """
        instance._mcp_server_adapter = None
        instance.load_configurations()
        instance._all_functions = _get_all_functions(instance)
        instance.map_all_agent_variables()
        instance.map_all_task_variables()

        instance._original_functions = {
            name: method
            for name, method in cls.__dict__.items()
            if any(
                hasattr(method, attr)
                for attr in [
                    "is_task",
                    "is_agent",
                    "is_before_kickoff",
                    "is_after_kickoff",
                    "is_kickoff",
                ]
            )
        }

        instance._original_tasks = _filter_functions(
            instance._original_functions, "is_task"
        )
        instance._original_agents = _filter_functions(
            instance._original_functions, "is_agent"
        )
        instance._before_kickoff = _filter_functions(
            instance._original_functions, "is_before_kickoff"
        )
        instance._after_kickoff = _filter_functions(
            instance._original_functions, "is_after_kickoff"
        )
        instance._kickoff = _filter_functions(
            instance._original_functions, "is_kickoff"
        )

        instance._after_kickoff["_close_mcp_server"] = instance._close_mcp_server


def _close_mcp_server(
    self: CrewInstance, _instance: CrewInstance, outputs: CrewOutput
) -> CrewOutput:
    """Stop MCP server adapter and return outputs.

    Args:
        self: Crew instance with MCP server adapter.
        _instance: Crew instance (unused, required by callback signature).
        outputs: Crew execution outputs.

    Returns:
        Unmodified crew outputs.
    """
    if self._mcp_server_adapter is not None:
        try:
            self._mcp_server_adapter.stop()
        except Exception as e:
            logging.warning(f"Error stopping MCP server: {e}")
    return outputs


def get_mcp_tools(self: Any, *tool_names: str) -> list[BaseTool]:
    """Get MCP tools filtered by name.

    Args:
        self: Crew instance with MCP server configuration.
        *tool_names: Optional tool names to filter by.

    Returns:
        List of filtered MCP tools, or empty list if no MCP server configured.
    """
    if not self.mcp_server_params:
        return []

    from crewai_tools import MCPServerAdapter  # type: ignore[import-untyped]

    if self._mcp_server_adapter is None:
        self._mcp_server_adapter = MCPServerAdapter(
            self.mcp_server_params, connect_timeout=self.mcp_connect_timeout
        )

    return self._mcp_server_adapter.tools.filter_by_names(tool_names or None)


def _load_config(
    self: Any, config_path: str | None, config_type: str
) -> dict[str, Any]:
    """Load YAML config file or return empty dict if not found.

    Args:
        config_path: Relative path to config file.
        config_type: Config type for logging (e.g., "agent", "task").

    Returns:
        Config dictionary or empty dict.
    """
    if isinstance(config_path, str):
        full_path = self.base_directory / config_path
        try:
            return self.load_yaml(full_path)
        except FileNotFoundError:
            logging.warning(
                f"{config_type.capitalize()} config file not found at {full_path}. "
                f"Proceeding with empty {config_type} configurations."
            )
            return {}
    else:
        logging.warning(
            f"No {config_type} configuration path provided. "
            f"Proceeding with empty {config_type} configurations."
        )
        return {}


def load_configurations(self: Any) -> None:
    """Load agent and task YAML configurations.

    Args:
        self: Crew instance with configuration paths.
    """
    self.agents_config = self._load_config(self.original_agents_config_path, "agent")
    self.tasks_config = self._load_config(self.original_tasks_config_path, "task")


def load_yaml(config_path: Path) -> Any:
    """Load and parse YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed YAML content.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    try:
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File not found: {config_path}")
        raise


def _get_all_functions(self: Any) -> dict[str, Callable[..., Any]]:
    """Return all non-dunder callable attributes.

    Args:
        self: Instance to inspect for callable attributes.

    Returns:
        Dictionary mapping attribute names to callable objects.
    """
    return {
        name: getattr(self, name)
        for name in dir(self)
        if not (name.startswith("__") and name.endswith("__"))
        and callable(getattr(self, name, None))
    }


def _filter_functions(
    functions: dict[str, Callable[..., Any]], attribute: str
) -> dict[str, Callable[..., Any]]:
    """Filter functions by attribute presence.

    Args:
        functions: Dictionary of functions to filter.
        attribute: Attribute name to check for.

    Returns:
        Dictionary containing only functions with the specified attribute.
    """
    return {name: func for name, func in functions.items() if hasattr(func, attribute)}


def map_all_agent_variables(self: Any) -> None:
    """Map agent configuration variables to callable instances.

    Args:
        self: Crew instance with agent configurations to map.
    """
    llms = _filter_functions(self._all_functions, "is_llm")
    tool_functions = _filter_functions(self._all_functions, "is_tool")
    cache_handler_functions = _filter_functions(self._all_functions, "is_cache_handler")
    callbacks = _filter_functions(self._all_functions, "is_callback")

    for agent_name, agent_info in self.agents_config.items():
        self._map_agent_variables(
            agent_name,
            agent_info,
            llms,
            tool_functions,
            cache_handler_functions,
            callbacks,
        )


def _map_agent_variables(
    self: Any,
    agent_name: str,
    agent_info: dict[str, Any],
    llms: dict[str, Callable[..., Any]],
    tool_functions: dict[str, Callable[..., Any]],
    cache_handler_functions: dict[str, Callable[..., Any]],
    callbacks: dict[str, Callable[..., Any]],
) -> None:
    """Resolve and map variables for a single agent.

    Args:
        self: Crew instance with agent configurations.
        agent_name: Name of agent to configure.
        agent_info: Agent configuration dictionary.
        llms: Dictionary of available LLM providers.
        tool_functions: Dictionary of available tools.
        cache_handler_functions: Dictionary of available cache handlers.
        callbacks: Dictionary of available callbacks.
    """
    if llm := agent_info.get("llm"):
        try:
            self.agents_config[agent_name]["llm"] = llms[llm]()
        except KeyError:
            self.agents_config[agent_name]["llm"] = llm

    if tools := agent_info.get("tools"):
        self.agents_config[agent_name]["tools"] = [
            tool_functions[tool]() for tool in tools
        ]

    if function_calling_llm := agent_info.get("function_calling_llm"):
        try:
            self.agents_config[agent_name]["function_calling_llm"] = llms[
                function_calling_llm
            ]()
        except KeyError:
            self.agents_config[agent_name]["function_calling_llm"] = (
                function_calling_llm
            )

    if step_callback := agent_info.get("step_callback"):
        self.agents_config[agent_name]["step_callback"] = callbacks[step_callback]()

    if cache_handler := agent_info.get("cache_handler"):
        self.agents_config[agent_name]["cache_handler"] = cache_handler_functions[
            cache_handler
        ]()


def map_all_task_variables(self: Any) -> None:
    """Map task configuration variables to callable instances.

    Args:
        self: Crew instance with task configurations to map.
    """
    agents = _filter_functions(self._all_functions, "is_agent")
    tasks = _filter_functions(self._all_functions, "is_task")
    output_json_functions = _filter_functions(self._all_functions, "is_output_json")
    tool_functions = _filter_functions(self._all_functions, "is_tool")
    callback_functions = _filter_functions(self._all_functions, "is_callback")
    output_pydantic_functions = _filter_functions(
        self._all_functions, "is_output_pydantic"
    )

    for task_name, task_info in self.tasks_config.items():
        self._map_task_variables(
            task_name,
            task_info,
            agents,
            tasks,
            output_json_functions,
            tool_functions,
            callback_functions,
            output_pydantic_functions,
        )


def _map_task_variables(
    self: Any,
    task_name: str,
    task_info: dict[str, Any],
    agents: dict[str, Callable[..., Any]],
    tasks: dict[str, Callable[..., Any]],
    output_json_functions: dict[str, Callable[..., Any]],
    tool_functions: dict[str, Callable[..., Any]],
    callback_functions: dict[str, Callable[..., Any]],
    output_pydantic_functions: dict[str, Callable[..., Any]],
) -> None:
    """Resolve and map variables for a single task.

    Args:
        self: Crew instance with task configurations.
        task_name: Name of task to configure.
        task_info: Task configuration dictionary.
        agents: Dictionary of available agents.
        tasks: Dictionary of available tasks.
        output_json_functions: Dictionary of available JSON output schemas.
        tool_functions: Dictionary of available tools.
        callback_functions: Dictionary of available callbacks.
        output_pydantic_functions: Dictionary of available Pydantic output schemas.
    """
    if context_list := task_info.get("context"):
        self.tasks_config[task_name]["context"] = [
            tasks[context_task_name]() for context_task_name in context_list
        ]

    if tools := task_info.get("tools"):
        self.tasks_config[task_name]["tools"] = [
            tool_functions[tool]() for tool in tools
        ]

    if agent_name := task_info.get("agent"):
        self.tasks_config[task_name]["agent"] = agents[agent_name]()

    if output_json := task_info.get("output_json"):
        self.tasks_config[task_name]["output_json"] = output_json_functions[output_json]

    if output_pydantic := task_info.get("output_pydantic"):
        self.tasks_config[task_name]["output_pydantic"] = output_pydantic_functions[
            output_pydantic
        ]

    if callbacks := task_info.get("callbacks"):
        self.tasks_config[task_name]["callbacks"] = [
            callback_functions[callback]() for callback in callbacks
        ]

    if guardrail := task_info.get("guardrail"):
        self.tasks_config[task_name]["guardrail"] = guardrail


def CrewBase(cls: type) -> type:  # noqa: N802
    """Apply CrewBaseMeta metaclass to a class for decorator syntax compatibility.

    Args:
        cls: Class to apply metaclass to.

    Returns:
        New class with CrewBaseMeta metaclass applied.
    """
    if isinstance(cls, CrewBaseMeta):
        return cls

    namespace = {
        key: value
        for key, value in cls.__dict__.items()
        if not key.startswith("__")
        or key in ("__module__", "__qualname__", "__annotations__")
    }

    return CrewBaseMeta(cls.__name__, cls.__bases__, namespace)
