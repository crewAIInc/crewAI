"""Base metaclass for creating crew classes with configuration and method management."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeGuard,
    TypeVar,
    TypedDict,
    cast,
)

from dotenv import load_dotenv
import yaml

from crewai.project.wrappers import CrewClass, CrewMetadata
from crewai.tools import BaseTool


if TYPE_CHECKING:
    from crewai import Agent, Task
    from crewai.agents.cache.cache_handler import CacheHandler
    from crewai.crews.crew_output import CrewOutput
    from crewai.project.wrappers import (
        CrewInstance,
        OutputJsonClass,
        OutputPydanticClass,
    )
    from crewai.tasks.task_output import TaskOutput


class AgentConfig(TypedDict, total=False):
    """Type definition for agent configuration dictionary.

    All fields are optional as they come from YAML configuration files.
    Fields can be either string references (from YAML) or actual instances (after processing).
    """

    # Core agent attributes (from BaseAgent)
    role: str
    goal: str
    backstory: str
    cache: bool
    verbose: bool
    max_rpm: int
    allow_delegation: bool
    max_iter: int
    max_tokens: int
    callbacks: list[str]

    # LLM configuration
    llm: str
    function_calling_llm: str
    use_system_prompt: bool

    # Template configuration
    system_template: str
    prompt_template: str
    response_template: str

    # Tools and handlers (can be string references or instances)
    tools: list[str] | list[BaseTool]
    step_callback: str
    cache_handler: str | CacheHandler

    # Code execution
    allow_code_execution: bool
    code_execution_mode: Literal["safe", "unsafe"]

    # Context and performance
    respect_context_window: bool
    max_retry_limit: int

    # Multimodal and reasoning
    multimodal: bool
    reasoning: bool
    max_reasoning_attempts: int

    # Knowledge configuration
    knowledge_sources: list[str] | list[Any]
    knowledge_storage: str | Any
    knowledge_config: dict[str, Any]
    embedder: dict[str, Any]
    agent_knowledge_context: str
    crew_knowledge_context: str
    knowledge_search_query: str

    # Misc configuration
    inject_date: bool
    date_format: str
    from_repository: str
    guardrail: Callable[[Any], tuple[bool, Any]] | str
    guardrail_max_retries: int


class TaskConfig(TypedDict, total=False):
    """Type definition for task configuration dictionary.

    All fields are optional as they come from YAML configuration files.
    Fields can be either string references (from YAML) or actual instances (after processing).
    """

    # Core task attributes
    name: str
    description: str
    expected_output: str

    # Agent and context
    agent: str
    context: list[str]

    # Tools and callbacks (can be string references or instances)
    tools: list[str] | list[BaseTool]
    callback: str
    callbacks: list[str]

    # Output configuration
    output_json: str
    output_pydantic: str
    output_file: str
    create_directory: bool

    # Execution configuration
    async_execution: bool
    human_input: bool
    markdown: bool

    # Guardrail configuration
    guardrail: Callable[[TaskOutput], tuple[bool, Any]] | str
    guardrail_max_retries: int

    # Misc configuration
    allow_crewai_trigger_context: bool


load_dotenv()

CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def _set_base_directory(cls: type[CrewClass]) -> None:
    """Set the base directory for the crew class.

    Args:
        cls: Crew class to configure.
    """
    try:
        cls.base_directory = Path(inspect.getfile(cls)).parent
    except (TypeError, OSError):
        cls.base_directory = Path.cwd()


def _set_config_paths(cls: type[CrewClass]) -> None:
    """Set the configuration file paths for the crew class.

    Args:
        cls: Crew class to configure.
    """
    cls.original_agents_config_path = getattr(
        cls, "agents_config", "config/agents.yaml"
    )
    cls.original_tasks_config_path = getattr(cls, "tasks_config", "config/tasks.yaml")


def _set_mcp_params(cls: type[CrewClass]) -> None:
    """Set the MCP server parameters for the crew class.

    Args:
        cls: Crew class to configure.
    """
    cls.mcp_server_params = getattr(cls, "mcp_server_params", None)
    cls.mcp_connect_timeout = getattr(cls, "mcp_connect_timeout", 30)


def _is_string_list(value: list[str] | list[BaseTool]) -> TypeGuard[list[str]]:
    """Type guard to check if list contains strings rather than BaseTool instances.

    Args:
        value: List that may contain strings or BaseTool instances.

    Returns:
        True if all elements are strings, False otherwise.
    """
    return all(isinstance(item, str) for item in value)


def _is_string_value(value: str | CacheHandler) -> TypeGuard[str]:
    """Type guard to check if value is a string rather than a CacheHandler instance.

    Args:
        value: Value that may be a string or CacheHandler instance.

    Returns:
        True if value is a string, False otherwise.
    """
    return isinstance(value, str)


class CrewBaseMeta(type):
    """Metaclass that adds crew functionality to classes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type[CrewClass]:
        """Create crew class with configuration and method injection.

        Args:
            name: Class name.
            bases: Base classes.
            namespace: Class namespace dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            New crew class with injected methods and attributes.
        """
        cls = cast(
            type[CrewClass], cast(object, super().__new__(mcs, name, bases, namespace))
        )

        cls.is_crew_class = True
        cls._crew_name = name

        for setup_fn in _CLASS_SETUP_FUNCTIONS:
            setup_fn(cls)

        for method in _METHODS_TO_INJECT:
            setattr(cls, method.__name__, method)

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
        instance._all_methods = _get_all_methods(instance)
        instance.map_all_agent_variables()
        instance.map_all_task_variables()

        original_methods = {
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

        after_kickoff_callbacks = _filter_methods(original_methods, "is_after_kickoff")
        after_kickoff_callbacks["close_mcp_server"] = instance.close_mcp_server

        instance.__crew_metadata__ = CrewMetadata(
            original_methods=original_methods,
            original_tasks=_filter_methods(original_methods, "is_task"),
            original_agents=_filter_methods(original_methods, "is_agent"),
            before_kickoff=_filter_methods(original_methods, "is_before_kickoff"),
            after_kickoff=after_kickoff_callbacks,
            kickoff=_filter_methods(original_methods, "is_kickoff"),
        )

        _register_crew_hooks(instance, cls)


def close_mcp_server(
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


def get_mcp_tools(self: CrewInstance, *tool_names: str) -> list[BaseTool]:
    """Get MCP tools filtered by name.

    Args:
        self: Crew instance with MCP server configuration.
        *tool_names: Optional tool names to filter by.

    Returns:
        List of filtered MCP tools, or empty list if no MCP server configured.
    """
    if not self.mcp_server_params:
        return []

    from crewai_tools import MCPServerAdapter

    if self._mcp_server_adapter is None:
        self._mcp_server_adapter = MCPServerAdapter(
            self.mcp_server_params, connect_timeout=self.mcp_connect_timeout
        )

    return cast(
        list[BaseTool],
        self._mcp_server_adapter.tools.filter_by_names(tool_names or None),
    )


def _load_config(
    self: CrewInstance, config_path: str | None, config_type: Literal["agent", "task"]
) -> dict[str, Any]:
    """Load YAML config file or return empty dict if not found.

    Args:
        self: Crew instance with base directory and load_yaml method.
        config_path: Relative path to config file.
        config_type: Config type for logging, either "agent" or "task".

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


def load_configurations(self: CrewInstance) -> None:
    """Load agent and task YAML configurations.

    Args:
        self: Crew instance with configuration paths.
    """
    self.agents_config = self._load_config(self.original_agents_config_path, "agent")
    self.tasks_config = self._load_config(self.original_tasks_config_path, "task")


def load_yaml(config_path: Path) -> dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary. Returns empty dict if file is empty.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    try:
        with open(config_path, encoding="utf-8") as file:
            content = yaml.safe_load(file)
        return content if isinstance(content, dict) else {}
    except FileNotFoundError:
        logging.warning(f"File not found: {config_path}")
        raise


def _get_all_methods(self: CrewInstance) -> dict[str, Callable[..., Any]]:
    """Return all non-dunder callable attributes (methods).

    Args:
        self: Instance to inspect for callable attributes.

    Returns:
        Dictionary mapping method names to bound method objects.
    """
    return {
        name: getattr(self, name)
        for name in dir(self)
        if not (name.startswith("__") and name.endswith("__"))
        and callable(getattr(self, name, None))
    }


def _filter_methods(
    methods: dict[str, CallableT], attribute: str
) -> dict[str, CallableT]:
    """Filter methods by attribute presence, preserving exact callable types.

    Args:
        methods: Dictionary of methods to filter.
        attribute: Attribute name to check for.

    Returns:
        Dictionary containing only methods with the specified attribute.
        The return type matches the input callable type exactly.
    """
    return {
        name: method for name, method in methods.items() if hasattr(method, attribute)
    }


def _register_crew_hooks(instance: CrewInstance, cls: type) -> None:
    """Detect and register crew-scoped hook methods.

    Args:
        instance: Crew instance to register hooks for.
        cls: Crew class type.
    """
    hook_methods = {
        name: method
        for name, method in cls.__dict__.items()
        if any(
            hasattr(method, attr)
            for attr in [
                "is_before_llm_call_hook",
                "is_after_llm_call_hook",
                "is_before_tool_call_hook",
                "is_after_tool_call_hook",
            ]
        )
    }

    if not hook_methods:
        return

    from crewai.hooks import (
        register_after_llm_call_hook,
        register_after_tool_call_hook,
        register_before_llm_call_hook,
        register_before_tool_call_hook,
    )

    instance._registered_hook_functions = []

    instance._hooks_being_registered = True

    for hook_method in hook_methods.values():
        bound_hook = hook_method.__get__(instance, cls)

        has_tool_filter = hasattr(hook_method, "_filter_tools")
        has_agent_filter = hasattr(hook_method, "_filter_agents")

        if hasattr(hook_method, "is_before_llm_call_hook"):
            if has_agent_filter:
                agents_filter = hook_method._filter_agents

                def make_filtered_before_llm(bound_fn, agents_list):
                    def filtered(context):
                        if context.agent and context.agent.role not in agents_list:
                            return None
                        return bound_fn(context)

                    return filtered

                final_hook = make_filtered_before_llm(bound_hook, agents_filter)
            else:
                final_hook = bound_hook

            register_before_llm_call_hook(final_hook)
            instance._registered_hook_functions.append(("before_llm_call", final_hook))

        if hasattr(hook_method, "is_after_llm_call_hook"):
            if has_agent_filter:
                agents_filter = hook_method._filter_agents

                def make_filtered_after_llm(bound_fn, agents_list):
                    def filtered(context):
                        if context.agent and context.agent.role not in agents_list:
                            return None
                        return bound_fn(context)

                    return filtered

                final_hook = make_filtered_after_llm(bound_hook, agents_filter)
            else:
                final_hook = bound_hook

            register_after_llm_call_hook(final_hook)
            instance._registered_hook_functions.append(("after_llm_call", final_hook))

        if hasattr(hook_method, "is_before_tool_call_hook"):
            if has_tool_filter or has_agent_filter:
                tools_filter = getattr(hook_method, "_filter_tools", None)
                agents_filter = getattr(hook_method, "_filter_agents", None)

                def make_filtered_before_tool(bound_fn, tools_list, agents_list):
                    def filtered(context):
                        if tools_list and context.tool_name not in tools_list:
                            return None
                        if (
                            agents_list
                            and context.agent
                            and context.agent.role not in agents_list
                        ):
                            return None
                        return bound_fn(context)

                    return filtered

                final_hook = make_filtered_before_tool(
                    bound_hook, tools_filter, agents_filter
                )
            else:
                final_hook = bound_hook

            register_before_tool_call_hook(final_hook)
            instance._registered_hook_functions.append(("before_tool_call", final_hook))

        if hasattr(hook_method, "is_after_tool_call_hook"):
            if has_tool_filter or has_agent_filter:
                tools_filter = getattr(hook_method, "_filter_tools", None)
                agents_filter = getattr(hook_method, "_filter_agents", None)

                def make_filtered_after_tool(bound_fn, tools_list, agents_list):
                    def filtered(context):
                        if tools_list and context.tool_name not in tools_list:
                            return None
                        if (
                            agents_list
                            and context.agent
                            and context.agent.role not in agents_list
                        ):
                            return None
                        return bound_fn(context)

                    return filtered

                final_hook = make_filtered_after_tool(
                    bound_hook, tools_filter, agents_filter
                )
            else:
                final_hook = bound_hook

            register_after_tool_call_hook(final_hook)
            instance._registered_hook_functions.append(("after_tool_call", final_hook))

    instance._hooks_being_registered = False


def map_all_agent_variables(self: CrewInstance) -> None:
    """Map agent configuration variables to callable instances.

    Args:
        self: Crew instance with agent configurations to map.
    """
    llms = _filter_methods(self._all_methods, "is_llm")
    tool_functions = _filter_methods(self._all_methods, "is_tool")
    cache_handler_functions = _filter_methods(self._all_methods, "is_cache_handler")
    callbacks = _filter_methods(self._all_methods, "is_callback")

    for agent_name, agent_info in self.agents_config.items():
        self._map_agent_variables(
            agent_name=agent_name,
            agent_info=agent_info,
            llms=llms,
            tool_functions=tool_functions,
            cache_handler_functions=cache_handler_functions,
            callbacks=callbacks,
        )


def _map_agent_variables(
    self: CrewInstance,
    agent_name: str,
    agent_info: AgentConfig,
    llms: dict[str, Callable[[], Any]],
    tool_functions: dict[str, Callable[[], BaseTool]],
    cache_handler_functions: dict[str, Callable[[], Any]],
    callbacks: dict[str, Callable[..., Any]],
) -> None:
    """Resolve and map variables for a single agent.

    Args:
        self: Crew instance with agent configurations.
        agent_name: Name of agent to configure.
        agent_info: Agent configuration dictionary with optional fields.
        llms: Dictionary mapping names to LLM factory functions.
        tool_functions: Dictionary mapping names to tool factory functions.
        cache_handler_functions: Dictionary mapping names to cache handler factory functions.
        callbacks: Dictionary of available callbacks.
    """
    if llm := agent_info.get("llm"):
        factory = llms.get(llm)
        self.agents_config[agent_name]["llm"] = factory() if factory else llm

    if tools := agent_info.get("tools"):
        if _is_string_list(tools):
            self.agents_config[agent_name]["tools"] = [
                tool_functions[tool]() for tool in tools
            ]

    if function_calling_llm := agent_info.get("function_calling_llm"):
        factory = llms.get(function_calling_llm)
        self.agents_config[agent_name]["function_calling_llm"] = (
            factory() if factory else function_calling_llm
        )

    if step_callback := agent_info.get("step_callback"):
        self.agents_config[agent_name]["step_callback"] = callbacks[step_callback]()

    if cache_handler := agent_info.get("cache_handler"):
        if _is_string_value(cache_handler):
            self.agents_config[agent_name]["cache_handler"] = cache_handler_functions[
                cache_handler
            ]()


def map_all_task_variables(self: CrewInstance) -> None:
    """Map task configuration variables to callable instances.

    Args:
        self: Crew instance with task configurations to map.
    """
    agents = _filter_methods(self._all_methods, "is_agent")
    tasks = _filter_methods(self._all_methods, "is_task")
    output_json_functions = _filter_methods(self._all_methods, "is_output_json")
    tool_functions = _filter_methods(self._all_methods, "is_tool")
    callback_functions = _filter_methods(self._all_methods, "is_callback")
    output_pydantic_functions = _filter_methods(self._all_methods, "is_output_pydantic")

    for task_name, task_info in self.tasks_config.items():
        self._map_task_variables(
            task_name=task_name,
            task_info=task_info,
            agents=agents,
            tasks=tasks,
            output_json_functions=output_json_functions,
            tool_functions=tool_functions,
            callback_functions=callback_functions,
            output_pydantic_functions=output_pydantic_functions,
        )


def _map_task_variables(
    self: CrewInstance,
    task_name: str,
    task_info: TaskConfig,
    agents: dict[str, Callable[[], Agent]],
    tasks: dict[str, Callable[[], Task]],
    output_json_functions: dict[str, OutputJsonClass[Any]],
    tool_functions: dict[str, Callable[[], BaseTool]],
    callback_functions: dict[str, Callable[..., Any]],
    output_pydantic_functions: dict[str, OutputPydanticClass[Any]],
) -> None:
    """Resolve and map variables for a single task.

    Args:
        self: Crew instance with task configurations.
        task_name: Name of task to configure.
        task_info: Task configuration dictionary with optional fields.
        agents: Dictionary mapping names to agent factory functions.
        tasks: Dictionary mapping names to task factory functions.
        output_json_functions: Dictionary of JSON output class wrappers.
        tool_functions: Dictionary mapping names to tool factory functions.
        callback_functions: Dictionary of available callbacks.
        output_pydantic_functions: Dictionary of Pydantic output class wrappers.
    """
    if context_list := task_info.get("context"):
        self.tasks_config[task_name]["context"] = [
            tasks[context_task_name]() for context_task_name in context_list
        ]

    if tools := task_info.get("tools"):
        if _is_string_list(tools):
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


_CLASS_SETUP_FUNCTIONS: tuple[Callable[[type[CrewClass]], None], ...] = (
    _set_base_directory,
    _set_config_paths,
    _set_mcp_params,
)

_METHODS_TO_INJECT = (
    close_mcp_server,
    get_mcp_tools,
    _load_config,
    load_configurations,
    staticmethod(load_yaml),
    map_all_agent_variables,
    _map_agent_variables,
    map_all_task_variables,
    _map_task_variables,
)


class _CrewBaseType(type):
    """Metaclass for CrewBase that makes it callable as a decorator."""

    def __call__(cls, decorated_cls: type) -> type[CrewClass]:
        """Apply CrewBaseMeta to the decorated class.

        Args:
            decorated_cls: Class to transform with CrewBaseMeta metaclass.

        Returns:
            New class with CrewBaseMeta metaclass applied.
        """
        __name = str(decorated_cls.__name__)
        __bases = tuple(decorated_cls.__bases__)
        __dict = {
            key: value
            for key, value in decorated_cls.__dict__.items()
            if key not in ("__dict__", "__weakref__")
        }
        for slot in __dict.get("__slots__", tuple()):
            __dict.pop(slot, None)
        __dict["__metaclass__"] = CrewBaseMeta
        return cast(type[CrewClass], CrewBaseMeta(__name, __bases, __dict))


class CrewBase(metaclass=_CrewBaseType):
    """Class decorator that applies CrewBaseMeta metaclass.

    Applies CrewBaseMeta metaclass to a class via decorator syntax rather than
    explicit metaclass declaration. Use as @CrewBase instead of
    class Foo(metaclass=CrewBaseMeta).

    Note:
        Reference: https://stackoverflow.com/questions/11091609/setting-a-class-metaclass-using-a-decorator
    """

    # e
    if TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Type stub for decorator usage.

            Args:
                decorated_cls: Class to transform with CrewBaseMeta metaclass.

            Returns:
                New class with CrewBaseMeta metaclass applied.
            """
            ...
