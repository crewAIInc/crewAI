"""Type protocols for OpenAI agents modules."""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict, runtime_checkable

from crewai.tools.base_tool import BaseTool


class AgentKwargs(TypedDict, total=False):
    """Typed dict for agent initialization kwargs."""

    role: str
    goal: str
    backstory: str
    model: str
    tools: list[BaseTool] | None
    agent_config: dict[str, Any] | None


@runtime_checkable
class OpenAIAgent(Protocol):
    """Protocol for OpenAI Agent."""

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI agent."""
        ...

    tools: list[Any]
    output_type: Any


@runtime_checkable
class OpenAIRunner(Protocol):
    """Protocol for OpenAI Runner."""

    @classmethod
    def run_sync(cls, agent: OpenAIAgent, message: str) -> Any:
        """Run agent synchronously with a message."""
        ...


@runtime_checkable
class OpenAIAgentsModule(Protocol):
    """Protocol for OpenAI agents module."""

    Agent: type[OpenAIAgent]
    Runner: type[OpenAIRunner]
    enable_verbose_stdout_logging: Callable[[], None]


@runtime_checkable
class OpenAITool(Protocol):
    """Protocol for OpenAI Tool."""


@runtime_checkable
class OpenAIFunctionTool(Protocol):
    """Protocol for OpenAI FunctionTool."""

    def __init__(
        self,
        name: str,
        description: str,
        params_json_schema: dict[str, Any],
        on_invoke_tool: Any,
    ) -> None:
        """Initialize the function tool."""
        ...
