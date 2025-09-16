"""Type protocols for LangGraph modules."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LangGraphMemorySaver(Protocol):
    """Protocol for LangGraph MemorySaver.

    Defines the interface for LangGraph's memory persistence mechanism.
    """

    def __init__(self) -> None:
        """Initialize the memory saver."""
        ...


@runtime_checkable
class LangGraphCheckPointMemoryModule(Protocol):
    """Protocol for LangGraph checkpoint memory module.

    Defines the interface for modules containing memory checkpoint functionality.
    """

    MemorySaver: type[LangGraphMemorySaver]


@runtime_checkable
class LangGraphPrebuiltModule(Protocol):
    """Protocol for LangGraph prebuilt module.

    Defines the interface for modules containing prebuilt agent factories.
    """

    def create_react_agent(
        self,
        model: Any,
        tools: list[Any],
        checkpointer: Any,
        debug: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a ReAct agent with the given configuration.

        Args:
            model: The language model to use for the agent.
            tools: List of tools available to the agent.
            checkpointer: Memory checkpointer for state persistence.
            debug: Whether to enable debug mode.
            **kwargs: Additional configuration options.

        Returns:
            The configured ReAct agent instance.
        """
        ...
