from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentExecutorProtocol(Protocol):
    """Protocol defining the expected interface for an agent executor."""

    @property
    def agent(self) -> Any: ...

    @property
    def task(self) -> Any: ...
