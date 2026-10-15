from typing import TYPE_CHECKING, Any

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction, AgentFinish, OutputParserError, parse
from crewai.agents.tools_handler import ToolsHandler


if TYPE_CHECKING:
    from crewai.agents.agent_executor import AgentExecutor, AgentExecutorState


__all__ = [
    "AgentAction",
    "AgentExecutor",
    "AgentExecutorState",
    "AgentFinish",
    "CacheHandler",
    "OutputParserError",
    "ToolsHandler",
    "parse",
]

_LAZY_EXPORTS = {
    "AgentExecutor": ("crewai.agents.agent_executor", "AgentExecutor"),
    "AgentExecutorState": ("crewai.agents.agent_executor", "AgentExecutorState"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
