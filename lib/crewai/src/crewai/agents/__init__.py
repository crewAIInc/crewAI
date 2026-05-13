from typing import TYPE_CHECKING, Any

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction, AgentFinish, OutputParserError, parse
from crewai.agents.tools_handler import ToolsHandler


if TYPE_CHECKING:
    from crewai.agents.crew_agent_executor import CrewAgentExecutor


__all__ = [
    "AgentAction",
    "AgentFinish",
    "CacheHandler",
    "CrewAgentExecutor",
    "OutputParserError",
    "ToolsHandler",
    "parse",
]


def __getattr__(name: str) -> Any:
    if name == "CrewAgentExecutor":
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        return CrewAgentExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
