"""Tools handler for managing tool execution and caching."""

from crewai.tools.cache_tools.cache_tools import CacheTools
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.agents.cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage.

    Attributes:
        last_used_tool: The most recently used tool calling instance.
        cache: Optional cache handler for storing tool outputs.
    """

    def __init__(self, cache: CacheHandler | None = None) -> None:
        """Initialize the callback handler.

        Args:
            cache: Optional cache handler for storing tool outputs.
        """
        self.cache: CacheHandler | None = cache
        self.last_used_tool: ToolCalling | InstructorToolCalling | None = None

    def on_tool_use(
        self,
        calling: ToolCalling | InstructorToolCalling,
        output: str,
        should_cache: bool = True,
    ) -> None:
        """Run when tool ends running.

        Args:
            calling: The tool calling instance.
            output: The output from the tool execution.
            should_cache: Whether to cache the tool output.
        """
        self.last_used_tool = calling
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )
