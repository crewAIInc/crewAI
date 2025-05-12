from typing import Any

from crewai.tools.cache_tools.cache_tools import CacheTools
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling

from .cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage."""

    last_used_tool: ToolCalling = {}  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Dict[...]", variable has type "ToolCalling")
    cache: CacheHandler | None

    def __init__(self, cache: CacheHandler | None = None) -> None:
        """Initialize the callback handler."""
        self.cache = cache
        self.last_used_tool = {}  # type: ignore # BUG?: same as above

    def on_tool_use(
        self,
        calling: ToolCalling | InstructorToolCalling,
        output: str,
        should_cache: bool = True,
    ) -> Any:
        """Run when tool ends running."""
        self.last_used_tool = calling  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Union[ToolCalling, InstructorToolCalling]", variable has type "ToolCalling")
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )
