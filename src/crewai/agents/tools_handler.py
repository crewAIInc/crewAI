from typing import Any

from ..tools.cache_tools import CacheTools
from ..tools.tool_calling import ToolCalling
from .cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage."""

    last_used_tool: ToolCalling = {}
    cache: CacheHandler

    def __init__(self, cache: CacheHandler):
        """Initialize the callback handler."""
        self.cache = cache
        self.last_used_tool = {}

    def on_tool_start(self, calling: ToolCalling) -> Any:
        """Run when tool starts running."""
        self.last_used_tool = calling

    def on_tool_end(self, calling: ToolCalling, output: str) -> Any:
        """Run when tool ends running."""
        if self.last_used_tool.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )
