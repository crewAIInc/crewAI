from typing import Any, Optional, Union

from ..tools.cache_tools.cache_tools import CacheTools
from ..tools.tool_calling import InstructorToolCalling, ToolCalling
from .cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage."""

    last_used_tool: ToolCalling = {}  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Dict[...]", variable has type "ToolCalling")
    cache: Optional[CacheHandler]

    def __init__(self, cache: Optional[CacheHandler] = None):
        """Initialize the callback handler."""
        self.cache = cache
        self.last_used_tool = {}  # type: ignore # BUG?: same as above

    def on_tool_use(
        self,
        calling: Union[ToolCalling, InstructorToolCalling],
        output: str,
        should_cache: bool = True,
        agentops: Optional[Any] = None,
    ) -> Any:
        """Run when tool ends running."""
        self.last_used_tool = calling  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Union[ToolCalling, InstructorToolCalling]", variable has type "ToolCalling")

        if agentops:
            agentops.record(
                agentops.ActionEvent(
                    name=calling.tool_name,
                    action_type="on_tool_use",
                    params=calling.arguments,
                    returns=output,
                    logs={
                        "tool": calling.tool_name,
                        "tool_calling": calling,
                        "should_cache": should_cache,
                    },
                )
            )

        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )
