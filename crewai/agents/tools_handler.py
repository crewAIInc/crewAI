from typing import Any, Dict

from langchain.callbacks.base import BaseCallbackHandler

from ..tools.cache_tools import CacheTools
from .cache.cache_handler import CacheHandler


class ToolsHandler(BaseCallbackHandler):
    """Callback handler for tool usage."""

    last_used_tool: Dict[str, Any] = {}
    cache: CacheHandler = None

    def __init__(self, cache: CacheHandler = None, **kwargs: Any):
        """Initialize the callback handler."""
        self.cache = cache
        super().__init__(**kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        name = serialized.get("name")
        if name not in ["invalid_tool", "_Exception"]:
            tools_usage = {
                "tool": name,
                "input": input_str,
            }
            self.last_used_tool = tools_usage

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if (
            "is not a valid tool" not in output
            and "Invalid or incomplete response" not in output
            and "Invalid Format" not in output
        ):
            if self.last_used_tool["tool"] != CacheTools().name:
                self.cache.add(
                    tool=self.last_used_tool["tool"],
                    input=self.last_used_tool["input"],
                    output=output,
                )
