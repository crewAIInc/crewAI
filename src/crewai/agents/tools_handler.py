from typing import Any

from ..tools.cache_tools import CacheTools
from ..tools.tool_calling import ToolCalling
from .cache.cache_handler import CacheHandler


class ToolsHandler:
    """
    This class serves as a callback handler for tool usage. It keeps track of the last used tool and 
    manages the caching of tool usage results.
    """

    last_used_tool: ToolCalling = {}  # Stores the last tool that was used along with its arguments
    cache: CacheHandler  # CacheHandler object for caching tool usage results

    def __init__(self, cache: CacheHandler):
        """
        Initialize the callback handler with a CacheHandler object.

        Parameters:
        cache: A CacheHandler object for caching tool usage results.
        """
        self.cache = cache
        self.last_used_tool = {}

    def on_tool_use(self, calling: ToolCalling, output: str) -> Any:
        """
        This method is called when a tool finishes running. It updates the last used tool and 
        adds the tool usage result to the cache (unless the tool is the CacheTools tool).

        Parameters:
        calling: A ToolCalling object representing the tool usage.
        output: The output of the tool usage.

        Returns:
        None
        """
        self.last_used_tool = calling
        if calling.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )
