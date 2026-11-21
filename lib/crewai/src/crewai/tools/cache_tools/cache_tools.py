from pydantic import BaseModel, Field

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.string_utils import sanitize_tool_name


_TOOL_MARKER = "tool:"
_INPUT_MARKER = "|input:"


class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",
        default_factory=CacheHandler,
    )

    def tool(self) -> CrewStructuredTool:
        return CrewStructuredTool.from_function(
            func=self.hit_cache,
            name=sanitize_tool_name(self.name),
            description="Reads directly from the cache",
        )

    def hit_cache(self, key: str) -> str | None:
        """Read a cached tool result from a composite cache key.

        The key is the ``tool:<tool name>|input:<tool input>`` form. Any other
        key -- the plain ``<tool>-<input>`` key the handler stores, a bare tool
        name, or an empty string -- is a cache miss, matching
        :meth:`CacheHandler.read`, instead of an IndexError that aborts the
        tool call.

        Args:
            key: The composite cache key.

        Returns:
            The cached result, or None when the key names no cached result.
        """
        if not key:
            return None

        _, tool_marker, remainder = key.partition(_TOOL_MARKER)
        if not tool_marker:
            return None

        tool, input_marker, tool_input = remainder.partition(_INPUT_MARKER)
        if not input_marker:
            return None

        tool = tool.strip()
        if not tool:
            return None

        return self.cache_handler.read(tool, tool_input.strip())
