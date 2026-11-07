from pydantic import BaseModel, Field

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.string_utils import sanitize_tool_name


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
        """Read a formatted cache key without splitting delimiters in its input."""
        tool, tool_input = key.split("tool:", 1)[1].split("|input:", 1)
        return self.cache_handler.read(tool.strip(), tool_input.strip())
