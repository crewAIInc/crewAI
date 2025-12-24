from pydantic import BaseModel, Field

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.structured_tool import CrewStructuredTool


class CacheTools(BaseModel):
    """Standaard tools om de cache te raadplegen."""

    name: str = "Cache Raadplegen"
    cache_handler: CacheHandler = Field(
        description="Cache Handler voor de crew",
        default_factory=CacheHandler,
    )

    def tool(self):
        return CrewStructuredTool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Leest direct uit de cache",
        )

    def hit_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
