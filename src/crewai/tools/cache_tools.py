from langchain.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from crewai.agents.cache import CacheHandler


class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",
        default=CacheHandler(),
    )

    def tool(self):
        return StructuredTool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
