from langchain.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from crewai.agents.cache import CacheHandler


class CacheTools(BaseModel):
    """
    This class represents the default tools for interacting with the cache. It contains a cache handler and a name.
    It also provides a method to hit the cache.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Hit Cache"  # The name of this tool
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",  # The cache handler used by this tool
        default=CacheHandler(),
    )

    def tool(self):
        """
        This method returns a StructuredTool object. The tool is created from the hit_cache function
        and has a name and a description.
        """
        return StructuredTool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        """
        This method is used to read data from the cache. It takes a key as input, which is split to extract the tool name and input.
        It then calls the read method on the cache handler with the tool name and input.
        """
        split = key.split("tool:")  # Split the key on "tool:"
        tool = split[1].split("|input:")[0].strip()  # Extract the tool name
        tool_input = split[1].split("|input:")[1].strip()  # Extract the tool input
        return self.cache_handler.read(tool, tool_input)  # Read from the cache using the tool name and input
