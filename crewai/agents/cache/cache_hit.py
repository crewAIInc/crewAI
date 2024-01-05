from langchain_core.agents import AgentAction
from pydantic.v1 import BaseModel, Field

from .cache_handler import CacheHandler


class CacheHit(BaseModel):
    """Cache Hit Object."""

    class Config:
        arbitrary_types_allowed = True

    action: AgentAction = Field(description="Action taken")
    cache: CacheHandler = Field(description="Cache Handler for the tool")
