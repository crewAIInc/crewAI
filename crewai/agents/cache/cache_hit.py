from typing import Any

from pydantic import BaseModel, Field

from .cache_handler import CacheHandler


class CacheHit(BaseModel):
    """Cache Hit Object."""

    class Config:
        arbitrary_types_allowed = True

    # Making it Any instead of AgentAction to avoind
    # pydantic v1 vs v2 incompatibility, langchain should
    # soon be updated to pydantic v2
    action: Any = Field(description="Action taken")
    cache: CacheHandler = Field(description="Cache Handler for the tool")
