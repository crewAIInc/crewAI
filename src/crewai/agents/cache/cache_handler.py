from typing import Any

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    """Callback handler for tool usage."""

    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool, input, output) -> None:
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> str | None:
        return self._cache.get(f"{tool}-{input}")
