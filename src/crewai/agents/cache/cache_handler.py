from typing import Any, Dict, Optional

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    """Callback handler for tool usage."""

    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool, input, output):
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        return self._cache.get(f"{tool}-{input}")
