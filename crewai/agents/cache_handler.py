from typing import Optional

from pydantic import PrivateAttr


class CacheHandler:
    """Callback handler for tool usage."""

    _cache: PrivateAttr = {}

    def __init__(self):
        self._cache = {}

    def add(self, tool, input, output):
        input = input.strip()
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        input = input.strip()
        return self._cache.get(f"{tool}-{input}")
