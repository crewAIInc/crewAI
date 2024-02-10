from typing import Optional


class CacheHandler:
    """Callback handler for tool usage."""

    _cache: dict = {}

    def __init__(self):
        self._cache = {}

    def add(self, tool, input, output):
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        return self._cache.get(f"{tool}-{input}")
