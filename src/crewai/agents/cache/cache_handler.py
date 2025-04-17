from typing import Any, Dict, Optional
import threading
from threading import local

from pydantic import BaseModel, PrivateAttr


_thread_local = local()


class CacheHandler(BaseModel):
    """Callback handler for tool usage."""

    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def _get_lock(self):
        """Get a thread-local lock to avoid pickling issues."""
        if not hasattr(_thread_local, "cache_lock"):
            _thread_local.cache_lock = threading.Lock()
        return _thread_local.cache_lock

    def add(self, tool, input, output):
        with self._get_lock():
            self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        with self._get_lock():
            return self._cache.get(f"{tool}-{input}")
