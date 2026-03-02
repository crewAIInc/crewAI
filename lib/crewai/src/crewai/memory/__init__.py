"""Memory module: unified Memory with LLM analysis and pluggable storage.

Heavy dependencies are lazily imported so that
``import crewai`` does not initialise at runtime — critical for
Celery pre-fork and similar deployment patterns.
"""

from __future__ import annotations

from typing import Any

from crewai.memory.memory_scope import MemoryScope, MemorySlice
from crewai.memory.types import (
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
    embed_text,
    embed_texts,
)

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Memory": ("crewai.memory.unified_memory", "Memory"),
    "EncodingFlow": ("crewai.memory.encoding_flow", "EncodingFlow"),
}


def __getattr__(name: str) -> Any:
    """Lazily import Memory / EncodingFlow to avoid pulling in lancedb at import time."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EncodingFlow",
    "Memory",
    "MemoryMatch",
    "MemoryRecord",
    "MemoryScope",
    "MemorySlice",
    "ScopeInfo",
    "compute_composite_score",
    "embed_text",
    "embed_texts",
]
