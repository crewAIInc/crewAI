"""Memory module: unified Memory with LLM analysis and pluggable storage."""

from crewai.memory.memory_scope import MemoryScope, MemorySlice
from crewai.memory.unified_memory import Memory
from crewai.memory.types import (
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
)

__all__ = [
    "Memory",
    "MemoryConfig",
    "MemoryMatch",
    "MemoryRecord",
    "MemoryScope",
    "MemorySlice",
    "ScopeInfo",
    "compute_composite_score",
]
