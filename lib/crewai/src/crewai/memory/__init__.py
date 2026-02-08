"""Memory module: unified Memory with LLM analysis and pluggable storage."""

from crewai.memory.consolidation_flow import ConsolidationFlow
from crewai.memory.memory_scope import MemoryScope, MemorySlice
from crewai.memory.unified_memory import Memory
from crewai.memory.types import (
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
    embed_text,
)

__all__ = [
    "ConsolidationFlow",
    "Memory",
    "MemoryMatch",
    "MemoryRecord",
    "MemoryScope",
    "MemorySlice",
    "ScopeInfo",
    "compute_composite_score",
    "embed_text",
]
