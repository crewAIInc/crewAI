"""Memory module: unified Memory with LLM analysis and pluggable storage."""

from crewai.memory.encoding_flow import EncodingFlow
from crewai.memory.memory_scope import MemoryScope, MemorySlice
from crewai.memory.types import (
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
    embed_text,
    embed_texts,
)
from crewai.memory.unified_memory import Memory


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
