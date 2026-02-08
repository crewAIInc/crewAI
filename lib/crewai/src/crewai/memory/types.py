"""Data types for the unified memory system."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """A single memory entry stored in the memory system."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the memory record.",
    )
    content: str = Field(description="The textual content of the memory.")
    scope: str = Field(
        default="/",
        description="Hierarchical path organizing the memory (e.g. /company/team/user).",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Categories or tags for the memory.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the memory.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 to 1.0, affects retrieval ranking.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was created.",
    )
    last_accessed: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was last accessed.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for semantic search. Computed on save if not provided.",
    )


class MemoryMatch(BaseModel):
    """A memory record with relevance score from a recall operation."""

    record: MemoryRecord = Field(description="The matched memory record.")
    score: float = Field(
        description="Combined relevance score (semantic, recency, importance).",
    )
    match_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for the match (e.g. semantic, recency, importance).",
    )


class ScopeInfo(BaseModel):
    """Information about a scope in the memory hierarchy."""

    path: str = Field(description="The scope path (e.g. /company/engineering).")
    record_count: int = Field(
        default=0,
        description="Number of records in this scope (including subscopes if applicable).",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Categories used in this scope.",
    )
    oldest_record: datetime | None = Field(
        default=None,
        description="Timestamp of the oldest record in this scope.",
    )
    newest_record: datetime | None = Field(
        default=None,
        description="Timestamp of the newest record in this scope.",
    )
    child_scopes: list[str] = Field(
        default_factory=list,
        description="Immediate child scope paths.",
    )


class MemoryConfig(BaseModel):
    """Configuration for memory retrieval scoring and behavior."""

    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for recency in the combined relevance score.",
    )
    semantic_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity in the combined relevance score.",
    )
    importance_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for explicit importance in the combined relevance score.",
    )
    recency_half_life_days: int = Field(
        default=30,
        ge=1,
        description="For exponential decay: score halves every N days.",
    )


def compute_composite_score(
    record: MemoryRecord,
    semantic_score: float,
    config: MemoryConfig,
) -> tuple[float, list[str]]:
    """Compute a weighted composite relevance score from semantic, recency, and importance.

    composite = w_semantic * semantic + w_recency * decay + w_importance * importance
    where decay = 0.5^(age_days / half_life_days).

    Args:
        record: The memory record (provides created_at and importance).
        semantic_score: Raw semantic similarity from vector search, in [0, 1].
        config: Weights and recency half-life.

    Returns:
        Tuple of (composite_score, match_reasons). match_reasons includes
        "semantic" always; "recency" if decay > 0.5; "importance" if record.importance > 0.5.
    """
    age_seconds = (datetime.utcnow() - record.created_at).total_seconds()
    age_days = max(age_seconds / 86400.0, 0.0)
    decay = 0.5 ** (age_days / config.recency_half_life_days)

    composite = (
        config.semantic_weight * semantic_score
        + config.recency_weight * decay
        + config.importance_weight * record.importance
    )

    reasons: list[str] = ["semantic"]
    if decay > 0.5:
        reasons.append("recency")
    if record.importance > 0.5:
        reasons.append("importance")

    return composite, reasons
