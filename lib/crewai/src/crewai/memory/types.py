"""Data types for the unified memory system."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# When searching the vector store, we ask for more results than the caller
# requested so that post-search steps (composite scoring, deduplication,
# category filtering) have enough candidates to fill the final result set.
# For example, if the caller asks for 10 results and this is 2, we fetch 20
# from the vector store and then trim down after scoring.
_RECALL_OVERSAMPLE_FACTOR = 2


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
    source: str | None = Field(
        default=None,
        description=(
            "Origin of this memory (e.g. user ID, session ID). "
            "Used for provenance tracking and privacy filtering."
        ),
    )
    private: bool = Field(
        default=False,
        description=(
            "If True, this memory is only visible to recall requests from the same source, "
            "or when include_private=True is passed."
        ),
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
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Information the system looked for but could not find.",
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
    """Internal configuration for memory scoring, consolidation, and recall behavior.

    Users configure these values via ``Memory(...)`` keyword arguments.
    This model is not part of the public API -- it exists so that the config
    can be passed as a single object to RecallFlow, EncodingFlow, and
    compute_composite_score.
    """

    # -- Composite score weights --
    # The recall composite score is:
    #   semantic_weight * similarity + recency_weight * decay + importance_weight * importance
    # These should sum to ~1.0 for intuitive 0-1 scoring.

    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for recency in the composite relevance score. "
            "Higher values favor recently created memories over older ones."
        ),
    )
    semantic_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for semantic similarity in the composite relevance score. "
            "Higher values make recall rely more on vector-search closeness."
        ),
    )
    importance_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for explicit importance in the composite relevance score. "
            "Higher values make high-importance memories surface more often."
        ),
    )
    recency_half_life_days: int = Field(
        default=30,
        ge=1,
        description=(
            "Number of days for the recency score to halve (exponential decay). "
            "Lower values make memories lose relevance faster; higher values "
            "keep old memories relevant longer."
        ),
    )

    # -- Consolidation (on save) --

    consolidation_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic similarity above which the consolidation flow is triggered "
            "when saving new content. The LLM then decides whether to merge, "
            "update, or delete overlapping records. Set to 1.0 to disable."
        ),
    )
    consolidation_limit: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum number of existing records to compare against when checking "
            "for consolidation during a save."
        ),
    )
    batch_dedup_threshold: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold for dropping near-exact duplicates "
            "within a single remember_many() batch. Only items with similarity "
            ">= this value are dropped. Set very high (0.98) to avoid "
            "discarding useful memories that are merely similar."
        ),
    )

    # -- Save defaults --

    default_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Importance assigned to new memories when no explicit value is given "
            "and the LLM analysis path is skipped (i.e. all fields provided by "
            "the caller)."
        ),
    )

    # -- Recall depth control --
    # The RecallFlow router uses these thresholds to decide between returning
    # results immediately ("synthesize") and doing an extra LLM-driven
    # exploration round ("explore_deeper").

    confidence_threshold_high: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "When recall confidence is at or above this value, results are "
            "returned directly without deeper exploration."
        ),
    )
    confidence_threshold_low: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "When recall confidence is below this value and exploration budget "
            "remains, a deeper LLM-driven exploration round is triggered."
        ),
    )
    complex_query_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "For queries classified as 'complex' by the LLM, deeper exploration "
            "is triggered when confidence is below this value."
        ),
    )
    exploration_budget: int = Field(
        default=1,
        ge=0,
        description=(
            "Number of LLM-driven exploration rounds allowed during deep recall. "
            "0 means recall always uses direct vector search only; higher values "
            "allow more thorough but slower retrieval."
        ),
    )
    recall_oversample_factor: int = Field(
        default=_RECALL_OVERSAMPLE_FACTOR,
        ge=1,
        description=(
            "When searching the vector store, fetch this many times more results "
            "than the caller requested so that post-search steps (composite "
            "scoring, deduplication, category filtering) have enough candidates "
            "to fill the final result set."
        ),
    )
    query_analysis_threshold: int = Field(
        default=250,
        ge=0,
        description=(
            "Character count threshold for LLM query analysis during deep recall. "
            "Queries shorter than this are embedded directly without an LLM call "
            "to distill sub-queries or infer scopes (saving ~1-3s). Longer queries "
            "(e.g. full task descriptions) benefit from LLM distillation. "
            "Set to 0 to always use LLM analysis."
        ),
    )


def embed_text(embedder: Any, text: str) -> list[float]:
    """Embed a single text string and return a list of floats.

    Args:
        embedder: Callable that accepts a list of strings and returns embeddings.
        text: The text to embed.

    Returns:
        List of floats representing the embedding, or empty list on failure.
    """
    if not text or not text.strip():
        return []
    result = embedder([text])
    if not result:
        return []
    first = result[0]
    if hasattr(first, "tolist"):
        return first.tolist()
    if isinstance(first, list):
        return [float(x) for x in first]
    return list(first)


def embed_texts(embedder: Any, texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call.

    The embedder already accepts ``list[str]``, so this just calls it once
    with the full batch and normalises the output format.

    Args:
        embedder: Callable that accepts a list of strings and returns embeddings.
        texts: List of texts to embed.

    Returns:
        List of embeddings, one per input text. Empty texts produce empty lists.
    """
    if not texts:
        return []
    # Filter out empty texts, remembering their positions
    valid: list[tuple[int, str]] = [
        (i, t) for i, t in enumerate(texts) if t and t.strip()
    ]
    if not valid:
        return [[] for _ in texts]

    result = embedder([t for _, t in valid])
    embeddings: list[list[float]] = [[] for _ in texts]
    for (orig_idx, _), emb in zip(valid, result, strict=False):
        if hasattr(emb, "tolist"):
            embeddings[orig_idx] = emb.tolist()
        elif isinstance(emb, list):
            embeddings[orig_idx] = [float(x) for x in emb]
        else:
            embeddings[orig_idx] = list(emb)
    return embeddings


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
