"""Batch-native encoding flow: full save pipeline for one or more memories.

Orchestrates the encoding side of memory in a single Flow with 5 steps:
1. Batch embed (ONE embedder call for all items)
2. Intra-batch dedup (cosine matrix, drop near-exact duplicates)
3. Parallel find similar (concurrent storage searches)
4. Parallel analyze (N concurrent LLM calls -- field resolution + consolidation)
5. Execute plans (batch re-embed updates + bulk insert)
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
from datetime import datetime
import logging
import math
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, start
from crewai.memory.analyze import (
    ConsolidationPlan,
    MemoryAnalysis,
    analyze_for_consolidation,
    analyze_for_save,
)
from crewai.memory.types import MemoryConfig, MemoryRecord, embed_texts
from crewai.memory.utils import join_scope_paths


logger = logging.getLogger(__name__)


class ItemState(BaseModel):
    """Per-item tracking within a batch."""

    content: str = ""
    # Caller-provided (None = infer via LLM)
    scope: str | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] | None = None
    importance: float | None = None
    source: str | None = None
    private: bool = False
    # Structural root scope prefix for hierarchical scoping
    root_scope: str | None = None
    resolved_scope: str = "/"
    resolved_categories: list[str] = Field(default_factory=list)
    resolved_metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_importance: float = 0.5
    resolved_source: str | None = None
    resolved_private: bool = False
    embedding: list[float] = Field(default_factory=list)
    dropped: bool = False
    similar_records: list[MemoryRecord] = Field(default_factory=list)
    top_similarity: float = 0.0
    plan: ConsolidationPlan | None = None
    result_record: MemoryRecord | None = None


class EncodingState(BaseModel):
    """Batch-level state for the encoding flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    items: list[ItemState] = Field(default_factory=list)
    records_inserted: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    items_dropped_dedup: int = 0


class EncodingFlow(Flow[EncodingState]):
    """Batch-native encoding pipeline for memory.remember() / remember_many().

    Processes N items through 5 sequential steps, maximising parallelism:
    - ONE embedder call for all items
    - N concurrent storage searches
    - N concurrent individual LLM calls (field resolution + consolidation)
    - ONE batch re-embed for updates + ONE bulk storage write
    """

    _skip_auto_memory: bool = True

    initial_state: type[EncodingState] = EncodingState

    def __init__(
        self,
        storage: Any,
        llm: Any,
        embedder: Any,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize the encoding flow.

        Args:
            storage: Storage backend for persisting memories.
            llm: LLM instance for analysis.
            embedder: Embedder for generating vectors.
            config: Optional memory configuration.
        """
        super().__init__(suppress_flow_events=True)
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()

    @start()
    def batch_embed(self) -> None:
        """Embed all items in a single embedder call."""
        items = list(self.state.items)
        texts = [item.content for item in items]
        embeddings = embed_texts(self._embedder, texts)
        for item, emb in zip(items, embeddings, strict=False):
            item.embedding = emb

    @listen(batch_embed)
    def intra_batch_dedup(self) -> None:
        """Drop near-exact duplicates within the batch.

        Computes the pairwise cosine-similarity matrix in one vectorized pass
        (normalize rows once, then a single ``X @ Xᵀ`` BLAS call) instead of the
        previous O(n²) loop of pure-Python cosine calls, each of which also
        recomputed both vector norms from scratch (O(n²·d)). The greedy
        "first occurrence wins" selection is preserved exactly: item ``j`` is
        dropped iff some earlier *kept* item is at least ``threshold`` similar.
        """
        items = list(self.state.items)
        if len(items) <= 1:
            return

        threshold = self._config.batch_dedup_threshold

        # Only items carrying an embedding participate; pre-dropped items are
        # excluded so they neither get re-dropped nor suppress others — exactly
        # as the scalar reference skips them.
        active: list[tuple[int, list[float]]] = [
            (idx, item.embedding)
            for idx, item in enumerate(items)
            if item.embedding and not item.dropped
        ]
        if len(active) <= 1:
            return

        dim = len(active[0][1])
        if any(len(emb) != dim for _, emb in active):
            # Ragged embeddings cannot form a matrix; this should not happen for
            # a single embedder, but fall back to the scalar reference so the
            # len-mismatch-as-zero-similarity behavior is preserved exactly.
            self._dedup_scalar(items, threshold)
            return

        matrix = np.asarray([emb for _, emb in active], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0.0
        normalized = np.zeros_like(matrix)
        normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
        # Cosine-similarity matrix; zero-norm rows contribute 0.0, matching
        # _cosine_similarity's zero-norm guard.
        sims = normalized @ normalized.T

        m = len(active)
        dropped = np.zeros(m, dtype=bool)
        for j in range(1, m):
            # Drop j iff an earlier, still-kept item is near-identical.
            if bool((~dropped[:j] & (sims[:j, j] >= threshold)).any()):
                dropped[j] = True

        for local_idx in range(m):
            if dropped[local_idx]:
                items[active[local_idx][0]].dropped = True
                self.state.items_dropped_dedup += 1

    def _dedup_scalar(self, items: list[ItemState], threshold: float) -> None:
        """Reference O(n²) dedup using scalar cosine similarity.

        Retained as the exact behavioral reference and as a fallback for the
        (unexpected) ragged-embedding case.
        """
        n = len(items)
        for j in range(1, n):
            if items[j].dropped or not items[j].embedding:
                continue
            for i in range(j):
                if items[i].dropped or not items[i].embedding:
                    continue
                sim = self._cosine_similarity(items[i].embedding, items[j].embedding)
                if sim >= threshold:
                    items[j].dropped = True
                    self.state.items_dropped_dedup += 1
                    break

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @listen(intra_batch_dedup)
    def parallel_find_similar(self) -> None:
        """Search storage for similar records, concurrently for all active items."""
        items = list(self.state.items)
        active = [
            (i, item)
            for i, item in enumerate(items)
            if not item.dropped and item.embedding
        ]

        if not active:
            return

        def _search_one(
            item: ItemState,
        ) -> list[tuple[MemoryRecord, float]]:
            # Use root_scope as the search boundary, then narrow by explicit scope if provided
            effective_prefix = None
            if item.root_scope:
                effective_prefix = item.root_scope.rstrip("/")
                if item.scope and item.scope.strip("/"):
                    effective_prefix = effective_prefix + "/" + item.scope.strip("/")
            elif item.scope and item.scope.strip("/"):
                effective_prefix = item.scope

            return self._storage.search(  # type: ignore[no-any-return]
                item.embedding,
                scope_prefix=effective_prefix,
                categories=None,
                limit=self._config.consolidation_limit,
                min_score=0.0,
            )

        if len(active) == 1:
            _, item = active[0]
            try:
                raw = _search_one(item)
            except Exception:
                logger.warning(
                    "Storage search failed in parallel_find_similar, "
                    "treating item as new",
                    exc_info=True,
                )
                raw = []
            item.similar_records = [r for r, _ in raw]
            item.top_similarity = float(raw[0][1]) if raw else 0.0
        else:
            with ThreadPoolExecutor(max_workers=min(len(active), 8)) as pool:
                futures = [
                    (
                        i,
                        item,
                        pool.submit(contextvars.copy_context().run, _search_one, item),
                    )
                    for i, item in active
                ]
                for _, item, future in futures:
                    try:
                        raw = future.result()
                    except Exception:
                        logger.warning(
                            "Storage search failed in parallel_find_similar, "
                            "treating item as new",
                            exc_info=True,
                        )
                        raw = []
                    item.similar_records = [r for r, _ in raw]
                    item.top_similarity = float(raw[0][1]) if raw else 0.0

    @listen(parallel_find_similar)
    def parallel_analyze(self) -> None:
        """Field resolution + consolidation via parallel individual LLM calls.

        Classifies each active item into one of four groups:
        - Group A: fields provided + no similar records -> fast insert, 0 LLM calls.
        - Group B: fields provided + similar records above threshold -> 1 consolidation call.
        - Group C: fields missing + no similar records -> 1 field-resolution call.
        - Group D: fields missing + similar records above threshold -> 2 concurrent calls.

        All LLM calls across all items run in parallel via ThreadPoolExecutor.
        """
        items = list(self.state.items)
        threshold = self._config.consolidation_threshold

        # Pre-fetch scope/category info (shared across all field-resolution calls)
        any_needs_fields = any(
            not it.dropped
            and (it.scope is None or it.categories is None or it.importance is None)
            for it in items
        )
        existing_scopes: list[str] = []
        existing_categories: list[str] = []
        if any_needs_fields:
            # Constrain scope/category suggestions to root_scope boundary
            active_root = next(
                (it.root_scope for it in items if not it.dropped and it.root_scope),
                None,
            )
            scope_search_root = active_root if active_root else "/"
            existing_scopes = self._storage.list_scopes(scope_search_root) or ["/"]
            existing_categories = list(
                self._storage.list_categories(scope_prefix=active_root).keys()
            )

        save_futures: dict[int, Future[MemoryAnalysis]] = {}
        consol_futures: dict[int, Future[ConsolidationPlan]] = {}

        pool = ThreadPoolExecutor(max_workers=10)
        try:
            for i, item in enumerate(items):
                if item.dropped:
                    continue

                fields_provided = (
                    item.scope is not None
                    and item.categories is not None
                    and item.importance is not None
                )
                has_similar = item.top_similarity >= threshold

                if fields_provided and not has_similar:
                    self._apply_defaults(item)
                    item.plan = ConsolidationPlan(actions=[], insert_new=True)
                elif fields_provided and has_similar:
                    self._apply_defaults(item)
                    consol_futures[i] = pool.submit(
                        contextvars.copy_context().run,
                        analyze_for_consolidation,
                        item.content,
                        list(item.similar_records),
                        self._llm,
                    )
                elif not fields_provided and not has_similar:
                    save_futures[i] = pool.submit(
                        contextvars.copy_context().run,
                        analyze_for_save,
                        item.content,
                        existing_scopes,
                        existing_categories,
                        self._llm,
                    )
                else:
                    save_futures[i] = pool.submit(
                        contextvars.copy_context().run,
                        analyze_for_save,
                        item.content,
                        existing_scopes,
                        existing_categories,
                        self._llm,
                    )
                    consol_futures[i] = pool.submit(
                        contextvars.copy_context().run,
                        analyze_for_consolidation,
                        item.content,
                        list(item.similar_records),
                        self._llm,
                    )

            for i, future in save_futures.items():
                analysis = future.result()
                item = items[i]
                inner_scope = item.scope or analysis.suggested_scope or "/"
                if item.root_scope:
                    item.resolved_scope = join_scope_paths(item.root_scope, inner_scope)
                else:
                    item.resolved_scope = inner_scope
                item.resolved_categories = (
                    item.categories
                    if item.categories is not None
                    else analysis.categories
                )
                item.resolved_importance = (
                    item.importance
                    if item.importance is not None
                    else analysis.importance
                )
                item.resolved_metadata = dict(
                    item.metadata or {},
                    **(
                        analysis.extracted_metadata.model_dump()
                        if analysis.extracted_metadata
                        else {}
                    ),
                )
                item.resolved_source = item.source
                item.resolved_private = item.private
                # If no consolidation future, it's Group C -> insert
                if i not in consol_futures:
                    item.plan = ConsolidationPlan(actions=[], insert_new=True)

            for i, consol_future in consol_futures.items():
                items[i].plan = consol_future.result()
        finally:
            pool.shutdown(wait=False)

    def _apply_defaults(self, item: ItemState) -> None:
        """Apply caller values with config defaults (fast path).

        If root_scope is set, prepends it to the inner scope to create the
        final resolved_scope.
        """
        inner_scope = item.scope or "/"
        if item.root_scope:
            item.resolved_scope = join_scope_paths(item.root_scope, inner_scope)
        else:
            item.resolved_scope = inner_scope if inner_scope != "/" else "/"

        item.resolved_categories = item.categories or []
        item.resolved_metadata = item.metadata or {}
        item.resolved_importance = (
            item.importance
            if item.importance is not None
            else self._config.default_importance
        )
        item.resolved_source = item.source
        item.resolved_private = item.private

    @listen(parallel_analyze)
    def execute_plans(self) -> None:
        """Apply all consolidation plans with batch re-embedding and bulk insert.

        Actions are deduplicated across items before applying: when multiple
        items reference the same existing record (e.g. both want to delete it),
        only the first action is applied. This prevents LanceDB commit
        conflicts from two operations targeting the same record.
        """
        items = list(self.state.items)
        now = datetime.utcnow()

        # Multiple items may reference the same existing record (because their
        # similar_records overlap). Collect one action per record_id, first wins.
        # Also build a map from record_id to the original MemoryRecord for updates.
        dedup_deletes: set[str] = set()  # record_ids to delete
        dedup_updates: dict[
            str, tuple[int, str]
        ] = {}  # record_id -> (item_idx, new_content)
        all_similar: dict[str, MemoryRecord] = {}  # record_id -> MemoryRecord

        for i, item in enumerate(items):
            if item.dropped or item.plan is None:
                continue
            for r in item.similar_records:
                if r.id not in all_similar:
                    all_similar[r.id] = r
            for action in item.plan.actions:
                rid = action.record_id
                if (
                    action.action == "delete"
                    and rid not in dedup_deletes
                    and rid not in dedup_updates
                ):
                    dedup_deletes.add(rid)
                elif (
                    action.action == "update"
                    and action.new_content
                    and rid not in dedup_deletes
                    and rid not in dedup_updates
                ):
                    dedup_updates[rid] = (i, action.new_content)

        update_list = list(
            dedup_updates.items()
        )  # [(record_id, (item_idx, new_content)), ...]
        update_embeddings: list[list[float]] = []
        if update_list:
            update_contents = [content for _, (_, content) in update_list]
            update_embeddings = embed_texts(self._embedder, update_contents)

        update_emb_map: dict[str, list[float]] = {}
        for (rid, _), emb in zip(update_list, update_embeddings, strict=False):
            update_emb_map[rid] = emb

        # Hold the write lock for the entire delete + update + insert sequence
        # so no other pipeline can interleave and cause version conflicts.
        # The lock is reentrant (RLock), so the individual storage methods
        # can re-acquire it without deadlocking.
        # Collect records to insert (outside lock -- pure data assembly)
        to_insert: list[tuple[int, MemoryRecord]] = []
        for i, item in enumerate(items):
            if item.dropped or item.plan is None:
                continue
            if item.plan.insert_new:
                to_insert.append(
                    (
                        i,
                        MemoryRecord(
                            content=item.content,
                            scope=item.resolved_scope,
                            categories=item.resolved_categories,
                            metadata=item.resolved_metadata,
                            importance=item.resolved_importance,
                            embedding=item.embedding if item.embedding else None,
                            source=item.resolved_source,
                            private=item.resolved_private,
                        ),
                    )
                )

        updated_records: dict[str, MemoryRecord] = {}
        if dedup_deletes:
            self._storage.delete(record_ids=list(dedup_deletes))
            self.state.records_deleted += len(dedup_deletes)

        for rid, (_item_idx, new_content) in dedup_updates.items():
            existing = all_similar.get(rid)
            if existing is not None:
                new_emb = update_emb_map.get(rid, [])
                updated = MemoryRecord(
                    id=existing.id,
                    content=new_content,
                    scope=existing.scope,
                    categories=existing.categories,
                    metadata=existing.metadata,
                    importance=existing.importance,
                    created_at=existing.created_at,
                    last_accessed=now,
                    embedding=new_emb if new_emb else existing.embedding,
                )
                self._storage.update(updated)
                self.state.records_updated += 1
                updated_records[rid] = updated

        if to_insert:
            records = [r for _, r in to_insert]
            self._storage.save(records)
            self.state.records_inserted += len(records)
            for idx, record in to_insert:
                items[idx].result_record = record

        # Set result_record for non-insert items (after lock, using updated_records)
        for _i, item in enumerate(items):
            if item.dropped or item.plan is None or item.plan.insert_new:
                continue
            if item.result_record is not None:
                continue
            first_updated = next(
                (
                    updated_records[a.record_id]
                    for a in item.plan.actions
                    if a.action == "update" and a.record_id in updated_records
                ),
                None,
            )
            item.result_record = (
                first_updated
                if first_updated is not None
                else (item.similar_records[0] if item.similar_records else None)
            )
