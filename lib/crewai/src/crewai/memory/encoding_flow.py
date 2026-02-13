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
from datetime import datetime
import math
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, start
from crewai.memory.analyze import (
    ConsolidationPlan,
    MemoryAnalysis,
    analyze_for_consolidation,
    analyze_for_save,
)
from crewai.memory.types import MemoryConfig, MemoryRecord, embed_texts


# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------


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
    # Resolved values
    resolved_scope: str = "/"
    resolved_categories: list[str] = Field(default_factory=list)
    resolved_metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_importance: float = 0.5
    resolved_source: str | None = None
    resolved_private: bool = False
    # Embedding
    embedding: list[float] = Field(default_factory=list)
    # Intra-batch dedup
    dropped: bool = False
    # Consolidation
    similar_records: list[MemoryRecord] = Field(default_factory=list)
    top_similarity: float = 0.0
    plan: ConsolidationPlan | None = None
    result_record: MemoryRecord | None = None


class EncodingState(BaseModel):
    """Batch-level state for the encoding flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    items: list[ItemState] = Field(default_factory=list)
    # Aggregate stats
    records_inserted: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    items_dropped_dedup: int = 0


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


class EncodingFlow(Flow[EncodingState]):
    """Batch-native encoding pipeline for memory.remember() / remember_many().

    Processes N items through 5 sequential steps, maximising parallelism:
    - ONE embedder call for all items
    - N concurrent storage searches
    - N concurrent individual LLM calls (field resolution + consolidation)
    - ONE batch re-embed for updates + ONE bulk storage write
    """

    _skip_auto_memory: bool = True

    initial_state = EncodingState

    def __init__(
        self,
        storage: Any,
        llm: Any,
        embedder: Any,
        config: MemoryConfig | None = None,
    ) -> None:
        super().__init__(suppress_flow_events=True)
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()

    # ------------------------------------------------------------------
    # Step 1: Batch embed (ONE embedder call)
    # ------------------------------------------------------------------

    @start()
    def batch_embed(self) -> None:
        """Embed all items in a single embedder call."""
        items = list(self.state.items)
        texts = [item.content for item in items]
        embeddings = embed_texts(self._embedder, texts)
        for item, emb in zip(items, embeddings, strict=False):
            item.embedding = emb

    # ------------------------------------------------------------------
    # Step 2: Intra-batch dedup (cosine similarity matrix)
    # ------------------------------------------------------------------

    @listen(batch_embed)
    def intra_batch_dedup(self) -> None:
        """Drop near-exact duplicates within the batch."""
        items = list(self.state.items)
        if len(items) <= 1:
            return

        threshold = self._config.batch_dedup_threshold
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

    # ------------------------------------------------------------------
    # Step 3: Parallel find similar (concurrent storage searches)
    # ------------------------------------------------------------------

    @listen(intra_batch_dedup)
    def parallel_find_similar(self) -> None:
        """Search storage for similar records, concurrently for all active items."""
        items = list(self.state.items)
        active = [(i, item) for i, item in enumerate(items) if not item.dropped and item.embedding]

        if not active:
            return

        def _search_one(item: ItemState) -> list[tuple[MemoryRecord, float]]:
            scope_prefix = item.scope if item.scope and item.scope.strip("/") else None
            return self._storage.search(
                item.embedding,
                scope_prefix=scope_prefix,
                categories=None,
                limit=self._config.consolidation_limit,
                min_score=0.0,
            )

        if len(active) == 1:
            _, item = active[0]
            raw = _search_one(item)
            item.similar_records = [r for r, _ in raw]
            item.top_similarity = float(raw[0][1]) if raw else 0.0
        else:
            with ThreadPoolExecutor(max_workers=min(len(active), 8)) as pool:
                futures = [(i, item, pool.submit(_search_one, item)) for i, item in active]
                for _, item, future in futures:
                    raw = future.result()
                    item.similar_records = [r for r, _ in raw]
                    item.top_similarity = float(raw[0][1]) if raw else 0.0

    # ------------------------------------------------------------------
    # Step 4: Parallel analyze (N concurrent LLM calls)
    # ------------------------------------------------------------------

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
            existing_scopes = self._storage.list_scopes("/") or ["/"]
            existing_categories = list(
                self._storage.list_categories(scope_prefix=None).keys()
            )

        # Classify items and submit LLM calls
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
                    # Group A: fast path
                    self._apply_defaults(item)
                    item.plan = ConsolidationPlan(actions=[], insert_new=True)
                elif fields_provided and has_similar:
                    # Group B: consolidation only
                    self._apply_defaults(item)
                    consol_futures[i] = pool.submit(
                        analyze_for_consolidation,
                        item.content, list(item.similar_records), self._llm,
                    )
                elif not fields_provided and not has_similar:
                    # Group C: field resolution only
                    save_futures[i] = pool.submit(
                        analyze_for_save,
                        item.content, existing_scopes, existing_categories, self._llm,
                    )
                else:
                    # Group D: both in parallel
                    save_futures[i] = pool.submit(
                        analyze_for_save,
                        item.content, existing_scopes, existing_categories, self._llm,
                    )
                    consol_futures[i] = pool.submit(
                        analyze_for_consolidation,
                        item.content, list(item.similar_records), self._llm,
                    )

            # Collect field-resolution results
            for i, future in save_futures.items():
                analysis = future.result()
                item = items[i]
                item.resolved_scope = item.scope or analysis.suggested_scope or "/"
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

            # Collect consolidation results
            for i, future in consol_futures.items():
                items[i].plan = future.result()
        finally:
            pool.shutdown(wait=False)

    def _apply_defaults(self, item: ItemState) -> None:
        """Apply caller values with config defaults (fast path)."""
        item.resolved_scope = item.scope or "/"
        item.resolved_categories = item.categories or []
        item.resolved_metadata = item.metadata or {}
        item.resolved_importance = (
            item.importance
            if item.importance is not None
            else self._config.default_importance
        )
        item.resolved_source = item.source
        item.resolved_private = item.private

    # ------------------------------------------------------------------
    # Step 5: Execute plans (batch re-embed + bulk insert)
    # ------------------------------------------------------------------

    @listen(parallel_analyze)
    def execute_plans(self) -> None:
        """Apply all consolidation plans with batch re-embedding and bulk insert."""
        items = list(self.state.items)
        now = datetime.utcnow()

        # --- First pass: collect update actions that need re-embedding ---
        update_tasks: list[tuple[int, str, str]] = []  # (item_idx, record_id, new_content)
        for i, item in enumerate(items):
            if item.dropped or item.plan is None:
                continue
            update_tasks.extend(
                (i, action.record_id, action.new_content)
                for action in item.plan.actions
                if action.action == "update" and action.new_content
            )

        # --- Batch re-embed all update contents in ONE call ---
        update_embeddings: list[list[float]] = []
        if update_tasks:
            update_contents = [content for _, _, content in update_tasks]
            update_embeddings = embed_texts(self._embedder, update_contents)

        # Build lookup: (item_idx, record_id) -> embedding
        update_emb_map: dict[tuple[int, str], list[float]] = {}
        for (item_idx, record_id, _), emb in zip(update_tasks, update_embeddings, strict=False):
            update_emb_map[(item_idx, record_id)] = emb

        # --- Second pass: apply all plans ---
        to_insert: list[tuple[int, MemoryRecord]] = []

        for i, item in enumerate(items):
            if item.dropped or item.plan is None:
                continue

            plan = item.plan
            record_by_id = {r.id: r for r in item.similar_records}

            # Apply update/delete actions
            for action in plan.actions:
                if action.action == "delete":
                    self._storage.delete(record_ids=[action.record_id])
                    self.state.records_deleted += 1
                elif action.action == "update" and action.new_content:
                    existing = record_by_id.get(action.record_id)
                    if existing is not None:
                        new_emb = update_emb_map.get((i, action.record_id), [])
                        updated = MemoryRecord(
                            id=existing.id,
                            content=action.new_content,
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
                        record_by_id[action.record_id] = updated

            if plan.insert_new:
                new_record = MemoryRecord(
                    content=item.content,
                    scope=item.resolved_scope,
                    categories=item.resolved_categories,
                    metadata=item.resolved_metadata,
                    importance=item.resolved_importance,
                    embedding=item.embedding if item.embedding else None,
                    source=item.resolved_source,
                    private=item.resolved_private,
                )
                to_insert.append((i, new_record))
            else:
                first_updated = next(
                    (
                        record_by_id[a.record_id]
                        for a in plan.actions
                        if a.action == "update" and a.record_id in record_by_id
                    ),
                    None,
                )
                item.result_record = (
                    first_updated
                    if first_updated is not None
                    else (item.similar_records[0] if item.similar_records else None)
                )

        # Bulk insert all new records at once
        if to_insert:
            records = [r for _, r in to_insert]
            self._storage.save(records)
            self.state.records_inserted += len(records)
            for idx, record in to_insert:
                items[idx].result_record = record
