"""Unified Memory class: single intelligent memory with LLM analysis and pluggable storage."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import threading
import time
from typing import Any, Literal

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.llms.base_llm import BaseLLM
from crewai.memory.analyze import extract_memories_from_content
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.storage.backend import StorageBackend
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import (
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
    embed_text,
)


def _default_embedder() -> Any:
    """Build default OpenAI embedder for memory."""
    from crewai.rag.embeddings.factory import build_embedder

    return build_embedder({"provider": "openai", "config": {}})


class Memory:
    """Unified memory: standalone, LLM-analyzed, with intelligent recall flow.

    Works without agent/crew. Uses LLM to infer scope, categories, importance on save.
    Uses RecallFlow for adaptive-depth recall. Supports scope/slice views and
    pluggable storage (LanceDB default).
    """

    def __init__(
        self,
        llm: BaseLLM | str = "gpt-4o-mini",
        storage: StorageBackend | str = "lancedb",
        embedder: Any = None,
        # -- Scoring weights --
        # These three weights control how recall results are ranked.
        # The composite score is: semantic_weight * similarity + recency_weight * decay + importance_weight * importance.
        # They should sum to ~1.0 for intuitive scoring.
        recency_weight: float = 0.3,
        semantic_weight: float = 0.5,
        importance_weight: float = 0.2,
        # How quickly old memories lose relevance. The recency score halves every
        # N days (exponential decay). Lower = faster forgetting; higher = longer relevance.
        recency_half_life_days: int = 30,
        # -- Consolidation --
        # When remembering new content, if an existing record has similarity >= this
        # threshold, the LLM is asked to merge/update/delete. Set to 1.0 to disable.
        consolidation_threshold: float = 0.85,
        # Max existing records to compare against when checking for consolidation.
        consolidation_limit: int = 5,
        # -- Save defaults --
        # Importance assigned to new memories when no explicit value is given and
        # the LLM analysis path is skipped (all fields provided by the caller).
        default_importance: float = 0.5,
        # -- Recall depth control --
        # These thresholds govern the RecallFlow router that decides between
        # returning results immediately ("synthesize") vs. doing an extra
        # LLM-driven exploration round ("explore_deeper").
        #   confidence >= confidence_threshold_high  => always synthesize
        #   confidence <  confidence_threshold_low   => explore deeper (if budget > 0)
        #   complex query + confidence < complex_query_threshold => explore deeper
        confidence_threshold_high: float = 0.8,
        confidence_threshold_low: float = 0.5,
        complex_query_threshold: float = 0.7,
        # How many LLM-driven exploration rounds the RecallFlow is allowed to run.
        # 0 = always shallow (vector search only); higher = more thorough but slower.
        exploration_budget: int = 1,
        # Queries shorter than this skip LLM analysis (saving ~1-3s).
        # Longer queries (full task descriptions) benefit from LLM distillation.
        query_analysis_threshold: int = 200,
    ) -> None:
        """Initialize Memory.

        Args:
            llm: LLM for analysis (model name or BaseLLM instance).
            storage: Backend: "lancedb" or a StorageBackend instance.
            embedder: Embedding callable, provider config dict, or None (default OpenAI).
            recency_weight: Weight for recency in the composite relevance score.
            semantic_weight: Weight for semantic similarity in the composite relevance score.
            importance_weight: Weight for importance in the composite relevance score.
            recency_half_life_days: Recency score halves every N days (exponential decay).
            consolidation_threshold: Similarity above which consolidation is triggered on save.
            consolidation_limit: Max existing records to compare during consolidation.
            default_importance: Default importance when not provided or inferred.
            confidence_threshold_high: Recall confidence above which results are returned directly.
            confidence_threshold_low: Recall confidence below which deeper exploration is triggered.
            complex_query_threshold: For complex queries, explore deeper below this confidence.
            exploration_budget: Number of LLM-driven exploration rounds during deep recall.
            query_analysis_threshold: Queries shorter than this skip LLM analysis during deep recall.
        """
        self._config = MemoryConfig(
            recency_weight=recency_weight,
            semantic_weight=semantic_weight,
            importance_weight=importance_weight,
            recency_half_life_days=recency_half_life_days,
            consolidation_threshold=consolidation_threshold,
            consolidation_limit=consolidation_limit,
            default_importance=default_importance,
            confidence_threshold_high=confidence_threshold_high,
            confidence_threshold_low=confidence_threshold_low,
            complex_query_threshold=complex_query_threshold,
            exploration_budget=exploration_budget,
            query_analysis_threshold=query_analysis_threshold,
        )

        # Store raw config for lazy initialization. LLM and embedder are only
        # built on first access so that Memory() never fails at construction
        # time (e.g. when auto-created by Flow without an API key set).
        self._llm_config: BaseLLM | str = llm
        self._llm_instance: BaseLLM | None = None if isinstance(llm, str) else llm
        self._embedder_config: Any = embedder
        self._embedder_instance: Any = (
            embedder if (embedder is not None and not isinstance(embedder, dict)) else None
        )

        # Storage is initialized eagerly (local, no API key needed).
        if storage == "lancedb":
            self._storage = LanceDBStorage()
        elif isinstance(storage, str):
            self._storage = LanceDBStorage(path=storage)
        else:
            self._storage = storage

        # Background save queue. max_workers=1 serializes saves to avoid
        # concurrent storage mutations (two saves finding the same similar
        # record and both trying to update/delete it). Within each save,
        # the parallel LLM calls still run on their own thread pool.
        self._save_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memory-save"
        )
        self._pending_saves: list[Future[Any]] = []
        self._pending_lock = threading.Lock()

    _MEMORY_DOCS_URL = "https://docs.crewai.com/concepts/memory"

    @property
    def _llm(self) -> BaseLLM:
        """Lazy LLM initialization -- only created when first needed."""
        if self._llm_instance is None:
            from crewai.llm import LLM

            try:
                self._llm_instance = LLM(model=self._llm_config)
            except Exception as e:
                raise RuntimeError(
                    f"Memory requires an LLM for analysis but initialization failed: {e}\n\n"
                    "To fix this, do one of the following:\n"
                    '  - Set OPENAI_API_KEY for the default model (gpt-4o-mini)\n'
                    '  - Pass a different model: Memory(llm="anthropic/claude-3-haiku-20240307")\n'
                    '  - Pass any LLM instance: Memory(llm=LLM(model="your-model"))\n'
                    "  - To skip LLM analysis, pass all fields explicitly to remember()\n"
                    '    and use depth="shallow" for recall.\n\n'
                    f"Docs: {self._MEMORY_DOCS_URL}"
                ) from e
        return self._llm_instance

    @property
    def _embedder(self) -> Any:
        """Lazy embedder initialization -- only created when first needed."""
        if self._embedder_instance is None:
            try:
                if isinstance(self._embedder_config, dict):
                    from crewai.rag.embeddings.factory import build_embedder

                    self._embedder_instance = build_embedder(self._embedder_config)
                else:
                    self._embedder_instance = _default_embedder()
            except Exception as e:
                raise RuntimeError(
                    f"Memory requires an embedder for vector search but initialization failed: {e}\n\n"
                    "To fix this, do one of the following:\n"
                    "  - Set OPENAI_API_KEY for the default embedder (text-embedding-3-small)\n"
                    '  - Pass a different embedder: Memory(embedder={{"provider": "google", "config": {{...}}}})\n'
                    "  - Pass a callable: Memory(embedder=my_embedding_function)\n\n"
                    f"Docs: {self._MEMORY_DOCS_URL}"
                ) from e
        return self._embedder_instance

    # ------------------------------------------------------------------
    # Background write queue
    # ------------------------------------------------------------------

    def _submit_save(self, fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a save operation to the background thread pool.

        The future is tracked so that ``drain_writes()`` can wait for it.
        If the pool has been shut down (e.g. after ``close()``), the save
        runs synchronously as a fallback so late saves still succeed.
        """
        try:
            future: Future[Any] = self._save_pool.submit(fn, *args, **kwargs)
        except RuntimeError:
            # Pool shut down -- run synchronously as fallback
            future = Future()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            return future
        with self._pending_lock:
            self._pending_saves.append(future)
        future.add_done_callback(self._on_save_done)
        return future

    def _on_save_done(self, future: Future[Any]) -> None:
        """Remove a completed future from the pending list and emit failure event if needed."""
        with self._pending_lock:
            try:
                self._pending_saves.remove(future)
            except ValueError:
                pass  # already removed
        exc = future.exception()
        if exc is not None:
            crewai_event_bus.emit(
                self,
                MemorySaveFailedEvent(
                    value="background save",
                    error=str(exc),
                    source_type="unified_memory",
                ),
            )

    def drain_writes(self) -> None:
        """Block until all pending background saves have completed.

        Called automatically by ``recall()`` and should be called by the
        crew at shutdown to ensure no saves are lost.
        """
        with self._pending_lock:
            pending = list(self._pending_saves)
        for future in pending:
            future.result()  # blocks until done; re-raises exceptions

    def close(self) -> None:
        """Drain pending saves and shut down the background thread pool."""
        self.drain_writes()
        self._save_pool.shutdown(wait=True)

    def _encode_batch(
        self,
        contents: list[str],
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
    ) -> list[MemoryRecord]:
        """Run the batch EncodingFlow for one or more items. No event emission.

        This is the core encoding logic shared by ``remember()`` and
        ``remember_many()``. Events are managed by the calling method.
        """
        from crewai.memory.encoding_flow import EncodingFlow

        flow = EncodingFlow(
            storage=self._storage,
            llm=self._llm,
            embedder=self._embedder,
            config=self._config,
        )
        items_input = [
            {
                "content": c,
                "scope": scope,
                "categories": categories,
                "metadata": metadata,
                "importance": importance,
                "source": source,
                "private": private,
            }
            for c in contents
        ]
        flow.kickoff(inputs={"items": items_input})
        return [
            item.result_record
            for item in flow.state.items
            if not item.dropped and item.result_record is not None
        ]

    def remember(
        self,
        content: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
    ) -> MemoryRecord:
        """Store a single item in memory (synchronous).

        Routes through the same serialized save pool as ``remember_many``
        to prevent races, but blocks until the save completes so the caller
        gets the ``MemoryRecord`` back immediately.

        Args:
            content: Text to remember.
            scope: Optional scope path; inferred if None.
            categories: Optional categories; inferred if None.
            metadata: Optional metadata; merged with LLM-extracted if inferred.
            importance: Optional importance 0-1; inferred if None.
            source: Optional provenance identifier (e.g. user ID, session ID).
            private: If True, only visible to recall from the same source.
            agent_role: Optional agent role for event metadata.

        Returns:
            The created MemoryRecord.

        Raises:
            Exception: On save failure (events emitted).
        """
        _source_type = "unified_memory"
        try:
            crewai_event_bus.emit(
                self,
                MemorySaveStartedEvent(
                    value=content,
                    metadata=metadata,
                    source_type=_source_type,
                ),
            )
            start = time.perf_counter()

            # Submit through the save pool for proper serialization,
            # then immediately wait for the result.
            future = self._submit_save(
                self._encode_batch,
                [content], scope, categories, metadata, importance, source, private,
            )
            records = future.result()
            record = records[0] if records else None

            elapsed_ms = (time.perf_counter() - start) * 1000
            crewai_event_bus.emit(
                self,
                MemorySaveCompletedEvent(
                    value=content,
                    metadata=metadata or {},
                    agent_role=agent_role,
                    save_time_ms=elapsed_ms,
                    source_type=_source_type,
                ),
            )
            return record
        except Exception as e:
            crewai_event_bus.emit(
                self,
                MemorySaveFailedEvent(
                    value=content,
                    metadata=metadata,
                    error=str(e),
                    source_type=_source_type,
                ),
            )
            raise

    def remember_many(
        self,
        contents: list[str],
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
    ) -> list[MemoryRecord]:
        """Store multiple items in memory (non-blocking).

        The encoding pipeline runs in a background thread. This method
        returns immediately so the caller (e.g. agent) is not blocked.
        A ``MemorySaveStartedEvent`` is emitted immediately; the
        ``MemorySaveCompletedEvent`` is emitted when the background
        save finishes.

        Any subsequent ``recall()`` call will automatically wait for
        pending saves to complete before searching (read barrier).

        Args:
            contents: List of text items to remember.
            scope: Optional scope applied to all items.
            categories: Optional categories applied to all items.
            metadata: Optional metadata applied to all items.
            importance: Optional importance applied to all items.
            source: Optional provenance identifier applied to all items.
            private: Privacy flag applied to all items.
            agent_role: Optional agent role for event metadata.

        Returns:
            Empty list (records are not available until the background save completes).
        """
        if not contents:
            return []

        self._submit_save(
            self._background_encode_batch,
            contents, scope, categories, metadata,
            importance, source, private, agent_role,
        )
        return []

    def _background_encode_batch(
        self,
        contents: list[str],
        scope: str | None,
        categories: list[str] | None,
        metadata: dict[str, Any] | None,
        importance: float | None,
        source: str | None,
        private: bool,
        agent_role: str | None,
    ) -> list[MemoryRecord]:
        """Run the encoding pipeline in a background thread with event emission.

        Both started and completed events are emitted here (in the background
        thread) so they pair correctly on the event bus scope stack.
        """
        crewai_event_bus.emit(
            self,
            MemorySaveStartedEvent(
                value=f"{len(contents)} memories (background)",
                metadata=metadata,
                source_type="unified_memory",
            ),
        )
        start = time.perf_counter()
        records = self._encode_batch(
            contents, scope, categories, metadata, importance, source, private
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        crewai_event_bus.emit(
            self,
            MemorySaveCompletedEvent(
                value=f"{len(records)} memories saved",
                metadata=metadata or {},
                agent_role=agent_role,
                save_time_ms=elapsed_ms,
                source_type="unified_memory",
            ),
        )
        return records

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from a raw content blob using the LLM.

        This is a pure helper -- it does NOT store anything.
        Call remember() on each returned string to persist them.

        Args:
            content: Raw text (e.g. task + result dump).

        Returns:
            List of short, self-contained memory statements.
        """
        return extract_memories_from_content(content, self._llm)

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: Literal["shallow", "deep"] = "deep",
        source: str | None = None,
        include_private: bool = False,
    ) -> list[MemoryMatch]:
        """Retrieve relevant memories.

        ``shallow`` embeds the query directly and runs a single vector search.
        ``deep`` (default) uses the RecallFlow: the LLM distills the query into
        targeted sub-queries, selects scopes, searches in parallel, and applies
        confidence-based routing for optional deeper exploration.

        Args:
            query: Natural language query.
            scope: Optional scope prefix to search within.
            categories: Optional category filter.
            limit: Max number of results.
            depth: "shallow" for direct vector search, "deep" for intelligent flow.
            source: Optional provenance filter. Private records are only visible
                    when this matches the record's source.
            include_private: If True, all private records are visible regardless of source.

        Returns:
            List of MemoryMatch, ordered by relevance.
        """
        # Read barrier: wait for any pending background saves to finish
        # so that the search sees all persisted records.
        self.drain_writes()

        _source = "unified_memory"
        try:
            crewai_event_bus.emit(
                self,
                MemoryQueryStartedEvent(
                    query=query,
                    limit=limit,
                    score_threshold=None,
                    source_type=_source,
                ),
            )
            start = time.perf_counter()

            if depth == "shallow":
                embedding = embed_text(self._embedder, query)
                if not embedding:
                    results: list[MemoryMatch] = []
                else:
                    raw = self._storage.search(
                        embedding,
                        scope_prefix=scope,
                        categories=categories,
                        limit=limit,
                        min_score=0.0,
                    )
                    # Privacy filter
                    if not include_private:
                        raw = [
                            (r, s) for r, s in raw
                            if not r.private or r.source == source
                        ]
                    results = []
                    for r, s in raw:
                        composite, reasons = compute_composite_score(
                            r, s, self._config
                        )
                        results.append(
                            MemoryMatch(
                                record=r,
                                score=composite,
                                match_reasons=reasons,
                            )
                        )
                    results.sort(key=lambda m: m.score, reverse=True)
            else:
                flow = RecallFlow(
                    storage=self._storage,
                    llm=self._llm,
                    embedder=self._embedder,
                    config=self._config,
                )
                flow.kickoff(
                    inputs={
                        "query": query,
                        "scope": scope,
                        "categories": categories or [],
                        "limit": limit,
                        "source": source,
                        "include_private": include_private,
                    }
                )
                results = flow.state.final_results

            # Update last_accessed for recalled records
            if results:
                try:
                    touch = getattr(self._storage, "touch_records", None)
                    if touch is not None:
                        touch([m.record.id for m in results])
                except Exception:  # noqa: S110
                    pass  # Non-critical: don't fail recall because of touch

            elapsed_ms = (time.perf_counter() - start) * 1000
            crewai_event_bus.emit(
                self,
                MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
                    score_threshold=None,
                    query_time_ms=elapsed_ms,
                    source_type=_source,
                ),
            )
            return results
        except Exception as e:
            crewai_event_bus.emit(
                self,
                MemoryQueryFailedEvent(
                    query=query,
                    limit=limit,
                    score_threshold=None,
                    error=str(e),
                    source_type=_source,
                ),
            )
            raise

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        """Delete memories matching criteria.

        Returns:
            Number of records deleted.
        """
        return self._storage.delete(
            scope_prefix=scope,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )

    def update(
        self,
        record_id: str,
        content: str | None = None,
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        """Update an existing memory record by ID.

        Args:
            record_id: ID of the record to update.
            content: New content; re-embedded if provided.
            scope: New scope path.
            categories: New categories.
            metadata: New metadata.
            importance: New importance score.

        Returns:
            The updated MemoryRecord.

        Raises:
            ValueError: If the record is not found.
        """
        existing = self._storage.get_record(record_id)
        if existing is None:
            raise ValueError(f"Record not found: {record_id}")
        now = datetime.utcnow()
        updates: dict[str, Any] = {"last_accessed": now}
        if content is not None:
            updates["content"] = content
            embedding = embed_text(self._embedder, content)
            updates["embedding"] = embedding if embedding else existing.embedding
        if scope is not None:
            updates["scope"] = scope
        if categories is not None:
            updates["categories"] = categories
        if metadata is not None:
            updates["metadata"] = metadata
        if importance is not None:
            updates["importance"] = importance
        updated = existing.model_copy(update=updates)
        self._storage.update(updated)
        return updated

    def scope(self, path: str) -> Any:
        """Return a scoped view of this memory."""
        from crewai.memory.memory_scope import MemoryScope

        return MemoryScope(memory=self, root_path=path)

    def slice(
        self,
        scopes: list[str],
        categories: list[str] | None = None,
        read_only: bool = True,
    ) -> Any:
        """Return a multi-scope view (slice) of this memory."""
        from crewai.memory.memory_scope import MemorySlice

        return MemorySlice(
            memory=self,
            scopes=scopes,
            categories=categories,
            read_only=read_only,
        )

    def list_scopes(self, path: str = "/") -> list[str]:
        """List immediate child scopes under path."""
        return self._storage.list_scopes(path)

    def list_records(
        self, scope: str | None = None, limit: int = 200, offset: int = 0
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first.

        Args:
            scope: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
        """
        return self._storage.list_records(scope_prefix=scope, limit=limit, offset=offset)

    def info(self, path: str = "/") -> ScopeInfo:
        """Return scope info for path."""
        return self._storage.get_scope_info(path)

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        """Return a formatted tree of scopes (string)."""
        lines: list[str] = []

        def _walk(p: str, depth: int, prefix: str) -> None:
            if depth > max_depth:
                return
            info = self._storage.get_scope_info(p)
            lines.append(f"{prefix}{p or '/'} ({info.record_count} records)")
            for child in info.child_scopes[:20]:
                _walk(child, depth + 1, prefix + "  ")

        _walk(path.rstrip("/") or "/", 0, "")
        return "\n".join(lines) if lines else f"{path or '/'} (0 records)"

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        """List categories and counts; path=None means global."""
        return self._storage.list_categories(scope_prefix=path)

    def reset(self, scope: str | None = None) -> None:
        """Reset (delete all) memories in scope. None = all."""
        self._storage.reset(scope_prefix=scope)

    async def aextract_memories(self, content: str) -> list[str]:
        """Async variant of extract_memories."""
        return self.extract_memories(content)

    async def aremember(
        self,
        content: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
    ) -> MemoryRecord:
        """Async remember: delegates to sync for now."""
        return self.remember(
            content,
            scope=scope,
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
        )

    async def aremember_many(
        self,
        contents: list[str],
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
    ) -> list[MemoryRecord]:
        """Async remember_many: delegates to sync for now."""
        return self.remember_many(
            contents,
            scope=scope,
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
            agent_role=agent_role,
        )

    async def arecall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: Literal["shallow", "deep"] = "deep",
        source: str | None = None,
        include_private: bool = False,
    ) -> list[MemoryMatch]:
        """Async recall: delegates to sync for now."""
        return self.recall(
            query,
            scope=scope,
            categories=categories,
            limit=limit,
            depth=depth,
            source=source,
            include_private=include_private,
        )
