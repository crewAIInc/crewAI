"""Unified Memory class: single intelligent memory with LLM analysis and pluggable storage."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
from datetime import datetime
import threading
import time
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainValidator, PrivateAttr

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
from crewai.memory.storage.backend import StorageBackend
from crewai.memory.types import (
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
    embed_text,
)
from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.openai.types import OpenAIProviderSpec


if TYPE_CHECKING:
    from chromadb.utils.embedding_functions.openai_embedding_function import (
        OpenAIEmbeddingFunction,
    )


def _passthrough(v: Any) -> Any:
    """PlainValidator that accepts any value, bypassing strict union discrimination."""
    return v


def _default_embedder() -> OpenAIEmbeddingFunction:
    """Build default OpenAI embedder for memory."""
    spec: OpenAIProviderSpec = {"provider": "openai", "config": {}}
    return build_embedder(spec)


class Memory(BaseModel):
    """Unified memory: standalone, LLM-analyzed, with intelligent recall flow.

    Works without agent/crew. Uses LLM to infer scope, categories, importance on save.
    Uses RecallFlow for adaptive-depth recall. Supports scope/slice views and
    pluggable storage (LanceDB default).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Annotated[BaseLLM | str, PlainValidator(_passthrough)] = Field(
        default="gpt-4o-mini",
        description="LLM for analysis (model name or BaseLLM instance).",
    )
    storage: Annotated[StorageBackend | str, PlainValidator(_passthrough)] = Field(
        default="lancedb",
        description="Storage backend instance or path string.",
    )
    embedder: Any = Field(
        default=None,
        description="Embedding callable, provider config dict, or None for default OpenAI.",
    )
    recency_weight: float = Field(
        default=0.3,
        description="Weight for recency in the composite relevance score.",
    )
    semantic_weight: float = Field(
        default=0.5,
        description="Weight for semantic similarity in the composite relevance score.",
    )
    importance_weight: float = Field(
        default=0.2,
        description="Weight for importance in the composite relevance score.",
    )
    recency_half_life_days: int = Field(
        default=30,
        description="Recency score halves every N days (exponential decay).",
    )
    consolidation_threshold: float = Field(
        default=0.85,
        description="Similarity above which consolidation is triggered on save.",
    )
    consolidation_limit: int = Field(
        default=5,
        description="Max existing records to compare during consolidation.",
    )
    default_importance: float = Field(
        default=0.5,
        description="Default importance when not provided or inferred.",
    )
    confidence_threshold_high: float = Field(
        default=0.8,
        description="Recall confidence above which results are returned directly.",
    )
    confidence_threshold_low: float = Field(
        default=0.5,
        description="Recall confidence below which deeper exploration is triggered.",
    )
    complex_query_threshold: float = Field(
        default=0.7,
        description="For complex queries, explore deeper below this confidence.",
    )
    exploration_budget: int = Field(
        default=1,
        description="Number of LLM-driven exploration rounds during deep recall.",
    )
    query_analysis_threshold: int = Field(
        default=200,
        description="Queries shorter than this skip LLM analysis during deep recall.",
    )
    read_only: bool = Field(
        default=False,
        description="If True, remember() and remember_many() are silent no-ops.",
    )

    _config: MemoryConfig = PrivateAttr()
    _llm_instance: BaseLLM | None = PrivateAttr(default=None)
    _embedder_instance: Any = PrivateAttr(default=None)
    _storage: StorageBackend = PrivateAttr()
    _save_pool: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memory-save"
        )
    )
    _pending_saves: list[Future[Any]] = PrivateAttr(default_factory=list)
    _pending_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context: Any) -> None:
        """Initialize runtime state from field values."""
        self._config = MemoryConfig(
            recency_weight=self.recency_weight,
            semantic_weight=self.semantic_weight,
            importance_weight=self.importance_weight,
            recency_half_life_days=self.recency_half_life_days,
            consolidation_threshold=self.consolidation_threshold,
            consolidation_limit=self.consolidation_limit,
            default_importance=self.default_importance,
            confidence_threshold_high=self.confidence_threshold_high,
            confidence_threshold_low=self.confidence_threshold_low,
            complex_query_threshold=self.complex_query_threshold,
            exploration_budget=self.exploration_budget,
            query_analysis_threshold=self.query_analysis_threshold,
        )

        self._llm_instance = None if isinstance(self.llm, str) else self.llm
        self._embedder_instance = (
            self.embedder
            if (self.embedder is not None and not isinstance(self.embedder, dict))
            else None
        )

        if isinstance(self.storage, str):
            from crewai.memory.storage.lancedb_storage import LanceDBStorage

            self._storage = (
                LanceDBStorage()
                if self.storage == "lancedb"
                else LanceDBStorage(path=self.storage)
            )
        else:
            self._storage = self.storage

    _MEMORY_DOCS_URL = "https://docs.crewai.com/concepts/memory"

    @property
    def _llm(self) -> BaseLLM:
        """Lazy LLM initialization -- only created when first needed."""
        if self._llm_instance is None:
            from crewai.llm import LLM

            try:
                model_name = self.llm if isinstance(self.llm, str) else str(self.llm)
                self._llm_instance = LLM(model=model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Memory requires an LLM for analysis but initialization failed: {e}\n\n"
                    "To fix this, do one of the following:\n"
                    "  - Set OPENAI_API_KEY for the default model (gpt-4o-mini)\n"
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
                if isinstance(self.embedder, dict):
                    self._embedder_instance = build_embedder(self.embedder)
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
        ctx = contextvars.copy_context()
        try:
            future: Future[Any] = self._save_pool.submit(ctx.run, fn, *args, **kwargs)
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
        """Remove a completed future from the pending list and emit failure event if needed.

        This callback must never raise -- it runs from the thread pool's
        internal machinery during process shutdown when executors and the
        event bus may already be closed.
        """
        try:
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
        except Exception:  # noqa: S110
            pass  # swallow everything during shutdown

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
    ) -> MemoryRecord | None:
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
            The created MemoryRecord, or None if this memory is read-only.

        Raises:
            Exception: On save failure (events emitted).
        """
        if self.read_only:
            return None
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
                [content],
                scope,
                categories,
                metadata,
                importance,
                source,
                private,
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
        if not contents or self.read_only:
            return []

        self._submit_save(
            self._background_encode_batch,
            contents,
            scope,
            categories,
            metadata,
            importance,
            source,
            private,
            agent_role,
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

        All ``emit`` calls are wrapped in try/except to handle the case where
        the event bus shuts down before the background save finishes (e.g.
        during process exit).
        """
        try:
            crewai_event_bus.emit(
                self,
                MemorySaveStartedEvent(
                    value=f"{len(contents)} memories (background)",
                    metadata=metadata,
                    source_type="unified_memory",
                ),
            )
        except RuntimeError:
            pass  # event bus shut down during process exit

        try:
            start = time.perf_counter()
            records = self._encode_batch(
                contents, scope, categories, metadata, importance, source, private
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
        except RuntimeError:
            # The encoding pipeline uses asyncio.run() -> to_thread() internally.
            # If the process is shutting down, the default executor is closed and
            # to_thread raises "cannot schedule new futures after shutdown".
            # Silently abandon the save -- the process is exiting anyway.
            return []

        try:
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
        except RuntimeError:
            pass  # event bus shut down during process exit
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
                            (r, s)
                            for r, s in raw
                            if not r.private or r.source == source
                        ]
                    results = []
                    for r, s in raw:
                        composite, reasons = compute_composite_score(r, s, self._config)
                        results.append(
                            MemoryMatch(
                                record=r,
                                score=composite,
                                match_reasons=reasons,
                            )
                        )
                    results.sort(key=lambda m: m.score, reverse=True)
            else:
                from crewai.memory.recall_flow import RecallFlow

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
        return self._storage.list_records(
            scope_prefix=scope, limit=limit, offset=offset
        )

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
    ) -> MemoryRecord | None:
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
