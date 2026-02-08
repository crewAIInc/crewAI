"""Unified Memory class: single intelligent memory with LLM analysis and pluggable storage."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.llms.base_llm import BaseLLM
from crewai.memory.analyze import (
    MemoryAnalysis,
    analyze_for_save,
    analyze_query,
    extract_memories_from_content,
)
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
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize Memory.

        Args:
            llm: LLM for analysis (model name or BaseLLM instance).
            storage: Backend: "lancedb" or a StorageBackend instance.
            embedder: Embedding function; None => default OpenAI.
            config: Optional retrieval config; None => defaults.
        """
        from crewai.llm import LLM

        self._config = config or MemoryConfig()
        self._llm = LLM(model=llm) if isinstance(llm, str) else llm
        if storage == "lancedb":
            self._storage = LanceDBStorage()
        elif isinstance(storage, str):
            self._storage = LanceDBStorage(path=storage)
        else:
            self._storage = storage
        self._embedder = embedder if embedder is not None else _default_embedder()

    def remember(
        self,
        content: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        """Store content in memory. Infers scope/categories/importance via LLM when not provided.

        Args:
            content: Text to remember.
            scope: Optional scope path; inferred if None.
            categories: Optional categories; inferred if None.
            metadata: Optional metadata; merged with LLM-extracted if scope/categories/importance inferred.
            importance: Optional importance 0-1; inferred if None.

        Returns:
            The created MemoryRecord.

        Raises:
            Exception: On save failure (events emitted).
        """
        _source = "unified_memory"
        try:
            crewai_event_bus.emit(
                self,
                MemorySaveStartedEvent(
                    value=content,
                    metadata=metadata,
                    source_type=_source,
                ),
            )
            start = time.perf_counter()

            need_analysis = (
                scope is None
                or categories is None
                or importance is None
                or (metadata is None and scope is None)
            )
            if need_analysis:
                existing_scopes = self._storage.list_scopes("/")
                if not existing_scopes:
                    existing_scopes = ["/"]
                existing_categories = list(
                    self._storage.list_categories(scope_prefix=None).keys()
                )
                analysis: MemoryAnalysis = analyze_for_save(
                    content,
                    existing_scopes,
                    existing_categories,
                    self._llm,
                )
                scope = scope or analysis.suggested_scope or "/"
                categories = categories if categories is not None else analysis.categories
                importance = (
                    importance
                    if importance is not None
                    else analysis.importance
                )
                metadata = dict(
                    metadata or {},
                    **(
                        analysis.extracted_metadata.model_dump()
                        if analysis.extracted_metadata
                        else {}
                    ),
                )
            else:
                scope = scope or "/"
                categories = categories or []
                metadata = metadata or {}
                importance = importance if importance is not None else 0.5

            embedding = embed_text(self._embedder, content)
            from crewai.memory.consolidation_flow import ConsolidationFlow

            flow = ConsolidationFlow(
                storage=self._storage,
                llm=self._llm,
                embedder=self._embedder,
                config=self._config,
            )
            flow.kickoff(
                inputs={
                    "new_content": content,
                    "new_embedding": embedding,
                    "scope": scope,
                    "categories": categories,
                    "metadata": metadata or {},
                    "importance": importance,
                },
            )
            record = flow.state.result_record
            if record is None:
                record = MemoryRecord(
                    content=content,
                    scope=scope,
                    categories=categories,
                    metadata=metadata or {},
                    importance=importance,
                    embedding=embedding if embedding else None,
                )
                self._storage.save([record])
            elapsed_ms = (time.perf_counter() - start) * 1000
            crewai_event_bus.emit(
                self,
                MemorySaveCompletedEvent(
                    value=content,
                    metadata=metadata or {},
                    agent_role=None,
                    save_time_ms=elapsed_ms,
                    source_type=_source,
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
                    source_type=_source,
                ),
            )
            raise

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
        depth: Literal["shallow", "deep", "auto"] = "auto",
    ) -> list[MemoryMatch]:
        """Retrieve relevant memories. Shallow = direct vector search; deep/auto = RecallFlow.

        Args:
            query: Natural language query.
            scope: Optional scope prefix to search within.
            categories: Optional category filter.
            limit: Max number of results.
            depth: "shallow" for direct search, "deep" or "auto" for intelligent flow.

        Returns:
            List of MemoryMatch, ordered by relevance.
        """
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
                    config=self._config,
                )
                embedding = embed_text(self._embedder, query)
                flow.kickoff(
                    inputs={
                        "query": query,
                        "scope": scope,
                        "categories": categories or [],
                        "limit": limit,
                        "query_embedding": embedding,
                    }
                )
                results = flow.state.final_results

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
            ValueError: If the storage does not support get_record or the record is not found.
        """
        get_record = getattr(self._storage, "get_record", None)
        if get_record is None:
            raise ValueError("This storage backend does not support update by ID")
        existing = get_record(record_id)
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
        self, scope: str | None = None, limit: int = 200
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first."""
        return self._storage.list_records(scope_prefix=scope, limit=limit)

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
    ) -> MemoryRecord:
        """Async remember: delegates to sync for now."""
        return self.remember(
            content,
            scope=scope,
            categories=categories,
            metadata=metadata,
            importance=importance,
        )

    async def arecall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: Literal["shallow", "deep", "auto"] = "auto",
    ) -> list[MemoryMatch]:
        """Async recall: delegates to sync for now."""
        return self.recall(
            query,
            scope=scope,
            categories=categories,
            limit=limit,
            depth=depth,
        )
