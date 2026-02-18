"""CrewAI Memory backend powered by Hindsight.

Implements the same duck-type interface as CrewAI's ``Memory`` class so it
can be passed directly to ``Crew(memory=hindsight_memory)``.  All memory
operations (remember, recall, reset) are delegated to Hindsight's
retain/recall APIs, giving crews persistent memory with fact extraction,
entity tracking, and temporal awareness.

Usage::

    from crewai.memory.storage.hindsight_storage import HindsightMemory, HindsightConfig

    memory = HindsightMemory(HindsightConfig(
        api_url="http://localhost:8888",
        bank_id="my-crew",
        mission="Track research findings",
    ))

    crew = Crew(agents=[...], tasks=[...], memory=memory)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async compatibility helpers
# ---------------------------------------------------------------------------
# CrewAI runs inside an async event loop. The Hindsight client's sync
# methods internally call loop.run_until_complete(), which fails when a
# loop is already running. We use a dedicated worker thread with a
# persistent event loop for all Hindsight API calls.

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_thread_init_lock = threading.Lock()
_initialized_threads: set[int] = set()


def _ensure_thread_loop() -> None:
    """Ensure the current thread has a persistent event loop."""
    tid = threading.get_ident()
    if tid not in _initialized_threads:
        with _thread_init_lock:
            if tid not in _initialized_threads:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _initialized_threads.add(tid)


def _call_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a sync Hindsight client method in a dedicated thread pool."""

    def _run() -> Any:
        _ensure_thread_loop()
        return fn(*args, **kwargs)

    future = _thread_pool.submit(_run)
    return future.result(timeout=60)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HindsightConfig(BaseModel):
    """Configuration for the Hindsight memory backend.

    Attributes:
        api_url: Hindsight API URL (e.g. ``http://localhost:8888``).
        api_key: API key for authentication. Falls back to
            ``HINDSIGHT_API_KEY`` environment variable.
        bank_id: Memory bank ID to use for this crew.
        budget: Recall budget level (``low``, ``mid``, or ``high``).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        mission: Optional bank mission for organizing memories.
    """

    api_url: str
    api_key: str | None = None
    bank_id: str
    budget: str = "mid"
    max_tokens: int = 4096
    tags: list[str] | None = None
    mission: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("api_url")
    @classmethod
    def api_url_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("api_url must not be empty")
        return v.strip().rstrip("/")

    @field_validator("bank_id")
    @classmethod
    def bank_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("bank_id must not be empty")
        return v.strip()

    @field_validator("budget")
    @classmethod
    def budget_valid(cls, v: str) -> str:
        allowed = {"low", "mid", "high"}
        if v not in allowed:
            raise ValueError(f"budget must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="before")
    @classmethod
    def resolve_api_key(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("api_key"):
            values["api_key"] = os.environ.get("HINDSIGHT_API_KEY")
        return values


# ---------------------------------------------------------------------------
# HindsightMemory — duck-type compatible with CrewAI's Memory class
# ---------------------------------------------------------------------------


class HindsightMemory:
    """CrewAI Memory backend that persists memories to Hindsight.

    Duck-type compatible with :class:`crewai.memory.unified_memory.Memory`.
    Pass an instance directly to ``Crew(memory=hindsight_memory)``.

    Maps CrewAI's Memory interface to Hindsight's memory API:

    - ``remember(content)``  -> ``client.retain(bank_id, content)``
    - ``recall(query)``      -> ``client.recall(bank_id, query)``
    - ``reset()``            -> ``client.delete_bank()`` + recreate

    Args:
        config: A :class:`HindsightConfig` instance with connection details.
    """

    def __init__(self, config: HindsightConfig) -> None:
        self.config = config
        self._local = threading.local()
        self._created_banks: set[str] = set()

        # Eagerly create the bank if a mission is provided
        if config.mission:
            self._ensure_bank(config.bank_id)

    def _get_client(self) -> Any:
        """Get or create a thread-local Hindsight client."""
        client = getattr(self._local, "client", None)
        if client is None:
            try:
                from hindsight_client import Hindsight
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    "The 'hindsight-client' package is required for Hindsight memory. "
                    "Install it with: pip install 'crewai[hindsight]'"
                ) from err
            client = Hindsight(
                base_url=self.config.api_url,
                api_key=self.config.api_key,
                timeout=30.0,
            )
            self._local.client = client
        return client

    def _ensure_bank(self, bank_id: str) -> None:
        """Create bank if not already created in this session."""
        if bank_id in self._created_banks:
            return

        def _create() -> None:
            self._get_client().create_bank(
                bank_id=bank_id,
                name=bank_id,
                mission=self.config.mission,
            )

        try:
            _call_sync(_create)
            self._created_banks.add(bank_id)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "409" in exc_str or "already exists" in exc_str or "conflict" in exc_str:
                self._created_banks.add(bank_id)
            else:
                logger.debug("Bank creation for %s failed (will retry): %s", bank_id, exc)

    # ------------------------------------------------------------------
    # Memory interface — remember
    # ------------------------------------------------------------------

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
        """Store a single item in memory via Hindsight's retain API.

        Args:
            content: Text to remember.
            scope: Ignored (Hindsight uses bank_id for isolation).
            categories: Mapped to Hindsight tags.
            metadata: Passed through to Hindsight retain metadata.
            importance: Ignored (Hindsight infers importance).
            source: Included in retain metadata.
            private: Ignored (Hindsight manages access via API keys).
            agent_role: Included in retain metadata.

        Returns:
            A ``MemoryRecord`` representing the stored memory.
        """
        bank_id = self.config.bank_id
        self._ensure_bank(bank_id)

        # Build retain metadata — Hindsight requires dict[str, str]
        retain_metadata: dict[str, str] = {"source": "crewai"}
        if metadata:
            for k, v in metadata.items():
                retain_metadata[k] = str(v)
        if source:
            retain_metadata["provenance"] = source
        if agent_role:
            retain_metadata["agent_role"] = agent_role

        # Merge config tags with category tags
        tags = list(self.config.tags) if self.config.tags else []
        if categories:
            tags.extend(categories)

        def _retain() -> None:
            self._get_client().retain(
                bank_id=bank_id,
                content=str(content),
                metadata=retain_metadata,
                tags=tags or None,
            )

        try:
            _call_sync(_retain)
        except Exception as e:
            raise RuntimeError(f"Hindsight retain failed: {e}") from e

        return MemoryRecord(
            content=content,
            scope=scope or "/",
            categories=categories or [],
            metadata=metadata or {},
            importance=importance or 0.5,
        )

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
        """Store multiple items in memory.

        Calls ``remember()`` for each item sequentially. Returns the
        list of created ``MemoryRecord`` instances.
        """
        if not contents:
            return []
        return [
            self.remember(
                c,
                scope=scope,
                categories=categories,
                metadata=metadata,
                importance=importance,
                source=source,
                private=private,
                agent_role=agent_role,
            )
            for c in contents
        ]

    # ------------------------------------------------------------------
    # Memory interface — recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: str = "deep",
        source: str | None = None,
        include_private: bool = False,
    ) -> list[MemoryMatch]:
        """Retrieve relevant memories via Hindsight's recall API.

        Args:
            query: Natural language query.
            scope: Ignored (Hindsight uses bank_id for isolation).
            categories: Ignored (Hindsight does its own retrieval).
            limit: Maximum number of results to return.
            depth: Ignored (Hindsight always uses multi-strategy retrieval).
            source: Ignored (Hindsight manages access via API keys).
            include_private: Ignored.

        Returns:
            List of ``MemoryMatch`` objects ordered by relevance.
        """
        bank_id = self.config.bank_id

        def _recall() -> Any:
            return self._get_client().recall(
                bank_id=bank_id,
                query=query,
                budget=self.config.budget,
                max_tokens=self.config.max_tokens,
            )

        try:
            response = _call_sync(_recall)

            recall_results = response.results if hasattr(response, "results") else []
            sliced = recall_results[:limit]
            denom = max(len(sliced) * 2, 1)

            matches: list[MemoryMatch] = []
            for i, r in enumerate(sliced):
                score = round(1.0 - (i / denom), 4)

                result_metadata: dict[str, Any] = {}
                if hasattr(r, "type") and r.type:
                    result_metadata["type"] = r.type
                if hasattr(r, "context") and r.context:
                    result_metadata["source_context"] = r.context
                if hasattr(r, "occurred_start") and r.occurred_start:
                    result_metadata["occurred_start"] = r.occurred_start
                if hasattr(r, "document_id") and r.document_id:
                    result_metadata["document_id"] = r.document_id
                if hasattr(r, "metadata") and r.metadata:
                    result_metadata.update(r.metadata)
                if hasattr(r, "tags") and r.tags:
                    result_metadata["tags"] = r.tags

                record = MemoryRecord(
                    content=r.text,
                    scope="/",
                    categories=list(r.tags) if hasattr(r, "tags") and r.tags else [],
                    metadata=result_metadata,
                )

                matches.append(
                    MemoryMatch(
                        record=record,
                        score=score,
                        match_reasons=["semantic"],
                    )
                )

            return matches

        except Exception as e:
            raise RuntimeError(f"Hindsight recall failed: {e}") from e

    # ------------------------------------------------------------------
    # Memory interface — other methods
    # ------------------------------------------------------------------

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from raw content.

        Simple implementation: returns the content as a single item.
        Hindsight's retain API handles its own fact extraction.
        """
        return [content]

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: Any = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        """Delete memories. Delegates to reset() since Hindsight manages its own storage."""
        self.reset()
        return 0

    def reset(self, scope: str | None = None) -> None:
        """Clear all memories by deleting and recreating the bank."""
        bank_id = self.config.bank_id

        def _delete() -> None:
            self._get_client().delete_bank(bank_id)

        try:
            _call_sync(_delete)
            self._created_banks.discard(bank_id)
            self._ensure_bank(bank_id)
        except Exception:
            logger.debug("Failed to reset bank %s", bank_id, exc_info=True)

    def drain_writes(self) -> None:
        """No-op. Hindsight retain is synchronous from the client perspective."""

    def close(self) -> None:
        """No-op. Thread-local clients are cleaned up by the GC."""

    def scope(self, path: str) -> HindsightMemory:
        """Return self. Hindsight uses bank_id for isolation, not scopes."""
        return self

    def info(self, path: str = "/") -> ScopeInfo:
        """Return basic scope info."""
        return ScopeInfo(path=path)

    # Async variants — delegate to sync implementations

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
        """Async remember: delegates to sync."""
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
        """Async remember_many: delegates to sync."""
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
        depth: str = "deep",
        source: str | None = None,
        include_private: bool = False,
    ) -> list[MemoryMatch]:
        """Async recall: delegates to sync."""
        return self.recall(
            query,
            scope=scope,
            categories=categories,
            limit=limit,
            depth=depth,
            source=source,
            include_private=include_private,
        )

    async def aextract_memories(self, content: str) -> list[str]:
        """Async extract_memories: delegates to sync."""
        return self.extract_memories(content)
