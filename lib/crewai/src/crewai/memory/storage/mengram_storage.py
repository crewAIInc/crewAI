"""CrewAI memory backend powered by Mengram.

Mengram (https://mengram.io) provides human-like memory for AI agents with
three memory types: semantic (facts & knowledge graph), episodic (events),
and procedural (learned workflows with success/failure tracking).

``MengramMemory`` duck-types CrewAI's ``Memory`` interface so it can be
passed directly to ``Crew(memory=mengram_memory)``.  All memory operations
are delegated to the Mengram cloud API -- no local LLM or embedder needed.

Usage::

    from crewai.memory.storage.mengram_storage import MengramMemory, MengramConfig

    memory = MengramMemory(MengramConfig(api_key="om-..."))
    crew = Crew(agents=[...], tasks=[...], memory=memory)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, field_validator, model_validator

from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo

logger = logging.getLogger(__name__)

_UPGRADE_URL = "https://mengram.io/billing"


class _QuotaExceededError(RuntimeError):
    """Raised when the Mengram API returns HTTP 402 (quota exceeded)."""

    def __init__(self, action: str, plan: str, limit: int, used: int) -> None:
        self.action = action
        self.plan = plan
        self.limit = limit
        self.used = used
        super().__init__(
            f"Mengram {plan} plan quota exceeded for {action} "
            f"({used}/{limit}). Upgrade at {_UPGRADE_URL}"
        )


# ---------------------------------------------------------------------------
# Lightweight Mengram HTTP client (stdlib only -- no extra dependency)
# ---------------------------------------------------------------------------

class _MengramClient:
    """Thin HTTP wrapper for the Mengram REST API.

    Uses :mod:`urllib.request` (stdlib) so the integration has zero external
    dependencies beyond what CrewAI already ships.  Handles authentication,
    retries on transient errors (429 / 5xx), and JSON (de)serialisation.
    """

    def __init__(self, api_key: str, base_url: str = "https://mengram.io") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        url = f"{self._base_url}{path}"
        if params:
            qs = "&".join(
                f"{k}={urllib.parse.quote(str(v))}"
                for k, v in params.items()
                if v is not None
            )
            if qs:
                url = f"{url}?{qs}"

        body = json.dumps(data).encode() if data else None

        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=body,
                method=method,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                resp_body = exc.read().decode()
                if exc.code == 402:
                    try:
                        detail = json.loads(resp_body).get("detail", {})
                    except Exception:
                        detail = {}
                    if isinstance(detail, dict):
                        raise _QuotaExceededError(
                            action=detail.get("action", "unknown"),
                            plan=detail.get("plan", "free"),
                            limit=detail.get("limit", 0),
                            used=detail.get("used", 0),
                        ) from exc
                    raise _QuotaExceededError(
                        action="unknown", plan="free", limit=0, used=0,
                    ) from exc
                if exc.code in (429, 502, 503, 504) and attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    last_err = exc
                    continue
                try:
                    detail = json.loads(resp_body).get("detail", resp_body)
                except Exception:
                    detail = resp_body
                raise RuntimeError(
                    f"Mengram API error {exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    last_err = exc
                    continue
                raise RuntimeError(f"Mengram network error: {exc}") from exc
        raise RuntimeError(f"Mengram request failed after 3 attempts: {last_err}")

    # -- Convenience wrappers for the endpoints we use -----------------------

    def add_text(self, text: str, user_id: str = "default") -> dict:
        """POST /v1/add_text -- extract memories from plain text."""
        return self._request("POST", "/v1/add_text", {"text": text, "user_id": user_id})

    def search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        graph_depth: int = 2,
    ) -> list[dict]:
        """POST /v1/search -- semantic search with knowledge graph traversal."""
        result = self._request(
            "POST",
            "/v1/search",
            {"query": query, "user_id": user_id, "limit": limit, "graph_depth": graph_depth},
        )
        return result.get("results", [])

    def search_all(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        graph_depth: int = 2,
    ) -> dict:
        """POST /v1/search/all -- unified search across semantic + episodic + procedural."""
        return self._request(
            "POST",
            "/v1/search/all",
            {"query": query, "user_id": user_id, "limit": limit, "graph_depth": graph_depth},
        )

    def delete_entity(self, name: str, user_id: str = "default") -> dict:
        """DELETE /v1/memory/{name} -- remove a single entity by name."""
        params: dict[str, str] = {}
        if user_id and user_id != "default":
            params["sub_user_id"] = user_id
        return self._request(
            "DELETE",
            f"/v1/memory/{urllib.parse.quote(name, safe='')}",
            params=params,
        )

    def delete_all(self, user_id: str = "default") -> dict:
        """DELETE /v1/memories/all -- remove all memories for the user."""
        params: dict[str, str] = {}
        if user_id and user_id != "default":
            params["sub_user_id"] = user_id
        return self._request("DELETE", "/v1/memories/all", params=params)

    def stats(self, user_id: str = "default") -> dict:
        """GET /v1/stats -- usage statistics."""
        params: dict[str, str] = {}
        if user_id and user_id != "default":
            params["sub_user_id"] = user_id
        return self._request("GET", "/v1/stats", params=params)

    def job_status(self, job_id: str) -> dict:
        """GET /v1/jobs/{job_id} -- check background job status."""
        return self._request("GET", f"/v1/jobs/{job_id}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MengramConfig(BaseModel):
    """Configuration for the Mengram memory backend.

    Attributes:
        api_key: Mengram API key (``om-...``).  Falls back to the
            ``MENGRAM_API_KEY`` environment variable.  Get a free key at
            `mengram.io <https://mengram.io>`_.
        base_url: Mengram API URL.  Falls back to ``MENGRAM_BASE_URL``.
            Default: ``https://mengram.io``.
        user_id: User identifier for memory isolation.  Each ``user_id``
            gets its own memory space.  Default: ``"default"``.
        graph_depth: How many hops to traverse in the knowledge graph
            during search.  Default: ``2``.
        search_limit: Default max results per memory type in recall.
            Default: ``5``.
    """

    api_key: str | None = None
    base_url: str = "https://mengram.io"
    user_id: str = "default"
    graph_depth: int = 2
    search_limit: int = 5

    model_config = {"extra": "forbid"}

    @field_validator("base_url")
    @classmethod
    def _base_url_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("base_url must not be empty")
        return v.strip().rstrip("/")

    @field_validator("user_id")
    @classmethod
    def _user_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_id must not be empty")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def _resolve_env_vars(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("api_key"):
            values["api_key"] = os.environ.get("MENGRAM_API_KEY")
        if not values.get("api_key"):
            raise ValueError(
                "api_key is required. Pass it directly or set the MENGRAM_API_KEY "
                "environment variable. Get a free key at https://mengram.io"
            )
        env_base = os.environ.get("MENGRAM_BASE_URL")
        if env_base and values.get("base_url", "https://mengram.io") == "https://mengram.io":
            values["base_url"] = env_base
        return values


# ---------------------------------------------------------------------------
# Result conversion helpers
# ---------------------------------------------------------------------------

def _semantic_to_match(result: dict) -> MemoryMatch:
    """Convert a Mengram semantic search result to a ``MemoryMatch``."""
    entity = result.get("entity", "")
    facts = result.get("facts", [])
    knowledge = result.get("knowledge", [])

    parts: list[str] = []
    if facts:
        parts.append(f"{entity}: {'; '.join(str(f) for f in facts[:10])}")
    elif entity:
        parts.append(entity)
    for k in knowledge[:3]:
        title = k.get("title", "") if isinstance(k, dict) else ""
        body = k.get("content", "") if isinstance(k, dict) else str(k)
        if title and body:
            parts.append(f"{title}: {body[:200]}")

    content = "\n".join(parts) if parts else entity
    score = float(result.get("score", 0.5))

    record = MemoryRecord(
        content=content,
        scope="/",
        categories=[result.get("type", "entity")],
        metadata={
            "entity": entity,
            "type": result.get("type", ""),
            "memory_type": "semantic",
            "fact_count": len(facts),
        },
        importance=min(max(score, 0.0), 1.0),
    )
    return MemoryMatch(record=record, score=score, match_reasons=["semantic"])


def _episodic_to_match(episode: dict, rank: int = 0) -> MemoryMatch:
    """Convert a Mengram episodic result to a ``MemoryMatch``."""
    summary = episode.get("summary", "")
    outcome = episode.get("outcome", "")
    when = episode.get("when", "")

    content = summary
    if outcome:
        content += f" -> Outcome: {outcome}"
    if when:
        content += f" ({when})"

    score = float(episode.get("score", round(0.8 - rank * 0.05, 4)))
    score = max(score, 0.1)

    record = MemoryRecord(
        content=content,
        scope="/",
        categories=["episode"],
        metadata={
            "memory_type": "episodic",
            "outcome": outcome,
            "when": when,
        },
        importance=0.6,
    )
    return MemoryMatch(record=record, score=score, match_reasons=["episodic"])


def _procedural_to_match(procedure: dict, rank: int = 0) -> MemoryMatch:
    """Convert a Mengram procedural result to a ``MemoryMatch``."""
    name = procedure.get("name", "")
    steps = procedure.get("steps", [])
    success_count = procedure.get("success_count", 0)
    fail_count = procedure.get("fail_count", 0)

    steps_str = " -> ".join(
        s.get("action", "") if isinstance(s, dict) else str(s) for s in steps[:10]
    )
    content = f"Procedure: {name}."
    if steps_str:
        content += f" Steps: {steps_str}."
    content += f" Success: {success_count}, Fail: {fail_count}"

    total = success_count + fail_count
    reliability = success_count / total if total > 0 else 0.5
    score = float(procedure.get("score", round(0.75 - rank * 0.05, 4)))
    score = max(score, 0.1)

    record = MemoryRecord(
        content=content,
        scope="/",
        categories=["procedure"],
        metadata={
            "memory_type": "procedural",
            "procedure_name": name,
            "procedure_id": procedure.get("id", ""),
            "success_count": success_count,
            "fail_count": fail_count,
            "reliability": reliability,
        },
        importance=min(0.5 + reliability * 0.3, 1.0),
    )
    return MemoryMatch(record=record, score=score, match_reasons=["procedural"])


def _chunk_to_match(chunk: dict) -> MemoryMatch:
    """Convert a Mengram raw-chunk result to a ``MemoryMatch``."""
    content = chunk.get("content", chunk.get("text", ""))
    score = float(chunk.get("score", 0.3))

    record = MemoryRecord(
        content=content,
        scope="/",
        categories=["chunk"],
        metadata={"memory_type": "chunk"},
        importance=0.3,
    )
    return MemoryMatch(record=record, score=score, match_reasons=["chunk"])


# ---------------------------------------------------------------------------
# MengramMemory -- duck-types crewai.memory.unified_memory.Memory
# ---------------------------------------------------------------------------

class MengramMemory:
    """CrewAI memory backend backed by Mengram's cloud API.

    Implements the same public interface as
    :class:`~crewai.memory.unified_memory.Memory` so it can be passed
    directly to ``Crew(memory=mengram_memory)``.

    Mengram handles LLM analysis, embedding, entity extraction, knowledge
    graph construction, episodic memory, and procedural learning server-side.
    ``MengramMemory`` therefore bypasses CrewAI's built-in encoding and
    recall pipelines -- no local LLM or embedder is required.

    Args:
        config: A :class:`MengramConfig` with connection details.

    Example::

        memory = MengramMemory(MengramConfig(api_key="om-..."))
        crew = Crew(agents=[...], tasks=[...], memory=memory)
    """

    def __init__(self, config: MengramConfig) -> None:
        self.config = config
        self._client = _MengramClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._save_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mengram-save"
        )
        self._pending_saves: list[Future[Any]] = []
        self._pending_lock = threading.Lock()
        self._quota_warned = False

    # -- Background write helpers -------------------------------------------

    def _submit_save(self, fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a save operation to the background thread pool."""
        try:
            future: Future[Any] = self._save_pool.submit(fn, *args, **kwargs)
        except RuntimeError:
            # Pool already shut down -- run inline as fallback.
            future: Future[Any] = Future()
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
        with self._pending_lock:
            try:
                self._pending_saves.remove(future)
            except ValueError:
                pass

    # -- Core interface: remember -------------------------------------------

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
        """Store content in Mengram via ``POST /v1/add_text``.

        The extraction pipeline runs server-side and automatically discovers
        entities, facts, relationships, episodes, and procedures.  The API
        returns immediately with a ``job_id``; extraction completes in the
        background.
        """
        enriched = content
        if agent_role:
            enriched = f"[Agent: {agent_role}] {enriched}"

        job_id = ""
        try:
            result = self._client.add_text(
                text=enriched, user_id=self.config.user_id,
            )
            job_id = result.get("job_id", "")
        except _QuotaExceededError as exc:
            if not self._quota_warned:
                logger.warning(
                    "Mengram %s plan quota reached — memory writes disabled. "
                    "Your crew will continue working without memory. "
                    "Upgrade at %s",
                    exc.plan,
                    _UPGRADE_URL,
                )
                self._quota_warned = True
        except Exception as exc:
            logger.warning("Mengram remember failed: %s", exc)

        return MemoryRecord(
            content=content,
            scope=scope or "/",
            categories=categories or [],
            metadata={
                **(metadata or {}),
                "mengram_job_id": job_id,
                **({"source": source} if source else {}),
                **({"agent_role": agent_role} if agent_role else {}),
            },
            importance=importance if importance is not None else 0.5,
            source=source,
            private=private,
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
        """Store multiple items (non-blocking, fire-and-forget).

        All items are combined into a single ``add_text`` call and sent
        to Mengram in a background thread.  Returns immediately.
        """
        if not contents:
            return []

        combined = "\n\n".join(contents)
        if agent_role:
            combined = f"[Agent: {agent_role}]\n{combined}"

        def _background() -> None:
            try:
                self._client.add_text(text=combined, user_id=self.config.user_id)
            except _QuotaExceededError as exc:
                if not self._quota_warned:
                    logger.warning(
                        "Mengram %s plan quota reached — memory writes disabled. "
                        "Your crew will continue working without memory. "
                        "Upgrade at %s",
                        exc.plan,
                        _UPGRADE_URL,
                    )
                    self._quota_warned = True
            except Exception as exc:
                logger.warning("Mengram background save failed: %s", exc)

        self._submit_save(_background)
        return []

    # -- Core interface: recall ---------------------------------------------

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
        """Search Mengram for relevant memories.

        ``depth="deep"`` (default) searches all three memory types via
        ``POST /v1/search/all``.  ``depth="shallow"`` does a faster
        semantic-only search via ``POST /v1/search``.
        """
        self.drain_writes()
        try:
            if depth == "shallow":
                return self._recall_shallow(query, limit)
            return self._recall_deep(query, limit)
        except _QuotaExceededError as exc:
            if not self._quota_warned:
                logger.warning(
                    "Mengram %s plan quota reached — memory search disabled. "
                    "Your crew will continue working without memory context. "
                    "Upgrade at %s",
                    exc.plan,
                    _UPGRADE_URL,
                )
                self._quota_warned = True
            return []
        except Exception as exc:
            logger.warning("Mengram recall failed: %s", exc)
            return []

    def _recall_shallow(self, query: str, limit: int) -> list[MemoryMatch]:
        results = self._client.search(
            query=query,
            user_id=self.config.user_id,
            limit=limit,
            graph_depth=self.config.graph_depth,
        )
        return [_semantic_to_match(r) for r in results[:limit]]

    def _recall_deep(self, query: str, limit: int) -> list[MemoryMatch]:
        # Request enough results per type so we can fill the caller's limit
        # after merging and sorting across all memory types.
        per_type = max(limit, self.config.search_limit)
        result = self._client.search_all(
            query=query,
            user_id=self.config.user_id,
            limit=per_type,
            graph_depth=self.config.graph_depth,
        )

        matches: list[MemoryMatch] = []

        for r in result.get("semantic", []):
            matches.append(_semantic_to_match(r))

        for i, ep in enumerate(result.get("episodic", [])):
            matches.append(_episodic_to_match(ep, rank=i))

        for i, pr in enumerate(result.get("procedural", [])):
            matches.append(_procedural_to_match(pr, rank=i))

        for ch in result.get("chunks", []):
            matches.append(_chunk_to_match(ch))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    # -- Core interface: forget / reset -------------------------------------

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        """Delete memories.

        If *metadata_filter* contains an ``"entity"`` key, deletes that
        specific entity via ``DELETE /v1/memory/{name}``.  Otherwise
        delegates to :meth:`reset` to clear all memories.
        """
        entity_name = (metadata_filter or {}).get("entity")
        if entity_name:
            try:
                self._client.delete_entity(entity_name, user_id=self.config.user_id)
                return 1
            except Exception as exc:
                logger.warning("Mengram forget entity=%s failed: %s", entity_name, exc)
                return 0

        self.reset()
        return 0

    def reset(self, scope: str | None = None) -> None:
        """Delete all memories for the configured ``user_id``."""
        try:
            self._client.delete_all(user_id=self.config.user_id)
        except Exception as exc:
            logger.warning("Mengram reset failed: %s", exc)

    # -- Extraction ---------------------------------------------------------

    def extract_memories(self, content: str) -> list[str]:
        """Return content as a single-item list.

        Mengram's server-side pipeline handles the actual entity / fact /
        episode / procedure extraction when :meth:`remember` is called.
        """
        return [content] if content else []

    # -- Lifecycle ----------------------------------------------------------

    def drain_writes(self) -> None:
        """Block until all pending background saves have completed."""
        with self._pending_lock:
            pending = list(self._pending_saves)
        for future in pending:
            try:
                future.result(timeout=30)
            except Exception as exc:
                logger.debug("Mengram pending save error: %s", exc)

    def close(self) -> None:
        """Drain pending saves and shut down the background thread pool."""
        self.drain_writes()
        self._save_pool.shutdown(wait=True)

    # -- Scope / info -------------------------------------------------------

    def scope(self, path: str) -> MengramMemory:
        """Return *self* (Mengram uses ``user_id`` for isolation, not scopes)."""
        return self

    def info(self, path: str = "/") -> ScopeInfo:
        """Return memory statistics as a :class:`ScopeInfo`."""
        try:
            s = self._client.stats(user_id=self.config.user_id)
            return ScopeInfo(
                path=path,
                record_count=s.get("entities", 0) + s.get("facts", 0),
                categories=list(s.get("by_type", {}).keys()),
            )
        except Exception as exc:
            logger.warning("Mengram info failed: %s", exc)
            return ScopeInfo(path=path)

    def list_scopes(self, path: str = "/") -> list[str]:
        """Return ``["/"]`` (Mengram uses ``user_id``, not hierarchical scopes)."""
        return ["/"]

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        """Return a single-line scope tree."""
        count = self.info(path).record_count
        return f"{path} ({count} records)"

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        """Return category counts from Mengram stats."""
        try:
            s = self._client.stats(user_id=self.config.user_id)
            return s.get("by_type", {})
        except Exception:
            return {}

    def list_records(
        self, scope: str | None = None, limit: int = 200, offset: int = 0,
    ) -> list[MemoryRecord]:
        """Not supported by Mengram's API -- returns empty list."""
        return []

    # -- Async variants (delegate to sync) ----------------------------------

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
        """Async :meth:`remember` -- delegates to sync."""
        return self.remember(
            content, scope=scope, categories=categories, metadata=metadata,
            importance=importance, source=source, private=private,
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
        """Async :meth:`remember_many` -- delegates to sync."""
        return self.remember_many(
            contents, scope=scope, categories=categories, metadata=metadata,
            importance=importance, source=source, private=private,
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
        """Async :meth:`recall` -- delegates to sync."""
        return self.recall(
            query, scope=scope, categories=categories, limit=limit,
            depth=depth, source=source, include_private=include_private,
        )

    async def aextract_memories(self, content: str) -> list[str]:
        """Async :meth:`extract_memories` -- delegates to sync."""
        return self.extract_memories(content)
