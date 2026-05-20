"""Scoped and sliced views over unified Memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.memory.types import (
    _RECALL_OVERSAMPLE_FACTOR,
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
)
from crewai.memory.unified_memory import Memory


def _ensure_memory_kind(value: Any) -> Any:
    """Backfill ``memory_kind`` on legacy dicts that predate the discriminator.

    Lets pre-1.14.6 configs/checkpoints flow into the discriminated
    ``Memory | MemoryScope | MemorySlice`` union without crashing. Inference:
    ``scopes`` key → ``slice``; ``root_path`` → ``scope``; else ``memory``.
    Pass-through for non-dict values (instances, ``bool``, ``None``).
    """
    if isinstance(value, dict) and "memory_kind" not in value:
        if "scopes" in value:
            value["memory_kind"] = "slice"
        elif "root_path" in value:
            value["memory_kind"] = "scope"
        else:
            value["memory_kind"] = "memory"
    return value


class MemoryScope(BaseModel):
    """View of Memory restricted to a root path. All operations are scoped under that path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_kind: Literal["scope"] = "scope"

    root_path: str = Field(default="/")

    _memory: Memory | None = PrivateAttr(default=None)
    _root: str = PrivateAttr(default="")

    @model_validator(mode="wrap")
    @classmethod
    def _accept_memory(cls, data: Any, handler: Any) -> MemoryScope:
        """Extract memory dependency and normalize root path before validation."""
        if isinstance(data, MemoryScope):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict or MemoryScope, got {type(data).__name__}")
        memory = data.pop("memory", None)
        instance: MemoryScope = handler(data)
        if memory is not None:
            instance._memory = memory
        root = instance.root_path.rstrip("/") or ""
        if root and not root.startswith("/"):
            root = "/" + root
        instance._root = root
        return instance

    def bind(self, memory: Memory) -> Self:
        """Rebind the runtime ``Memory`` dependency after restore.

        Required after deserializing from a checkpoint, since the live
        ``Memory`` cannot be serialized.
        """
        self._memory = memory
        return self

    def _require_memory(self) -> Memory:
        """Return the bound ``Memory`` or raise a clear error if missing."""
        if self._memory is None:
            raise RuntimeError(
                "MemoryScope is not bound to a Memory; call .bind(memory) "
                "after restore."
            )
        return self._memory

    @property
    def read_only(self) -> bool:
        """Whether the underlying memory is read-only."""
        return self._require_memory().read_only

    def _scope_path(self, scope: str | None) -> str:
        if not scope or scope == "/":
            return self._root or "/"
        s = scope.rstrip("/")
        if not s.startswith("/"):
            s = "/" + s
        if not self._root:
            return s
        base = self._root.rstrip("/")
        return f"{base}{s}"

    def remember(
        self,
        content: str,
        scope: str | None = "/",
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
    ) -> MemoryRecord | None:
        """Remember content; scope is relative to this scope's root."""
        path = self._scope_path(scope)
        return self._require_memory().remember(
            content,
            scope=path,
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
        )

    def remember_many(
        self,
        contents: list[str],
        scope: str | None = "/",
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
    ) -> list[MemoryRecord]:
        """Remember multiple items; scope is relative to this scope's root."""
        path = self._scope_path(scope)
        return self._require_memory().remember_many(
            contents,
            scope=path,
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
            agent_role=agent_role,
        )

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
        """Recall within this scope (root path and below)."""
        search_scope = self._scope_path(scope) if scope else (self._root or "/")
        return self._require_memory().recall(
            query,
            scope=search_scope,
            categories=categories,
            limit=limit,
            depth=depth,
            source=source,
            include_private=include_private,
        )

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from content; delegates to underlying Memory."""
        return self._require_memory().extract_memories(content)

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        """Forget within this scope."""
        prefix = self._scope_path(scope) if scope else (self._root or "/")
        return self._require_memory().forget(
            scope=prefix,
            categories=categories,
            older_than=older_than,
            metadata_filter=metadata_filter,
            record_ids=record_ids,
        )

    def list_scopes(self, path: str = "/") -> list[str]:
        """List child scopes under path (relative to this scope's root)."""
        full = self._scope_path(path)
        return self._require_memory().list_scopes(full)

    def info(self, path: str = "/") -> ScopeInfo:
        """Info for path under this scope."""
        full = self._scope_path(path)
        return self._require_memory().info(full)

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        """Tree under path within this scope."""
        full = self._scope_path(path)
        return self._require_memory().tree(full, max_depth=max_depth)

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        """Categories in this scope; path None means this scope root."""
        full = self._scope_path(path) if path else (self._root or "/")
        return self._require_memory().list_categories(full)

    def reset(self, scope: str | None = None) -> None:
        """Reset within this scope."""
        prefix = self._scope_path(scope) if scope else (self._root or "/")
        self._require_memory().reset(scope=prefix)

    def subscope(self, path: str) -> MemoryScope:
        """Return a narrower scope under this scope."""
        child = path.strip("/")
        if not child:
            return MemoryScope(memory=self._memory, root_path=self._root or "/")
        base = self._root.rstrip("/") or ""
        new_root = f"{base}/{child}" if base else f"/{child}"
        return MemoryScope(memory=self._memory, root_path=new_root)


class MemorySlice(BaseModel):
    """View over multiple scopes: recall searches all, remember is a no-op when read_only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_kind: Literal["slice"] = "slice"

    scopes: list[str] = Field(default_factory=list)
    categories: list[str] | None = Field(default=None)
    read_only: bool = Field(default=True)

    _memory: Memory | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _accept_memory(cls, data: Any, handler: Any) -> MemorySlice:
        """Extract memory dependency and normalize scopes before validation."""
        if isinstance(data, MemorySlice):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict or MemorySlice, got {type(data).__name__}")
        memory = data.pop("memory", None)
        data["scopes"] = [s.rstrip("/") or "/" for s in data.get("scopes", [])]
        instance: MemorySlice = handler(data)
        if memory is not None:
            instance._memory = memory
        return instance

    def bind(self, memory: Memory) -> Self:
        """Rebind the runtime ``Memory`` dependency after restore."""
        self._memory = memory
        return self

    def _require_memory(self) -> Memory:
        """Return the bound ``Memory`` or raise a clear error if missing."""
        if self._memory is None:
            raise RuntimeError(
                "MemorySlice is not bound to a Memory; call .bind(memory) "
                "after restore."
            )
        return self._memory

    def remember(
        self,
        content: str,
        scope: str,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
    ) -> MemoryRecord | None:
        """Remember into an explicit scope. No-op when read_only=True."""
        if self.read_only:
            return None
        return self._require_memory().remember(
            content,
            scope=scope,
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
        )

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
        """Recall across all slice scopes; results merged and re-ranked."""
        cats = categories or self.categories
        all_matches: list[MemoryMatch] = []
        for sc in self.scopes:
            matches = self._require_memory().recall(
                query,
                scope=sc,
                categories=cats,
                limit=limit * _RECALL_OVERSAMPLE_FACTOR,
                depth=depth,
                source=source,
                include_private=include_private,
            )
            all_matches.extend(matches)
        seen_ids: set[str] = set()
        unique: list[MemoryMatch] = []
        for m in sorted(all_matches, key=lambda x: x.score, reverse=True):
            if m.record.id not in seen_ids:
                seen_ids.add(m.record.id)
                unique.append(m)
                if len(unique) >= limit:
                    break
        return unique

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from content; delegates to underlying Memory."""
        return self._require_memory().extract_memories(content)

    def list_scopes(self, path: str = "/") -> list[str]:
        """List scopes across all slice roots."""
        out: list[str] = []
        for sc in self.scopes:
            full = f"{sc.rstrip('/')}{path}" if sc != "/" else path
            out.extend(self._require_memory().list_scopes(full))
        return sorted(set(out))

    def info(self, path: str = "/") -> ScopeInfo:
        """Aggregate info across slice scopes (record counts summed)."""
        total_records = 0
        all_categories: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None
        children: list[str] = []
        for sc in self.scopes:
            full = f"{sc.rstrip('/')}{path}" if sc != "/" else path
            inf = self._require_memory().info(full)
            total_records += inf.record_count
            all_categories.update(inf.categories)
            if inf.oldest_record:
                oldest = (
                    inf.oldest_record
                    if oldest is None
                    else min(oldest, inf.oldest_record)
                )
            if inf.newest_record:
                newest = (
                    inf.newest_record
                    if newest is None
                    else max(newest, inf.newest_record)
                )
            children.extend(inf.child_scopes)
        return ScopeInfo(
            path=path,
            record_count=total_records,
            categories=sorted(all_categories),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(set(children)),
        )

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        """Categories and counts across slice scopes."""
        counts: dict[str, int] = {}
        for sc in self.scopes:
            full = (f"{sc.rstrip('/')}{path}" if sc != "/" else path) if path else sc
            for k, v in self._require_memory().list_categories(full).items():
                counts[k] = counts.get(k, 0) + v
        return counts
