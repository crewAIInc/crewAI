"""Scoped and sliced views over unified Memory."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai.memory.unified_memory import Memory

from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo


class MemoryScope:
    """View of Memory restricted to a root path. All operations are scoped under that path."""

    def __init__(self, memory: "Memory", root_path: str) -> None:
        """Initialize scope.

        Args:
            memory: The underlying Memory instance.
            root_path: Root path for this scope (e.g. /agent/1).
        """
        self._memory = memory
        self._root = root_path.rstrip("/") or ""
        if self._root and not self._root.startswith("/"):
            self._root = "/" + self._root

    def _scope_path(self, scope: str | None) -> str:
        if not scope or scope == "/":
            return self._root or "/"
        s = scope.rstrip("/")
        if not s.startswith("/"):
            s = "/" + s
        if not self._root:
            return s
        base = self._root.rstrip("/")
        return f"{base}{s}" if s == "/" else f"{base}{s}"

    def remember(
        self,
        content: str,
        scope: str | None = "/",
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        """Remember content; scope is relative to this scope's root."""
        path = self._scope_path(scope)
        return self._memory.remember(
            content,
            scope=path,
            categories=categories,
            metadata=metadata,
            importance=importance,
        )

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: str = "auto",
    ) -> list[MemoryMatch]:
        """Recall within this scope (root path and below)."""
        search_scope = self._scope_path(scope) if scope else (self._root or "/")
        return self._memory.recall(
            query,
            scope=search_scope,
            categories=categories,
            limit=limit,
            depth=depth,
        )

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from content; delegates to underlying Memory."""
        return self._memory.extract_memories(content)

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
        return self._memory.forget(
            scope=prefix,
            categories=categories,
            older_than=older_than,
            metadata_filter=metadata_filter,
            record_ids=record_ids,
        )

    def list_scopes(self, path: str = "/") -> list[str]:
        """List child scopes under path (relative to this scope's root)."""
        full = self._scope_path(path)
        return self._memory.list_scopes(full)

    def info(self, path: str = "/") -> ScopeInfo:
        """Info for path under this scope."""
        full = self._scope_path(path)
        return self._memory.info(full)

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        """Tree under path within this scope."""
        full = self._scope_path(path)
        return self._memory.tree(full, max_depth=max_depth)

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        """Categories in this scope; path None means this scope root."""
        full = self._scope_path(path) if path else (self._root or "/")
        return self._memory.list_categories(full)

    def reset(self, scope: str | None = None) -> None:
        """Reset within this scope."""
        prefix = self._scope_path(scope) if scope else (self._root or "/")
        self._memory.reset(scope=prefix)

    def subscope(self, path: str) -> MemoryScope:
        """Return a narrower scope under this scope."""
        child = path.strip("/")
        if not child:
            return MemoryScope(self._memory, self._root or "/")
        base = self._root.rstrip("/") or ""
        new_root = f"{base}/{child}" if base else f"/{child}"
        return MemoryScope(self._memory, new_root)


class MemorySlice:
    """View over multiple scopes: recall searches all, remember requires explicit scope unless read_only."""

    def __init__(
        self,
        memory: "Memory",
        scopes: list[str],
        categories: list[str] | None = None,
        read_only: bool = True,
    ) -> None:
        """Initialize slice.

        Args:
            memory: The underlying Memory instance.
            scopes: List of scope paths to include.
            categories: Optional category filter for recall.
            read_only: If True, remember() raises PermissionError.
        """
        self._memory = memory
        self._scopes = [s.rstrip("/") or "/" for s in scopes]
        self._categories = categories
        self._read_only = read_only

    def remember(
        self,
        content: str,
        scope: str,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        """Remember into an explicit scope. Required when read_only=False."""
        if self._read_only:
            raise PermissionError("This MemorySlice is read-only")
        return self._memory.remember(
            content,
            scope=scope,
            categories=categories,
            metadata=metadata,
            importance=importance,
        )

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: str = "auto",
    ) -> list[MemoryMatch]:
        """Recall across all slice scopes; results merged and re-ranked."""
        cats = categories or self._categories
        all_matches: list[MemoryMatch] = []
        for sc in self._scopes:
            matches = self._memory.recall(
                query,
                scope=sc,
                categories=cats,
                limit=limit * 2,
                depth=depth,
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
        return self._memory.extract_memories(content)

    def list_scopes(self, path: str = "/") -> list[str]:
        """List scopes across all slice roots."""
        out: list[str] = []
        for sc in self._scopes:
            full = f"{sc.rstrip('/')}{path}" if sc != "/" else path
            out.extend(self._memory.list_scopes(full))
        return sorted(set(out))

    def info(self, path: str = "/") -> ScopeInfo:
        """Aggregate info across slice scopes (record counts summed)."""
        total_records = 0
        all_categories: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None
        children: list[str] = []
        for sc in self._scopes:
            full = f"{sc.rstrip('/')}{path}" if sc != "/" else path
            inf = self._memory.info(full)
            total_records += inf.record_count
            all_categories.update(inf.categories)
            if inf.oldest_record:
                oldest = inf.oldest_record if oldest is None else min(oldest, inf.oldest_record)
            if inf.newest_record:
                newest = inf.newest_record if newest is None else max(newest, inf.newest_record)
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
        for sc in self._scopes:
            full = (f"{sc.rstrip('/')}{path}" if sc != "/" else path) if path else sc
            for k, v in self._memory.list_categories(full).items():
                counts[k] = counts.get(k, 0) + v
        return counts
