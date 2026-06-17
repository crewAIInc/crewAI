from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any

from mimir_client import MimirClient  # The official client required for Mimir
from crewai.memory.types import MemoryRecord, ScopeInfo

_logger = logging.getLogger(__name__)

class MimirStorage:
    """Mimir-backed storage for persistent cross-session agent memory."""

    def __init__(
        self,
        path: str | None = None,
        table_name: str = "crewai_memory",
        **kwargs: Any
    ) -> None:
        """Initialize connection with Mimir memory engine."""
        # If no path is provided, Mimir defaults to a local-first database
        self.db_path = path or "./mimir_memory.db"
        self._table_name = table_name
        self.client = MimirClient(self.db_path)
        _logger.info(f"Mimir Storage Backend initialized at {self.db_path}")

    def save(self, records: list[MemoryRecord]) -> None:
        """Save memory records into Mimir persistent database using 'remember'."""
        if not records:
            return
        
        for record in records:
            # Create structured metadata for cross-session persistence
            metadata = record.metadata or {}
            metadata.update({
                "scope": record.scope,
                "categories": record.categories,
                "created_at": record.created_at.isoformat()
            })
            
            # Mimir uses 'remember' method to store long-term stable information
            self.client.remember(
                content=record.content,
                metadata=metadata
            )

    def search(
        self,
        query_embedding: list[float], # Provided by CrewAI interface
        scope_prefix: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        **kwargs: Any
    ) -> list[tuple[MemoryRecord, float]]:
        """Search memories using Mimir hybrid 'recall' function."""
        # Mimir supports hybrid search. We use recall to query relevant memories
        raw_results = self.client.recall(query=kwargs.get("query_text", ""), limit=limit)
        
        out: list[tuple[MemoryRecord, float]] = []
        for row in raw_results:
            # Convert Mimir output row into CrewAI MemoryRecord format
            record = MemoryRecord(
                id=str(row.get("id")),
                content=str(row.get("content")),
                scope=str(row.get("metadata", {}).get("scope", "/")),
                categories=row.get("metadata", {}).get("categories", []),
                metadata=row.get("metadata", {}),
                created_at=datetime.fromisoformat(row.get("metadata", {}).get("created_at", datetime.utcnow().isoformat()))
            )
            score = float(row.get("score", 1.0))
            if score >= min_score:
                out.append((record, score))
                
        return out[:limit]

    def delete(self, record_ids: list[str] | None = None, **kwargs: Any) -> int:
        """Remove memories using Mimir 'forget' function."""
        if not record_ids:
            return 0
        count = 0
        for rid in record_ids:
            # Mimir uses 'forget' to remove or decay a specific memory record
            self.client.forget(record_id=rid)
            count += 1
        return count

    # --- Async implementations required by CrewAI architecture ---
    async def asave(self, records: list[MemoryRecord]) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        **kwargs: Any
    ) -> list[tuple[MemoryRecord, float]]:
        return self.search(query_embedding, scope_prefix, limit, min_score, **kwargs)

    async def adelete(self, record_ids: list[str] | None = None, **kwargs: Any) -> int:
        return self.delete(record_ids, **kwargs)