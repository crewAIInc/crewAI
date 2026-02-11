"""Consolidation flow: decide insert/update/delete when new content is similar to existing memories."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, router, start
from crewai.memory.analyze import (
    ConsolidationPlan,
    analyze_for_consolidation,
)
from crewai.memory.types import MemoryConfig, MemoryRecord, embed_text


class ConsolidationState(BaseModel):
    """State for the consolidation flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    new_content: str = ""
    new_embedding: list[float] = Field(default_factory=list)
    scope: str = "/"
    categories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5
    similar_records: list[MemoryRecord] = Field(default_factory=list)
    top_similarity: float = 0.0
    plan: ConsolidationPlan | None = None
    result_record: MemoryRecord | None = None
    records_updated: int = 0
    records_deleted: int = 0


class ConsolidationFlow(Flow[ConsolidationState]):
    """Flow that gates and runs memory consolidation on remember()."""

    _skip_auto_memory: bool = True

    initial_state = ConsolidationState

    def __init__(
        self,
        storage: Any,
        llm: Any,
        embedder: Any,
        config: MemoryConfig | None = None,
    ) -> None:
        super().__init__()
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()

    @start()
    def find_similar(self) -> list[MemoryRecord]:
        """Search for existing records similar to the new content (cheap pre-check)."""
        if not self.state.new_embedding:
            self.state.top_similarity = 0.0
            return []
        scope_prefix = self.state.scope if self.state.scope.strip("/") else None
        raw = self._storage.search(
            self.state.new_embedding,
            scope_prefix=scope_prefix,
            categories=None,
            limit=self._config.consolidation_limit,
            min_score=0.0,
        )
        records = [r for r, _ in raw]
        self.state.similar_records = records
        if raw:
            _, top_score = raw[0]
            self.state.top_similarity = float(top_score)
        else:
            self.state.top_similarity = 0.0
        return records

    @router(find_similar)
    def check_threshold(self) -> str:
        """Gate: only run LLM consolidation when similarity is above threshold."""
        if self.state.top_similarity < self._config.consolidation_threshold:
            return "insert_only"
        return "needs_consolidation"

    @listen("needs_consolidation")
    def analyze_conflicts(self) -> ConsolidationPlan:
        """Run LLM to decide keep/update/delete per record and whether to insert new."""
        plan = analyze_for_consolidation(
            self.state.new_content,
            self.state.similar_records,
            self._llm,
        )
        self.state.plan = plan
        return plan

    @listen("insert_only")
    def insert_only_path(self) -> None:
        """Fast path: no similar records above threshold; plan stays None."""
        return None

    @listen(insert_only_path)
    @listen(analyze_conflicts)
    def execute_plan(self) -> MemoryRecord | None:
        """Apply consolidation plan or insert new record; set state.result_record."""
        now = datetime.utcnow()
        plan = self.state.plan
        record_by_id = {r.id: r for r in self.state.similar_records}

        if plan is not None:
            for action in plan.actions:
                if action.action == "delete":
                    self._storage.delete(record_ids=[action.record_id])
                    self.state.records_deleted += 1
                elif action.action == "update" and action.new_content:
                    existing = record_by_id.get(action.record_id)
                    if existing is not None:
                        new_embedding = embed_text(
                            self._embedder, action.new_content
                        )
                        updated = MemoryRecord(
                            id=existing.id,
                            content=action.new_content,
                            scope=existing.scope,
                            categories=existing.categories,
                            metadata=existing.metadata,
                            importance=existing.importance,
                            created_at=existing.created_at,
                            last_accessed=now,
                            embedding=new_embedding if new_embedding else existing.embedding,
                        )
                        self._storage.update(updated)
                        self.state.records_updated += 1
                        record_by_id[action.record_id] = updated

            if not plan.insert_new:
                first_updated = next(
                    (
                        record_by_id[a.record_id]
                        for a in plan.actions
                        if a.action == "update" and a.record_id in record_by_id
                    ),
                    None,
                )
                if first_updated is not None:
                    self.state.result_record = first_updated
                    return first_updated
                if self.state.similar_records:
                    self.state.result_record = self.state.similar_records[0]
                    return self.state.similar_records[0]

        new_record = MemoryRecord(
            content=self.state.new_content,
            scope=self.state.scope,
            categories=self.state.categories,
            metadata=self.state.metadata,
            importance=self.state.importance,
            embedding=self.state.new_embedding if self.state.new_embedding else None,
        )
        if not new_record.embedding:
            new_embedding = embed_text(self._embedder, self.state.new_content)
            new_record = new_record.model_copy(update={"embedding": new_embedding if new_embedding else None})
        self._storage.save([new_record])
        self.state.result_record = new_record
        return new_record
