"""Encoding flow: full save pipeline from analysis through consolidation.

Orchestrates the encoding side of memory:
1. Decide whether LLM analysis is needed (router)
2. If yes: infer scope, categories, importance, metadata via LLM
3. If no: use caller-provided values with config defaults
4. Embed the content
5. Delegate to ConsolidationFlow for conflict resolution and storage
6. Finalize: ensure a record is persisted
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.memory.analyze import MemoryAnalysis, analyze_for_save
from crewai.memory.types import MemoryConfig, MemoryRecord, embed_text


class EncodingState(BaseModel):
    """State for the encoding flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = ""

    # Caller-provided values (None = infer via LLM)
    scope: str | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] | None = None
    importance: float | None = None
    source: str | None = None
    private: bool = False

    # Resolved values (populated by analysis or defaults step)
    resolved_scope: str = "/"
    resolved_categories: list[str] = Field(default_factory=list)
    resolved_metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_importance: float = 0.5
    resolved_source: str | None = None
    resolved_private: bool = False

    # Embedding and result
    embedding: list[float] = Field(default_factory=list)
    result_record: MemoryRecord | None = None
    consolidation_stats: dict[str, int] = Field(default_factory=dict)

    # Internal routing flag
    needs_analysis: bool = False


class EncodingFlow(Flow[EncodingState]):
    """Flow that owns the full encoding pipeline for memory.remember().

    Routes between LLM-powered deep analysis and a fast skip path,
    embeds the content, then delegates to ConsolidationFlow for
    conflict resolution and storage.
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
        super().__init__()
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()

    # ------------------------------------------------------------------
    # Step 1: Decide whether LLM analysis is needed
    # ------------------------------------------------------------------

    @start()
    def check_inputs(self) -> bool:
        """Check if LLM analysis is needed based on caller-provided fields."""
        self.state.needs_analysis = (
            self.state.scope is None
            or self.state.categories is None
            or self.state.importance is None
            or (self.state.metadata is None and self.state.scope is None)
        )
        return self.state.needs_analysis

    @router(check_inputs)
    def route_analysis(self) -> str:
        """Route to deep analysis or skip based on whether fields are provided."""
        if self.state.needs_analysis:
            return "deep_analysis"
        return "skip_analysis"

    # ------------------------------------------------------------------
    # Step 2a: Deep analysis path (LLM infers scope/categories/importance)
    # ------------------------------------------------------------------

    @listen("deep_analysis")
    def analyze_with_llm(self) -> MemoryAnalysis:
        """Call analyze_for_save to infer scope, categories, importance, metadata."""
        existing_scopes = self._storage.list_scopes("/")
        if not existing_scopes:
            existing_scopes = ["/"]
        existing_categories = list(
            self._storage.list_categories(scope_prefix=None).keys()
        )
        analysis = analyze_for_save(
            self.state.content,
            existing_scopes,
            existing_categories,
            self._llm,
        )
        self.state.resolved_scope = self.state.scope or analysis.suggested_scope or "/"
        self.state.resolved_categories = (
            self.state.categories
            if self.state.categories is not None
            else analysis.categories
        )
        self.state.resolved_importance = (
            self.state.importance
            if self.state.importance is not None
            else analysis.importance
        )
        self.state.resolved_metadata = dict(
            self.state.metadata or {},
            **(
                analysis.extracted_metadata.model_dump()
                if analysis.extracted_metadata
                else {}
            ),
        )
        # Source and private are never LLM-inferred
        self.state.resolved_source = self.state.source
        self.state.resolved_private = self.state.private
        return analysis

    # ------------------------------------------------------------------
    # Step 2b: Skip path (use caller values with config defaults)
    # ------------------------------------------------------------------

    @listen("skip_analysis")
    def use_defaults(self) -> str:
        """Populate resolved fields from caller values and config defaults."""
        self.state.resolved_scope = self.state.scope or "/"
        self.state.resolved_categories = self.state.categories or []
        self.state.resolved_metadata = self.state.metadata or {}
        self.state.resolved_importance = (
            self.state.importance
            if self.state.importance is not None
            else self._config.default_importance
        )
        self.state.resolved_source = self.state.source
        self.state.resolved_private = self.state.private
        return "defaults_applied"

    # ------------------------------------------------------------------
    # Step 3: Embed (both paths converge here)
    # ------------------------------------------------------------------

    @listen(or_(analyze_with_llm, use_defaults))
    def embed_content(self) -> list[float]:
        """Embed the content for vector storage and similarity search."""
        self.state.embedding = embed_text(self._embedder, self.state.content)
        return self.state.embedding

    # ------------------------------------------------------------------
    # Step 4: Consolidate (delegates to ConsolidationFlow)
    # ------------------------------------------------------------------

    @listen(embed_content)
    async def consolidate(self) -> MemoryRecord | None:
        """Run ConsolidationFlow for conflict resolution and storage.

        Defined as ``async`` so the Flow executor awaits it directly in
        the event loop instead of dispatching to a thread.  This lets us
        ``await cflow.kickoff_async()``, the standard Flow entry-point
        that properly initialises state via ``_initialize_state(inputs)``
        -- avoiding the nested ``asyncio.run()`` crash that the sync
        ``kickoff()`` would cause.

        Note: ``list()`` / ``dict()`` calls materialise the
        ``LockedListProxy`` / ``LockedDictProxy`` wrappers returned by
        ``self.state.*`` into plain Python objects.  This is necessary
        because Pydantic's ``model_validate`` (used inside
        ``_initialize_state``) reads list/dict data via C-level internals
        that bypass the proxy's Python-level ``__iter__`` / ``__getitem__``
        and see the empty builtin storage instead.
        """
        from crewai.memory.consolidation_flow import ConsolidationFlow

        cflow = ConsolidationFlow(
            storage=self._storage,
            llm=self._llm,
            embedder=self._embedder,
            config=self._config,
        )
        await cflow.kickoff_async(
            inputs={
                "new_content": self.state.content,
                "new_embedding": list(self.state.embedding),
                "scope": self.state.resolved_scope,
                "categories": list(self.state.resolved_categories),
                "metadata": dict(self.state.resolved_metadata),
                "importance": self.state.resolved_importance,
                "source": self.state.resolved_source,
                "private": self.state.resolved_private,
            },
        )
        self.state.result_record = cflow.state.result_record
        self.state.consolidation_stats = {
            "records_updated": cflow.state.records_updated,
            "records_deleted": cflow.state.records_deleted,
        }
        return self.state.result_record

    # ------------------------------------------------------------------
    # Step 5: Finalize (fallback save if consolidation didn't persist)
    # ------------------------------------------------------------------

    @listen(consolidate)
    def finalize(self) -> MemoryRecord:
        """Ensure a record is persisted. Fallback if ConsolidationFlow returned None."""
        if self.state.result_record is not None:
            return self.state.result_record
        record = MemoryRecord(
            content=self.state.content,
            scope=self.state.resolved_scope,
            categories=self.state.resolved_categories,
            metadata=self.state.resolved_metadata,
            importance=self.state.resolved_importance,
            embedding=self.state.embedding if self.state.embedding else None,
        )
        self._storage.save([record])
        self.state.result_record = record
        return record
