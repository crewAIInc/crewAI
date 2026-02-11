"""RLM-inspired intelligent recall flow for memory retrieval.

Implements adaptive-depth retrieval with:
- LLM query distillation into targeted sub-queries
- Keyword-driven category filtering
- Time-based filtering from temporal hints
- Parallel multi-query, multi-scope search
- Confidence-based routing with iterative deepening (budget loop)
- Evidence gap tracking propagated to results
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, router, start
from crewai.memory.analyze import QueryAnalysis, analyze_query
from crewai.memory.types import (
    _RECALL_OVERSAMPLE_FACTOR,
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    compute_composite_score,
    embed_text,
)


class RecallState(BaseModel):
    """State for the recall flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = ""
    scope: str | None = None
    categories: list[str] | None = None
    inferred_categories: list[str] = Field(default_factory=list)
    time_cutoff: datetime | None = None
    limit: int = 10
    query_embeddings: list[tuple[str, list[float]]] = Field(default_factory=list)
    query_analysis: QueryAnalysis | None = None
    candidate_scopes: list[str] = Field(default_factory=list)
    chunk_findings: list[Any] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    final_results: list[MemoryMatch] = Field(default_factory=list)
    exploration_budget: int = 1


class RecallFlow(Flow[RecallState]):
    """RLM-inspired intelligent memory recall flow.

    Analyzes the query via LLM to produce targeted sub-queries and filters,
    embeds each sub-query, searches across candidate scopes in parallel,
    and iteratively deepens exploration when confidence is low.
    """

    _skip_auto_memory: bool = True

    initial_state = RecallState

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
    # Helpers
    # ------------------------------------------------------------------

    def _merged_categories(self) -> list[str] | None:
        """Merge caller-supplied and LLM-inferred categories."""
        merged = list(
            set((self.state.categories or []) + self.state.inferred_categories)
        )
        return merged or None

    def _do_search(self) -> list[dict[str, Any]]:
        """Run parallel search across (embeddings x scopes) with filters.

        Populates ``state.chunk_findings`` and ``state.confidence``.
        Returns the findings list.
        """
        search_categories = self._merged_categories()

        def _search_one(
            embedding: list[float], scope: str
        ) -> tuple[str, list[tuple[MemoryRecord, float]]]:
            raw = self._storage.search(
                embedding,
                scope_prefix=scope,
                categories=search_categories,
                limit=self.state.limit * _RECALL_OVERSAMPLE_FACTOR,
                min_score=0.0,
            )
            # Post-filter by time cutoff
            if self.state.time_cutoff and raw:
                raw = [
                    (r, s) for r, s in raw if r.created_at >= self.state.time_cutoff
                ]
            return scope, raw

        # Build (embedding, scope) task list
        tasks: list[tuple[list[float], str]] = [
            (embedding, scope)
            for _query_text, embedding in self.state.query_embeddings
            for scope in self.state.candidate_scopes
        ]

        findings: list[dict[str, Any]] = []

        if len(tasks) <= 1:
            for emb, sc in tasks:
                scope, results = _search_one(emb, sc)
                if results:
                    top_composite, _ = compute_composite_score(
                        results[0][0], results[0][1], self._config
                    )
                    findings.append({
                        "scope": scope,
                        "results": results,
                        "top_score": top_composite,
                    })
        else:
            with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
                futures = {
                    pool.submit(_search_one, emb, sc): (emb, sc)
                    for emb, sc in tasks
                }
                for future in as_completed(futures):
                    scope, results = future.result()
                    if results:
                        top_composite, _ = compute_composite_score(
                            results[0][0], results[0][1], self._config
                        )
                        findings.append({
                            "scope": scope,
                            "results": results,
                            "top_score": top_composite,
                        })

        self.state.chunk_findings = findings
        self.state.confidence = max(
            (f["top_score"] for f in findings), default=0.0
        )
        return findings

    # ------------------------------------------------------------------
    # Flow steps
    # ------------------------------------------------------------------

    @start()
    def analyze_query_step(self) -> QueryAnalysis:
        """Analyze the query, embed distilled sub-queries, extract filters."""
        self.state.exploration_budget = self._config.exploration_budget
        available = self._storage.list_scopes(self.state.scope or "/")
        if not available:
            available = ["/"]
        scope_info = (
            self._storage.get_scope_info(self.state.scope or "/")
            if self.state.scope
            else None
        )
        analysis = analyze_query(
            self.state.query,
            available,
            scope_info,
            self._llm,
        )
        self.state.query_analysis = analysis

        # Wire keywords -> category filter
        if analysis.keywords:
            self.state.inferred_categories = analysis.keywords

        # Parse time_filter into a datetime cutoff
        if analysis.time_filter:
            try:
                self.state.time_cutoff = datetime.fromisoformat(analysis.time_filter)
            except ValueError:
                # If the time filter isn't a valid ISO format, ignore it and proceed without a cutoff.
                pass

        # Embed distilled recall queries (or fall back to original query)
        queries = analysis.recall_queries if analysis.recall_queries else [self.state.query]
        pairs: list[tuple[str, list[float]]] = []
        for q in queries[:3]:
            emb = embed_text(self._embedder, q)
            if emb:
                pairs.append((q, emb))
        if not pairs:
            emb = embed_text(self._embedder, self.state.query)
            if emb:
                pairs.append((self.state.query, emb))
        self.state.query_embeddings = pairs
        return analysis

    @listen(analyze_query_step)
    def filter_and_chunk(self) -> list[str]:
        """Select candidate scopes based on LLM analysis."""
        analysis = self.state.query_analysis
        scope_prefix = (self.state.scope or "/").rstrip("/") or "/"
        if analysis and analysis.suggested_scopes:
            candidates = [s for s in analysis.suggested_scopes if s]
        else:
            candidates = self._storage.list_scopes(scope_prefix)
        if not candidates:
            info = self._storage.get_scope_info(scope_prefix)
            if info.record_count > 0:
                candidates = [scope_prefix]
            else:
                candidates = [scope_prefix]
        self.state.candidate_scopes = candidates[:20]
        return self.state.candidate_scopes

    @listen(filter_and_chunk)
    def search_chunks(self) -> list[Any]:
        """Initial parallel search across (embeddings x scopes) with filters."""
        return self._do_search()

    @router(search_chunks)
    def decide_depth(self) -> str:
        """Route based on confidence, complexity, and remaining budget."""
        analysis = self.state.query_analysis
        if (
            analysis
            and analysis.complexity == "complex"
            and self.state.confidence < self._config.complex_query_threshold
        ):
            if self.state.exploration_budget > 0:
                return "explore_deeper"
        if self.state.confidence >= self._config.confidence_threshold_high:
            return "synthesize"
        if (
            self.state.exploration_budget > 0
            and self.state.confidence < self._config.confidence_threshold_low
        ):
            return "explore_deeper"
        return "synthesize"

    @listen("explore_deeper")
    def recursive_exploration(self) -> list[Any]:
        """Feed top results back to LLM for deeper context extraction.

        Decrements the exploration budget so the loop terminates.
        """
        self.state.exploration_budget -= 1

        enhanced = []
        for finding in self.state.chunk_findings:
            if not finding.get("results"):
                continue
            content_parts = [r[0].content for r in finding["results"][:5]]
            chunk_text = "\n---\n".join(content_parts)
            prompt = (
                f"Query: {self.state.query}\n\n"
                f"Relevant memory excerpts:\n{chunk_text}\n\n"
                "Extract the most relevant information for the query. "
                "If something is missing, say what's missing in one short line."
            )
            try:
                response = self._llm.call([{"role": "user", "content": prompt}])
                if isinstance(response, str) and "missing" in response.lower():
                    self.state.evidence_gaps.append(response[:200])
                enhanced.append({
                    "scope": finding["scope"],
                    "extraction": response,
                    "results": finding["results"],
                })
            except Exception:
                enhanced.append({
                    "scope": finding["scope"],
                    "extraction": "",
                    "results": finding["results"],
                })
        self.state.chunk_findings = enhanced
        return enhanced

    @listen(recursive_exploration)
    def re_search(self) -> list[Any]:
        """Re-search after exploration to update confidence for the router loop."""
        return self._do_search()

    @router(re_search)
    def re_decide_depth(self) -> str:
        """Re-evaluate depth after re-search. Same logic as decide_depth."""
        return self.decide_depth()

    @listen("synthesize")
    def synthesize_results(self) -> list[MemoryMatch]:
        """Deduplicate, composite-score, rank, and attach evidence gaps."""
        seen_ids: set[str] = set()
        matches: list[MemoryMatch] = []
        for finding in self.state.chunk_findings:
            if not isinstance(finding, dict):
                continue
            results = finding.get("results", [])
            if not isinstance(results, list):
                continue
            for item in results:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    record, score = item[0], item[1]
                else:
                    continue
                if isinstance(record, MemoryRecord) and record.id not in seen_ids:
                    seen_ids.add(record.id)
                    composite, reasons = compute_composite_score(
                        record, float(score), self._config
                    )
                    matches.append(
                        MemoryMatch(
                            record=record,
                            score=composite,
                            match_reasons=reasons,
                        )
                    )
        matches.sort(key=lambda m: m.score, reverse=True)
        self.state.final_results = matches[: self.state.limit]

        # Attach evidence gaps to the first result so callers can inspect them
        if self.state.evidence_gaps and self.state.final_results:
            self.state.final_results[0].evidence_gaps = list(self.state.evidence_gaps)

        return self.state.final_results
