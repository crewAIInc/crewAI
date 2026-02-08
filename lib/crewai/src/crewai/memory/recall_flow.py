"""RLM-inspired intelligent recall flow for memory retrieval."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, router, start
from crewai.memory.analyze import QueryAnalysis, analyze_query
from crewai.memory.types import (
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    compute_composite_score,
)


class RecallState(BaseModel):
    """State for the recall flow."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = ""
    scope: str | None = None
    categories: list[str] | None = None
    limit: int = 10
    query_embedding: list[float] = Field(default_factory=list)
    query_analysis: QueryAnalysis | None = None
    candidate_scopes: list[str] = Field(default_factory=list)
    chunk_findings: list[Any] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    final_results: list[MemoryMatch] = Field(default_factory=list)
    exploration_budget: int = 1


class RecallFlow(Flow[RecallState]):
    """RLM-inspired intelligent memory recall flow."""

    initial_state = RecallState

    def __init__(
        self,
        storage: Any,
        llm: Any,
        config: MemoryConfig | None = None,
    ) -> None:
        super().__init__()
        self._storage = storage
        self._llm = llm
        self._config = config or MemoryConfig()

    @start()
    def analyze_query_step(self) -> QueryAnalysis:
        available = self._storage.list_scopes(self.state.scope or "/")
        if not available:
            available = ["/"]
        scope_info = self._storage.get_scope_info(self.state.scope or "/") if self.state.scope else None
        analysis = analyze_query(
            self.state.query,
            available,
            scope_info,
            self._llm,
        )
        self.state.query_analysis = analysis
        return analysis

    @listen(analyze_query_step)
    def filter_and_chunk(self) -> list[str]:
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
        findings = []
        for scope in self.state.candidate_scopes:
            results = self._storage.search(
                self.state.query_embedding,
                scope_prefix=scope,
                categories=self.state.categories,
                limit=self.state.limit * 2,
                min_score=0.0,
            )
            if results:
                top_composite, _ = compute_composite_score(
                    results[0][0], results[0][1], self._config
                )
                findings.append(
                    {
                        "scope": scope,
                        "results": results,
                        "top_score": top_composite,
                    }
                )
        self.state.chunk_findings = findings
        if findings:
            self.state.confidence = max(f["top_score"] for f in findings)
        else:
            self.state.confidence = 0.0
        return findings

    @router(search_chunks)
    def decide_depth(self) -> str:
        analysis = self.state.query_analysis
        if analysis and analysis.complexity == "complex" and self.state.confidence < 0.7:
            if self.state.exploration_budget > 0:
                return "explore_deeper"
        if self.state.confidence >= 0.8:
            return "synthesize"
        if self.state.exploration_budget > 0 and self.state.confidence < 0.5:
            return "explore_deeper"
        return "synthesize"

    @listen("explore_deeper")
    def recursive_exploration(self) -> list[Any]:
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
                enhanced.append({"scope": finding["scope"], "extraction": response, "results": finding["results"]})
            except Exception:
                enhanced.append({"scope": finding["scope"], "extraction": "", "results": finding["results"]})
        self.state.chunk_findings = enhanced
        return enhanced

    @listen("synthesize")
    @listen(recursive_exploration)
    def synthesize_results(self) -> list[MemoryMatch]:
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
        return self.state.final_results
