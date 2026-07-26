import json
import os
from typing import Any, ClassVar

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, model_validator


class BGPTPaperToolSchema(BaseModel):
    """Input schema for BGPTPaperTool."""

    search_query: str | None = Field(
        None,
        description="Natural-language scientific search query, e.g. 'CAR-T response rates'",
    )
    doi: str | None = Field(
        None,
        description="Paper DOI for direct lookup, e.g. '10.1038/s41586-024-07386-0'",
    )
    num_results: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of papers to return for keyword search",
    )
    days_back: int | None = Field(
        None,
        ge=1,
        description="Only return papers published within the last N days",
    )

    @model_validator(mode="after")
    def require_query_or_doi(self) -> "BGPTPaperToolSchema":
        if not self.search_query and not self.doi:
            raise ValueError("Provide either search_query or doi")
        return self


class BGPTPaperTool(BaseTool):
    """
    BGPTPaperTool - Search scientific papers with structured full-text evidence.

    Uses the plain HTTP REST API at bgpt.pro (not MCP transport):
    - POST https://bgpt.pro/api/mcp-search
    - POST https://bgpt.pro/api/mcp-doi-lookup

    Returns methods, sample sizes, limitations, conflicts of interest,
    falsifiability prompts, and other evidence fields beyond abstracts.
    """

    name: str = "BGPT Scientific Paper Evidence Search"
    description: str = (
        "Search scientific papers or look up a DOI via BGPT and return structured "
        "experimental evidence fields (methods, limitations, COI, falsifiability)."
    )
    args_schema: type[BaseModel] = BGPTPaperToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    search_url: str = "https://bgpt.pro/api/mcp-search"
    lookup_url: str = "https://bgpt.pro/api/mcp-doi-lookup"
    timeout: int = 30
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BGPT_API_KEY",
                description="Optional paid-tier API key (free tier needs no key)",
                required=False,
            ),
        ]
    )
    EVIDENCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "title",
        "doi",
        "publication_date",
        "publication_name",
        "one_sentence_summary",
        "methods_and_experimental_techniques",
        "sample_size_and_population",
        "results_and_conclusions",
        "paper_limitations_and_biases",
        "conflict_of_interest",
        "data_availability_statements",
        "code_and_data_links",
        "how_to_falsify",
        "study_blindspots",
    )

    def _format_paper(self, paper: dict[str, Any]) -> str:
        lines = []
        for field in self.EVIDENCE_FIELDS:
            value = paper.get(field)
            if value:
                lines.append(f"{field}: {value}")
        if not lines:
            lines.append(json.dumps(paper, indent=2))
        return "\n".join(lines)

    def _run(
        self,
        search_query: str | None = None,
        doi: str | None = None,
        num_results: int = 5,
        days_back: int | None = None,
        **_: Any,
    ) -> str:
        api_key = os.environ.get("BGPT_API_KEY")
        try:
            if doi:
                payload: dict[str, Any] = {"doi": doi}
                if api_key:
                    payload["api_key"] = api_key
                response = requests.post(
                    self.lookup_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                paper = data.get("result") or data.get("paper")
                if not paper:
                    return f"No paper found for DOI {doi}"
                return self._format_paper(paper)

            payload = {"query": search_query, "num_results": num_results}
            if days_back is not None:
                payload["days_back"] = days_back
            if api_key:
                payload["api_key"] = api_key

            response = requests.post(
                self.search_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            papers = data.get("results") or data.get("papers") or []
            if not papers:
                return f"No papers found for query: {search_query}"
            formatted = [self._format_paper(paper) for paper in papers]
            return "\n\n---\n\n".join(formatted)
        except requests.RequestException as exc:
            return f"BGPT request failed: {exc}"
        except Exception as exc:
            return f"Unexpected BGPT error: {exc}"
