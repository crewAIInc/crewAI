import logging
import os
from typing import Any, List, Literal, Optional, Type

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SemanticScholarSearchToolSchema(BaseModel):
    """Input for Semantic Scholar Search Tool."""

    search_query: Optional[str] = Field(
        None,
        description="Search query for academic papers (used in search mode)",
    )
    mode: Literal["search", "lookup"] = Field(
        default="search",
        description="Operation mode: 'search' for keyword search, 'lookup' for paper details by ID",
    )
    paper_id: Optional[str] = Field(
        None,
        description="Semantic Scholar paper ID (required for lookup mode)",
    )
    year: Optional[str] = Field(
        None,
        description="Year filter for papers (e.g., '2023-', '2020-2023')",
    )
    fields: Optional[str] = Field(
        default="title,url,year,citationCount,abstract,authors",
        description="Comma-separated list of fields to return (e.g., title,url,year,citationCount,abstract,authors)",
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return (max 1000 for search)",
    )
    sort: Optional[str] = Field(
        None,
        description="Sort results by: 'citationCount', 'publicationDate', or 'paperId'",
    )
    min_citation_count: Optional[int] = Field(
        None,
        description="Minimum number of citations to include in results",
    )
    fields_of_study: Optional[str] = Field(
        None,
        description="Filter by fields of study (e.g., 'Computer Science,Physics')",
    )
    publication_types: Optional[str] = Field(
        None,
        description="Filter by publication types (e.g., 'JournalArticle,ConferencePaper')",
    )
    open_access_pdf: Optional[bool] = Field(
        None,
        description="Filter to only papers with open access PDFs",
    )
    api_key: Optional[str] = Field(
        None,
        description="Semantic Scholar API key (optional, can also set SEMANTIC_SCHOLAR_API_KEY env variable)",
    )


class SemanticScholarSearchTool(BaseTool):
    """Tool for searching academic papers on Semantic Scholar.

    Supports two modes:
    - search: Keyword search for academic papers using the bulk search endpoint
    - lookup: Get paper details by Semantic Scholar paper ID
    """

    name: str = "Search Semantic Scholar"
    description: str = (
        "A tool to search academic papers on Semantic Scholar. "
        "Supports keyword search (bulk) and paper lookup by ID. "
        "Use 'search' mode for keyword queries, 'lookup' mode for specific paper details."
    )
    args_schema: Type[BaseModel] = SemanticScholarSearchToolSchema
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: Optional[str] = None
    env_vars: List[EnvVar] = [
        EnvVar(
            name="SEMANTIC_SCHOLAR_API_KEY",
            description="API key for Semantic Scholar",
            required=False,
        ),
    ]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Semantic Scholar search tool.

        Args:
            api_key: Optional API key for Semantic Scholar. If not provided,
                     will look for SEMANTIC_SCHOLAR_API_KEY environment variable.
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    def _get_api_key(self) -> Optional[str]:
        """Get the API key from instance attribute."""
        return self.api_key

    def _build_headers(self) -> dict:
        """Build request headers with optional API key."""
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _search_papers(self, search_query: str, **kwargs) -> dict:
        """Search for papers using the bulk search endpoint."""
        endpoint = f"{self.base_url}/paper/search/bulk"

        params = {
            "query": search_query,
            "limit": kwargs.get("limit", 10),
            "fields": kwargs.get(
                "fields", "title,url,year,citationCount,abstract,authors"
            ),
        }

        if kwargs.get("year"):
            params["year"] = kwargs["year"]
        if kwargs.get("sort"):
            params["sort"] = kwargs["sort"]
        if kwargs.get("min_citation_count"):
            params["minCitationCount"] = kwargs["min_citation_count"]
        if kwargs.get("fields_of_study"):
            params["fieldsOfStudy"] = kwargs["fields_of_study"]
        if kwargs.get("publication_types"):
            params["publicationTypes"] = kwargs["publication_types"]
        if kwargs.get("open_access_pdf") is not None:
            params["openAccessPdf"] = kwargs["open_access_pdf"]

        headers = {"Content-Type": "application/json"}
        api_key = kwargs.get("api_key") or self.api_key
        if api_key:
            headers["x-api-key"] = api_key

        try:
            response = requests.get(
                endpoint, params=params, headers=headers, timeout=30
            )
            if response.status_code == 403 and api_key:
                logger.warning("API key rejected, retrying without authentication")
                headers.pop("x-api-key", None)
                response = requests.get(
                    endpoint, params=params, headers=headers, timeout=30
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403 and api_key:
                logger.warning("API key rejected, retrying without authentication")
                headers.pop("x-api-key", None)
                response = requests.get(
                    endpoint, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                return response.json()
            raise

    def _lookup_paper(self, paper_id: str, **kwargs) -> dict:
        """Get paper details by paper ID."""
        endpoint = f"{self.base_url}/paper/{paper_id}"

        params = {
            "fields": kwargs.get(
                "fields", "title,url,year,citationCount,abstract,authors"
            )
        }

        headers = {"Content-Type": "application/json"}
        api_key = kwargs.get("api_key") or self.api_key
        if api_key:
            headers["x-api-key"] = api_key

        try:
            response = requests.get(
                endpoint, params=params, headers=headers, timeout=30
            )
            if response.status_code == 403 and api_key:
                logger.warning("API key rejected, retrying without authentication")
                headers.pop("x-api-key", None)
                response = requests.get(
                    endpoint, params=params, headers=headers, timeout=30
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403 and api_key:
                logger.warning("API key rejected, retrying without authentication")
                headers.pop("x-api-key", None)
                response = requests.get(
                    endpoint, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                return response.json()
            raise

    def _format_paper(self, paper: dict) -> str:
        """Format a single paper result as a readable string."""
        lines = []

        title = paper.get("title", "N/A")
        lines.append(f"Title: {title}")

        paper_id = paper.get("paperId", "N/A")
        lines.append(f"Paper ID: {paper_id}")

        url = paper.get("url", "N/A")
        lines.append(f"URL: {url}")

        year = paper.get("year", "N/A")
        lines.append(f"Year: {year}")

        citation_count = paper.get("citationCount", "N/A")
        lines.append(f"Citations: {citation_count}")

        abstract = paper.get("abstract")
        if abstract:
            abstract_short = (
                abstract[:500] + "..." if len(abstract) > 500 else abstract
            )
            lines.append(f"Abstract: {abstract_short}")

        authors = paper.get("authors", [])
        if authors:
            author_names = [a.get("name", "Unknown") for a in authors[:5]]
            lines.append(f"Authors: {', '.join(author_names)}")

        return "\n".join(lines)

    def _format_papers_list(self, papers: list) -> str:
        """Format a list of papers as a readable string."""
        if not papers:
            return "No papers found."

        lines = []
        for i, paper in enumerate(papers, 1):
            lines.append(f"\n--- Paper {i} ---")
            lines.append(self._format_paper(paper))
            lines.append("")

        return "\n".join(lines)

    def _run(self, **kwargs: Any) -> Any:
        """Execute the search or lookup operation."""
        api_key = kwargs.pop("api_key", None) or self.api_key
        mode = kwargs.get("mode", "search")

        if mode == "lookup":
            paper_id = kwargs.get("paper_id")
            if not paper_id:
                raise ValueError("paper_id is required for lookup mode")

            paper = self._lookup_paper(
                paper_id,
                fields=kwargs.get("fields"),
                api_key=api_key,
            )
            return self._format_paper(paper)

        else:
            search_query = kwargs.get("search_query")
            if not search_query:
                raise ValueError("search_query is required for search mode")

            result = self._search_papers(
                search_query,
                year=kwargs.get("year"),
                fields=kwargs.get("fields"),
                limit=kwargs.get("limit", 10),
                sort=kwargs.get("sort"),
                min_citation_count=kwargs.get("min_citation_count"),
                fields_of_study=kwargs.get("fields_of_study"),
                publication_types=kwargs.get("publication_types"),
                open_access_pdf=kwargs.get("open_access_pdf"),
                api_key=api_key,
            )

            total = result.get("total", 0)
            papers = result.get("data", [])

            output = f"Found {total} papers.\n"
            output += self._format_papers_list(papers)

            return output