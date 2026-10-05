import requests
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class OpenAlexResearchToolInput(BaseModel):
    """Input schema for OpenAlexResearchTool."""

    query: str = Field(
        description="The search query or research topic (e.g., 'retrieval augmented generation', 'CRISPR gene editing')."
    )
    limit: int = Field(
        default=5,
        description="Number of relevant paper results to return (1 to 20).",
    )
    email: Optional[str] = Field(
        default=None,
        description="Optional email address to join OpenAlex's polite pool for faster response times.",
    )


class OpenAlexResearchTool(BaseTool):
    name: str = "OpenAlex Research Search"
    description: str = (
        "Searches over 250M+ open-access scholarly works, scientific papers, "
        "authors, and venues via the OpenAlex REST API."
    )
    args_schema: Type[BaseModel] = OpenAlexResearchToolInput

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Reconstruct abstract text from OpenAlex abstract_inverted_index."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return "N/A"

        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        abstract_text = " ".join(word for _, word in word_positions)

        if len(abstract_text) > 300:
            return abstract_text[:300] + "..."
        return abstract_text

    def _run(self, query: str, limit: int = 5, email: Optional[str] = None) -> str:
        """Fetch scholarly works from OpenAlex API."""
        if not query or not query.strip():
            return "Error: Search query must be a non-empty string."

        clamped_limit = max(1, min(limit, 20))
        clean_query = query.strip()

        url = f"https://api.openalex.org/works?search={clean_query}&per_page={clamped_limit}"
        if email and email.strip():
            url += f"&mailto={email.strip()}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results or not isinstance(results, list):
                return f"No scholarly works found matching query: '{clean_query}'"

            output_lines = [f"Found {len(results)} paper(s) for '{clean_query}':\n"]

            for idx, work in enumerate(results, 1):
                if not work or not isinstance(work, dict):
                    continue

                title = work.get("title") or "Untitled Work"
                pub_year = work.get("publication_year", "N/A")
                cited_count = work.get("cited_by_count", 0)

                primary_location = work.get("primary_location") or {}
                source = primary_location.get("source") or {}
                venue = source.get("display_name", "N/A")

                authorships = work.get("authorships") or []
                author_names = []
                for auth in authorships[:3]:
                    author_obj = auth.get("author") or {}
                    if author_obj.get("display_name"):
                        author_names.append(author_obj["display_name"])
                authors_str = ", ".join(author_names) if author_names else "Unknown Authors"

                oa_info = work.get("open_access") or {}
                oa_url = oa_info.get("oa_url") or work.get("doi") or work.get("id", "N/A")

                abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

                entry = (
                    f"{idx}. {title} ({pub_year})\n"
                    f"   Authors: {authors_str}\n"
                    f"   Venue: {venue} | Citations: {cited_count}\n"
                    f"   URL: {oa_url}\n"
                    f"   Abstract: {abstract}\n"
                )
                output_lines.append(entry)

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error retrieving OpenAlex research data: {str(e)}"
