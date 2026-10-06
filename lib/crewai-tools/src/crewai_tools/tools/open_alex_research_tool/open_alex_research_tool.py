from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class OpenAlexResearchToolInput(BaseModel):
    """Input schema for OpenAlexResearchTool."""

    query: str = Field(
        description=(
            "The search query or research topic "
            "(e.g., 'retrieval augmented generation', 'CRISPR gene editing')."
        )
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant paper results to return (1 to 20).",
    )
    email: str | None = Field(
        default=None,
        description="Optional email address for OpenAlex's polite pool.",
    )


class OpenAlexResearchTool(BaseTool):
    """Search scholarly works using the OpenAlex REST API."""

    name: str = "OpenAlex Research Search"
    description: str = (
        "Searches scholarly works across academic disciplines via the "
        "OpenAlex REST API, returning publication metadata, citation counts, "
        "available open-access URLs, and abstract snippets."
    )
    args_schema: type[BaseModel] = OpenAlexResearchToolInput

    def _reconstruct_abstract(
        self,
        inverted_index: dict[str, list[int]] | None,
    ) -> str:
        """Reconstruct abstract text from OpenAlex's inverted index."""
        if not isinstance(inverted_index, dict) or not inverted_index:
            return "N/A"

        word_positions: list[tuple[int, str]] = []

        for word, positions in inverted_index.items():
            if not isinstance(word, str) or not isinstance(positions, list):
                continue

            word_positions.extend(
                (position, word) for position in positions if isinstance(position, int)
            )

        word_positions.sort(key=lambda item: item[0])
        abstract_text = " ".join(word for _, word in word_positions)

        if len(abstract_text) > 300:
            return abstract_text[:300] + "..."

        return abstract_text or "N/A"

    def _run(
        self,
        query: str,
        limit: int = 5,
        email: str | None = None,
    ) -> str:
        """Search OpenAlex for scholarly works matching the query."""
        validated_input = OpenAlexResearchToolInput(
            query=query,
            limit=limit,
            email=email,
        )
        query = validated_input.query
        limit = validated_input.limit
        email = validated_input.email

        if not query or not query.strip():
            return "Error: Search query must be a non-empty string."

        clean_query = query.strip()

        try:
            params = {
                "search": clean_query,
                "per_page": limit,
            }

            if email and email.strip():
                params["mailto"] = email.strip()

            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data: Any = response.json()

            if not isinstance(data, dict):
                return "Error: Unexpected response format from OpenAlex."

            results = data.get("results")

            if not isinstance(results, list):
                return "Error: Unexpected response format from OpenAlex."

            if not results:
                return f"No scholarly works found matching query: '{clean_query}'"

            output_lines: list[str] = []

            for work in results:
                if not isinstance(work, dict):
                    continue

                title = work.get("title") or "Untitled Work"
                pub_year = work.get("publication_year", "N/A")
                cited_count = work.get("cited_by_count", 0)

                primary_location = work.get("primary_location")
                if not isinstance(primary_location, dict):
                    primary_location = {}

                source = primary_location.get("source")
                if not isinstance(source, dict):
                    source = {}

                venue = source.get("display_name", "N/A")

                authorships = work.get("authorships")
                if not isinstance(authorships, list):
                    authorships = []

                author_names: list[str] = []

                for authorship in authorships[:3]:
                    if not isinstance(authorship, dict):
                        continue

                    author = authorship.get("author")
                    if not isinstance(author, dict):
                        continue

                    author_name = author.get("display_name")
                    if isinstance(author_name, str) and author_name:
                        author_names.append(author_name)

                authors_str = (
                    ", ".join(author_names) if author_names else "Unknown Authors"
                )

                open_access = work.get("open_access")
                if not isinstance(open_access, dict):
                    open_access = {}

                oa_url = open_access.get("oa_url")
                doi = work.get("doi")
                work_id = work.get("id")

                if isinstance(oa_url, str) and oa_url:
                    url = oa_url
                elif isinstance(doi, str) and doi:
                    url = doi
                elif isinstance(work_id, str) and work_id:
                    url = work_id
                else:
                    url = "N/A"

                abstract = self._reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )

                output_lines.append(
                    f"{len(output_lines) + 1}. {title} ({pub_year})\n"
                    f"   Authors: {authors_str}\n"
                    f"   Venue: {venue} | Citations: {cited_count}\n"
                    f"   URL: {url}\n"
                    f"   Abstract: {abstract}"
                )

            if not output_lines:
                return f"No scholarly works found matching query: '{clean_query}'"

            return (
                f"Found {len(output_lines)} paper(s) for "
                f"'{clean_query}':\n\n" + "\n\n".join(output_lines)
            )

        except requests.RequestException as exc:
            return f"Error retrieving OpenAlex research data: {exc}"
        except (ValueError, TypeError) as exc:
            return f"Error retrieving OpenAlex research data: {exc}"
