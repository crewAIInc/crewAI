from typing import Any, ClassVar
from urllib.parse import quote

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_BASE_URL = "https://www.edgepedia.org/api/v1"
DEFAULT_USER_AGENT = "crewai-tools-edgepedia/1.0"


def _truncate_preserving_credit(body: str, credit: str, max_chars: int) -> str:
    """Truncate article text to max_chars while keeping the attribution line intact."""
    credit_block = f"\n\nCredit: {credit}" if credit else ""
    if max_chars <= 0:
        return body + credit_block
    if len(body) + len(credit_block) <= max_chars:
        return body + credit_block
    budget = max_chars - len(credit_block)
    if budget <= 3:
        return credit_block.strip()[:max_chars] if credit_block else body[:max_chars]
    return body[: budget - 3].rstrip() + "..." + credit_block


class EdgepediaToolInput(BaseModel):
    """Input schema for EdgepediaTool."""

    query: str = Field(
        ...,
        description=(
            "Subject name or title words to search in Edgepedia (for example, "
            "'Mount Fuji' or 'photosynthesis'), rather than a full question."
        ),
    )
    lang: str | None = Field(
        default=None,
        description=(
            "Optional language code for stored translations across 31 languages "
            "(for example, 'en' for English, 'zh' for Chinese, 'es' for Spanish, "
            "'ar' for Arabic, or 'hi' for Hindi). Falls back to English when unavailable."
        ),
    )
    limit: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of matching articles to retrieve (1 to 5).",
    )
    citations: bool | None = Field(
        default=None,
        description=(
            "Include inline source URLs when True, or shorter '[n]' citation markers "
            "when False. Defaults to the tool's configured setting."
        ),
    )


class EdgepediaTool(BaseTool):
    """Search Edgepedia and retrieve cited Markdown encyclopedia articles.

    Connects directly to Edgepedia's keyless public REST API
    (https://www.edgepedia.org/api/v1), covering more than 519,000 cited articles
    with stored translations across 31 languages.
    """

    BASE_API_URL: ClassVar[str] = DEFAULT_BASE_URL
    name: str = "Edgepedia Search"
    description: str = (
        "Search Edgepedia, a free cited encyclopedia with more than 519,000 articles "
        "across 31 languages, and retrieve full Markdown articles with citations and "
        "credit lines. Pass a subject name or title words."
    )
    args_schema: type[BaseModel] = EdgepediaToolInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(default_factory=list)
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description="Base URL for the Edgepedia v1 REST API.",
    )
    default_lang: str = Field(
        default="en",
        description="Default language code used when lang is not provided per call.",
    )
    default_citations: bool = Field(
        default=True,
        description="Default inline citation setting when citations is not provided per call.",
    )
    max_chars_per_article: int = Field(
        default=4000,
        ge=0,
        description="Maximum character length per returned article (0 disables truncation).",
    )
    timeout: float = Field(
        default=15.0,
        gt=0,
        description="HTTP request timeout in seconds.",
    )

    def _get_json(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Perform a GET request against the Edgepedia REST API and parse JSON."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected JSON object from {url}, got {type(payload).__name__}"
            )
        return payload

    def _fetch_article(
        self,
        slug: str,
        *,
        lang: str,
        citations: bool,
    ) -> dict[str, Any]:
        """Fetch an individual article by slug with optional language and citations."""
        params: dict[str, str] = {}
        if lang != "en":
            params["lang"] = lang
        if not citations:
            params["citations"] = "false"
        encoded_slug = quote(slug.strip(), safe="")
        return self._get_json(f"/articles/{encoded_slug}", params=params or None)

    def _format_article(self, article: dict[str, Any]) -> str:
        """Format an Edgepedia article response into Markdown with attribution."""
        title = str(article.get("title") or article.get("slug") or "Untitled")
        url = str(article.get("url") or "")
        credit = str(article.get("credit") or "")
        body = str(
            article.get("markdown")
            or article.get("text")
            or article.get("excerpt")
            or ""
        ).strip()
        header = f"Article: {title}\nURL: {url}\n\n" if url else f"Article: {title}\n\n"
        if self.max_chars_per_article > 0:
            remaining = max(self.max_chars_per_article - len(header), 0)
            formatted_body = _truncate_preserving_credit(body, credit, remaining)
        else:
            formatted_body = f"{body}\n\nCredit: {credit}" if credit else body
        return f"{header}{formatted_body}"

    def _run(
        self,
        query: str,
        lang: str | None = None,
        limit: int = 1,
        citations: bool | None = None,
    ) -> str:
        """Search Edgepedia and return cited Markdown articles with credit lines.

        Args:
            query: Subject name or title words to look up.
            lang: Optional language code for stored translations across 31 languages.
            limit: Number of matching articles to retrieve (1 to 5).
            citations: Include inline source URLs (True) or compact '[n]' markers (False).

        Returns:
            Formatted Markdown text for matching articles, including each article's
            credit line, or a descriptive error message if the lookup fails.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return "Please provide a non-empty search query for Edgepedia."

        effective_lang = (
            lang if lang is not None else self.default_lang
        ).strip().lower() or "en"
        effective_citations = self.default_citations if citations is None else citations

        try:
            search_payload = self._get_json(
                "/search",
                params={"q": cleaned_query, "limit": limit},
            )
            results = search_payload.get("results")
            if not isinstance(results, list) or not results:
                return f"No Edgepedia articles found for '{cleaned_query}'."

            formatted_blocks: list[str] = []
            for entry in results[:limit]:
                if not isinstance(entry, dict):
                    continue
                slug = entry.get("slug")
                if not isinstance(slug, str) or not slug:
                    continue
                article = self._fetch_article(
                    slug,
                    lang=effective_lang,
                    citations=effective_citations,
                )
                formatted_blocks.append(self._format_article(article))

            if not formatted_blocks:
                return f"No Edgepedia articles found for '{cleaned_query}'."

            return "\n\n---\n\n".join(formatted_blocks)
        except requests.RequestException as exc:
            return f"Failed to query Edgepedia API: {exc}"
        except (TypeError, ValueError) as exc:
            return f"Invalid response from Edgepedia API: {exc}"
