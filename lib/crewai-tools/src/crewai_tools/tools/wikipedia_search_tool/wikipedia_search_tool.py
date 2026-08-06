import logging
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

try:
    import wikipedia  # type: ignore[import-untyped]

    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class WikipediaSearchToolSchema(BaseModel):
    """Input schema for WikipediaSearchTool."""

    search_query: str = Field(
        ...,
        description="Search query to find relevant Wikipedia articles, e.g. 'Artificial Intelligence' or 'Python programming'",
    )
    lang: str | None = Field(
        default=None,
        description="Optional Wikipedia language code (e.g., 'en', 'tr', 'es', 'fr', 'de'). Defaults to 'en'.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional maximum number of Wikipedia search results to return (between 1 and 10). Defaults to 3.",
    )
    load_full_content: bool | None = Field(
        default=None,
        description="Optional flag. If True, returns full article content instead of lead section summary. Defaults to False.",
    )


class WikipediaSearchTool(BaseTool):
    """A tool that searches Wikipedia and retrieves article summaries or full content."""

    name: str = "Search Wikipedia"
    description: str = "A tool to search Wikipedia for information, returning titles, URLs, and summaries or full content of matching articles."
    args_schema: type[BaseModel] = WikipediaSearchToolSchema
    model_config = ConfigDict(extra="allow")

    package_dependencies: list[str] = Field(default_factory=lambda: ["wikipedia"])
    env_vars: list[EnvVar] = Field(default_factory=list)

    lang: str = "en"
    limit: int = 3
    load_full_content: bool = False
    user_agent: str = (
        "CrewAIWikipediaSearchTool/1.0 (https://crewai.com; contact@crewai.com)"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initialize_wikipedia()

    def _initialize_wikipedia(self) -> None:
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError(
                "The 'wikipedia' package is required to use the WikipediaSearchTool. "
                "Please install it using your package manager (e.g., `pip install wikipedia` or `uv add wikipedia`)."
            )

    def _run(
        self,
        search_query: str,
        lang: str | None = None,
        limit: int | None = None,
        load_full_content: bool | None = None,
        **kwargs: Any,
    ) -> str:
        if not WIKIPEDIA_AVAILABLE:
            return (
                "Error: The 'wikipedia' package is required to use WikipediaSearchTool. "
                "Please install it using your package manager (e.g., `pip install wikipedia` or `uv add wikipedia`)."
            )

        target_lang = lang or self.lang
        target_limit = limit or self.limit
        should_load_full_content = (
            self.load_full_content if load_full_content is None else load_full_content
        )

        wikipedia.set_user_agent(self.user_agent)
        wikipedia.set_lang(target_lang)

        try:
            search_results = wikipedia.search(search_query, results=target_limit)
        except Exception as e:
            logger.error(f"Wikipedia search failed for query '{search_query}': {e}")
            return f"Error searching Wikipedia for '{search_query}': {e!s}"

        if not search_results:
            return f"No Wikipedia results found for query: '{search_query}'"

        formatted_results: list[str] = []

        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                if should_load_full_content:
                    body = f"Content: {page.content}"
                else:
                    summary = wikipedia.summary(title, auto_suggest=False)
                    body = f"Summary: {summary}"

                formatted_results.append(
                    f"Title: {page.title}\nURL: {page.url}\n{body}"
                )
            except wikipedia.exceptions.DisambiguationError as e:  # noqa: PERF203
                options_str = ", ".join(e.options[:5])
                formatted_results.append(
                    f"Title: {title} (Disambiguation)\n"
                    f"Note: '{title}' refers to multiple topics: {options_str}..."
                )
            except wikipedia.exceptions.PageError:
                formatted_results.append(
                    f"Title: {title}\nNote: Could not retrieve page details."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch details for Wikipedia page '{title}': {e}"
                )
                formatted_results.append(
                    f"Title: {title}\nNote: Error retrieving details: {e!s}"
                )

        return "\n\n" + "-" * 80 + "\n\n".join(formatted_results)


if WIKIPEDIA_AVAILABLE:
    if not getattr(WikipediaSearchTool, "_model_rebuilt", False):
        WikipediaSearchTool.model_rebuild()
        WikipediaSearchTool._model_rebuilt = True  # type: ignore[attr-defined]
