import logging
from typing import Any, ClassVar

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__file__)

try:
    import wikipedia
except ImportError:
    wikipedia = None


class WikipediaSearchToolInput(BaseModel):
    """Input for WikipediaSearchTool."""

    search_query: str = Field(
        ..., description="The topic or query to search for on Wikipedia"
    )


class WikipediaSearchTool(BaseTool):
    """Tool for searching Wikipedia and retrieving article summaries.

    Uses the `wikipedia` Python library to search for topics and return
    concise summaries. Handles disambiguation and missing pages gracefully.
    """

    name: str = "Wikipedia Search"
    description: str = (
        "A tool that searches Wikipedia for a given topic and returns "
        "a concise summary. Useful for quick factual lookups and research."
    )
    args_schema: type[BaseModel] = WikipediaSearchToolInput
    sentences: int = 5
    language: str = "en"
    top_k: int = 1

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    def __init__(
        self,
        sentences: int = 5,
        language: str = "en",
        top_k: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.sentences = sentences
        self.language = language
        self.top_k = top_k

    def _run(self, search_query: str, **kwargs: Any) -> str:
        if wikipedia is None:
            return (
                "The 'wikipedia' package is required to use WikipediaSearchTool. "
                "Install it with: pip install wikipedia"
            )

        wikipedia.set_lang(self.language)

        try:
            results = wikipedia.search(search_query, results=self.top_k)
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Error searching Wikipedia: {e}"

        if not results:
            return f"No Wikipedia articles found for '{search_query}'."

        output_parts = [self._fetch_article(title) for title in results]

        return "\n\n---\n\n".join(output_parts)

    def _fetch_article(self, title: str) -> str:
        try:
            summary = wikipedia.summary(title, sentences=self.sentences)
            page = wikipedia.page(title, auto_suggest=False)
            return f"Title: {page.title}\nURL: {page.url}\nSummary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            options = ", ".join(e.options[:5])
            return (
                f"Title: {title}\n"
                f"Disambiguation: The term '{title}' is ambiguous. "
                f"Possible options: {options}. "
                f"Please refine your search query."
            )
        except wikipedia.exceptions.PageError:
            return f"Title: {title}\nError: No Wikipedia page found for '{title}'."
        except Exception as e:
            logger.error(f"Error retrieving Wikipedia page '{title}': {e}")
            return f"Title: {title}\nError: Failed to retrieve page: {e}"
