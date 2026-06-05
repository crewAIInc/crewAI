from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


try:
    import wikipedia

    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    wikipedia = Any  # type: ignore[assignment]


class WikipediaSearchToolSchema(BaseModel):
    """Input schema for WikipediaSearchTool."""

    query: str = Field(..., description="Topic to search for on Wikipedia")
    sentences: int = Field(
        3, ge=1, description="Number of summary sentences to return"
    )


class WikipediaSearchTool(BaseTool):
    name: str = "Wikipedia Search Tool"
    description: str = (
        "Search Wikipedia for a topic and return a concise summary."
    )
    args_schema: type[BaseModel] = WikipediaSearchToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["wikipedia"])

    def _run(self, query: str, sentences: int = 3) -> str:
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError(
                "The 'wikipedia' package is required to use WikipediaSearchTool. "
                "Please install it with: uv add wikipedia"
            )

        try:
            return wikipedia.summary(
                query,
                sentences=sentences,
                auto_suggest=False,
            )
        except wikipedia.exceptions.DisambiguationError as e:
            options = ", ".join(e.options[:5])
            return (
                f"The query '{query}' is ambiguous. "
                f"Please be more specific. Possible options: {options}"
            )
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page was found for '{query}'."
        except Exception as e:
            return f"An error occurred while searching Wikipedia: {e!s}"
