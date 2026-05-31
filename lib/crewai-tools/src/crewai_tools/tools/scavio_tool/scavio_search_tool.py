from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.scavio_tool.base import ScavioBaseTool


class ScavioSearchToolSchema(BaseModel):
    """Input schema for ScavioSearchTool."""

    query: str = Field(..., description="The search query string.")


class ScavioSearchTool(ScavioBaseTool):
    """Tool that uses the Scavio Search API to perform web searches.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Scavio API key.
        search_type: The type of search to perform.
        max_results: The maximum number of results to return.
        country_code: ISO 3166-1 alpha-2 country code for localized results.
        language: ISO 639-1 language code for results.
        device: Device type for search results.
        include_knowledge_graph: Whether to include the knowledge graph.
        include_questions: Whether to include related questions.
    """

    name: str = "Scavio Web Search"
    description: str = (
        "A tool that performs web searches using the Scavio Search API. "
        "It returns search results with titles, URLs, descriptions, "
        "knowledge graphs, and related questions as a JSON string. "
        "Use for any query requiring real-time or recent web information."
    )
    args_schema: type[BaseModel] = ScavioSearchToolSchema

    search_type: Literal["classic", "news", "maps", "images"] = Field(
        default="classic",
        description="The type of search to perform.",
    )
    country_code: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code for localized results.",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code for results.",
    )
    device: Literal["desktop", "mobile"] = Field(
        default="desktop",
        description="Device type for search results.",
    )
    include_knowledge_graph: bool = Field(
        default=True,
        description="Whether to include the knowledge graph in results.",
    )
    include_questions: bool = Field(
        default=True,
        description="Whether to include related questions in results.",
    )

    def _run(self, query: str) -> str:
        """Synchronously performs a web search using the Scavio API.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results.
        """
        raw = self.client.google.search(
            query=query,
            search_type=self.search_type,
            country_code=self.country_code,
            language=self.language,
            device=self.device,
        )

        raw = self._truncate_results(raw, "results")

        if not self.include_knowledge_graph:
            raw.pop("knowledge_graph", None)
        if not self.include_questions:
            raw.pop("questions", None)

        return self._format_response(raw)

    async def _arun(self, query: str) -> str:
        """Asynchronously performs a web search using the Scavio API.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results.
        """
        raw = await self.async_client.google.search(
            query=query,
            search_type=self.search_type,
            country_code=self.country_code,
            language=self.language,
            device=self.device,
        )

        raw = self._truncate_results(raw, "results")

        if not self.include_knowledge_graph:
            raw.pop("knowledge_graph", None)
        if not self.include_questions:
            raw.pop("questions", None)

        return self._format_response(raw)
