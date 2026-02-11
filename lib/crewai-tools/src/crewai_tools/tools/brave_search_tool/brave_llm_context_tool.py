from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.response_types import LLMContext
from crewai_tools.tools.brave_search_tool.schemas import (
    LLMContextHeaders,
    LLMContextParams,
)


class BraveLLMContextTool(BraveSearchToolBase):
    """A tool that retrieves context for LLM usage from the Brave Search API."""

    name: str = "Brave LLM Context"
    args_schema: type[BaseModel] = LLMContextParams
    header_schema: type[BaseModel] = LLMContextHeaders

    description: str = (
        "A tool that retrieves context for LLM usage from the Brave Search API. "
        "Results are returned as structured JSON data."
    )

    search_url: str = "https://api.search.brave.com/res/v1/llm/context"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: LLMContext.Response) -> dict[str, Any]:
        """The LLM Context response schema is fairly simple. Return as is."""
        return response
