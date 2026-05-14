"""Tool for web search via the TinyFish search endpoint."""

from __future__ import annotations

import json

from pydantic import BaseModel

from crewai_tools.tools.tinyfish_tool.base import TinyfishToolBase
from crewai_tools.tools.tinyfish_tool.schemas import TinyfishSearchParams


class TinyfishSearchTool(TinyfishToolBase):
    """Search the web using TinyFish."""

    name: str = "Tinyfish Search"
    description: str = (
        "Search the web using TinyFish and return structured search results "
        "(title, url, snippet, site name). Use to discover relevant URLs or "
        "current information before fetching content or running an automation."
    )
    args_schema: type[BaseModel] = TinyfishSearchParams

    def _run(
        self,
        query: str,
        location: str | None = None,
        language: str | None = None,
    ) -> str:
        client, err = self._get_client()
        if err is not None or client is None:
            return err or "Error: failed to initialise TinyFish client."
        try:
            response = client.search.query(
                query=query,
                location=location,
                language=language,
            )
        except Exception as exc:
            return self._format_error(exc)
        return json.dumps(response.model_dump())
