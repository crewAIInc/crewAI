"""Tool for fetching clean page content via the TinyFish fetch endpoint."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from crewai_tools.tools.tinyfish_tool.base import TinyfishToolBase
from crewai_tools.tools.tinyfish_tool.schemas import TinyfishFetchParams


class TinyfishFetchTool(TinyfishToolBase):
    """Fetch clean readable content from URLs using TinyFish."""

    name: str = "Tinyfish Fetch"
    description: str = (
        "Fetch clean readable content from one to ten URLs using TinyFish. "
        "Use when the target URL is known and the agent needs page text, "
        "metadata, links, or image links without controlling a browser."
    )
    args_schema: type[BaseModel] = TinyfishFetchParams

    def _run(
        self,
        urls: list[str],
        format: Literal["markdown", "html", "json"] = "markdown",
        links: bool = False,
        image_links: bool = False,
    ) -> str:
        client, err = self._get_client()
        if err is not None or client is None:
            return err or "Error: failed to initialise TinyFish client."
        try:
            response = client.fetch.get_contents(
                urls=urls,
                format=format,
                links=links,
                image_links=image_links,
            )
        except Exception as exc:
            return self._format_error(exc)
        return json.dumps(response.model_dump())
