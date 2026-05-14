"""Synchronous TinyFish browser-automation tool."""

from __future__ import annotations

import json

from pydantic import BaseModel

from crewai_tools.tools.tinyfish_tool.base import TinyfishToolBase
from crewai_tools.tools.tinyfish_tool.schemas import TinyfishAgentParams


class TinyfishAgentTool(TinyfishToolBase):
    """Run a TinyFish browser automation synchronously and return the result."""

    name: str = "Tinyfish Agent"
    description: str = (
        "Automate any website using natural language. Provide a URL and a "
        "goal describing what to accomplish — extract data, fill forms, click "
        "buttons, navigate pages, and more. Waits for completion and returns "
        "the structured JSON result, including status, run_id, and any "
        "extracted data."
    )
    args_schema: type[BaseModel] = TinyfishAgentParams

    def _run(
        self,
        url: str,
        goal: str,
        browser_profile: str = "lite",
    ) -> str:
        from tinyfish import BrowserProfile

        client, err = self._get_client()
        if err is not None or client is None:
            return err or "Error: failed to initialise TinyFish client."
        try:
            response = client.agent.run(
                url=url,
                goal=goal,
                browser_profile=BrowserProfile(browser_profile),
            )
        except Exception as exc:
            return self._format_error(exc)
        return json.dumps(response.model_dump())
