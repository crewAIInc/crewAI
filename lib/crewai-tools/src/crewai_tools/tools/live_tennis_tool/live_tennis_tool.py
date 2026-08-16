"""Tool for querying professional tennis data from the Live Tennis API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, ValidationError
import requests

from crewai_tools.tools.live_tennis_tool.schemas import LiveTennisToolSchema


API_KEY_ENV_VAR = "LIVETENNIS_API_KEY"


class LiveTennisTool(BaseTool):
    """Query the Live Tennis API (https://livetennisapi.com).

    Covers the endpoints available on the free tier: live match scores,
    upcoming matches, scheduled fixtures, player search and profiles
    (including current ranking), and API usage. The 'rankings' action
    (full published ranking tables) requires a paid plan.
    """

    name: str = "Live Tennis API"
    description: str = (
        "Fetches professional tennis data from the Live Tennis API: matches "
        "currently in play with live scores, upcoming matches, scheduled "
        "fixtures, player search and profiles (including current ranking and "
        "ranking points), published ranking tables, and your API quota usage. "
        "Results are returned as JSON."
    )
    args_schema: type[BaseModel] = LiveTennisToolSchema
    base_url: str = "https://api.livetennisapi.com/api/public/v1"
    request_timeout: float = 30.0
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=API_KEY_ENV_VAR,
                description="API key for the Live Tennis API (free tier available)",
                required=True,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> str:
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            return (
                f"The {API_KEY_ENV_VAR} environment variable is not set. "
                "Get a free API key at https://livetennisapi.com and set "
                f"{API_KEY_ENV_VAR} to use this tool."
            )

        try:
            params = LiveTennisToolSchema(**kwargs)
        except ValidationError as e:
            return f"Invalid arguments for the Live Tennis API tool: {e}"

        path, query = self._build_request(params)
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {api_key}"},
                params=query,
                timeout=self.request_timeout,
            )
        except requests.RequestException as e:
            return f"Live Tennis API request failed: {e}"

        if response.status_code == 401:
            return (
                "Live Tennis API rejected the request (401): the "
                f"{API_KEY_ENV_VAR} key is missing, unknown, or disabled."
            )
        if response.status_code == 403:
            return (
                "Live Tennis API returned 403: this endpoint requires a higher "
                f"subscription tier than the current key provides. {response.text}"
            )
        if response.status_code == 429:
            return f"Live Tennis API rate limit exceeded (429): {response.text}"
        if not response.ok:
            return (
                f"Live Tennis API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            return json.dumps(response.json())
        except ValueError:
            return response.text

    @staticmethod
    def _build_request(params: LiveTennisToolSchema) -> tuple[str, dict[str, Any]]:
        """Map a validated action to its endpoint path and query parameters."""
        query: dict[str, Any] = {}
        if params.limit is not None:
            query["limit"] = params.limit
        if params.offset is not None:
            query["offset"] = params.offset

        if params.action in ("live_matches", "upcoming_matches"):
            query["status"] = "live" if params.action == "live_matches" else "upcoming"
            if params.tour:
                query["tour"] = params.tour
            return "/matches", query
        if params.action == "fixtures":
            if params.tour:
                query["tour"] = params.tour
            return "/fixtures", query
        if params.action == "search_players":
            query["search"] = params.search
            return "/players", query
        if params.action == "player_profile":
            return f"/players/{quote(str(params.player_id), safe='')}", {}
        if params.action == "rankings":
            query["system"] = params.system
            return "/rankings", query
        return "/usage", {}
