import logging
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)

_CATEGORIES = [
    "crypto",
    "tradfi",
    "business",
    "tech",
    "politics",
    "world",
    "science",
    "health",
    "energy",
    "sports",
]

_MAX_API_LIMIT = 50


class NewsflashNewsToolSchema(BaseModel):
    """Input for NewsflashNewsTool."""

    query: str = Field(
        ..., description="Search query for news events, e.g. 'bitcoin etf approval'"
    )
    semantic: bool = Field(
        True,
        description=(
            "Use semantic (meaning-based) search. Set to False for exact keyword "
            "matching."
        ),
    )
    category: str | None = Field(
        None,
        description=(
            "Optional category filter. One of: " + ", ".join(_CATEGORIES) + "."
        ),
    )
    limit: int = Field(
        10, ge=1, le=_MAX_API_LIMIT, description="Maximum number of events to return"
    )
    min_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Only return events with confidence >= this value. Confidence is "
            "min(1, corroborating_sources / 3): 1.0 means at least 3 independent "
            "outlets reported the same happening. Raise this to filter out "
            "single-source, unverified stories."
        ),
    )


class NewsflashNewsTool(BaseTool):
    name: str = "Newsflash News Search"
    description: str = (
        "Search real-time news as deduplicated, corroborated events. Newsflash "
        "clusters articles from many outlets into one event with a corroboration "
        "count and a confidence score (min(1, sources/3)), so agents can filter "
        "out single-source or fabricated stories via min_confidence. Works "
        "without an API key (50 requests/day)."
    )
    args_schema: type[BaseModel] = NewsflashNewsToolSchema
    base_url: str = "https://newsflash.sh/api"
    request_timeout: int = 15
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="NEWSFLASH_API_KEY",
                description=(
                    "Optional Newsflash API key for higher rate limits and deeper "
                    "history. Keyless access works out of the box (50 requests/day)."
                ),
                required=False,
            ),
        ]
    )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        api_key = os.environ.get("NEWSFLASH_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def _first_article_url(self, event_id: Any) -> str | None:
        """Fetch event details and return the URL of its first article."""
        try:
            details = self._get(f"/events/{event_id}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch details for event {event_id}: {e}")
            return None
        articles = details.get("articles") or []
        if articles:
            url = articles[0].get("url")
            return url if isinstance(url, str) else None
        return None

    def _format_event(self, index: int, event: dict[str, Any], link: str | None) -> str:
        sources = event.get("sources") or []
        corroboration = event.get("corroboration", len(sources))
        confidence = event.get("confidence", 0.0)
        lines = [
            f"{index}. {event.get('canonical_title', 'Untitled event')}",
            (
                f"   Category: {event.get('category', 'unknown')} | "
                f"Confidence: {confidence:.2f} | "
                f"Corroboration: {corroboration} source(s)"
                + (f" ({', '.join(sources)})" if sources else "")
            ),
            f"   First seen: {event.get('first_seen_at', 'unknown')}",
        ]
        summary = (event.get("summary") or "").strip()
        if summary:
            lines.append(f"   Summary: {summary}")
        if link:
            lines.append(f"   Link: {link}")
        return "\n".join(lines)

    def _run(self, **kwargs: Any) -> str:
        query = kwargs.get("query") or kwargs.get("search_query")
        if not query:
            return "Error: a search query is required."
        semantic = kwargs.get("semantic", True)
        category = kwargs.get("category")
        limit = kwargs.get("limit", 10)
        min_confidence = kwargs.get("min_confidence", 0.0)

        # Over-fetch when a confidence floor is set so filtering can still
        # yield up to `limit` events; the API caps limit at 50.
        request_limit = limit
        if min_confidence > 0:
            request_limit = min(_MAX_API_LIMIT, max(limit * 3, limit))

        params = {
            "q": query,
            "semantic": "1" if semantic else "0",
            "limit": request_limit,
        }
        if category:
            params["category"] = category

        try:
            data = self._get("/events", params=params)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying the Newsflash API: {e}")
            return f"Error querying the Newsflash API: {e}"

        events = [
            event
            for event in data.get("events", [])
            if event.get("confidence", 0.0) >= min_confidence
        ][:limit]

        if not events:
            note = (data.get("window") or {}).get("note")
            message = (
                f"No corroborated news events found for '{query}'"
                + (f" in category '{category}'" if category else "")
                + (
                    f" with confidence >= {min_confidence}"
                    if min_confidence > 0
                    else ""
                )
                + "."
            )
            return f"{message} ({note})" if note else message

        # Fetch article links only for the events actually returned.
        formatted = [
            self._format_event(i, event, self._first_article_url(event.get("id")))
            for i, event in enumerate(events, start=1)
        ]

        header = f"Newsflash events for '{query}' ({len(events)} shown"
        if min_confidence > 0:
            header += f", confidence >= {min_confidence}"
        header += "):"
        return "\n\n".join([header, *formatted])

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)
