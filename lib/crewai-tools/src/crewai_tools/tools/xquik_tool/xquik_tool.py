from __future__ import annotations

import json
import os
from typing import Any, Literal
from urllib.parse import quote

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_XQUIK_BASE_URL = "https://xquik.com/api/v1"
XQUIK_API_CONTRACT = "2026-04-29"


def _drop_empty(params: dict[str, Any]) -> dict[str, Any]:
    """Remove parameters that should not be sent to the API."""
    return {
        key: value for key, value in params.items() if value is not None and value != ""
    }


def _validate_range(value: int, *, minimum: int, maximum: int, name: str) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _normalize_path_identifier(value: str, *, name: str) -> str:
    cleaned = value.strip().lstrip("@")
    if not cleaned:
        raise ValueError(f"{name} is required")
    return quote(cleaned, safe="")


def _require_text(value: str, *, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} is required")
    return cleaned


def _error_body(response: requests.Response) -> str:
    try:
        return json.dumps(response.json())
    except ValueError:
        return response.text[:500]


class XquikToolBase(BaseTool):
    """Base class for Xquik REST API tools."""

    base_url: str = DEFAULT_XQUIK_BASE_URL
    timeout: int = 30
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="XQUIK_API_KEY",
                description="API key for Xquik",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        api_key_value = api_key or os.environ.get("XQUIK_API_KEY")
        if not api_key_value:
            raise ValueError("XQUIK_API_KEY environment variable is required")
        self._api_key = api_key_value
        self.base_url = (base_url or DEFAULT_XQUIK_BASE_URL).rstrip("/")
        self.timeout = int(timeout)

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "x-api-key": self._api_key,
            "xquik-api-contract": XQUIK_API_CONTRACT,
        }

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                headers=self.headers,
                params=_drop_empty(params or {}),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Xquik API request failed: {exc}") from exc

        if response.ok:
            try:
                result: dict[str, Any] = response.json()
                return result
            except ValueError as exc:
                raise RuntimeError(
                    f"Xquik API returned invalid JSON (HTTP {response.status_code})"
                ) from exc

        raise RuntimeError(
            f"Xquik API error (HTTP {response.status_code}): {_error_body(response)}"
        )


class XquikSearchTweetsToolSchema(BaseModel):
    """Input for XquikSearchTweetsTool."""

    search_query: str = Field(
        ...,
        description="X search query. Supports operators such as from:, has:media, lang:, since:, and min_faves:.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum tweets to return, from 1 to 200.",
    )
    query_type: Literal["Latest", "Top"] = Field(
        default="Latest",
        description="Sort order for the search results.",
    )
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )
    since_time: str | None = Field(
        default=None,
        description="ISO 8601 timestamp. Only return tweets after this time.",
    )
    until_time: str | None = Field(
        default=None,
        description="ISO 8601 timestamp. Only return tweets before this time.",
    )


class XquikGetTweetToolSchema(BaseModel):
    """Input for XquikGetTweetTool."""

    tweet_id: str = Field(..., description="Tweet ID to look up.")


class XquikGetUserToolSchema(BaseModel):
    """Input for XquikGetUserTool."""

    user: str = Field(..., description="X username, @username, or user ID.")


class XquikGetUserTweetsToolSchema(BaseModel):
    """Input for XquikGetUserTweetsTool."""

    user: str = Field(..., description="X username, @username, or user ID.")
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )
    include_replies: bool = Field(
        default=False,
        description="Include reply tweets in the returned page.",
    )
    include_parent_tweet: bool = Field(
        default=False,
        description="Include parent tweets for replies when available.",
    )


class XquikGetTrendsToolSchema(BaseModel):
    """Input for XquikGetTrendsTool."""

    woeid: int = Field(
        default=1,
        ge=1,
        description="Region WOEID. Use 1 for worldwide, 23424977 for US, 23424975 for UK, or 23424969 for Turkey.",
    )
    count: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Number of trending topics to return, from 1 to 50.",
    )


class XquikSearchTweetsTool(XquikToolBase):
    """Search public X/Twitter posts with Xquik."""

    name: str = "Xquik Search Tweets"
    description: str = (
        "Search public X/Twitter posts with X query operators. "
        "Returns structured tweet, author, metric, and pagination data."
    )
    args_schema: type[BaseModel] = XquikSearchTweetsToolSchema

    def _run(
        self,
        search_query: str,
        limit: int = 20,
        query_type: Literal["Latest", "Top"] = "Latest",
        cursor: str | None = None,
        since_time: str | None = None,
        until_time: str | None = None,
    ) -> dict[str, Any]:
        query = _require_text(search_query, name="search_query")
        if query_type not in {"Latest", "Top"}:
            raise ValueError("query_type must be either 'Latest' or 'Top'")

        return self._get(
            "/x/tweets/search",
            {
                "q": query,
                "limit": _validate_range(limit, minimum=1, maximum=200, name="limit"),
                "queryType": query_type,
                "cursor": cursor,
                "sinceTime": since_time,
                "untilTime": until_time,
            },
        )


class XquikGetTweetTool(XquikToolBase):
    """Look up one X/Twitter post with Xquik."""

    name: str = "Xquik Get Tweet"
    description: str = (
        "Look up one X/Twitter post by ID. "
        "Returns full text, author, metrics, media, and creation time when available."
    )
    args_schema: type[BaseModel] = XquikGetTweetToolSchema

    def _run(self, tweet_id: str) -> dict[str, Any]:
        normalized_tweet_id = _normalize_path_identifier(tweet_id, name="tweet_id")
        return self._get(f"/x/tweets/{normalized_tweet_id}")


class XquikGetUserTool(XquikToolBase):
    """Look up one X/Twitter user profile with Xquik."""

    name: str = "Xquik Get User"
    description: str = (
        "Look up an X/Twitter user by username or user ID. "
        "Returns profile metadata, follower counts, and verification data."
    )
    args_schema: type[BaseModel] = XquikGetUserToolSchema

    def _run(self, user: str) -> dict[str, Any]:
        normalized_user = _normalize_path_identifier(user, name="user")
        return self._get(f"/x/users/{normalized_user}")


class XquikGetUserTweetsTool(XquikToolBase):
    """List recent posts from one X/Twitter user with Xquik."""

    name: str = "Xquik Get User Tweets"
    description: str = (
        "List recent posts from an X/Twitter user by username or user ID. "
        "Supports cursor pagination and optional replies."
    )
    args_schema: type[BaseModel] = XquikGetUserTweetsToolSchema

    def _run(
        self,
        user: str,
        cursor: str | None = None,
        include_replies: bool = False,
        include_parent_tweet: bool = False,
    ) -> dict[str, Any]:
        normalized_user = _normalize_path_identifier(user, name="user")
        params: dict[str, Any] = {"cursor": cursor}
        if include_replies:
            params["includeReplies"] = "true"
        if include_parent_tweet:
            params["includeParentTweet"] = "true"
        return self._get(f"/x/users/{normalized_user}/tweets", params)


class XquikGetTrendsTool(XquikToolBase):
    """Get regional X/Twitter trends with Xquik."""

    name: str = "Xquik Get Trends"
    description: str = (
        "Get trending X/Twitter hashtags and topics by region WOEID. "
        "Returns trend names, queries, ranks, and descriptions when available."
    )
    args_schema: type[BaseModel] = XquikGetTrendsToolSchema

    def _run(self, woeid: int = 1, count: int = 30) -> dict[str, Any]:
        if woeid < 1:
            raise ValueError("woeid must be greater than or equal to 1")

        return self._get(
            "/x/trends",
            {
                "woeid": woeid,
                "count": _validate_range(count, minimum=1, maximum=50, name="count"),
            },
        )
