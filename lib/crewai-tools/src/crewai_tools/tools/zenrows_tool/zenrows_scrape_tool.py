"""Zenrows web scraping tool for CrewAI.

Wraps the Zenrows Scraper API (https://api.zenrows.com/v1/): anti-bot bypass,
headless-browser JavaScript rendering, and residential-proxy rotation behind
one HTTP endpoint. See https://www.zenrows.com/ for account signup and
https://docs.zenrows.com/universal-scraper-api/api-reference for the full
parameter reference.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import requests

from crewai_tools.security.safe_path import validate_url


logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.zenrows.com/v1/"
_RESPONSE_TYPES = ("html", "markdown", "plaintext")
_DEFAULT_TIMEOUT = 140.0  # JS-rendered pages can legitimately take ~90s+ upstream.
# Parameters Adaptive Stealth Mode (`mode: "auto"`) manages itself; Zenrows
# returns a 400 REQ_INVALID_PARAMS if either is also set manually.
_MODE_AUTO_MANAGED_PARAMS = ("js_render", "premium_proxy")


class ZenRowsScrapeToolSchema(BaseModel):
    """Input schema for ZenRowsScrapeTool."""

    url: str = Field(
        ...,
        description=(
            "Full URL of the web page to scrape, including the scheme "
            "(e.g. 'https://example.com/product/123')."
        ),
    )
    response_type: Literal["html", "markdown", "plaintext"] = Field(
        default="markdown",
        description=(
            "Output format for the scraped content. 'markdown' (default) returns "
            "clean, LLM-friendly markdown; 'plaintext' strips all markup and tags; "
            "'html' returns the raw rendered HTML."
        ),
    )


class ZenRowsScrapeTool(BaseTool):
    """Scrape a single web page through the Zenrows API and return its content.

    Zenrows handles anti-bot bypass, JavaScript rendering, and residential
    proxy rotation server-side. By default every request is sent in
    `Adaptive Stealth Mode
    <https://docs.zenrows.com/universal-scraper-api/features/adaptive-stealth-mode>`_
    (``mode: "auto"``):
    Zenrows starts with the cheapest viable request configuration and
    automatically escalates to JS rendering or premium proxies only when the
    target page needs it, billing only for the configuration that succeeds.

    This deliberately keeps the tool's per-call surface to just ``url`` and
    ``response_type`` -- deciding *how hard* a given page is to scrape (does
    it need a headless browser? a residential proxy? which country?) is left
    to Zenrows' own adaptive routing rather than the calling model, which
    otherwise tends to guess at that decision from a large raw parameter
    list rather than reason about it.

    Advanced Zenrows parameters that a page consistently needs (for example
    ``proxy_country`` to pin a geography, ``js_instructions`` to interact
    with the page, ``wait_for``/``wait`` for slow-rendering content, or
    turning Adaptive Stealth off in favor of manually pinned
    ``js_render``/``premium_proxy`` values) can be set once at construction
    time via ``config`` and then apply to every request made by this tool
    instance. ``proxy_country`` works alongside ``mode: "auto"`` --
    Zenrows enables ``premium_proxy`` for it automatically -- but outside
    auto mode it requires ``premium_proxy: True`` to be set explicitly, or
    it has no effect; see :meth:`__init__` validation below. See
    https://docs.zenrows.com/universal-scraper-api/api-reference for the
    full parameter reference.

    Args:
        api_key: Your Zenrows API key. Defaults to the ``ZENROWS_API_KEY``
            environment variable.
        config: Optional. Extra Zenrows API request parameters merged into
            every request made by this tool instance. Defaults to Adaptive
            Stealth Mode (``{"mode": "auto"}``).
        timeout: Request timeout in seconds. Defaults to 140s to accommodate
            JS-rendered pages.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )

    name: str = "Zenrows web scrape tool"
    description: str = (
        "Scrape a single web page through Zenrows and return its content. Handles "
        "JavaScript-rendered pages, anti-bot protection, and geo-restricted content "
        "automatically -- just pass the URL."
    )
    args_schema: type[BaseModel] = ZenRowsScrapeToolSchema

    api_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    timeout: float = _DEFAULT_TIMEOUT
    config: dict[str, Any] = Field(default_factory=lambda: {"mode": "auto"})

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ZENROWS_API_KEY",
                description="API key for Zenrows (https://www.zenrows.com/)",
                required=True,
            ),
        ]
    )

    _resolved_api_key: str = PrivateAttr(default="")

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if config is not None:
            self.config = config
        self._validate_config()

        resolved_key = api_key or os.environ.get("ZENROWS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Zenrows API key is required. Set the ZENROWS_API_KEY environment "
                "variable or pass api_key=... when constructing ZenRowsScrapeTool."
            )
        # Kept off the public `api_key` field (and thus out of model_dump()/logs)
        # since it is a credential, not descriptive tool configuration.
        self._resolved_api_key = resolved_key

    def _validate_config(self) -> None:
        """Catch two config combinations that Zenrows would otherwise reject
        (or, worse, silently ignore) at request time.
        """
        if self.config.get("mode") == "auto":
            conflicting = [p for p in _MODE_AUTO_MANAGED_PARAMS if p in self.config]
            if conflicting:
                raise ValueError(
                    f'config sets {conflicting} together with mode: "auto" '
                    "(Adaptive Stealth Mode). Zenrows manages those parameters "
                    "itself in automatic mode and rejects the request "
                    "(400 REQ_INVALID_PARAMS) if they are also set manually. "
                    f"Either drop `mode` to control {conflicting} manually, or "
                    "remove them and let Adaptive Stealth Mode choose."
                )

        # Outside mode="auto", Zenrows only applies geolocation to premium
        # (residential) proxies -- proxy_country alone is silently ignored
        # there. Under mode="auto" this restriction doesn't apply: Zenrows
        # documents proxy_country as usable alongside Adaptive Stealth Mode
        # and enables premium_proxy for it automatically.
        if (
            self.config.get("proxy_country")
            and not self.config.get("premium_proxy")
            and self.config.get("mode") != "auto"
        ):
            raise ValueError(
                "config sets `proxy_country` without `premium_proxy: True` "
                'and without `mode: "auto"`. Outside Adaptive Stealth Mode, '
                "Zenrows only applies geolocation to premium (residential) "
                "proxies, so `proxy_country` alone has no effect -- it is "
                "silently ignored rather than raising an error. Either set "
                "`premium_proxy: True` alongside `proxy_country`, or add "
                '`mode: "auto"` and let Zenrows enable premium_proxy itself.'
            )

    def _run(
        self,
        url: str,
        response_type: Literal["html", "markdown", "plaintext"] = "markdown",
        **_: Any,
    ) -> str:
        if response_type not in _RESPONSE_TYPES:
            raise ValueError(
                f"Unsupported response_type: {response_type!r}. Must be one of "
                f"{_RESPONSE_TYPES}."
            )

        # `config` is only guaranteed valid as of the last time this ran. It's
        # a plain dict, so `validate_assignment` on the field only checks it's
        # still a dict, not that its contents are sane -- reassigning
        # `tool.config`, or mutating a dict object the caller kept a
        # reference to, would otherwise bypass the __init__ check silently.
        self._validate_config()

        # Zenrows fetches `url` on our behalf -- the actual HTTP request this
        # process makes always targets `self.base_url` (Zenrows' own,
        # trusted API host). Validation here rejects unsafe schemes
        # (e.g. file://) and malformed input before an API credit is spent,
        # matching the other scraping tools in this repo (Firecrawl,
        # Scrapfly, Bright Data) rather than protecting a direct fetch.
        validated_url = validate_url(url)

        params: dict[str, Any] = dict(self.config)
        params["url"] = validated_url
        # Set last so nothing in `config` can shadow the resolved credential.
        params["apikey"] = self._resolved_api_key
        if response_type == "html":
            # `config` isn't documented as a place to set `response_type`, but
            # nothing stops a caller from putting it there anyway -- drop it
            # so an explicit per-call `"html"` always wins.
            params.pop("response_type", None)
        else:
            params["response_type"] = response_type

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout:
            return (
                f"Zenrows request timed out after {self.timeout:.0f}s while scraping "
                f"'{validated_url}'. The page may be slow to render; consider raising "
                "the tool's `timeout` or checking whether the target is reachable."
            )
        except requests.HTTPError as exc:
            return self._format_http_error(exc, validated_url)
        except requests.RequestException as exc:
            return f"Zenrows request failed for '{validated_url}': {exc}"

        return response.text

    @staticmethod
    def _format_http_error(exc: requests.HTTPError, url: str) -> str:
        """Turn a failed Zenrows response into an actionable message.

        Zenrows returns a JSON error envelope (``{"code": ..., "message": ...}``)
        on most failures; this surfaces that detail instead of a bare status
        code, without leaking a stack trace to the agent.
        """
        response = exc.response
        status = response.status_code if response is not None else None
        detail = ""
        if response is not None:
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and body.get("message"):
                detail = f" {body['message']}"
                if body.get("code"):
                    detail += f" (code: {body['code']})"
            elif response.text:
                detail = f" {response.text[:300]}"

        if status == 400:
            return f"Zenrows rejected the request configuration (400).{detail}"
        if status == 401:
            return (
                f"Zenrows authentication failed (401): invalid or revoked API "
                f"key.{detail}"
            )
        if status == 402:
            return (
                "Zenrows request failed (402): plan/credit limit reached, or this "
                f"feature is not enabled for your account.{detail}"
            )
        if status == 422:
            return f"Zenrows rejected the request parameters (422) for '{url}'.{detail}"
        if status == 429:
            return (
                f"Zenrows rate limit exceeded (429) for '{url}'. Retry after a "
                f"short delay.{detail}"
            )
        return (
            f"Zenrows request failed with HTTP {status} while scraping '{url}'.{detail}"
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        # `_run` makes a blocking `requests.get()` call (up to `timeout`
        # seconds); running it directly here would stall the event loop for
        # the whole crew. Offload it to a worker thread instead.
        return await asyncio.to_thread(self._run, *args, **kwargs)
