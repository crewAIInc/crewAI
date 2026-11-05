import json
import time
from typing import Any, Literal

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_web_api_base_tool.oxylabs_web_api_base_tool import (
    OxylabsWebAPIBaseTool,
    _ScrapeTimeoutError,
)


__all__ = ["OxylabsWebScrapeTool"]


# Below this many characters, a page fetched without JavaScript is likely an
# empty shell that renders client-side.
_THIN_CONTENT_CHARS = 500


class OxylabsWebScrapeArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL of the page to read.")
    output: Literal["markdown", "html"] = Field(
        default="markdown",
        description=(
            "markdown to read the page: far fewer tokens, structure intact. html only "
            "when you need the markup itself. Ignored when `json_prompt` is set."
        ),
    )
    location: str | None = Field(
        default=None,
        description=(
            "Two-letter country code to fetch the page from. Use it when the page "
            "varies by country: pricing, availability, language."
        ),
        pattern="^[A-Za-z]{2}$",
    )
    device: Literal["desktop", "mobile"] | None = Field(
        default=None, description="Viewport to fetch as. The API's default is desktop."
    )
    run_js: bool = Field(
        default=False,
        description=(
            "Execute the page's JavaScript. Needed for pages that render client-side "
            "and come back nearly empty otherwise. Slower: try without it first."
        ),
    )
    json_prompt: str | None = Field(
        default=None,
        description=(
            "Fields to pull out as JSON, in plain words: name them and say what shape "
            "you want, e.g. 'the product title, price and a list of image links'. "
            "Leave unset to read the page yourself."
        ),
    )


class OxylabsWebScrapeTool(OxylabsWebAPIBaseTool):
    """Read any web page with the Oxylabs Web API.

    Get an API key from a Web API instance in the Oxylabs dashboard:
    https://dashboard.oxylabs.io

    Args:
        api_key (str): Oxylabs Web API key. Defaults to ``OXYLABS_WEB_API_KEY``.
        max_content_chars (int | None): Truncate page content to this many
            characters. ``None`` turns truncation off.
        max_wait (float): Seconds to wait for a slow scrape the API queued.
    """

    name: str = "Oxylabs Web Scrape tool"
    description: str = (
        "Fetch and read a single URL, including JavaScript-heavy and bot-protected "
        "pages that an ordinary fetch can't open. Returns the page as Markdown by "
        "default. If the page comes back nearly empty, call again with `run_js` set "
        "to true. Set `json_prompt` to get named fields as JSON instead of the page."
    )
    args_schema: type[BaseModel] = OxylabsWebScrapeArgs

    max_content_chars: int | None = Field(
        default=50_000,
        description=(
            "Truncate page content to this many characters, so one large page "
            "doesn't fill the model's context. None turns truncation off."
        ),
    )
    max_wait: float = Field(
        default=600, description="Seconds to wait for a slow scrape the API queued."
    )
    poll_interval: float = Field(
        default=5, description="Seconds between checks on a queued scrape."
    )

    def _run(
        self,
        url: str,
        output: Literal["markdown", "html"] = "markdown",
        location: str | None = None,
        device: Literal["desktop", "mobile"] | None = None,
        run_js: bool = False,
        json_prompt: str | None = None,
    ) -> str | ToolFailure:
        if not url.startswith(("http://", "https://")):
            return self._invalid_input(
                f"`url` must be an absolute http(s) URL, got: {url!r}"
            )

        payload: dict[str, Any] = {"url": url, "output": [output]}
        if json_prompt and json_prompt.strip():
            # The AI parser: returns the named fields under `json` instead of the page.
            payload["output"] = ["json"]
            payload["json"] = {"prompt": json_prompt}
        if location:
            payload["location"] = location.strip().upper()
        if device:
            payload["device"] = device
        if run_js:
            payload["run_js"] = True

        try:
            response = self._request("POST", "/v1/scrape", payload)
        except _ScrapeTimeoutError:
            # The page took longer than the synchronous call allows: the same
            # request goes to the API's queue, and the result is collected there.
            response = self._queue_and_wait(payload)
        if isinstance(response, ToolFailure):
            return response

        return self._format(response, output, run_js)

    def _queue_and_wait(self, payload: dict[str, Any]) -> dict[str, Any] | ToolFailure:
        queued = self._request("POST", "/v1/async/scrape", payload)
        if isinstance(queued, ToolFailure):
            return queued
        request_id = (queued.get("metadata") or {}).get("request_id")
        if not request_id:
            return ToolFailure(
                message="Oxylabs Web API queued the scrape but returned no request id.",
                code="invalid_response",
                retryable=True,
            )

        deadline = time.monotonic() + self.max_wait
        while True:
            body = self._request("GET", f"/v1/async/scrape/{request_id}")
            if isinstance(body, ToolFailure) or body.get("state") != "pending":
                return body
            if time.monotonic() >= deadline:
                return ToolFailure(
                    message=(
                        f"The page was still loading after {self.max_wait:.0f}s. "
                        "Raise `max_wait` on the tool for slow pages."
                    ),
                    code="pending",
                    retryable=True,
                    details={"request_id": request_id},
                )
            time.sleep(self.poll_interval)

    def _format(
        self, response: dict[str, Any], output: str, run_js: bool
    ) -> str | ToolFailure:
        results = [r for r in response.get("results") or [] if isinstance(r, dict)]
        if not results:
            return ToolFailure(
                message="Oxylabs Web API returned no results for this page.",
                code="empty_response",
                retryable=True,
            )

        if "json" in results[0]:
            parsed = [r.get("json") for r in results]
            return json.dumps(parsed[0] if len(parsed) == 1 else parsed)

        text = "\n\n".join(str(r.get(output) or "") for r in results)
        limit = self.max_content_chars
        if limit is not None and len(text) > limit:
            return (
                f"{text[:limit]}\n\n[Truncated: showing {limit} of {len(text)} "
                "characters. Use `json_prompt` to pull out only the fields you need.]"
            )
        if not run_js and len(text.strip()) < _THIN_CONTENT_CHARS:
            return (
                f"{text}\n\n[Very little content. If the page renders client-side, "
                "call again with `run_js` set to true.]"
            )
        return text
