import os
import random
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
from pydantic import Field, PrivateAttr
import requests

from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import sdk_type


__all__ = ["OxylabsWebAPIBaseTool"]


# Transient statuses worth another attempt. A spent quota, a faulted request and
# a scrape timeout are verdicts, and resending them changes nothing.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_TIMEOUT_TITLE = "REQUEST_FAILED_SCRAPE_TIMEOUT"
_QUOTA_TITLE = "QUOTA_EXCEEDED"


class _ScrapeTimeoutError(Exception):
    """The synchronous scrape ran out of time; the request can be queued instead."""


def _json_body(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _detail(body: dict[str, Any] | None, response: requests.Response) -> str:
    """The useful part of an error body: per-field errors first, then `detail`."""
    if body is None:
        return response.text[:300]
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(
            ": ".join(str(part) for part in (e.get("pointer"), e.get("detail")) if part)
            for e in errors
            if isinstance(e, dict)
        )
    return str(body.get("detail") or body.get("message") or body)[:300]


def _retryable(response: requests.Response, body: dict[str, Any] | None) -> bool:
    if response.status_code not in _RETRY_STATUS:
        return False
    if body is None:
        return True
    return body.get("title") not in (_TIMEOUT_TITLE, _QUOTA_TITLE) and (
        body.get("state") != "faulted"
    )


def _failure(response: requests.Response, path: str) -> ToolFailure | None:
    """Describe a failed response, or None when it succeeded."""
    body = _json_body(response)
    title = (body or {}).get("title")
    status = response.status_code

    if status == 401:
        return ToolFailure(
            message=(
                "Oxylabs Web API rejected the API key (401). Check OXYLABS_WEB_API_KEY: "
                "it must come from a Web API instance in the Oxylabs dashboard, and "
                "keys from other Oxylabs products don't work."
            ),
            code="401",
        )
    if title == _QUOTA_TITLE:
        return ToolFailure(
            message=(
                "Oxylabs Web API quota is spent. Check the plan in the Oxylabs "
                "dashboard; retrying will not help."
            ),
            code="quota_exceeded",
        )
    if status == 429:
        return ToolFailure(
            message="Oxylabs Web API is rate limiting requests (429).",
            code="429",
            retryable=True,
        )
    if body is not None and body.get("state") == "faulted":
        request_id = (body.get("metadata") or {}).get("request_id")
        reason = ": ".join(str(p) for p in (title, body.get("detail")) if p)
        return ToolFailure(
            message=(
                "Oxylabs Web API could not complete the request "
                f"({reason or 'no reason given'})."
            ),
            code=str(title or status),
            retryable=True,
            details={"request_id": request_id} if request_id else {},
        )
    if status >= 400:
        return ToolFailure(
            message=f"Oxylabs Web API {path} returned {status}: {_detail(body, response)}",
            code=str(status),
            retryable=status >= 500,
        )
    if body is None:
        return ToolFailure(
            message=f"Oxylabs Web API {path} returned a non-JSON response.",
            code="invalid_response",
            retryable=True,
        )
    return None


class OxylabsWebAPIBaseTool(BaseTool):
    """Base class for the Oxylabs Web API tools.

    Holds what the Web API tools share: the API key, the HTTP calls with
    retries, and the translation of a failed call into a :class:`ToolFailure`.

    Get an API key from a Web API instance in the Oxylabs dashboard:
    https://dashboard.oxylabs.io
    """

    base_url: str = "https://webapi.oxylabs.io"
    request_timeout: float = Field(
        default=200, description="Seconds to wait for one HTTP response."
    )
    max_retries: int = Field(
        default=2, description="Extra attempts for 429, 5xx and connection errors."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OXYLABS_WEB_API_KEY",
                description="API key of an Oxylabs Web API instance",
                required=True,
            ),
        ]
    )

    _api_key: str = PrivateAttr()

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # A key pasted or read from a file often carries a trailing newline.
        api_key = (api_key or os.environ.get("OXYLABS_WEB_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "You must pass api_key when instantiating the tool or set the "
                "OXYLABS_WEB_API_KEY environment variable"
            )
        if any(char.isspace() for char in api_key):
            # Rejected here because requests would otherwise refuse the header
            # with an error that quotes the whole key.
            raise ValueError(
                "OXYLABS_WEB_API_KEY contains whitespace or a line break; pass "
                "the key on its own"
            )
        self._api_key = api_key

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any] | ToolFailure:
        """Call the Web API. Returns the response body, or why there is none.

        Raises ``_ScrapeTimeoutError`` when a synchronous scrape outlived the API's
        own limit, so the caller can queue the same request instead.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "x-oxylabs-sdk": sdk_type(),
        }
        failure: ToolFailure | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                # Full jitter, so concurrent agents that hit the same 429 don't
                # retry in step.
                time.sleep(random.uniform(0, 2 ** (attempt - 1)))  # noqa: S311
            try:
                response = requests.request(
                    method,
                    f"{self.base_url.rstrip('/')}{path}",
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout,
                )
            except requests.Timeout:
                return ToolFailure(
                    message=f"Oxylabs Web API {path} did not respond in time.",
                    code="timeout",
                    retryable=True,
                )
            except requests.exceptions.InvalidHeader:
                # Never echo this one: requests quotes the rejected header value,
                # which is the API key.
                return ToolFailure(
                    message=(
                        "Oxylabs Web API request headers were rejected: check that "
                        "OXYLABS_WEB_API_KEY holds only the key."
                    ),
                    code="invalid_key",
                )
            except requests.RequestException as e:
                failure = ToolFailure(
                    message=f"Could not reach the Oxylabs Web API: {e}",
                    code="connection_error",
                    retryable=True,
                )
                continue

            body = _json_body(response)
            if (body or {}).get("title") == _TIMEOUT_TITLE:
                raise _ScrapeTimeoutError
            if attempt < self.max_retries and _retryable(response, body):
                continue
            failure = _failure(response, path)
            if failure is None and body is not None:
                return body
            break

        return failure or ToolFailure(
            message=f"Oxylabs Web API {path} failed after retries.",
            retryable=True,
        )

    @staticmethod
    def _invalid_input(message: str) -> ToolFailure:
        return ToolFailure(
            message=message,
            reason=ToolFailureReason.INVALID_INPUT,
            code="invalid_input",
        )
