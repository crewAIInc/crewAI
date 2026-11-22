"""Flatverify tools: check whether email addresses can receive mail."""

from __future__ import annotations

from collections.abc import Iterator
import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field, PrivateAttr
import requests


FLATVERIFY_API_URL = "https://app.flatverify.com/api/v1"
_ENV_API_KEY = "FLATVERIFY_API_KEY"
_DONE = "done"
_USER_AGENT = "crewai-tools-flatverify"


class FlatverifyAPIError(Exception):
    """An error answer from the Flatverify API, or no answer at all."""

    def __init__(
        self, message: str, *, status_code: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def _error_from(resp: requests.Response) -> FlatverifyAPIError:
    code = "error"
    message = resp.reason or "Request failed"
    try:
        body = resp.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        code = str(error.get("code") or code)
        message = str(error.get("message") or message)
    return FlatverifyAPIError(message, status_code=resp.status_code, code=code)


def _is_retryable(resp: requests.Response) -> bool:
    """503 busy and 429 rate_limited clear after Retry-After; too_many_jobs does not."""
    if resp.status_code == 503:
        return True
    return resp.status_code == 429 and _error_from(resp).code != "too_many_jobs"


def _retry_after(resp: requests.Response, attempt: int, max_wait: float) -> float:
    try:
        wait = float(resp.headers.get("Retry-After", 2**attempt))
    except ValueError:
        wait = float(2**attempt)
    return max(0.0, min(wait, max_wait))


class _FlatverifyClient:
    """Minimal Flatverify REST client used by the tools."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        max_retry_wait: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_retry_wait = max_retry_wait
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            }
        )

    def __repr__(self) -> str:
        return f"_FlatverifyClient(base_url={self._base_url!r})"

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            try:
                resp = self._session.request(
                    method, f"{self._base_url}{path}", timeout=self._timeout, **kwargs
                )
            except requests.RequestException as e:
                raise FlatverifyAPIError(
                    f"Could not reach the Flatverify API: {type(e).__name__}"
                ) from None
            if resp.ok:
                return resp.json()
            if _is_retryable(resp) and attempt < self._max_retries:
                time.sleep(_retry_after(resp, attempt, self._max_retry_wait))
                attempt += 1
                continue
            raise _error_from(resp)

    def iter_results(self, job_id: str) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"format": "json"}
        while True:
            page = self.request("GET", f"/jobs/{job_id}/results", params=params)
            yield from page.get("results") or []
            if page.get("next_after") is None:
                return
            params = {"format": "json", "after": page["next_after"]}


class _FlatverifyBaseTool(BaseTool):
    """Shared settings and API key handling for the Flatverify tools."""

    base_url: str = FLATVERIFY_API_URL
    timeout: float = 30.0
    max_retries: int = Field(
        default=3, description="Retries for rate limited (429) and busy (503) answers."
    )
    max_retry_wait: float = Field(
        default=30.0, description="Longest wait in seconds between two retries."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=_ENV_API_KEY,
                description="API key for Flatverify (create one at app.flatverify.com)",
                required=True,
            ),
        ]
    )
    _client: _FlatverifyClient = PrivateAttr()

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Create the tool.

        Args:
            api_key: Flatverify API key. Defaults to the FLATVERIFY_API_KEY
                environment variable.
            **kwargs: Other tool settings such as ``timeout`` or ``max_retries``.
        """
        super().__init__(**kwargs)
        key = api_key or os.getenv(_ENV_API_KEY)
        if not key:
            raise ValueError(
                f"Missing Flatverify API key. Set {_ENV_API_KEY} or pass api_key."
            )
        self._client = _FlatverifyClient(
            key, self.base_url, self.timeout, self.max_retries, self.max_retry_wait
        )

    @staticmethod
    def _failure(error: FlatverifyAPIError) -> ToolFailure:
        message = (
            f"Flatverify API error {error.status_code}: {error.message}"
            if error.status_code
            else error.message
        )
        return ToolFailure(
            message=message,
            code=error.code,
            retryable=error.status_code in (None, 429, 503),
        )


class FlatverifyVerifyEmailToolSchema(BaseModel):
    """Input for FlatverifyVerifyEmailTool."""

    email: str = Field(
        ..., description="The email address to check, for example maya.chen@amazon.com."
    )


class FlatverifyVerifyEmailTool(_FlatverifyBaseTool):
    """Check whether one email address can receive mail."""

    name: str = "Flatverify Verify Email"
    description: str = (
        "Check whether one email address can receive mail. Input is a single "
        "email address. Returns status (valid, invalid, risky or unknown), "
        "sub_status with the reason (for example accepted, mailbox_not_found, "
        "catch_all, role_account or disposable) and true or false flags for "
        "disposable, role, free and catch_all addresses."
    )
    args_schema: type[BaseModel] = FlatverifyVerifyEmailToolSchema

    def _run(self, email: str) -> dict[str, Any] | ToolFailure:
        try:
            result: dict[str, Any] = self._client.request(
                "POST", "/verify", json={"email": email}
            )
        except FlatverifyAPIError as e:
            return self._failure(e)
        return result


class FlatverifyBulkVerifyToolSchema(BaseModel):
    """Input for FlatverifyBulkVerifyTool."""

    emails: list[str] = Field(
        default_factory=list,
        description=(
            "The email addresses to check, for example "
            '["j.ortiz@salesforce.com", "sam.lee@stripe.com"].'
        ),
    )
    name: str | None = Field(
        default=None, description="Optional label for the job, such as the list name."
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Only to fetch an earlier job that was not finished yet: its job_id. "
            "Leave emails empty when you pass this."
        ),
    )


class FlatverifyBulkVerifyTool(_FlatverifyBaseTool):
    """Check a list of email addresses as one job and wait for the results."""

    name: str = "Flatverify Bulk Verify Emails"
    description: str = (
        "Check a list of email addresses as one job and wait for the results. "
        "Use this instead of checking many addresses one by one. Duplicates and "
        "malformed entries are skipped. Returns counts per status (valid, "
        "invalid, risky, unknown) and one row per address with status, "
        "sub_status and flags. If the job is not finished within the wait time, "
        "returns its job_id; call this tool again with only that job_id to get "
        "the results."
    )
    args_schema: type[BaseModel] = FlatverifyBulkVerifyToolSchema
    wait_timeout: float = Field(
        default=300.0,
        description="Seconds to wait for a job before returning its job_id.",
    )
    poll_interval: float = Field(
        default=3.0, description="Seconds between two job status checks."
    )
    max_results: int = Field(
        default=1000, description="Most result rows returned to the agent per call."
    )

    def _run(
        self,
        emails: list[str] | None = None,
        name: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any] | ToolFailure:
        if not emails and not job_id:
            return ToolFailure(
                message="Pass a list of email addresses, or the job_id of an earlier job.",
                code="invalid_request",
            )
        try:
            skipped: int | None = None
            if not job_id:
                body: dict[str, Any] = {"emails": list(emails or [])}
                if name:
                    body["name"] = name
                started = self._client.request("POST", "/verify/bulk", json=body)
                job_id = str(started["id"])
                skipped = started.get("skipped")
            job = self._wait(job_id)
            return self._summary(job, skipped)
        except FlatverifyAPIError as e:
            return self._failure(e)

    def _wait(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.wait_timeout
        while True:
            job: dict[str, Any] = self._client.request("GET", f"/jobs/{job_id}")
            if job.get("status") == _DONE or time.monotonic() >= deadline:
                return job
            time.sleep(self.poll_interval)

    def _summary(self, job: dict[str, Any], skipped: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": job.get("id"),
            "status": job.get("status"),
            "total": job.get("total"),
            "processed": job.get("processed"),
            "counts": job.get("counts"),
        }
        if skipped is not None:
            out["skipped"] = skipped
        if job.get("status") != _DONE:
            if job.get("waiting_reason"):
                out["waiting_reason"] = job["waiting_reason"]
            out["message"] = (
                "The job is not finished yet. Call this tool again with this job_id "
                "to get the results."
            )
            return out
        rows: list[dict[str, Any]] = []
        truncated = False
        for row in self._client.iter_results(str(job["id"])):
            if len(rows) >= self.max_results:
                truncated = True
                break
            rows.append(row)
        out["results"] = rows
        out["results_truncated"] = truncated
        return out
