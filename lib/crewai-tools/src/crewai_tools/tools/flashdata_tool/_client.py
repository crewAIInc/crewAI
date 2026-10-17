"""Bounded FlashData requests with no automatic retry or credential forwarding."""

from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


QUERY_URL = "https://data.flashdata.dev/v1/queries/realtime"
ASYNC_URL = "https://data.flashdata.dev/v1/queries"
JOBS_URL = "https://api.flashdata.dev/v2/jobs"
AUTH_URL = "https://api.flashdata.dev/v2/api-access"
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
UNKNOWN_OUTCOME = (
    "The request may already have been accepted and charged. "
    "Check Jobs in the FlashData console before resubmitting. No automatic retry was made."
)
READ_ERROR = (
    "Could not complete this read. Retry the same job read; do not submit a new job."
)


class FlashDataError(Exception):
    """An error safe to display without raw responses or credentials."""


class Client:
    def __init__(
        self, api_key: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.api_key = api_key
        self.transport = transport

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
    ) -> httpx.Response:
        headers = {"Accept": "application/json", "User-Agent": "flashdata-crewai/0.1.0"}
        if authenticated:
            headers.update({"X-API-Key": self.api_key, "X-Request-Id": str(uuid4())})
        try:
            with (
                httpx.Client(
                    timeout=httpx.Timeout(120.0, connect=15.0),
                    follow_redirects=False,
                    trust_env=False,
                    transport=self.transport,
                ) as client,
                client.stream(method, url, json=payload, headers=headers) as response,
            ):
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise FlashDataError(
                            "Response exceeds 20 MiB. Retrieve it from Jobs; do not resubmit."
                        )
                    body.extend(chunk)
                # iter_bytes already decodes content encodings; do not decode twice.
                return httpx.Response(
                    response.status_code,
                    content=bytes(body),
                    headers={"location": response.headers.get("location", "")},
                )
        except httpx.HTTPError:
            raise FlashDataError(
                UNKNOWN_OUTCOME if method == "POST" else READ_ERROR
            ) from None

    @staticmethod
    def _json(response: httpx.Response, *, submission: bool = False) -> dict[str, Any]:
        message = UNKNOWN_OUTCOME if submission else READ_ERROR
        errors = {
            401: "API key rejected. Check FLASHDATA_API_KEY.",
            402: "Insufficient FlashData Credits. Fund your account before a new request.",
            403: "This account or key cannot access the requested source.",
            429: "FlashData rate or quota limit reached. Wait before a new request.",
        }
        if response.status_code != 200:
            raise FlashDataError(
                f"{errors.get(response.status_code, message)} HTTP {response.status_code}."
            )
        try:
            data = response.json()
        except ValueError:
            raise FlashDataError(message) from None
        if not isinstance(data, dict):
            raise FlashDataError(message)
        return data

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._json(self._request("POST", QUERY_URL, payload), submission=True)
        if data.get("status") not in (None, "done", "completed") or not isinstance(
            data.get("results"), list
        ):
            raise FlashDataError(UNKNOWN_OUTCOME)
        job = data.get("job")
        if isinstance(job, dict) and job.get("status") == "failed":
            raise FlashDataError(
                "The job failed. Inspect it in Jobs before a new request."
            )
        return data

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        from crewai_tools.tools.flashdata_tool._inputs import DownloadJobInput

        data = self._json(self._request("POST", ASYNC_URL, payload), submission=True)
        try:
            DownloadJobInput.model_validate({"job_id": data.get("id")})
        except ValueError:
            raise FlashDataError(UNKNOWN_OUTCOME) from None
        if data.get("status") not in ("pending", "processing"):
            raise FlashDataError(UNKNOWN_OUTCOME)
        return {
            k: data[k] for k in ("id", "status", "source", "created_at") if k in data
        }

    def status(self, job_id: str) -> dict[str, Any]:
        from crewai_tools.tools.flashdata_tool._inputs import DownloadJobInput

        DownloadJobInput(job_id=job_id)
        data = self._json(self._request("GET", f"{JOBS_URL}/{job_id}"))
        job = data.get("job")
        if not isinstance(job, dict) or job.get("source") != "youtube_download":
            raise FlashDataError("This reader only accepts YouTube download jobs.")
        if job.get("id") != job_id:
            raise FlashDataError("Job receipt does not match the requested ID.")
        # Job params can contain storage credentials. Export lifecycle fields only.
        return {
            "job": {
                k: job[k]
                for k in (
                    "id",
                    "source",
                    "status",
                    "created_at",
                    "updated_at",
                    "completed_at",
                )
                if k in job
            }
        }

    def result(self, job_id: str) -> dict[str, Any]:
        status = self.status(job_id)
        if status["job"].get("status") != "completed":
            return status
        identity = self._json(self._request("GET", AUTH_URL))
        link = self._request("GET", f"{JOBS_URL}/{job_id}/results")
        try:
            origin = identity.get("result_origin")
            trusted = urlsplit(origin) if isinstance(origin, str) else None
            target = urlsplit(link.headers.get("location", ""))
            if (
                link.status_code != 302
                or not trusted
                or trusted.scheme != "https"
                or not trusted.hostname
                or trusted.username
                or trusted.password
                or trusted.path not in ("", "/")
                or trusted.query
                or trusted.fragment
                or target.scheme != "https"
                or target.username
                or target.password
                or target.fragment
                or (target.hostname, target.port or 443)
                != (trusted.hostname, trusted.port or 443)
            ):
                raise ValueError()
        except ValueError:
            raise FlashDataError(
                "Result URL does not match trusted FlashData storage."
            ) from None
        output = self._json(self._request("GET", target.geturl(), authenticated=False))
        return {**status, "output": output}
