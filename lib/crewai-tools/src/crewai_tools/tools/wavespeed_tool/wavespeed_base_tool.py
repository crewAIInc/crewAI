from __future__ import annotations

import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests


_FAILED_STATUSES = frozenset({"failed", "cancelled", "timeout", "deleted"})
# Poll answers worth retrying: the prediction keeps running while the API recovers.
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 30.0
_UNKNOWN_SUBMIT = (
    "The prediction may still have been created and billed; check your WaveSpeed "
    "history (https://wavespeed.ai/history) before running the tool again."
)


class WaveSpeedPredictionError(RuntimeError):
    """A WaveSpeed prediction was accepted but did not produce outputs.

    ``prediction_id`` identifies the run so its result can be read later instead
    of submitting (and paying for) it again.
    """

    def __init__(self, message: str, prediction_id: str) -> None:
        super().__init__(message)
        self.prediction_id = prediction_id


class WaveSpeedTimeoutError(TimeoutError):
    """A WaveSpeed prediction was still running when the tool stopped waiting."""

    def __init__(self, message: str, prediction_id: str) -> None:
        super().__init__(message)
        self.prediction_id = prediction_id


def _retry_after(response: requests.Response | None) -> float | None:
    """Seconds from a numeric Retry-After header, if any."""
    if response is None:
        return None
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return None


class WaveSpeedBaseTool(BaseTool):
    """Shared base for tools that run models on the WaveSpeed AI API.

    A run is submitted once and then polled until it completes. The submit
    request is never retried, because every accepted submission starts a new
    (billed) prediction; only the polling requests retry transient errors
    (408, 429, 5xx, network errors) with exponential backoff and Retry-After.
    Once a prediction id exists, every error carries it.
    """

    base_url: str = Field(
        default="https://api.wavespeed.ai",
        description="Base URL of the WaveSpeed AI API.",
    )
    poll_interval: float = Field(
        default=2.0,
        description="Seconds to wait between result polls.",
    )
    timeout: float = Field(
        default=300.0,
        description=(
            "Maximum seconds to wait for a prediction to finish, including the "
            "submit request."
        ),
    )
    request_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for each HTTP request.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional model inputs sent with every request, for parameters "
            "of the chosen model that the tool does not expose directly."
        ),
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="WAVESPEED_API_KEY",
                description="API key for WaveSpeed AI",
                required=True,
            ),
        ]
    )

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv("WAVESPEED_API_KEY")
        if not api_key:
            raise ValueError(
                "WAVESPEED_API_KEY environment variable is required to use "
                f"{self.__class__.__name__}."
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _unwrap(response: requests.Response) -> dict[str, Any]:
        """Return the ``data`` object of an API response or raise on errors."""
        try:
            body = response.json()
        except ValueError:
            body = None

        message = body.get("message") if isinstance(body, dict) else None
        if not response.ok:
            raise RuntimeError(
                f"WaveSpeed API request failed with HTTP {response.status_code}: "
                f"{message or response.text}"
            )
        if not isinstance(body, dict):
            raise RuntimeError("WaveSpeed API returned a non-JSON response.")
        if body.get("code") != 200:
            raise RuntimeError(
                f"WaveSpeed API error {body.get('code')}: {message or 'unknown error'}"
            )
        data = body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("WaveSpeed API response has no data.")
        return data

    def _submit(
        self,
        model: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        """Send the one submit request; never retried."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v3/{model}",
                json={**self.extra_params, **payload},
                headers=headers,
                timeout=timeout,
            )
        except requests.ConnectTimeout as exc:
            # The connection never opened, so no prediction was created.
            raise RuntimeError(f"Could not reach the WaveSpeed API: {exc}") from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"WaveSpeed did not answer the submit request in time. {_UNKNOWN_SUBMIT}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"The WaveSpeed submit request failed ({type(exc).__name__}). "
                f"{_UNKNOWN_SUBMIT}"
            ) from exc
        if response.status_code >= 500:
            raise RuntimeError(
                f"WaveSpeed API returned HTTP {response.status_code} to the submit "
                f"request. {_UNKNOWN_SUBMIT}"
            )
        return self._unwrap(response)

    def _generate(self, model: str, payload: dict[str, Any]) -> tuple[str, list[str]]:
        """Submit a prediction and wait for its outputs.

        Args:
            model: The WaveSpeed model id, e.g. ``bytedance/seedream-v5.5``.
            payload: The model input.

        Returns:
            The prediction id and the list of output URLs.

        Raises:
            WaveSpeedPredictionError: The prediction failed or returned no usable
                outputs; ``prediction_id`` is set.
            WaveSpeedTimeoutError: The prediction was still running after
                ``timeout`` seconds; ``prediction_id`` is set.
            RuntimeError: The submit request failed; no prediction id exists.
        """
        headers = self._headers()
        deadline = time.monotonic() + self.timeout
        submit_timeout = max(0.001, min(self.request_timeout, self.timeout))
        data = self._submit(model, payload, headers, submit_timeout)
        prediction_id = data.get("id")
        if not prediction_id:
            raise RuntimeError("WaveSpeed API did not return a prediction id.")

        result_url = f"{self.base_url}/api/v3/predictions/{prediction_id}/result"
        delay = self.poll_interval
        while True:
            status = data.get("status")
            if status == "completed":
                outputs = data.get("outputs")
                if not isinstance(outputs, list):
                    raise WaveSpeedPredictionError(
                        f"WaveSpeed prediction {prediction_id} completed, but its "
                        "outputs are not a list.",
                        prediction_id,
                    )
                urls = [url for url in outputs if isinstance(url, str) and url]
                if not urls:
                    raise WaveSpeedPredictionError(
                        f"WaveSpeed prediction {prediction_id} completed without outputs.",
                        prediction_id,
                    )
                return prediction_id, urls
            if status in _FAILED_STATUSES:
                raise WaveSpeedPredictionError(
                    f"WaveSpeed prediction {prediction_id} {status}: "
                    f"{data.get('error') or 'no error message'}",
                    prediction_id,
                )
            if time.monotonic() >= deadline:
                raise WaveSpeedTimeoutError(
                    f"WaveSpeed prediction {prediction_id} did not finish within "
                    f"{self.timeout:g} seconds (last status: {status}). It may still "
                    f"finish: read its result at {result_url} instead of running the "
                    "tool again.",
                    prediction_id,
                )
            time.sleep(delay)
            try:
                poll: requests.Response | None = requests.get(
                    result_url, headers=headers, timeout=self.request_timeout
                )
            except requests.RequestException:
                poll = None
            if poll is None or poll.status_code in _TRANSIENT_STATUS_CODES:
                hint = _retry_after(poll)
                delay = (
                    min(hint, _MAX_BACKOFF_SECONDS)
                    if hint is not None
                    else min(max(delay, self.poll_interval) * 2, _MAX_BACKOFF_SECONDS)
                )
                continue
            delay = self.poll_interval
            try:
                data = self._unwrap(poll)
            except RuntimeError as exc:
                raise WaveSpeedPredictionError(
                    f"{exc} (prediction {prediction_id}; result: {result_url})",
                    prediction_id,
                ) from exc
