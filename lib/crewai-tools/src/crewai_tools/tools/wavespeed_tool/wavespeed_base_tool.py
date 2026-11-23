from __future__ import annotations

import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests


_FAILED_STATUSES = frozenset({"failed", "cancelled", "timeout", "deleted"})


class WaveSpeedBaseTool(BaseTool):
    """Shared base for tools that run models on the WaveSpeed AI API.

    A run is submitted once and then polled until it completes. The submit
    request is never retried, because every accepted submission starts a new
    (billed) prediction; only the polling requests tolerate transient errors.
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
        description="Maximum seconds to wait for a prediction to finish.",
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

    def _generate(self, model: str, payload: dict[str, Any]) -> tuple[str, list[str]]:
        """Submit a prediction and wait for its outputs.

        Args:
            model: The WaveSpeed model id, e.g. ``bytedance/seedream-v5.5``.
            payload: The model input.

        Returns:
            The prediction id and the list of output URLs.
        """
        headers = self._headers()
        response = requests.post(
            f"{self.base_url}/api/v3/{model}",
            json={**self.extra_params, **payload},
            headers=headers,
            timeout=self.request_timeout,
        )
        data = self._unwrap(response)
        prediction_id = data.get("id")
        if not prediction_id:
            raise RuntimeError("WaveSpeed API did not return a prediction id.")

        result_url = f"{self.base_url}/api/v3/predictions/{prediction_id}/result"
        deadline = time.monotonic() + self.timeout
        while True:
            status = data.get("status")
            if status == "completed":
                outputs = data.get("outputs") or []
                if not outputs:
                    raise RuntimeError(
                        f"WaveSpeed prediction {prediction_id} completed without outputs."
                    )
                return prediction_id, [str(url) for url in outputs]
            if status in _FAILED_STATUSES:
                raise RuntimeError(
                    f"WaveSpeed prediction {prediction_id} {status}: "
                    f"{data.get('error') or 'no error message'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"WaveSpeed prediction {prediction_id} did not finish within "
                    f"{self.timeout:g} seconds (last status: {status})."
                )
            time.sleep(self.poll_interval)
            try:
                poll = requests.get(
                    result_url, headers=headers, timeout=self.request_timeout
                )
            except requests.RequestException:
                continue
            if poll.status_code >= 500:
                continue
            data = self._unwrap(poll)
