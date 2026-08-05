import logging
import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)


class TunovaMusicGenerationToolSchema(BaseModel):
    """Input for TunovaMusicGenerationTool."""

    prompt: str = Field(
        ...,
        description="Mandatory text prompt describing the music to generate, e.g. 'an upbeat synthwave track about summer nights'",
    )
    make_instrumental: bool = Field(
        False, description="Generate an instrumental track without vocals"
    )
    wait_seconds: int = Field(
        360,
        ge=10,
        le=360,
        description="Maximum number of seconds to wait for the render to finish",
    )


class TunovaMusicGenerationTool(BaseTool):
    name: str = "Tunova music generation tool"
    description: str = (
        "Generate a complete song from a text prompt with Tunova, a hosted "
        "Suno-quality music generation API. Returns the track title and audio "
        "URL. Songs are billed only on successful renders, so failed "
        "generations can be retried at no extra cost."
    )
    args_schema: type[BaseModel] = TunovaMusicGenerationToolSchema
    base_url: str = "https://api.tunova.ai"
    poll_interval: int = 10
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TUNOVA_API_KEY", description="API key for Tunova", required=True
            ),
        ]
    )

    def _headers(self) -> dict[str, str]:
        """Build request headers with the Tunova API key."""
        api_key = os.environ.get("TUNOVA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TUNOVA_API_KEY environment variable is required to use this tool"
            )
        return {"X-API-Key": api_key, "content-type": "application/json"}

    def _submit_job(self, prompt: str, make_instrumental: bool) -> str:
        """Submit a generation job and return its job id."""
        payload: dict[str, Any] = {"prompt": prompt}
        if make_instrumental:
            payload["make_instrumental"] = True
        response = requests.post(
            f"{self.base_url}/api/generate",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        job_id = response.json().get("job_id")
        if not job_id:
            raise ValueError("Tunova API did not return a job_id")
        return str(job_id)

    def _get_job(self, job_id: str) -> dict[str, Any]:
        """Fetch the current state of a generation job."""
        response = requests.get(
            f"{self.base_url}/api/jobs/{job_id}",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        return dict(response.json())

    @staticmethod
    def _format_clips(clips: list[dict[str, Any]]) -> str:
        """Format finished clips into a short human-readable summary."""
        lines: list[str] = []
        for clip in clips:
            title = clip.get("title") or "Untitled"
            duration = clip.get("duration")
            duration_note = f" ({duration:.0f}s)" if duration else ""
            lines.append(f"'{title}'{duration_note}: {clip.get('audio_url', '')}")
        return "\n".join(lines)

    def _run(
        self, prompt: str, make_instrumental: bool = False, wait_seconds: int = 360
    ) -> str:
        """Generate a song from a text prompt and wait for the render to finish."""
        try:
            job_id = self._submit_job(prompt, make_instrumental)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting generation job to Tunova API: {e}")
            return f"Failed to submit music generation job: {e}"

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                job = self._get_job(job_id)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error polling Tunova job {job_id}: {e}")
                continue

            status = job.get("status")
            if status == "complete":
                clips = job.get("clips") or []
                if not clips:
                    return f"Music generation job {job_id} completed but returned no clips."
                return f"Music generated successfully:\n{self._format_clips(clips)}"
            if status == "failed":
                return (
                    f"Music generation job {job_id} failed. Failed renders are "
                    "automatically refunded, so it is safe to retry this tool "
                    "at no extra cost."
                )

        return (
            f"Music generation job {job_id} did not finish within {wait_seconds} "
            "seconds. You are only billed for successful renders."
        )
