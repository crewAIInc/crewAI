"""ChronoVerify image capture-time and provenance verification tool for CrewAI."""

import json
import os
from pathlib import Path

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class ChronoVerifyImageVerificationToolSchema(BaseModel):
    """Input for ChronoVerifyImageVerificationTool."""

    image_path: str | None = Field(
        default=None,
        description=(
            "Local filesystem path of the image file to verify. "
            "Provide exactly one of image_path or image_url."
        ),
    )
    image_url: str | None = Field(
        default=None,
        description=(
            "Direct URL of the image to verify. Must be a direct link to the "
            "image bytes; redirects and hosts that block automated fetches fail, "
            "so prefer image_path when the file is available locally."
        ),
    )


class ChronoVerifyImageVerificationTool(BaseTool):
    """Verify a photo's capture time and provenance with the ChronoVerify API.

    Sends the image to the ChronoVerify verification endpoint, which checks
    EXIF and XMP metadata, validates C2PA Content Credentials against the
    official trust lists, and runs classical pixel forensics. The response is
    a typed verdict with a 0 to 100 confidence plus the extracted capture
    time, capture device, capture location, C2PA details, and file hashes.

    ChronoVerify is not a deepfake or AI-generation detector. Verdicts are
    investigative triage to support human review, not proof.

    An API key is optional. Without one, requests use the free keyless path,
    which is rate limited per IP. Set ``CHRONOVERIFY_API_KEY`` to meter
    requests against a key instead.
    """

    name: str = "ChronoVerify Image Verification"
    description: str = (
        "Verify when a photo was taken and its provenance. Checks EXIF and XMP "
        "metadata, validates C2PA Content Credentials against the official "
        "trust lists, and runs pixel forensics, returning one verdict "
        "(provenance_confirmed, consistent, inconclusive, metadata_anomaly, or "
        "manipulation_indicated) with a 0 to 100 confidence, plus the extracted "
        "capture time, capture device, and capture location. It does not detect "
        "deepfakes or AI generation; verdicts are investigative triage to "
        "support human review, not proof. Accepts a local image file path or a "
        "direct image URL."
    )
    args_schema: type[BaseModel] = ChronoVerifyImageVerificationToolSchema
    api_base_url: str = Field(
        default="https://chronoverify.com",
        description="Base URL of the ChronoVerify API.",
    )
    request_timeout: int = Field(
        default=120,
        description="Timeout in seconds for the verification request.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CHRONOVERIFY_API_KEY",
                description=(
                    "Optional ChronoVerify API key (cv_live_...). Without it, "
                    "requests use the free keyless path, which is rate limited "
                    "per IP. A free key is available at "
                    "https://chronoverify.com/v1/keys/free."
                ),
                required=False,
            ),
        ]
    )

    def _request_headers(self) -> dict[str, str]:
        """Build request headers, adding auth only when an API key is set."""
        api_key = os.environ.get("CHRONOVERIFY_API_KEY")
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
        return {}

    def _run(
        self,
        image_path: str | None = None,
        image_url: str | None = None,
    ) -> str:
        """Verify an image and return the verification result as JSON text.

        Args:
            image_path: Local path of the image file to verify.
            image_url: Direct URL of the image to verify.

        Returns:
            The ChronoVerify verification response serialized as JSON, or an
            error message describing what went wrong.
        """
        if not image_path and not image_url:
            return "Error: provide either image_path or image_url."
        if image_path and image_url:
            return "Error: provide only one of image_path or image_url, not both."

        endpoint = f"{self.api_base_url.rstrip('/')}/v1/verify"

        try:
            if image_path:
                path = Path(image_path)
                if not path.is_file():
                    return f"Error: image file not found: {image_path}"
                with path.open("rb") as image_file:
                    response = requests.post(
                        endpoint,
                        headers=self._request_headers(),
                        files={"file": (path.name, image_file)},
                        timeout=self.request_timeout,
                    )
            else:
                response = requests.post(
                    endpoint,
                    headers=self._request_headers(),
                    data={"url": image_url},
                    timeout=self.request_timeout,
                )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "?"
            detail = e.response.text[:500] if e.response is not None else str(e)
            return f"Error verifying image (HTTP {status_code}): {detail}"
        except requests.exceptions.RequestException as e:
            return f"Error communicating with the ChronoVerify API: {e!s}"
        except ValueError as e:
            return f"Error parsing the ChronoVerify API response: {e!s}"
