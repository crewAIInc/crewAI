import json
import time
import urllib.error
import urllib.request
from typing import Any, List, Literal, Optional

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


BASE_URL = "https://api.muapi.ai/api/v1"

IMAGE_MODELS = [
    "flux-schnell",
    "flux-dev",
    "flux-kontext-dev",
    "flux-kontext-pro",
    "flux-kontext-max",
    "hidream-fast",
    "hidream-dev",
    "hidream-full",
    "midjourney",
    "gpt4o",
    "gpt-image-2",
    "imagen4",
    "imagen4-fast",
    "seedream",
    "reve",
    "ideogram",
    "hunyuan",
    "wan2.1",
    "qwen",
]

VIDEO_MODELS = [
    "veo3",
    "veo3-fast",
    "kling-master",
    "wan2.1",
    "wan2.2",
    "seedance-pro",
    "seedance-pro-fast",
    "runway",
    "pixverse",
    "sora",
    "minimax-hailuo-02-pro",
]


class MuApiImageSchema(BaseModel):
    """Input for MuAPI Image Generation Tool."""

    prompt: str = Field(description="Text description of the image to generate.")
    model: str = Field(
        default="flux-schnell",
        description=(
            "Model to use. Options: " + ", ".join(IMAGE_MODELS)
        ),
    )
    width: Optional[int] = Field(default=None, description="Image width in pixels.")
    height: Optional[int] = Field(default=None, description="Image height in pixels.")


class MuApiVideoSchema(BaseModel):
    """Input for MuAPI Video Generation Tool."""

    prompt: str = Field(description="Text description of the video to generate.")
    model: str = Field(
        default="veo3-fast",
        description=(
            "Model to use. Options: " + ", ".join(VIDEO_MODELS)
        ),
    )
    duration: int = Field(default=5, description="Duration in seconds (3–60).")
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3"] = Field(
        default="16:9", description="Aspect ratio."
    )


def _submit_and_poll(api_key: str, endpoint: str, payload: dict, timeout: int = 300) -> str:
    """Submit a muapi.ai job and poll until completion, returning the output URL."""
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    body = json.dumps(payload).encode()

    # Submit
    try:
        req = urllib.request.Request(f"{BASE_URL}/{endpoint}", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"MuAPI submit failed [{exc.code}]: {exc.read().decode(errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MuAPI submit connection error: {exc.reason}") from exc

    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"MuAPI did not return a request_id: {data}")

    # Poll
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        try:
            poll_req = urllib.request.Request(
                f"{BASE_URL}/predictions/{request_id}/result",
                headers={"x-api-key": api_key},
            )
            with urllib.request.urlopen(poll_req, timeout=15) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"MuAPI poll failed [{exc.code}]: {exc.read().decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MuAPI poll connection error: {exc.reason}") from exc

        status = result.get("status", "pending")
        if status == "completed":
            outputs = result.get("outputs", [])
            if not outputs:
                raise RuntimeError("Generation completed but returned no outputs")
            return outputs[0]
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Generation {status}: {result.get('error', '')}")

    raise TimeoutError(f"Generation timed out after {timeout}s")


class MuApiImageTool(BaseTool):
    """Generate images using muapi.ai — a unified API for 400+ image generation models."""

    name: str = "MuAPI Image Generator"
    description: str = (
        "Generates images from text prompts using muapi.ai. "
        "Supports Flux, Midjourney, GPT-4o Image, Google Imagen 4, "
        "Seedream, HiDream, Reve, Ideogram, and more. "
        "Returns the URL of the generated image."
    )
    args_schema: type[BaseModel] = MuApiImageSchema

    env_vars: List[EnvVar] = [
        EnvVar(
            name="MUAPI_API_KEY",
            description="API key for muapi.ai. Get one at https://muapi.ai/dashboard/api-keys",
            required=True,
        )
    ]

    def _run(self, **kwargs: Any) -> str:
        import os

        api_key = os.environ.get("MUAPI_API_KEY", "")
        if not api_key:
            return "Error: MUAPI_API_KEY environment variable not set."

        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", "flux-schnell")
        payload: dict = {"prompt": prompt}
        if kwargs.get("width"):
            payload["width"] = kwargs["width"]
        if kwargs.get("height"):
            payload["height"] = kwargs["height"]

        try:
            url = _submit_and_poll(api_key, model, payload)
            return json.dumps({"image_url": url, "model": model, "prompt": prompt})
        except Exception as e:
            return f"Error generating image: {e}"


class MuApiVideoTool(BaseTool):
    """Generate videos using muapi.ai — a unified API for 400+ video generation models."""

    name: str = "MuAPI Video Generator"
    description: str = (
        "Generates short videos from text prompts using muapi.ai. "
        "Supports Veo3, Kling, Wan, Seedance, Runway, Pixverse, Sora, and more. "
        "Returns the URL of the generated MP4 video."
    )
    args_schema: type[BaseModel] = MuApiVideoSchema

    env_vars: List[EnvVar] = [
        EnvVar(
            name="MUAPI_API_KEY",
            description="API key for muapi.ai. Get one at https://muapi.ai/dashboard/api-keys",
            required=True,
        )
    ]

    def _run(self, **kwargs: Any) -> str:
        import os

        api_key = os.environ.get("MUAPI_API_KEY", "")
        if not api_key:
            return "Error: MUAPI_API_KEY environment variable not set."

        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", "veo3-fast")
        payload: dict = {
            "prompt": prompt,
            "duration": kwargs.get("duration", 5),
            "aspect_ratio": kwargs.get("aspect_ratio", "16:9"),
        }

        try:
            url = _submit_and_poll(api_key, model, payload, timeout=600)
            return json.dumps({"video_url": url, "model": model, "prompt": prompt})
        except Exception as e:
            return f"Error generating video: {e}"
