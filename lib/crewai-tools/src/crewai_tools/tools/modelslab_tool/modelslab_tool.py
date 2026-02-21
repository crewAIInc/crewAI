import json
import time
from typing import Optional

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

MODELSLAB_BASE_URL = "https://modelslab.com/api/v6"
MODELSLAB_POLLING_INTERVAL = 3  # seconds between polls
MODELSLAB_POLLING_TIMEOUT = 300  # 5 minute max


class ModelsLabImageSchema(BaseModel):
    """Input for ModelsLab Image Generation Tool."""

    image_description: str = Field(
        description="Text description of the image to generate."
    )


class ModelsLabImageGenerationTool(BaseTool):
    """Generates images using ModelsLab's text-to-image API.

    ModelsLab provides access to 200+ AI models including Flux, SDXL,
    and thousands of community fine-tunes via a unified API.

    Docs: https://docs.modelslab.com/image-generation/community-models/text2img
    """

    name: str = "ModelsLab Image Generation Tool"
    description: str = (
        "Generates images from text prompts using ModelsLab's AI API. "
        "Supports Flux, SDXL, Stable Diffusion, and 200+ community models. "
        "Returns a URL to the generated image."
    )
    args_schema: type[BaseModel] = ModelsLabImageSchema

    model: str = "flux"
    width: int = 512
    height: int = 512
    samples: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    api_base_url: str = MODELSLAB_BASE_URL

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MODELSLAB_API_KEY",
                description="API key for ModelsLab services. Get yours at https://modelslab.com/dashboard/api-keys",
                required=True,
            ),
        ]
    )

    def _poll_for_result(self, generation_id: int, api_key: str) -> list[str]:
        """Poll the fetch endpoint until image generation completes."""
        fetch_url = f"{self.api_base_url.rstrip('/')}/images/fetch/{generation_id}"
        start = time.time()

        while True:
            if time.time() - start > MODELSLAB_POLLING_TIMEOUT:
                raise TimeoutError(
                    f"ModelsLab image generation timed out after {MODELSLAB_POLLING_TIMEOUT}s "
                    f"(generation_id={generation_id})."
                )
            response = requests.post(
                fetch_url,
                json={"key": api_key},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            status = data.get("status", "")

            if status == "success":
                return data.get("output", [])
            elif status == "error":
                raise RuntimeError(
                    f"ModelsLab error: {data.get('message', 'unknown error')}"
                )
            elif status == "processing":
                time.sleep(MODELSLAB_POLLING_INTERVAL)
            else:
                raise RuntimeError(
                    f"ModelsLab: unexpected status '{status}' for generation {generation_id}"
                )

    def _run(self, **kwargs) -> str:
        import os

        image_description = kwargs.get("image_description", "")
        if not image_description:
            return "Image description is required."

        api_key = os.environ.get("MODELSLAB_API_KEY", "")
        if not api_key:
            return "MODELSLAB_API_KEY environment variable is not set."

        payload: dict = {
            "key": api_key,
            "prompt": image_description,
            "model_id": self.model,
            "width": self.width,
            "height": self.height,
            "samples": self.samples,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_checker": "no",
        }
        if self.negative_prompt:
            payload["negative_prompt"] = self.negative_prompt
        if self.seed is not None:
            payload["seed"] = self.seed

        text2img_url = f"{self.api_base_url.rstrip('/')}/images/text2img"
        try:
            response = requests.post(
                text2img_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return f"ModelsLab API request failed: {e}"

        status = data.get("status", "")

        if status == "error":
            return f"ModelsLab error: {data.get('message', 'unknown error')}"

        if status == "processing":
            generation_id = data.get("id")
            if not generation_id:
                return "ModelsLab returned 'processing' status without a generation ID."
            try:
                image_urls = self._poll_for_result(
                    generation_id=generation_id, api_key=api_key
                )
            except (TimeoutError, RuntimeError) as e:
                return f"ModelsLab polling failed: {e}"
        elif status == "success":
            image_urls = data.get("output", [])
        else:
            return f"ModelsLab: unexpected response status '{status}'."

        if not image_urls:
            return "ModelsLab returned no images."

        return json.dumps(
            {
                "image_url": image_urls[0],
                "all_image_urls": image_urls,
                "model": self.model,
                "prompt": image_description,
            }
        )
