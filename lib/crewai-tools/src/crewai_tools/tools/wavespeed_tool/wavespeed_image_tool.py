from __future__ import annotations

from builtins import type as type_
import json
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.wavespeed_tool.wavespeed_base_tool import WaveSpeedBaseTool


class WaveSpeedImageToolSchema(BaseModel):
    """Input for WaveSpeedImageTool."""

    prompt: str = Field(..., description="Description of the image to generate.")
    images: list[str] | None = Field(
        default=None,
        description=(
            "Optional URLs of reference images. When given, the image is "
            "produced by editing or combining these images according to the prompt."
        ),
    )


class WaveSpeedImageTool(WaveSpeedBaseTool):
    """Generate or edit images with a model hosted on WaveSpeed AI."""

    name: str = "WaveSpeed Image Generation"
    description: str = (
        "Generates an image from a text prompt using WaveSpeed AI. Optionally "
        "takes reference image URLs to edit or combine. Returns the URLs of "
        "the generated images."
    )
    args_schema: type_[BaseModel] = WaveSpeedImageToolSchema

    model: str = Field(
        default="bytedance/seedream-v5.5",
        description="WaveSpeed model id used for text-to-image generation.",
    )
    edit_model: str = Field(
        default="bytedance/seedream-v5.5/edit",
        description="WaveSpeed model id used when reference images are given.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description="Aspect ratio such as '1:1' or '16:9'. Uses the model default if unset.",
    )
    resolution: str | None = Field(
        default=None,
        description="Output resolution such as '1k' or '2k'. Uses the model default if unset.",
    )
    output_format: str | None = Field(
        default=None,
        description="Output format such as 'png' or 'jpeg'. Uses the model default if unset.",
    )

    def _run(self, prompt: str, images: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"prompt": prompt}
        if images:
            payload["images"] = images
        for key in ("aspect_ratio", "resolution", "output_format"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value

        model = self.edit_model if images else self.model
        prediction_id, outputs = self._generate(model, payload)
        return json.dumps(
            {"image_urls": outputs, "model": model, "prediction_id": prediction_id}
        )
