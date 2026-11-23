from __future__ import annotations

from builtins import type as type_
import json
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.wavespeed_tool.wavespeed_base_tool import WaveSpeedBaseTool


class WaveSpeedVideoToolSchema(BaseModel):
    """Input for WaveSpeedVideoTool."""

    prompt: str = Field(..., description="Description of the video to generate.")
    image: str | None = Field(
        default=None,
        description=(
            "Optional URL of an image to animate. When given, it is used as "
            "the first frame of the video."
        ),
    )


class WaveSpeedVideoTool(WaveSpeedBaseTool):
    """Generate videos with a model hosted on WaveSpeed AI."""

    name: str = "WaveSpeed Video Generation"
    description: str = (
        "Generates a video from a text prompt using WaveSpeed AI, optionally "
        "animating a starting image given by URL. Returns the URL of the "
        "generated video."
    )
    args_schema: type_[BaseModel] = WaveSpeedVideoToolSchema

    model: str = Field(
        default="alibaba/wan-3.0/text-to-video",
        description="WaveSpeed model id used for text-to-video generation.",
    )
    image_model: str = Field(
        default="alibaba/wan-3.0/image-to-video",
        description="WaveSpeed model id used when a starting image is given.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description="Aspect ratio such as '16:9' or '9:16'. Uses the model default if unset.",
    )
    duration: int | None = Field(
        default=None,
        description="Video length in seconds. Uses the model default if unset.",
    )
    resolution: str | None = Field(
        default=None,
        description="Output resolution such as '720p'. Uses the model default if unset.",
    )
    timeout: float = Field(
        default=900.0,
        description="Maximum seconds to wait for a prediction to finish.",
    )

    def _run(self, prompt: str, image: str | None = None) -> str:
        payload: dict[str, Any] = {"prompt": prompt}
        if image:
            payload["image"] = image
        for key in ("aspect_ratio", "duration", "resolution"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value

        model = self.image_model if image else self.model
        prediction_id, outputs = self._generate(model, payload)
        return json.dumps(
            {"video_url": outputs[0], "model": model, "prediction_id": prediction_id}
        )
