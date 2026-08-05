from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


class TwelveLabsAnalyzeToolSchema(BaseModel):
    """Input schema for the TwelveLabs video-analysis tool."""

    prompt: str = Field(
        ...,
        description=(
            "Instruction describing what to extract from the video, e.g. "
            "'Summarize this video' or 'List every product shown'."
        ),
    )
    video_url: str | None = Field(
        default=None,
        description=(
            "Publicly accessible URL of the video to analyze. Provide this or "
            "video_id (one is required)."
        ),
    )
    video_id: str | None = Field(
        default=None,
        description=(
            "ID of a video already indexed in TwelveLabs. Provide this or "
            "video_url (one is required)."
        ),
    )


class TwelveLabsAnalyzeTool(BaseTool):
    """Analyze video content using TwelveLabs' Pegasus video-understanding model.

    Given a video (either a public URL or an already-indexed ``video_id``) and a
    natural-language prompt, the tool returns a text answer generated from the
    video's visuals, speech, and on-screen text. Useful for summarization,
    question answering over video, content moderation, and metadata extraction.

    Get a free API key at https://twelvelabs.io and set ``TWELVELABS_API_KEY``.

    Attributes:
        model_name: TwelveLabs Pegasus model to use (default ``pegasus1.5``).
        max_tokens: Maximum number of tokens in the generated answer. Must be at
            least 512 for the Pegasus model.
        temperature: Sampling temperature for generation.
        api_key: TwelveLabs API key. Falls back to ``TWELVELABS_API_KEY``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "TwelveLabs Video Analysis"
    description: str = (
        "Analyze a video with TwelveLabs Pegasus and answer a natural-language "
        "prompt about its content (summaries, Q&A, moderation, metadata). "
        "Accepts a public video URL or an indexed TwelveLabs video_id."
    )
    args_schema: type[BaseModel] = TwelveLabsAnalyzeToolSchema

    model_name: str = "pegasus1.5"
    max_tokens: int = 2048
    temperature: float | None = None
    api_key: str | None = None
    _client: Any = None

    package_dependencies: list[str] = Field(default_factory=lambda: ["twelvelabs"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TWELVELABS_API_KEY",
                description="API key for TwelveLabs (https://twelvelabs.io)",
                required=False,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from twelvelabs import TwelveLabs
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'twelvelabs' package. Would you like to "
                "install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "twelvelabs"], check=True)  # noqa: S607
                from twelvelabs import TwelveLabs
            else:
                raise ImportError(
                    "`twelvelabs` package not found, please run `uv add twelvelabs`"
                ) from None

        self.api_key = api_key or os.getenv("TWELVELABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TwelveLabs API key is required. Set TWELVELABS_API_KEY or pass "
                "api_key. Get a free key at https://twelvelabs.io."
            )
        self._client = TwelveLabs(api_key=self.api_key)

    def _run(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt")
        video_url = kwargs.get("video_url")
        video_id = kwargs.get("video_id")

        if not prompt:
            return "A prompt is required."
        if not video_url and not video_id:
            return "Either video_url or video_id is required."

        if self._client is None:
            raise RuntimeError("TwelveLabs client not initialized")

        analyze_kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            analyze_kwargs["temperature"] = self.temperature

        if video_id:
            analyze_kwargs["video_id"] = video_id
        else:
            from twelvelabs.types.video_context import VideoContext_Url

            analyze_kwargs["video"] = VideoContext_Url(url=video_url)

        try:
            response = self._client.analyze(**analyze_kwargs)
        except Exception as e:
            raise RuntimeError(f"TwelveLabs analysis failed: {e!s}") from e

        return response.data or ""
