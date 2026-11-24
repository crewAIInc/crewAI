"""Treza tools: AI pictures and video clips (Veo, Kling, Seedance and more).

Each render tool asks Treza for a price, stops when the price is over
``max_cost_usd``, starts the render once and waits for the finished file.
The API key, ``max_cost_usd`` and the wait are set when the tool is built, not
by the model, so an agent cannot raise a spending cap you set.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from crewai_tools.tools.treza_tool.treza_client import (
    KEYS_PAGE,
    Data,
    TrezaClient,
    TrezaError,
    compact,
    finished_or_raise,
    run_item,
    to_number,
)


ImageShape = Literal["1:1", "16:9", "9:16", "4:5", "3:4", "4:3", "2:3", "3:2"]
VideoShape = Literal["16:9", "9:16", "1:1"]


def _treza_env_vars() -> list[EnvVar]:
    return [
        EnvVar(
            name="TREZA_API_KEY",
            description=f"Treza API key with the pipelines read, write and run permissions ({KEYS_PAGE})",
            required=True,
        )
    ]


class _TrezaBaseTool(BaseTool):
    """What every Treza tool shares: the key, the cap, the wait and how a render is reported."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Treza API key. Defaults to the TREZA_API_KEY environment variable.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        description=(
            "The most one render may cost, in US dollars. Treza prices the render "
            "first and spends nothing when it is higher. None means no cap."
        ),
    )
    wait_seconds: float = Field(
        default=300,
        description=(
            "How long to wait for a render. One still going comes back with "
            'status "running" and the IDs to look it up later.'
        ),
    )
    env_vars: list[EnvVar] = Field(default_factory=_treza_env_vars)
    _client: TrezaClient | None = PrivateAttr(default=None)

    def _treza(self) -> TrezaClient:
        if self._client is None:
            key = self.api_key.get_secret_value() if self.api_key else ""
            key = key or (os.environ.get("TREZA_API_KEY") or "").strip()
            if not key:
                raise TrezaError(
                    f"Set TREZA_API_KEY or pass api_key. Make a key at {KEYS_PAGE}",
                    401,
                    "MissingKey",
                )
            self._client = TrezaClient(key)
        return self._client

    def _cap(self) -> float:
        cap = to_number(self.max_cost_usd)
        return cap if cap is not None and cap > 0 else float("inf")

    def _render(self, tool: str, args: Data, output_key: str) -> str:
        treza = self._treza()
        cap = self._cap()
        args = compact({**args, "maxPriceUsd": cap if cap != float("inf") else None})
        started = treza.start_quoted(cap, lambda: treza.request(tool, args))
        run = treza.wait_for_run(
            started["pipelineId"], started["runId"], self.wait_seconds
        )
        run = {"runId": started["runId"], "pipelineId": started["pipelineId"], **run}
        finished_or_raise(run)
        model = (started.get("quoted") or {}).get("model")
        extra = {
            "price_usd": started.get("priceUsd"),
            "model": model.get("id") if isinstance(model, dict) else model,
        }
        return json.dumps(run_item(run, output_key, extra))

    @staticmethod
    def _guard(fn: Callable[[], str]) -> str:
        """Hand a refusal (over the cap, low balance, a bad key) back to the agent as text."""
        try:
            return fn()
        except TrezaError as err:
            return f"Error: {err}"


class TrezaImageToolSchema(BaseModel):
    """Input for TrezaImageTool."""

    prompt: str = Field(..., description="What the picture shows.")
    aspect_ratio: ImageShape = Field(
        default="1:1",
        description='The picture\'s shape, for example "16:9" wide or "9:16" tall.',
    )
    model: str | None = Field(
        default=None, description="A Treza model ID. Leave out for Treza's default."
    )


class TrezaImageTool(_TrezaBaseTool):
    """Make a picture from a description with Treza."""

    name: str = "Treza Image Generation"
    description: str = (
        "Make a picture from a text description with Treza. Returns JSON with the "
        "finished picture's url, the price in US dollars and the model. Uses the "
        "account's prepaid Treza credits."
    )
    args_schema: type[BaseModel] = TrezaImageToolSchema

    def _run(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        model: str | None = None,
        **_: Any,
    ) -> str:
        return self._guard(
            lambda: self._render(
                "make_image",
                {"prompt": prompt, "aspectRatio": aspect_ratio, "model": model},
                "image",
            )
        )


class TrezaVideoToolSchema(BaseModel):
    """Input for TrezaVideoTool."""

    prompt: str = Field(..., description="What happens in the clip.")
    aspect_ratio: VideoShape = Field(
        default="16:9",
        description='"16:9" wide, "9:16" vertical (Shorts, Reels, TikTok) or "1:1" square.',
    )
    duration_sec: float | None = Field(
        default=None,
        ge=1,
        le=60,
        description="Clip length in seconds, usually 4 to 10. Defaults to 4.",
    )
    with_audio: bool | None = Field(
        default=None,
        description="Generate sound with the clip where the model can. Defaults to true.",
    )
    model: str | None = Field(
        default=None, description="A Treza model ID. Leave out for Treza's default."
    )


class TrezaVideoTool(_TrezaBaseTool):
    """Make a video clip from a description with Treza."""

    name: str = "Treza Video Generation"
    description: str = (
        "Make a short video clip from a text description with Treza. Takes one to "
        "five minutes. Returns JSON with the finished clip's url, the price in US "
        'dollars and the model, or status "running" with a run_id and pipeline_id '
        "to pass to Treza Get Run when it is not done in time. Uses the account's "
        "prepaid Treza credits."
    )
    args_schema: type[BaseModel] = TrezaVideoToolSchema

    def _run(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        duration_sec: float | None = None,
        with_audio: bool | None = None,
        model: str | None = None,
        **_: Any,
    ) -> str:
        return self._guard(
            lambda: self._render(
                "make_video",
                {
                    "prompt": prompt,
                    "aspectRatio": aspect_ratio,
                    "durationSec": to_number(duration_sec),
                    "withAudio": with_audio,
                    "model": model,
                },
                "video",
            )
        )


class TrezaGetRunToolSchema(BaseModel):
    """Input for TrezaGetRunTool."""

    pipeline_id: str = Field(..., description="The pipeline_id a Treza tool returned.")
    run_id: str = Field(..., description="The run_id a Treza tool returned.")


class TrezaGetRunTool(_TrezaBaseTool):
    """Look up a render that was still going when its tool stopped waiting."""

    name: str = "Treza Get Run"
    description: str = (
        "Check a Treza render by its pipeline_id and run_id. Returns JSON with its "
        "status and, once finished, the url of the file. Free."
    )
    args_schema: type[BaseModel] = TrezaGetRunToolSchema

    def _run(self, pipeline_id: str, run_id: str, **_: Any) -> str:
        def look() -> str:
            run = self._treza().call(
                "get_run", {"pipelineId": pipeline_id, "runId": run_id}
            )
            run = {"runId": run_id, "pipelineId": pipeline_id, **run}
            finished_or_raise(run)
            return json.dumps(run_item(run))

        return self._guard(look)
