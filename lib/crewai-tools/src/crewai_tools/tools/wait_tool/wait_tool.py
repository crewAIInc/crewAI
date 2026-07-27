"""Tool that pauses execution for a given amount of time."""

import asyncio
import time
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


DEFAULT_MAX_SECONDS: float = 300.0


def _build_description(max_seconds: float) -> str:
    """Build the LLM-facing description, including the current per-call cap.

    Args:
        max_seconds: The per-call cap advertised to the model.

    Returns:
        The tool description shown to the model.
    """
    return (
        "Pause and do nothing for a set number of seconds before continuing.\n"
        "Use this when work that was already started somewhere else needs real time to "
        "finish before its status or result can be checked again, for example: a sandbox "
        "build, test run, or script that is still executing; a deployment or provisioning "
        "step that is still rolling out; a batch import, export, or training job; an async "
        "API that returned a job id to poll later; or a rate limit or backoff that has to "
        "cool down before retrying.\n"
        "The usual pattern is: start the job, wait, check its status, and wait again if it "
        "is still running.\n"
        "Do not use this to pace a conversation, to pretend to work, or when the "
        "information needed is already available. Waiting only lets clock time pass, it "
        "does not advance or check the job.\n"
        f"A single call waits at most {max_seconds:g} seconds. If more time is needed, "
        "call this tool again."
    )


class WaitToolSchema(BaseModel):
    """Input for WaitTool."""

    seconds: float = Field(
        ...,
        ge=0,
        description="How many seconds to wait before continuing.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional short note on what is being waited for, echoed back in the result.",
    )


class WaitTool(BaseTool):
    """Pause execution so a long-running job elsewhere has time to progress.

    Agents that kick off out-of-band work (a sandbox build, a deployment, an async
    API job) have no other way to let clock time pass: without this tool they either
    poll in a tight loop or give up before the work finishes.

    A single call waits at most ``max_seconds``. Longer requests are clamped to that
    cap and the result says so, so the model can call again instead of failing.
    """

    name: str = "Wait"
    description: str = _build_description(DEFAULT_MAX_SECONDS)
    args_schema: type[BaseModel] = WaitToolSchema
    max_seconds: float = Field(
        default=DEFAULT_MAX_SECONDS,
        gt=0,
        description="Upper bound, in seconds, for a single wait. Longer requests are capped to this value.",
    )

    def __init__(self, max_seconds: float | None = None, **kwargs: Any) -> None:
        if max_seconds is not None:
            kwargs["max_seconds"] = max_seconds
        super().__init__(**kwargs)
        if "description" not in kwargs:
            self.description = _build_description(self.max_seconds)

    def _resolve_duration(self, seconds: float) -> tuple[float, bool]:
        """Validate and clamp the requested duration to ``max_seconds``.

        ``BaseTool.run`` skips ``args_schema`` validation when called with
        positional arguments, so the non-negative bound is enforced here too
        rather than left to ``time.sleep`` to reject.

        Args:
            seconds: The requested wait duration.

        Returns:
            A tuple of the duration to actually wait and whether it was capped.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"seconds must be zero or greater, got {seconds:g}.")
        if seconds > self.max_seconds:
            return self.max_seconds, True
        return seconds, False

    def _format_result(
        self, waited: float, requested: float, reason: str | None
    ) -> str:
        """Describe the completed wait back to the model.

        Args:
            waited: The duration actually waited.
            requested: The duration the model asked for.
            reason: Optional note on what is being waited for.

        Returns:
            A summary of how long was waited and whether the request was capped.
        """
        parts = [f"Waited {waited:g} seconds."]
        if waited < requested:
            parts.append(
                f"Requested {requested:g} seconds, capped at {self.max_seconds:g} seconds per call - "
                "call this tool again if more waiting is needed."
            )
        if reason:
            parts.append(f"Reason: {reason}")
        return " ".join(parts)

    def _run(self, seconds: float, reason: str | None = None) -> str:
        """Block for ``seconds``, capped at ``max_seconds``.

        Args:
            seconds: How many seconds to wait.
            reason: Optional note on what is being waited for.

        Returns:
            A summary of how long was waited and whether the request was capped.
        """
        waited, _ = self._resolve_duration(seconds)
        time.sleep(waited)
        return self._format_result(waited, seconds, reason)

    async def _arun(self, seconds: float, reason: str | None = None) -> str:
        """Await for ``seconds``, capped at ``max_seconds``, without blocking the loop.

        Args:
            seconds: How many seconds to wait.
            reason: Optional note on what is being waited for.

        Returns:
            A summary of how long was waited and whether the request was capped.
        """
        waited, _ = self._resolve_duration(seconds)
        await asyncio.sleep(waited)
        return self._format_result(waited, seconds, reason)
