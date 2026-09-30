"""Tool that pauses execution for a given amount of time."""

import asyncio
import math
import time
from typing import Any

from crewai.tools import BaseTool
from crewai.types.callback import SerializableCallable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


DEFAULT_MAX_SECONDS: float = 300.0

_DESCRIPTION_TEMPLATE = (
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
    "A single call waits at most {cap}. If more time is needed, "
    "call this tool again."
)

# Everything before the cap figure; used to tell a generated description from one
# the caller wrote, so only generated text is kept in sync with ``max_seconds``.
_GENERATED_DESCRIPTION_PREFIX = _DESCRIPTION_TEMPLATE.split("{cap}")[0]


def _format_seconds(value: float) -> str:
    """Render a duration with a correctly pluralized unit.

    Args:
        value: The duration in seconds.

    Returns:
        The duration and its unit, e.g. ``"1 second"`` or ``"300 seconds"``.
    """
    return f"{value:g} second" if value == 1 else f"{value:g} seconds"


def _build_description(max_seconds: float) -> str:
    """Build the LLM-facing description, including the current per-call cap.

    Args:
        max_seconds: The per-call cap advertised to the model.

    Returns:
        The tool description shown to the model.
    """
    return _DESCRIPTION_TEMPLATE.format(cap=_format_seconds(max_seconds))


def _is_generated_description(description: str) -> bool:
    """Report whether a description came from :func:`_build_description`.

    Args:
        description: The description to inspect.

    Returns:
        True if the text was generated rather than authored by the caller.
    """
    return description.startswith(_GENERATED_DESCRIPTION_PREFIX)


def _never_cache(_args: Any = None, _result: Any = None) -> bool:
    """Refuse caching of wait results.

    The value of a wait is the time that passed, not the message returned. A
    cached hit would hand back "Waited 300 seconds." without waiting, silently
    turning a poll-wait-check loop into a busy loop.

    Args:
        _args: Unused, part of the ``cache_function`` signature.
        _result: Unused, part of the ``cache_function`` signature.

    Returns:
        Always False.
    """
    return False


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

    The generated description advertises the cap to the model, and is kept in sync
    with ``max_seconds``. A description supplied by the caller is left alone.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = "Wait"
    description: str = _build_description(DEFAULT_MAX_SECONDS)
    args_schema: type[BaseModel] = WaitToolSchema
    max_seconds: float = Field(
        default=DEFAULT_MAX_SECONDS,
        gt=0,
        description="Upper bound, in seconds, for a single wait. Longer requests are capped to this value.",
    )
    cache_function: SerializableCallable = Field(
        default=_never_cache,
        description="Waits are never cached: a cache hit would return the message without any time passing.",
    )

    def __init__(self, max_seconds: float | None = None, **kwargs: Any) -> None:
        if max_seconds is not None:
            kwargs["max_seconds"] = max_seconds
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def _sync_description_with_cap(self) -> Self:
        """Keep a generated description's advertised cap equal to ``max_seconds``.

        Runs on construction, ``model_validate``, and assignment, so the cap the
        model is told about cannot drift from the one enforced. ``model_copy``
        skips validation, as it does for any pydantic model, so prefer
        construction or assignment when changing the cap.

        Returns:
            The validated tool.
        """
        expected = _build_description(self.max_seconds)
        if self.description != expected and _is_generated_description(self.description):
            # Bypass the field setter: assigning here would re-enter validation.
            self.__dict__["description"] = expected
        return self

    def _resolve_duration(self, seconds: float) -> tuple[float, bool]:
        """Validate and clamp the requested duration to ``max_seconds``.

        ``BaseTool.run`` skips ``args_schema`` validation when called with
        positional arguments, so the bounds are enforced here too rather than
        left to ``time.sleep`` to reject. Infinity is a valid request: it clamps
        to the cap like any other oversized wait.

        Args:
            seconds: The requested wait duration.

        Returns:
            A tuple of the duration to actually wait and whether it was capped.

        Raises:
            ValueError: If ``seconds`` is negative or not a number.
        """
        if math.isnan(seconds):
            raise ValueError("seconds must be a number, got NaN.")
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
        parts = [f"Waited {_format_seconds(waited)}."]
        if waited < requested:
            parts.append(
                f"Requested {_format_seconds(requested)}, capped at "
                f"{_format_seconds(self.max_seconds)} per call - "
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
