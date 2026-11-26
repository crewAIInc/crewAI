"""Jev Choice decisions through the official TypeSafe Python SDK."""

from __future__ import annotations

from contextlib import suppress
import importlib
from typing import TYPE_CHECKING, Annotated, Any

from crewai.tools import BaseTool, EnvVar, ToolFailure
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, WithJsonSchema


if TYPE_CHECKING:
    from typesafe_sdk import SystemOneResponse, TypeSafeClient


_QUESTION_NAME = "decision"
# Any list is accepted. The schema types the items because strict function calling
# rejects an array whose items have no type.
_ITEM_TYPES = ("string", "number", "boolean", "object", "null")
_JsonArray = Annotated[
    list[Any],
    WithJsonSchema(
        {"type": "array", "items": {"anyOf": [{"type": t} for t in _ITEM_TYPES]}}
    ),
]


def _decision(
    response: SystemOneResponse, choices: dict[str, str | None]
) -> dict[str, Any] | ToolFailure:
    """Turn a System One response into the tool's result or a ``ToolFailure``."""
    from typesafe_sdk import TypeSafeError

    answer = response.choices.get(_QUESTION_NAME)
    if answer is None:
        return ToolFailure(
            message=f"TypeSafe returned no Choice answer named {_QUESTION_NAME!r}.",
            code="missing_choice_answer",
        )
    if answer.choice not in choices:
        return ToolFailure(
            message=(
                f"TypeSafe selected {answer.choice!r}, which is not one of "
                "the supplied choices."
            ),
            code="unknown_choice_label",
        )

    request_id: str | None = None
    with suppress(TypeSafeError):  # raised when the response has no request id
        request_id = response.request_id
    return {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
        "model": response.model,
        "usage": response.usage.model_dump(),
        "request_id": request_id,
    }


class JevDecisionToolSchema(BaseModel):
    """Input schema for JevDecisionTool."""

    model_config = ConfigDict(strict=True)  # reject values instead of coercing them

    state: str | dict[str, Any] | _JsonArray = Field(
        ..., description="Text, JSON object, or array for Jev to evaluate."
    )
    question: str = Field(
        ...,
        pattern=r"\S",
        description="The non-empty question or instruction to answer about the state.",
    )
    choices: dict[str, str | None] = Field(
        ...,
        description=(
            "Choice labels mapped to a description of when each applies, "
            "or null to let the label speak for itself."
        ),
    )


class JevDecisionResult(BaseModel):
    """Shape of a successful result; an agent receives it as JSON."""

    choice: str = Field(description="The selected label.")
    confidence: float = Field(description="TypeSafe's confidence in the selection.")
    probabilities: dict[str, float] = Field(description="Probability by label.")
    model: str = Field(description="The model that answered.")
    usage: dict[str, int | None] = Field(description="Token counts.")
    request_id: str | None = Field(default=None, description="The TypeSafe request id.")


class JevDecisionTool(BaseTool):
    """Answer one Choice question with TypeSafe's Jev model.

    Credentials and endpoint come from the SDK's own configuration
    (``TYPESAFE_API_KEY``, ``TYPESAFE_BASE_URL``), or from ``client``.
    """

    name: str = "Jev Decision Tool"
    description: str = (
        "Asks TypeSafe's Jev model one Choice question about a state and returns "
        "the selected label, its confidence, and the probability of each label."
    )
    args_schema: type[BaseModel] = JevDecisionToolSchema
    result_schema: type[BaseModel] | None = JevDecisionResult
    model: str | None = Field(
        default=None, description="Jev model name. None uses the SDK default."
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["typesafe-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TYPESAFE_API_KEY",
                description="API key for the TypeSafe API",
                required=True,
            ),
        ]
    )
    _client: TypeSafeClient | None = PrivateAttr(default=None)

    def __init__(self, *, client: TypeSafeClient | None = None, **kwargs: Any) -> None:
        """Create the tool.

        Args:
            client: A configured SDK client to reuse. The caller keeps ownership
                and closes it. When omitted, each call creates and closes its own.
            **kwargs: Tool fields, such as ``model``.

        Raises:
            ImportError: If the optional ``typesafe-sdk`` package is not installed.
        """
        super().__init__(**kwargs)
        try:
            importlib.import_module("typesafe_sdk")
        except ImportError as error:
            raise ImportError(
                "The 'typesafe-sdk' package is required to use JevDecisionTool. "
                "Install it with: uv add 'crewai-tools[typesafe-sdk]'"
            ) from error
        self._client = client

    def _run(
        self,
        state: str | dict[str, Any] | list[Any],
        question: str,
        choices: dict[str, str | None],
    ) -> dict[str, Any] | ToolFailure:
        """Return a ``JevDecisionResult`` as a dict, or a ``ToolFailure``."""
        from typesafe_sdk import Choice, TypeSafeClient

        try:
            ask = Choice(instructions=question, criteria=choices)
            questions = {_QUESTION_NAME: ask}
            if self._client is not None:
                response = self._client.system_one(state, questions, model=self.model)
            else:
                with TypeSafeClient() as client:
                    response = client.system_one(state, questions, model=self.model)
        except Exception as error:  # the SDK lets some errors through unwrapped
            return ToolFailure(
                message=f"Jev decision failed: {error}", code=type(error).__name__
            )
        return _decision(response, choices)
