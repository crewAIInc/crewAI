from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr, model_validator


_MISSING_SDK_MESSAGE = (
    "The 'typesafe-sdk' package is required to use JevScoreTool. "
    "Install it with: uv add 'crewai-tools[typesafe-sdk]'"
)


class JevScoreToolSchema(BaseModel):
    """Input schema for JevScoreTool."""

    state: Any = Field(
        ...,
        description="Text, JSON object, or array for Jev to evaluate.",
    )
    question: str = Field(
        ...,
        description="Score question or instruction to rate the state against.",
    )
    criteria: list[Any] = Field(
        ...,
        description=(
            "Ordered descriptions of the score levels, lowest first. "
            "Each level's position is its score, starting at 0."
        ),
    )

    @model_validator(mode="after")
    def validate_input(self) -> JevScoreToolSchema:
        if self.state is None:
            raise ValueError("state cannot be None")
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.criteria:
            raise ValueError("criteria must contain at least one level")
        if any(
            level is None or (isinstance(level, str) and not level.strip())
            for level in self.criteria
        ):
            raise ValueError("criteria levels cannot be empty")
        return self


class JevScoreTool(BaseTool):
    """Tool for rating state on one Jev Score rubric with TypeSafe System One.

    The returned ``score`` is the SDK's expected score: the probability-weighted
    average of the rubric levels, which may fall between levels. ``success``
    only reports whether a usable answer came back; it is not a pass/fail
    verdict on the score.
    """

    name: str = "Jev Score Tool"
    description: str = (
        "Rates state against one TypeSafe Jev Score rubric and returns the "
        "expected score, the probability of each level, confidence, and request "
        "metadata when available."
    )
    args_schema: type[BaseModel] = JevScoreToolSchema

    model: str | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["typesafe-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TYPESAFE_API_KEY",
                description="API key for the TypeSafe System One API",
                required=True,
            ),
            EnvVar(
                name="TYPESAFE_BASE_URL",
                description="Optional TypeSafe API base URL override",
                required=False,
            ),
            EnvVar(
                name="TYPESAFE_DEFAULT_MODEL",
                description="Optional default Jev model override",
                required=False,
            ),
        ]
    )
    _client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client
        self.model = model

    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as e:
            raise ImportError(_MISSING_SDK_MESSAGE) from e
        return TypeSafeClient(model=self.model)

    def _run(
        self,
        state: Any,
        question: str,
        criteria: Sequence[Any],
    ) -> dict[str, Any]:
        try:
            validated = JevScoreToolSchema(
                state=state,
                question=question,
                criteria=list(criteria),
            )
            try:
                from typesafe_sdk import Score
            except ImportError as e:
                raise ImportError(_MISSING_SDK_MESSAGE) from e

            score_question = Score(
                instructions=validated.question,
                criteria=validated.criteria,
            )

            client = self._build_client()
            created_client = self._client is None
            try:
                response = client.system_one(
                    state=validated.state,
                    questions={"score": score_question},
                )
            finally:
                if created_client and hasattr(client, "close"):
                    client.close()

            return _score_result(response)
        except Exception as e:
            return {"success": False, "error": str(e)}


def _score_result(response: Any) -> dict[str, Any]:
    """Convert a System One response into the tool's structured Score result."""
    answer = (getattr(response, "scores", None) or {}).get("score")
    if answer is None:
        return {
            "success": False,
            "error": "TypeSafe response did not include a Score answer named 'score'.",
        }

    score = getattr(answer, "score", None)
    if score is None:
        return {"success": False, "error": "Score answer did not include a score."}

    result: dict[str, Any] = {
        "success": True,
        "score": score,
        "probabilities": dict(getattr(answer, "probabilities", None) or {}),
        "confidence": getattr(answer, "confidence", None),
        "legend": dict(getattr(answer, "legend", None) or {}),
    }
    if response_model := getattr(response, "model", None):
        result["model"] = response_model
    if usage := getattr(response, "usage", None):
        model_dump = getattr(usage, "model_dump", None)
        result["usage"] = model_dump() if model_dump else usage
    return result
