from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class JevDecisionToolSchema(BaseModel):
    """Input schema for JevDecisionTool."""

    state: Any = Field(
        ...,
        description="Text, JSON object, or array for Jev to evaluate.",
    )
    question: str = Field(
        ...,
        description="Choice question or instruction to evaluate against the state.",
    )
    choices: dict[str, Any | None] = Field(
        ...,
        description=(
            "Mapping of choice labels to descriptions. "
            "At least two labels are required."
        ),
    )

    @model_validator(mode="after")
    def validate_input(self) -> JevDecisionToolSchema:
        if self.state is None:
            raise ValueError("state cannot be None")
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if len(self.choices) < 2:
            raise ValueError("choices must contain at least two labels")
        if any(not str(label).strip() for label in self.choices):
            raise ValueError("choice labels cannot be empty")
        return self


class JevDecisionTool(BaseTool):
    """Make one Jev Choice decision synchronously or asynchronously.

    Use ``await tool.arun(state=..., question=..., choices=...)`` for native
    async execution. Pass ``async_client=`` to reuse a caller-owned SDK client;
    ``client=`` is used only for synchronous calls. Injected clients are never
    closed by this tool. Otherwise, each async call creates and closes its own
    client, including on failure or cancellation. SDK configuration and retries
    remain controlled by the SDK or the injected client.
    """

    name: str = "Jev Decision Tool"
    description: str = (
        "Evaluates one TypeSafe Jev Choice question and returns the selected label, "
        "probabilities, confidence, and request metadata when available."
    )
    args_schema: type[BaseModel] = JevDecisionToolSchema

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
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        *,
        async_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._async_client = async_client
        self.model = model

    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as e:
            raise ImportError(
                "The 'typesafe-sdk' package is required to use JevDecisionTool. "
                "Install it with: uv add --optional typesafe-sdk typesafe-sdk"
            ) from e
        return TypeSafeClient(model=self.model)

    def _build_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as e:
            raise ImportError(
                "The 'typesafe-sdk' package is required to use JevDecisionTool. "
                "Install it with: uv add --optional typesafe-sdk typesafe-sdk"
            ) from e
        return AsyncTypeSafeClient(model=self.model)

    @staticmethod
    def _make_question(
        validated: JevDecisionToolSchema, *, injected_client: bool
    ) -> Any:
        try:
            from typesafe_sdk import Choice
        except ImportError as e:
            if not injected_client:
                raise ImportError(
                    "The 'typesafe-sdk' package is required to use JevDecisionTool. "
                    "Install it with: uv add --optional typesafe-sdk typesafe-sdk"
                ) from e
            return {
                "type": "choice",
                "instructions": validated.question,
                "criteria": validated.choices,
            }
        return Choice(instructions=validated.question, criteria=validated.choices)

    @staticmethod
    def _format_response(
        response: Any, choices: Mapping[str, Any | None]
    ) -> dict[str, Any]:
        answer = getattr(response, "choices", {}).get("decision")
        if answer is None:
            return {
                "success": False,
                "error": (
                    "TypeSafe response did not include a Choice answer named "
                    "'decision'."
                ),
            }

        label = getattr(answer, "choice", None)
        probabilities = getattr(answer, "probabilities", None)
        if label is None:
            return {
                "success": False,
                "error": "Choice answer did not include a selected label.",
            }
        if label not in choices:
            return {
                "success": False,
                "error": f"Choice answer selected unknown label: {label}",
            }

        confidence = getattr(answer, "confidence", None)
        result: dict[str, Any] = {
            "success": True,
            "choice": label,
            "probabilities": dict(probabilities or {}),
            "confidence": confidence,
        }
        if response_model := getattr(response, "model", None):
            result["model"] = response_model
        if usage := getattr(response, "usage", None):
            model_dump = getattr(usage, "model_dump", None)
            result["usage"] = model_dump() if model_dump else usage
        return result

    def _run(
        self,
        state: Any,
        question: str,
        choices: Mapping[str, Any | None],
    ) -> dict[str, Any]:
        try:
            validated = JevDecisionToolSchema(
                state=state, question=question, choices=dict(choices)
            )
            choice_question = self._make_question(
                validated, injected_client=self._client is not None
            )
            client = self._build_client()
            created_client = self._client is None
            try:
                response = client.system_one(
                    state=validated.state,
                    questions={"decision": choice_question},
                    model=self.model,
                )
            finally:
                if created_client and hasattr(client, "close"):
                    client.close()
            return self._format_response(response, validated.choices)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _arun(
        self,
        state: Any,
        question: str,
        choices: Mapping[str, Any | None],
    ) -> dict[str, Any]:
        """Await the SDK and close only owned clients; cancellation propagates."""
        try:
            validated = JevDecisionToolSchema(
                state=state, question=question, choices=dict(choices)
            )
            choice_question = self._make_question(
                validated, injected_client=self._async_client is not None
            )
            client = self._build_async_client()
            try:
                response = await client.system_one(
                    state=validated.state,
                    questions={"decision": choice_question},
                    model=self.model,
                )
            finally:
                if self._async_client is None:
                    await client.aclose()
            return self._format_response(response, validated.choices)
        except Exception as e:
            return {"success": False, "error": str(e)}
