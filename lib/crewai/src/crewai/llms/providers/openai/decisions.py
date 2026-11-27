"""Typed predicate client for the OpenAI Decisions API.

``POST /v1/decisions`` goes through ``OpenAI.post("/decisions")`` and
``AsyncOpenAI.post("/decisions")``. CrewAI depends on ``openai>=2.30.0,<3``
(locked at 2.41.0). That SDK has those methods and no ``client.decisions``
resource. The native resource ships in Python SDK 3.26.0, which this tree
does not allow.

``OpenAIDecisions.predicate`` is the sync ``PredicateDecider`` method proposed
in crewAI pull request 8167: ``predicate(input, *, name, instructions)``
returns a frozen answer or ``None``. ``None`` means no predicate answer was
produced for that question (the name is missing, the answer is a refusal, or
the answer is another question type). A predicate whose probability is
missing, non-numeric, boolean, non-finite, or outside ``[0, 1]`` raises
``DecisionResponseError``. OpenAI API and transport exceptions propagate.

PR 8167 defines its own frozen ``PredicateAnswer`` on
``crewai.tasks.decision_guardrail``. That module is not merged, so this
adapter does not import it. Both models expose ``name: str`` and
``probability: float`` in ``[0, 1]``, and the guardrail reads those attributes
at runtime, so this client works with ``DecisionGuardrail`` without a shared
import. mypy will not treat ``OpenAIDecisions`` as that protocol until both
changes share one ``PredicateAnswer`` class. The protocol's return type is the
guardrail's class, and a different class is not a subtype of it.
``apredicate`` is not part of the protocol: task guardrails call ``predicate``
synchronously, including from ``Task.aexecute_sync``.
"""

from __future__ import annotations

import math
import os
from typing import Any, TypeGuard

from openai import AsyncOpenAI, OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


DEFAULT_DECISIONS_MODEL: str = "gpt-6-luna"
_DECISIONS_PATH: str = "/decisions"


class DecisionResponseError(ValueError):
    """The Decisions payload was not a usable predicate answer."""


class PredicateAnswer(BaseModel):
    """Typed answer to one predicate question.

    Attributes:
        name: Question name echoed by the decider.
        probability: Estimated probability, from 0 to 1, that the predicate
            holds.

    This matches the ``PredicateAnswer`` shape in PR 8167. It lives here so
    the adapter does not import that unmerged module.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    probability: float = Field(ge=0.0, le=1.0)

    @field_validator("probability", mode="before")
    @classmethod
    def _validate_probability(cls, value: Any) -> float:
        """Reject bools and every value that is not a finite unit number."""
        candidate: object = value
        if not _is_unit_probability(candidate):
            raise ValueError("probability must be a finite number between 0 and 1")
        return float(candidate)


class OpenAIDecisions:
    """Ask one predicate question with ``POST /decisions``.

    The default model is ``gpt-6-luna``, the only model the Decisions API
    documents. Inject ``client`` and ``async_client`` in tests. This adapter
    does not close injected clients. Built clients use the same key and base
    URL as ``OpenAICompletion``, without subclassing it.

    Args:
        client: Sync client. Built on the first ``predicate`` call when omitted.
        async_client: Async client. Built on the first ``apredicate`` call
            when omitted.
        model: Decisions model id.
        api_key: API key. Falls back to ``OPENAI_API_KEY``.
        base_url: API base URL. Falls back to ``OPENAI_BASE_URL``, then
            ``OPENAI_API_BASE``. The SDK base URL already includes ``/v1``.
        organization: Optional OpenAI organization.
        project: Optional OpenAI project.
        timeout: Request timeout in seconds.
        max_retries: SDK retry count. Omit it to keep the SDK default of 2.
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        self.model = model
        self._client = client
        self._async_client = async_client
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._project = project
        self._timeout = timeout
        self._max_retries = max_retries

    def predicate(
        self, input: str, *, name: str, instructions: str
    ) -> PredicateAnswer | None:
        """Ask one predicate question about ``input``.

        Args:
            input: Text the question evaluates.
            name: Question name sent to the API and required on the answer.
            instructions: Condition to evaluate.

        Returns:
            The predicate answer, or None when that question is missing,
            refused, or answered as a different question type.

        Raises:
            ValueError: ``input``, ``name``, or ``instructions`` is blank, or
                a client has to be built and ``OPENAI_API_KEY`` is missing.
            DecisionResponseError: The body is malformed, or the predicate
                probability is missing or not a finite number in ``[0, 1]``.
            openai.APIStatusError: The API rejected the request.
            openai.APITimeoutError: The request timed out.
            openai.APIConnectionError: The connection failed.
        """
        body, question_name = self._question(
            input,
            name=name,
            instructions=instructions,
        )
        payload = self._get_sync_client().post(
            _DECISIONS_PATH,
            body=body,
            cast_to=object,
        )
        return _parse_predicate_answer(payload, name=question_name)

    async def apredicate(
        self, input: str, *, name: str, instructions: str
    ) -> PredicateAnswer | None:
        """Ask one predicate question about ``input`` asynchronously.

        This is the async form of :meth:`predicate`. It is not part of the
        ``PredicateDecider`` protocol.

        Args:
            input: Text the question evaluates.
            name: Question name sent to the API and required on the answer.
            instructions: Condition to evaluate.

        Returns:
            The predicate answer, or None when that question is missing,
            refused, or answered as a different question type.

        Raises:
            ValueError: ``input``, ``name``, or ``instructions`` is blank, or
                a client has to be built and ``OPENAI_API_KEY`` is missing.
            DecisionResponseError: The body is malformed, or the predicate
                probability is missing or not a finite number in ``[0, 1]``.
            openai.APIStatusError: The API rejected the request.
            openai.APITimeoutError: The request timed out.
            openai.APIConnectionError: The connection failed.
        """
        body, question_name = self._question(
            input,
            name=name,
            instructions=instructions,
        )
        payload = await self._get_async_client().post(
            _DECISIONS_PATH,
            body=body,
            cast_to=object,
        )
        return _parse_predicate_answer(payload, name=question_name)

    def _question(
        self, input: str, *, name: str, instructions: str
    ) -> tuple[dict[str, Any], str]:
        """Validate one predicate question and build its request body."""
        text = _require_text(input, "input")
        question_name = _require_text(name, "name")
        question = _require_text(instructions, "instructions")
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "questions": [
                {
                    "type": "predicate",
                    "name": question_name,
                    "instructions": question,
                }
            ],
        }
        return body, question_name

    def _get_sync_client(self) -> OpenAI:
        """Return the injected sync client, or build one on first use."""
        if self._client is None:
            self._client = OpenAI(**self._client_params())
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        """Return the injected async client, or build one on first use."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_params())
        return self._async_client

    def _client_params(self) -> dict[str, Any]:
        """Build OpenAI client kwargs, omitting anything the SDK can default."""
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        params: dict[str, Any] = {"api_key": api_key}
        base_url = (
            self._base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        if base_url:
            params["base_url"] = base_url
        if self._organization is not None:
            params["organization"] = self._organization
        if self._project is not None:
            params["project"] = self._project
        if self._timeout is not None:
            params["timeout"] = self._timeout
        if self._max_retries is not None:
            params["max_retries"] = self._max_retries
        return params


def _require_text(value: object, label: str) -> str:
    """Return ``value`` when it is a non-blank string."""
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{label} must be a non-empty string")


def _is_unit_probability(value: object) -> TypeGuard[int | float]:
    """Whether ``value`` is a real, finite number in ``[0, 1]``.

    Bools are rejected. ``bool`` is a subclass of ``int``, so that check has
    to run before the numeric one. Integers with hundreds of digits overflow
    when converted to float; that is an invalid probability, not an unexpected
    error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and 0.0 <= value <= 1.0
    except OverflowError:
        return False


def _parse_predicate_answer(payload: object, *, name: str) -> PredicateAnswer | None:
    """Read the named predicate from a Decisions JSON object."""
    if not isinstance(payload, dict):
        raise DecisionResponseError("Decisions response must be a JSON object.")
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise DecisionResponseError(
            "Decisions response field 'answers' must be a list."
        )
    for item in answers:
        if isinstance(item, dict) and item.get("name") == name:
            return _interpret_named_answer(item, name=name)
    return None


def _interpret_named_answer(
    answer: dict[Any, Any], *, name: str
) -> PredicateAnswer | None:
    """Interpret the first answer whose name matches the question.

    The first match wins. A refusal or a non-predicate type there hides any
    later answer with the same name.
    """
    if answer.get("type") != "predicate":
        return None
    try:
        return PredicateAnswer(name=name, probability=answer.get("probability"))
    except ValidationError as exc:
        raise DecisionResponseError(
            f"Decision for predicate {name!r} has no valid probability."
        ) from exc


__all__ = [
    "DEFAULT_DECISIONS_MODEL",
    "DecisionResponseError",
    "OpenAIDecisions",
    "PredicateAnswer",
]
