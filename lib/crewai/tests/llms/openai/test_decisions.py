"""Mocked tests for the OpenAI Decisions predicate adapter.

No test builds a live HTTP call. Sync and async clients are injected, and the
tests that exercise client construction patch ``OpenAI`` and ``AsyncOpenAI``.
Request and response bodies follow the documented ``POST /v1/decisions``
predicate shape: string ``input``, one ``type: predicate`` question, and an
``answers`` entry with ``type``, ``name``, and ``probability``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from crewai.llms.providers.openai.decisions import (
    DEFAULT_DECISIONS_MODEL,
    DecisionResponseError,
    OpenAIDecisions,
    PredicateAnswer,
)


TEXT = "I was charged twice for my order."
NAME = "refund_requested"
INSTRUCTIONS = "Does the customer claim they were charged more than once?"


def _predicate_response(
    name: str = NAME,
    probability: object = 0.92,
    *,
    answer_type: str = "predicate",
    include_probability: bool = True,
    extra_answer: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a documented predicate response, plus optional unknown fields."""
    answer: dict[str, object] = {"type": answer_type, "name": name}
    if include_probability:
        answer["probability"] = probability
    if extra_answer:
        answer.update(extra_answer)
    body: dict[str, object] = {"answers": [answer]}
    if extra_body:
        body.update(extra_body)
    return body


def _expected_body(
    text: str = TEXT,
    *,
    name: str = NAME,
    instructions: str = INSTRUCTIONS,
    model: str = DEFAULT_DECISIONS_MODEL,
) -> dict[str, object]:
    """Build the predicate request body sent to ``POST /decisions``."""
    return {
        "model": model,
        "input": text,
        "questions": [
            {
                "type": "predicate",
                "name": name,
                "instructions": instructions,
            }
        ],
    }


def _adapter(response: object) -> tuple[OpenAIDecisions, MagicMock]:
    """Return an adapter whose sync client returns ``response``."""
    client = MagicMock()
    client.post.return_value = response
    return OpenAIDecisions(client=client), client


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/decisions")


def _status_error(
    status: int, message: str, error_cls: type[APIStatusError]
) -> APIStatusError:
    response = httpx.Response(status, request=_request())
    return error_cls(message, response=response, body={"error": {"message": message}})


def _ask(adapter: OpenAIDecisions, **overrides: object) -> PredicateAnswer | None:
    arguments: dict[str, object] = {
        "input": TEXT,
        "name": NAME,
        "instructions": INSTRUCTIONS,
    }
    arguments.update(overrides)
    return adapter.predicate(
        arguments["input"],  # type: ignore[arg-type]
        name=arguments["name"],  # type: ignore[arg-type]
        instructions=arguments["instructions"],  # type: ignore[arg-type]
    )


class TestPredicateResponse:
    def test_returns_typed_probability_and_posts_documented_body(self) -> None:
        response = _predicate_response(
            extra_answer={"ignored": True},
            extra_body={"id": "dec_123", "usage": {"input_tokens": 12}},
        )
        adapter, client = _adapter(response)

        answer = adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer == PredicateAnswer(name=NAME, probability=0.92)
        client.post.assert_called_once_with(
            "/decisions",
            body=_expected_body(),
            cast_to=object,
        )

    @pytest.mark.parametrize("probability", [0, 1, 0.0, 1.0, 0.92])
    def test_accepts_boundary_and_fractional_probabilities(
        self, probability: float
    ) -> None:
        adapter, _client = _adapter(_predicate_response(probability=probability))

        answer = adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer is not None
        assert answer.probability == float(probability)
        assert isinstance(answer.probability, float)

    def test_selects_the_answer_with_the_requested_name(self) -> None:
        adapter, _client = _adapter(
            {
                "answers": [
                    {"type": "predicate", "name": "other", "probability": 0.1},
                    "not-an-object",
                    {"type": "predicate", "name": NAME, "probability": 0.4},
                ]
            }
        )

        answer = adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer == PredicateAnswer(name=NAME, probability=0.4)

    def test_forwards_a_custom_model(self) -> None:
        client = MagicMock()
        client.post.return_value = _predicate_response(probability=0.2)
        adapter = OpenAIDecisions(client=client, model="custom-decisions-model")

        adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert client.post.call_args.kwargs["body"]["model"] == "custom-decisions-model"

    def test_answer_is_frozen(self) -> None:
        adapter, _client = _adapter(_predicate_response())
        answer = adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer is not None
        with pytest.raises(ValidationError):
            answer.probability = 0.1  # type: ignore[misc]

    def test_sync_call_does_not_use_the_async_client(self) -> None:
        sync_client = MagicMock()
        sync_client.post.return_value = _predicate_response()
        async_client = MagicMock()
        adapter = OpenAIDecisions(client=sync_client, async_client=async_client)

        adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        async_client.post.assert_not_called()


class TestMissingDecisions:
    def test_empty_answer_list_returns_none(self) -> None:
        adapter, _client = _adapter({"answers": []})

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    def test_unmatched_name_returns_none(self) -> None:
        adapter, _client = _adapter(_predicate_response(name="other"))

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    def test_refusal_returns_none(self) -> None:
        adapter, _client = _adapter(
            _predicate_response(
                answer_type="refusal",
                include_probability=False,
                extra_answer={"probability": 0.99},
            )
        )

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    @pytest.mark.parametrize("answer_type", ["choice", "score"])
    def test_other_question_types_return_none(self, answer_type: str) -> None:
        adapter, _client = _adapter(_predicate_response(answer_type=answer_type))

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    def test_missing_type_returns_none(self) -> None:
        adapter, _client = _adapter({"answers": [{"name": NAME, "probability": 0.5}]})

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    def test_first_refusal_hides_a_later_predicate_with_the_same_name(self) -> None:
        adapter, _client = _adapter(
            {
                "answers": [
                    {"type": "refusal", "name": NAME},
                    {"type": "predicate", "name": NAME, "probability": 0.9},
                ]
            }
        )

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    def test_non_object_answers_with_no_match_return_none(self) -> None:
        adapter, _client = _adapter({"answers": ["nope", None, 1]})

        assert adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None


class TestMalformedResponses:
    @pytest.mark.parametrize(
        "probability",
        [True, False, float("nan"), float("inf"), float("-inf"), 1.01, -0.01, "0.5", None],
    )
    def test_invalid_probability_raises(self, probability: object) -> None:
        adapter, _client = _adapter(_predicate_response(probability=probability))

        with pytest.raises(DecisionResponseError, match="no valid probability"):
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

    @pytest.mark.parametrize("probability", [10**399, -(10**399)])
    def test_oversized_integer_probability_raises(self, probability: int) -> None:
        """A JSON integer too large for float() must stay a decision error."""
        adapter, _client = _adapter(_predicate_response(probability=probability))

        with pytest.raises(DecisionResponseError, match="no valid probability"):
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

    def test_missing_probability_raises(self) -> None:
        adapter, _client = _adapter(
            _predicate_response(include_probability=False)
        )

        with pytest.raises(DecisionResponseError, match="no valid probability"):
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

    @pytest.mark.parametrize(
        "payload",
        [None, [], "nope", {"answers": None}, {"answers": {}}, {}],
    )
    def test_malformed_body_raises(self, payload: object) -> None:
        adapter, _client = _adapter(payload)

        with pytest.raises(DecisionResponseError, match="Decisions response"):
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"input": "  "}, "input"),
            ({"input": None}, "input"),
            ({"name": ""}, "name"),
            ({"name": "\n"}, "name"),
            ({"instructions": "   "}, "instructions"),
            ({"instructions": 12}, "instructions"),
        ],
    )
    def test_blank_question_raises_before_http(
        self, overrides: dict[str, object], match: str
    ) -> None:
        adapter, client = _adapter(_predicate_response())

        with pytest.raises(ValueError, match=f"{match} must be a non-empty string"):
            _ask(adapter, **overrides)

        client.post.assert_not_called()


class TestApiErrors:
    @pytest.mark.parametrize(
        "error",
        [
            _status_error(429, "slow down", RateLimitError),
            _status_error(400, "bad request", APIStatusError),
            APITimeoutError(_request()),
            APIConnectionError(message="connection failed", request=_request()),
        ],
    )
    def test_api_and_transport_errors_propagate(self, error: Exception) -> None:
        client = MagicMock()
        client.post.side_effect = error
        adapter = OpenAIDecisions(client=client)

        with pytest.raises(type(error)) as caught:
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert caught.value is error


class TestAsyncPredicate:
    @pytest.mark.asyncio
    async def test_apredicate_posts_the_documented_body(self) -> None:
        async_client = MagicMock()
        async_client.post = AsyncMock(return_value=_predicate_response())
        sync_client = MagicMock()
        adapter = OpenAIDecisions(client=sync_client, async_client=async_client)

        answer = await adapter.apredicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer == PredicateAnswer(name=NAME, probability=0.92)
        async_client.post.assert_awaited_once_with(
            "/decisions",
            body=_expected_body(),
            cast_to=object,
        )
        sync_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_apredicate_refusal_returns_none(self) -> None:
        async_client = MagicMock()
        async_client.post = AsyncMock(
            return_value=_predicate_response(
                answer_type="refusal", include_probability=False
            )
        )
        adapter = OpenAIDecisions(async_client=async_client)

        assert await adapter.apredicate(TEXT, name=NAME, instructions=INSTRUCTIONS) is None

    @pytest.mark.asyncio
    async def test_apredicate_invalid_probability_raises(self) -> None:
        async_client = MagicMock()
        async_client.post = AsyncMock(
            return_value=_predicate_response(probability=float("nan"))
        )
        adapter = OpenAIDecisions(async_client=async_client)

        with pytest.raises(DecisionResponseError, match="no valid probability"):
            await adapter.apredicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

    @pytest.mark.asyncio
    async def test_apredicate_propagates_rate_limit(self) -> None:
        error = _status_error(429, "slow down", RateLimitError)
        async_client = MagicMock()
        async_client.post = AsyncMock(side_effect=error)
        adapter = OpenAIDecisions(async_client=async_client)

        with pytest.raises(RateLimitError) as caught:
            await adapter.apredicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert caught.value is error

    @pytest.mark.asyncio
    async def test_apredicate_blank_input_does_not_call_post(self) -> None:
        async_client = MagicMock()
        async_client.post = AsyncMock()
        adapter = OpenAIDecisions(async_client=async_client)

        with pytest.raises(ValueError, match="input must be a non-empty string"):
            await adapter.apredicate(" ", name=NAME, instructions=INSTRUCTIONS)

        async_client.post.assert_not_awaited()


class TestClientConstruction:
    def test_missing_api_key_raises_before_a_client_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = OpenAIDecisions()

        with (
            patch("crewai.llms.providers.openai.decisions.OpenAI") as openai_cls,
            pytest.raises(ValueError, match="OPENAI_API_KEY"),
        ):
            adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        openai_cls.assert_not_called()

    def test_injected_client_does_not_require_an_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter, client = _adapter(_predicate_response(probability=0.2))

        answer = adapter.predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer == PredicateAnswer(name=NAME, probability=0.2)
        client.post.assert_called_once()

    def test_explicit_client_options_override_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://base.example/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://api-base.example/v1")
        with patch("crewai.llms.providers.openai.decisions.OpenAI") as openai_cls:
            openai_cls.return_value.post.return_value = _predicate_response(
                probability=0.5
            )
            answer = OpenAIDecisions(
                api_key="sk-explicit",
                base_url="https://arg.example/v1",
                organization="org_123",
                project="proj_123",
                timeout=5.0,
                max_retries=0,
            ).predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert answer == PredicateAnswer(name=NAME, probability=0.5)
        openai_cls.assert_called_once_with(
            api_key="sk-explicit",
            base_url="https://arg.example/v1",
            organization="org_123",
            project="proj_123",
            timeout=5.0,
            max_retries=0,
        )

    def test_open_ai_base_url_precedes_open_ai_api_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://base.example/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://api-base.example/v1")
        with patch("crewai.llms.providers.openai.decisions.OpenAI") as openai_cls:
            openai_cls.return_value.post.return_value = _predicate_response()
            OpenAIDecisions().predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert openai_cls.call_args.kwargs["base_url"] == "https://base.example/v1"

    def test_open_ai_api_base_is_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_BASE", "https://api-base.example/v1")
        with patch("crewai.llms.providers.openai.decisions.OpenAI") as openai_cls:
            openai_cls.return_value.post.return_value = _predicate_response()
            OpenAIDecisions().predicate(TEXT, name=NAME, instructions=INSTRUCTIONS)

        assert (
            openai_cls.call_args.kwargs["base_url"] == "https://api-base.example/v1"
        )

    def test_blank_model_raises_before_http(self) -> None:
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            OpenAIDecisions(model="  ", client=MagicMock())

    @pytest.mark.asyncio
    async def test_builds_async_client_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        with patch(
            "crewai.llms.providers.openai.decisions.AsyncOpenAI"
        ) as async_cls:
            async_cls.return_value.post = AsyncMock(
                return_value=_predicate_response(probability=0.4)
            )
            answer = await OpenAIDecisions().apredicate(
                TEXT, name=NAME, instructions=INSTRUCTIONS
            )

        assert answer == PredicateAnswer(name=NAME, probability=0.4)
        async_cls.assert_called_once_with(api_key="sk-test")
