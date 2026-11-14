import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.exceptions.structured_response_exception import (
    StructuredResponseConversionError,
)


@pytest.mark.asyncio
async def test_async_streaming_applies_response_model():
    """The async stream must validate the accumulated text with `response_model`.

    The sync path converts the accumulated stream through `InternalInstructor`,
    but the async path returned the raw text, so async callers asking for a
    structured answer silently received a plain string.
    """

    class Answer(BaseModel):
        answer: str

    async def fake_acompletion(**kwargs):
        async def gen():
            yield {
                "id": "chunk-1",
                "choices": [{"delta": {"content": '{"answer": "42"}'}}],
            }
            yield {"id": "chunk-2", "choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()

    stub_litellm = MagicMock()
    stub_litellm.acompletion = fake_acompletion

    llm = MagicMock()
    llm.is_litellm = True
    llm._extract_finish_reason_and_response_id.return_value = (None, None)
    llm._usage_to_dict.return_value = None

    def _unused_type(name):
        # `isinstance()` needs a real type; the stream below uses plain dicts.
        return type(name, (), {})

    with (
        patch("crewai.llm.litellm", stub_litellm),
        patch("crewai.llm.ModelResponseBase", _unused_type("ModelResponseBase")),
        patch("crewai.llm.ModelResponseStream", _unused_type("ModelResponseStream")),
        patch("crewai.llm.LiteLLMStreamingChoices", _unused_type("LiteLLMStreamingChoices")),
        patch("crewai.llm.LiteLLMDelta", _unused_type("LiteLLMDelta")),
        patch("crewai.llm.InternalInstructor") as mock_instructor,
    ):
        mock_instructor.return_value.to_pydantic.return_value = Answer(answer="42")
        result = await LLM._ahandle_streaming_response(
            llm,
            params={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is the answer?"}],
            },
            response_model=Answer,
        )

    mock_instructor.assert_called_once()
    assert mock_instructor.call_args.kwargs["content"] == '{"answer": "42"}'
    assert json.loads(result) == {"answer": "42"}


async def _fake_stream(**kwargs):
    async def gen():
        yield {"id": "chunk-1", "choices": [{"delta": {"content": '{"answer": "42"}'}}]}
        yield {"id": "chunk-2", "choices": [{"delta": {}, "finish_reason": "stop"}]}

    return gen()


@pytest.mark.asyncio
async def test_conversion_failure_is_reported_instead_of_returning_raw_text():
    """A response_model that cannot be applied must not fall back to the raw stream.

    The streaming handler fails open on mid-stream errors by returning the text
    collected so far. A failed conversion used to take that same path, so the
    caller received unvalidated text as a successful structured answer.
    """

    class Answer(BaseModel):
        answer: str

    stub_litellm = MagicMock()
    stub_litellm.acompletion = _fake_stream

    llm = MagicMock()
    llm.is_litellm = True
    llm._extract_finish_reason_and_response_id.return_value = (None, None)
    llm._usage_to_dict.return_value = None

    instructor = MagicMock()
    instructor.return_value.to_pydantic.side_effect = ValueError("schema mismatch")

    with (
        patch.multiple(
            "crewai.llm",
            litellm=stub_litellm,
            ModelResponseBase=type("ModelResponseBase", (), {}),
            ModelResponseStream=type("ModelResponseStream", (), {}),
            LiteLLMStreamingChoices=type("LiteLLMStreamingChoices", (), {}),
            LiteLLMDelta=type("LiteLLMDelta", (), {}),
            InternalInstructor=instructor,
        ),
        pytest.raises(StructuredResponseConversionError) as error,
    ):
        await LLM._ahandle_streaming_response(
            llm,
            params={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            response_model=Answer,
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert "Answer" in str(error.value)


@pytest.mark.asyncio
async def test_context_length_failure_keeps_its_own_error_type():
    """`to_pydantic()` runs a completion, so an overflow is not a schema error.

    Callers retry on a context-length failure; reporting it as a conversion
    error would make a too-long stream look like a bad schema.
    """
    from litellm.exceptions import ContextWindowExceededError

    class Answer(BaseModel):
        answer: str

    stub_litellm = MagicMock()
    stub_litellm.acompletion = _fake_stream

    llm = MagicMock()
    llm.is_litellm = True
    llm._extract_finish_reason_and_response_id.return_value = (None, None)
    llm._usage_to_dict.return_value = None

    instructor = MagicMock()
    instructor.return_value.to_pydantic.side_effect = ContextWindowExceededError(
        message="maximum context length is 128000 tokens",
        model="gpt-4o-mini",
        llm_provider="openai",
    )

    with (
        patch.multiple(
            "crewai.llm",
            litellm=stub_litellm,
            ModelResponseBase=type("ModelResponseBase", (), {}),
            ModelResponseStream=type("ModelResponseStream", (), {}),
            LiteLLMStreamingChoices=type("LiteLLMStreamingChoices", (), {}),
            LiteLLMDelta=type("LiteLLMDelta", (), {}),
            InternalInstructor=instructor,
        ),
        pytest.raises(LLMContextLengthExceededError) as error,
    ):
        await LLM._ahandle_streaming_response(
            llm,
            params={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            response_model=Answer,
        )

    assert isinstance(error.value.__cause__, ContextWindowExceededError)


@pytest.mark.asyncio
async def test_conversion_runs_off_the_event_loop():
    """`to_pydantic()` is a blocking provider call, so it must not run inline."""

    class Answer(BaseModel):
        answer: str

    seen_threads = []
    main_thread = threading.get_ident()

    stub_litellm = MagicMock()
    stub_litellm.acompletion = _fake_stream

    llm = MagicMock()
    llm.is_litellm = True
    llm._extract_finish_reason_and_response_id.return_value = (None, None)
    llm._usage_to_dict.return_value = None

    instructor = MagicMock()
    instructor.return_value.to_pydantic.side_effect = lambda: (
        seen_threads.append(threading.get_ident()) or Answer(answer="42")
    )

    with patch.multiple(
        "crewai.llm",
        litellm=stub_litellm,
        ModelResponseBase=type("ModelResponseBase", (), {}),
        ModelResponseStream=type("ModelResponseStream", (), {}),
        LiteLLMStreamingChoices=type("LiteLLMStreamingChoices", (), {}),
        LiteLLMDelta=type("LiteLLMDelta", (), {}),
        InternalInstructor=instructor,
    ):
        await LLM._ahandle_streaming_response(
            llm,
            params={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            response_model=Answer,
        )

    assert seen_threads
    assert seen_threads[0] != main_thread
