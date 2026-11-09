import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.llm import LLM


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
