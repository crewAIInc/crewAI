from typing import Any

import pytest
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.converter import Converter, ConverterError
from pydantic import BaseModel, ValidationError


class Summary(BaseModel):
    summary: str


class OfflineLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(model="offline", temperature=0)
        self.responses = responses
        self.calls = 0

    def supports_function_calling(self) -> bool:
        return False

    def call(self, messages: Any, **kwargs: Any) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        return self.call(messages, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response", ["not JSON", '{"wrong":"field"}', "text {broken}"])
async def test_exhausted_retries_preserve_validation_cause(
    asynchronous: bool, response: str
) -> None:
    llm = OfflineLLM([response])
    converter = Converter(
        llm=llm, text="input", model=Summary, instructions="JSON", max_attempts=2
    )
    with pytest.raises(ConverterError) as caught:
        if asynchronous:
            await converter.ato_pydantic()
        else:
            converter.to_pydantic()
    assert llm.calls == 2
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "Agent must be provided" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("first", ["not JSON", '{"wrong":"field"}'])
async def test_retry_recovers_with_existing_llm(asynchronous: bool, first: str) -> None:
    llm = OfflineLLM([first, '{"summary":"recovered"}'])
    converter = Converter(llm=llm, text="input", model=Summary, instructions="JSON")
    result = await converter.ato_pydantic() if asynchronous else converter.to_pydantic()
    assert result == Summary(summary="recovered")
    assert llm.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "response",
    ['prefix {"summary":"ok"} suffix', 'prefix {"summary":"line\nnext"} suffix'],
)
async def test_embedded_json_needs_no_extra_model_call(
    asynchronous: bool, response: str
) -> None:
    llm = OfflineLLM([response])
    converter = Converter(llm=llm, text="input", model=Summary, instructions="JSON")
    result = await converter.ato_pydantic() if asynchronous else converter.to_pydantic()
    assert isinstance(result, Summary)
    assert result.summary in ("ok", "line\nnext")
    assert llm.calls == 1
