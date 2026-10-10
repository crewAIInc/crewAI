from typing import Any

import pytest
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.converter import Converter, ConverterError
from pydantic import BaseModel, ValidationError


class Summary(BaseModel):
    """Minimal output schema for offline conversion checks."""

    summary: str


class OfflineLLM(BaseLLM):
    """Return controlled responses without creating a provider client."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize the response sequence and call counter."""
        super().__init__(model="offline", temperature=0)
        self.responses = responses
        self.calls = 0

    def supports_function_calling(self) -> bool:
        """Exercise the plain-text conversion branch."""
        return False

    def call(self, messages: Any, **kwargs: Any) -> str:
        """Return the next response, repeating the last on exhaustion."""
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        """Expose the same controlled responses to async conversion."""
        return self.call(messages, **kwargs)


@pytest.mark.asyncio
async def test_async_retry_never_calls_sync_llm() -> None:
    """Async retries must use acall even after invalid JSON."""

    class AsyncOnlyLLM(OfflineLLM):
        def call(self, messages: Any, **kwargs: Any) -> str:
            """Fail if the converter enters the synchronous path."""
            raise AssertionError("Async conversion called the synchronous LLM")

        async def acall(self, messages: Any, **kwargs: Any) -> str:
            """Recover with valid JSON on the second asynchronous call."""
            self.calls += 1
            return "invalid" if self.calls == 1 else '{"summary":"recovered"}'

    llm = AsyncOnlyLLM([])
    converter = Converter(llm=llm, text="input", model=Summary, instructions="JSON")
    assert await converter.ato_pydantic() == Summary(summary="recovered")
    assert llm.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_hook_abort_is_not_retried(asynchronous: bool) -> None:
    """An explicit hook abort retains its identity and stops immediately."""
    from crewai.hooks.dispatch import HookAborted

    error = HookAborted("controlled abort")

    class AbortedLLM(OfflineLLM):
        def call(self, messages: Any, **kwargs: Any) -> str:
            """Raise the controlled abort on the first attempt."""
            self.calls += 1
            raise error

    llm = AbortedLLM([])
    converter = Converter(llm=llm, text="input", model=Summary, instructions="JSON")
    with pytest.raises(HookAborted) as caught:
        if asynchronous:
            await converter.ato_pydantic()
        else:
            converter.to_pydantic()
    assert caught.value is error
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response", ["not JSON", '{"wrong":"field"}', "text {broken}"])
async def test_exhausted_retries_preserve_validation_cause(
    asynchronous: bool, response: str
) -> None:
    """Retry exhaustion must retain validation failure, not a missing agent."""
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
    """A later valid response recovers within the existing retry budget."""
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
    """Valid embedded JSON is validated locally in a single model call."""
    llm = OfflineLLM([response])
    converter = Converter(llm=llm, text="input", model=Summary, instructions="JSON")
    result = await converter.ato_pydantic() if asynchronous else converter.to_pydantic()
    assert isinstance(result, Summary)
    assert result.summary in ("ok", "line\nnext")
    assert llm.calls == 1
