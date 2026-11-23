"""Validate structured responses from custom LLMs against the requested model."""

import json
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel
import pytest

from crewai.utilities.converter import Converter, ConverterError


class Report(BaseModel):
    count: int
    status: str = "ready"


class ProviderReport(BaseModel):
    count: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["to_pydantic", "ato_pydantic", "to_json", "ato_json"]
)
@pytest.mark.parametrize("valid", [False, True])
async def test_converter_validates_foreign_model(method: str, valid: bool) -> None:
    response = ProviderReport(count="3" if valid else "invalid")
    llm = Mock()
    llm.supports_function_calling.return_value = False
    llm.call.return_value = response
    llm.acall = AsyncMock(return_value=response)
    converter = Converter(
        llm=llm,
        text="three items",
        model=Report,
        instructions="Return a report.",
        max_attempts=2,
    )

    async def invoke() -> BaseModel | str | ConverterError:
        result = getattr(converter, method)()
        return await result if method.startswith("a") else result

    if valid:
        result = await invoke()
        if method.endswith("json"):
            assert json.loads(result) == {"count": 3, "status": "ready"}
        else:
            assert isinstance(result, Report)
            assert result.count == 3
            assert result.status == "ready"
    elif method.endswith("json"):
        assert isinstance(await invoke(), ConverterError)
    else:
        with pytest.raises(ConverterError):
            await invoke()
    calls = llm.acall.call_count if method.startswith("a") else llm.call.call_count
    assert calls == (1 if valid else 2)


def test_converter_preserves_requested_model_instance() -> None:
    response = Report(count=3)
    converter = Converter(
        llm=Mock(), text="three items", model=Report, instructions="Return a report."
    )
    assert converter._coerce_response_to_pydantic(response) is response
