"""A denied async conversion must not enter the ordinary retry loop."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from crewai.hooks.dispatch import HookAborted
from crewai.utilities.converter import Converter
from pydantic import BaseModel


class CountOutput(BaseModel):
    count: int


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output", [False, True], ids=["pydantic", "json"])
@pytest.mark.parametrize("function_calling", [False, True])
async def test_async_converter_propagates_abort_without_retrying(
    json_output: bool, function_calling: bool
) -> None:
    denied = HookAborted(reason="model denied by policy", source="test-policy")
    llm = Mock()
    llm.supports_function_calling.return_value = function_calling
    llm.acall = AsyncMock(side_effect=denied)
    converter = Converter(
        llm=llm,
        text="Count the items",
        model=CountOutput,
        instructions="Return the count",
        max_attempts=3,
    )
    instructor = Mock()
    instructor.to_json.side_effect = denied
    with patch.object(converter, "_create_instructor", return_value=instructor):
        with pytest.raises(HookAborted) as caught:
            if json_output:
                await converter.ato_json()
            else:
                await converter.ato_pydantic()
    assert caught.value is denied
    if json_output and function_calling:
        instructor.to_json.assert_called_once_with()
        llm.acall.assert_not_called()
    else:
        llm.acall.assert_awaited_once()
        instructor.to_json.assert_not_called()
