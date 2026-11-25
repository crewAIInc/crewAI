"""Check the task boundary when a converter returns an unusable JSON string."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from crewai import Task
from pydantic import BaseModel


class CountOutput(BaseModel):
    count: int
    status: str = "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("json_output", [False, True], ids=["pydantic", "json"])
@pytest.mark.parametrize("response", ["not JSON", '{"count": "invalid"}'])
async def test_invalid_converter_string_leaves_structured_fields_empty(
    use_async: bool, json_output: bool, response: str
) -> None:
    task = Task(
        description="Count the items",
        expected_output="A count",
        output_json=CountOutput if json_output else None,
        output_pydantic=None if json_output else CountOutput,
    )
    converter = (
        AsyncMock(return_value=response) if use_async else Mock(return_value=response)
    )
    entry = "async_convert_to_model" if use_async else "convert_to_model"
    with patch(f"crewai.task.{entry}", converter):
        result = (
            await task._aexport_output("raw answer")
            if use_async
            else task._export_output("raw answer")
        )
    assert result == (None, None)
    assert converter.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("json_output", [False, True], ids=["pydantic", "json"])
async def test_valid_converter_string_keeps_the_configured_output_format(
    use_async: bool, json_output: bool
) -> None:
    task = Task(
        description="Count the items",
        expected_output="A count",
        output_json=CountOutput if json_output else None,
        output_pydantic=None if json_output else CountOutput,
    )
    response = '{"count": "3"}'
    converter = (
        AsyncMock(return_value=response) if use_async else Mock(return_value=response)
    )
    entry = "async_convert_to_model" if use_async else "convert_to_model"
    with patch(f"crewai.task.{entry}", converter):
        model, data = (
            await task._aexport_output("raw answer")
            if use_async
            else task._export_output("raw answer")
        )
    if json_output:
        assert model is None
        assert data == {"count": 3, "status": "ready"}
    else:
        assert model == CountOutput(count=3)
        assert data is None
