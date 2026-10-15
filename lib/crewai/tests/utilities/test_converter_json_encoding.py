import json
from unittest.mock import AsyncMock, Mock

from crewai.task import Task
from crewai.utilities.converter import Converter
from pydantic import BaseModel
import pytest


class Report(BaseModel):
    name: str
    count: int


@pytest.fixture(params=[False, True], ids=["plain-json", "fenced-json"])
def response(request: pytest.FixtureRequest) -> str:
    """Supply realistic JSON text from a model without function calling."""
    raw = json.dumps({"name": 'a "quoted" report', "count": 2})
    return f"```json\n{raw}\n```" if request.param else raw


def test_to_json_preserves_object_shape(response: str) -> None:
    """Return JSON data rather than a JSON string containing escaped JSON."""
    llm = Mock()
    llm.supports_function_calling.return_value = False
    llm.call.return_value = response
    converter = Converter(
        llm=llm, text="report", model=Report, instructions="Extract JSON"
    )

    result = converter.to_json()

    assert json.loads(result) == {"name": 'a "quoted" report', "count": 2}
    assert Task._unpack_model_output(result) == (
        None,
        {"name": 'a "quoted" report', "count": 2},
    )
    assert llm.call.call_count == 1


@pytest.mark.asyncio
async def test_ato_json_preserves_object_shape(response: str) -> None:
    """Keep the same structured result through asynchronous conversion."""
    llm = Mock()
    llm.supports_function_calling.return_value = False
    llm.acall = AsyncMock(return_value=response)
    converter = Converter(
        llm=llm, text="report", model=Report, instructions="Extract JSON"
    )

    result = await converter.ato_json()

    assert json.loads(result) == {"name": 'a "quoted" report', "count": 2}
    assert Task._unpack_model_output(result) == (
        None,
        {"name": 'a "quoted" report', "count": 2},
    )
    assert llm.acall.await_count == 1
