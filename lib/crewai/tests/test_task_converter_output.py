from crewai import Agent, Task
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.converter import Converter, ConverterError
from pydantic import BaseModel
import pytest


class Result(BaseModel):
    count: int
    approved: bool = False


class JSONStringConverter(Converter):
    def to_json(self, current_attempt: int = 1) -> str:
        return '{"count": "3"}'

    async def ato_json(self, current_attempt: int = 1) -> str:
        return '{"count": "3"}'


class FailingPydanticConverter(Converter):
    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        raise ConverterError("Could not convert result")

    async def ato_pydantic(self, current_attempt: int = 1) -> BaseModel:
        raise ConverterError("Could not convert result")


@pytest.fixture
def agent() -> Agent:
    return Agent(
        role="Counter",
        goal="Count records",
        backstory="Returns record counts.",
        llm=OpenAICompletion(model="gpt-4o-mini", api_key="test-key"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True])
async def test_task_export_keeps_validated_json(async_path: bool, agent: Agent) -> None:
    task = Task(
        description="Count records",
        expected_output="The count",
        agent=agent,
        output_json=Result,
        converter_cls=JSONStringConverter,
    )

    if async_path:
        output = await task._aexport_output("raw result")
    else:
        output = task._export_output("raw result")

    assert output == (None, {"count": 3, "approved": False})


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True])
async def test_task_export_retains_pydantic_format_on_converter_fallback(
    async_path: bool,
    agent: Agent,
) -> None:
    task = Task(
        description="Count records",
        expected_output="The count",
        agent=agent,
        output_pydantic=Result,
        converter_cls=FailingPydanticConverter,
    )
    raw = '{"count": "3"}'

    if async_path:
        output = await task._aexport_output(raw)
    else:
        output = task._export_output(raw)

    assert output == (Result(count=3), None)
