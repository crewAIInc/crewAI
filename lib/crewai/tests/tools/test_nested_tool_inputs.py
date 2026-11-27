import json

from crewai.tools import tool
from crewai.tools.structured_tool import CrewStructuredTool
from pydantic import BaseModel
import pytest


class Location(BaseModel):
    city: str


def describe_locations(
    primary: Location, others: list[Location], separator: str = ", "
) -> str:
    """Describe the supplied locations."""
    return separator.join(location.city for location in [primary, *others])


async def describe_locations_async(
    primary: Location, others: list[Location], separator: str = ", "
) -> str:
    """Describe the supplied locations asynchronously."""
    return describe_locations(primary, others, separator)


@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize("json_input", [False, True])
def test_invoke_preserves_nested_models(async_function: bool, json_input: bool) -> None:
    structured = CrewStructuredTool.from_function(
        describe_locations_async if async_function else describe_locations
    )
    arguments = {"primary": {"city": "Paris"}, "others": [{"city": "Rome"}]}

    assert structured.invoke(json.dumps(arguments) if json_input else arguments) == (
        "Paris, Rome"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize("json_input", [False, True])
async def test_ainvoke_preserves_nested_models(
    async_function: bool, json_input: bool
) -> None:
    structured = CrewStructuredTool.from_function(
        describe_locations_async if async_function else describe_locations
    )
    arguments = {"primary": {"city": "Paris"}, "others": [{"city": "Rome"}]}

    assert (
        await structured.ainvoke(json.dumps(arguments) if json_input else arguments)
        == "Paris, Rome"
    )


def test_decorated_tool_run_preserves_nested_models() -> None:
    decorated = tool(describe_locations)

    assert (
        decorated.run(primary={"city": "Paris"}, others=[{"city": "Rome"}])
        == "Paris, Rome"
    )


@pytest.mark.asyncio
async def test_decorated_tool_arun_preserves_nested_models() -> None:
    decorated = tool(describe_locations_async)

    assert (
        await decorated.arun(primary={"city": "Paris"}, others=[{"city": "Rome"}])
        == "Paris, Rome"
    )


@pytest.mark.parametrize("structured", [False, True])
def test_nested_models_still_validate_required_fields(structured: bool) -> None:
    if structured:
        wrapped = CrewStructuredTool.from_function(describe_locations)
        with pytest.raises(ValueError, match=r"primary\.city"):
            wrapped.invoke({"primary": {}, "others": []})
    else:
        decorated = tool(describe_locations)
        with pytest.raises(ValueError, match=r"primary\.city"):
            decorated.run(primary={}, others=[])
