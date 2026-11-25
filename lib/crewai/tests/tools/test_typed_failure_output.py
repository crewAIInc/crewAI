"""A failure is not a successful typed payload, even on a typed tool."""

import warnings

import pytest
from crewai.tools.base_tool import Tool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel


class CountOutput(BaseModel):
    count: int


@pytest.mark.parametrize("structured", [False, True], ids=["tool", "structured"])
def test_failure_output_bypasses_success_schema(structured: bool) -> None:
    def count_items() -> dict[str, int]:
        return {"count": 3}

    tool = (
        CrewStructuredTool.from_function(
            count_items,
            name="count",
            description="Count items",
            result_schema=CountOutput,
        )
        if structured
        else Tool(
            name="count",
            description="Count items",
            func=count_items,
            result_schema=CountOutput,
        )
    )
    failure = ToolFailure(message="Collection unavailable", code="not_found")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = tool.format_output_for_agent(failure)
    assert rendered == "Collection unavailable (code: not_found)"
    assert not caught
    assert tool.format_output_for_agent({"count": "3"}) == '{"count":3}'
