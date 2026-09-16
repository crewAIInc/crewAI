from unittest.mock import AsyncMock, patch

from crewai_tools.tools.wait_tool import WaitTool
from pydantic import ValidationError
import pytest


@pytest.fixture
def tool():
    return WaitTool()


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_waits_requested_duration(mock_sleep, tool):
    result = tool.run(seconds=2)

    mock_sleep.assert_called_once_with(2)
    assert "Waited 2 seconds." in result
    assert "capped" not in result


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_caps_long_waits(mock_sleep, tool):
    result = tool.run(seconds=3600)

    mock_sleep.assert_called_once_with(300)
    assert "Waited 300 seconds." in result
    assert "Requested 3600 seconds, capped at 300 seconds per call" in result
    assert "call this tool again" in result


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_custom_max_seconds(mock_sleep):
    tool = WaitTool(max_seconds=10)

    result = tool.run(seconds=45)

    mock_sleep.assert_called_once_with(10)
    assert "Waited 10 seconds." in result
    assert "10 seconds" in tool.description


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_reason_is_echoed_back(mock_sleep, tool):
    result = tool.run(seconds=1, reason="sandbox build running")

    assert "Reason: sandbox build running" in result


def test_zero_seconds_is_allowed(tool):
    assert "Waited 0 seconds." in tool.run(seconds=0)


def test_negative_seconds_is_rejected(tool):
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        tool.run(seconds=-5)


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_negative_seconds_is_rejected_when_passed_positionally(mock_sleep, tool):
    # BaseTool.run() skips args_schema validation for positional arguments.
    with pytest.raises(ValueError, match="seconds must be zero or greater"):
        tool.run(-5)

    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_async_negative_seconds_is_rejected_when_passed_positionally(tool):
    with patch(
        "crewai_tools.tools.wait_tool.wait_tool.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        with pytest.raises(ValueError, match="seconds must be zero or greater"):
            await tool.arun(-5)

    mock_sleep.assert_not_awaited()


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_nan_seconds_is_rejected_when_passed_positionally(mock_sleep, tool):
    with pytest.raises(ValueError, match="seconds must be a number"):
        tool.run(float("nan"))

    mock_sleep.assert_not_called()


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_infinite_seconds_is_capped_like_any_other_long_wait(mock_sleep, tool):
    result = tool.run(float("inf"))

    mock_sleep.assert_called_once_with(300)
    assert "Waited 300 seconds." in result


@patch("crewai_tools.tools.wait_tool.wait_tool.time.sleep")
def test_singular_second_is_not_pluralized(mock_sleep):
    tool = WaitTool(max_seconds=1)

    assert "Waited 1 second." in tool.run(seconds=1)
    assert "capped at 1 second per call" in tool.run(seconds=5)
    assert "at most 1 second." in tool.description


def test_invalid_max_seconds_is_rejected():
    with pytest.raises(ValidationError):
        WaitTool(max_seconds=0)


@pytest.mark.asyncio
async def test_async_wait_caps_long_waits(tool):
    with patch(
        "crewai_tools.tools.wait_tool.wait_tool.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        result = await tool.arun(seconds=3600, reason="deployment")

    mock_sleep.assert_awaited_once_with(300)
    assert "Waited 300 seconds." in result
    assert "Reason: deployment" in result


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda d: WaitTool(max_seconds=10), id="init"),
        pytest.param(lambda d: WaitTool(10), id="init_positional"),
        pytest.param(lambda d: WaitTool(max_seconds=10, description=d), id="init_desc"),
        pytest.param(
            lambda d: WaitTool.model_validate({"max_seconds": 10, "description": d}),
            id="model_validate",
        ),
        pytest.param(
            lambda d: WaitTool.model_validate(WaitTool(max_seconds=10).model_dump()),
            id="round_trip",
        ),
    ],
)
def test_advertised_cap_matches_enforced_cap(build):
    tool = build(WaitTool().description)

    assert tool.max_seconds == 10
    assert "at most 10 seconds" in tool.description
    assert "300 seconds" not in tool.description


def test_advertised_cap_follows_assignment():
    tool = WaitTool()
    tool.max_seconds = 10

    assert "at most 10 seconds" in tool.description


def test_caller_supplied_description_is_left_alone():
    tool = WaitTool(max_seconds=10, description="Hold on for a bit.")

    assert tool.description == "Hold on for a bit."


def test_waits_are_never_cached():
    # A cache hit would hand back "Waited N seconds." without any time passing,
    # turning a poll-wait-check loop into a busy loop.
    tool = WaitTool()

    assert tool.cache_function("{}", "Waited 1 second.") is False
    assert WaitTool.model_validate(tool.model_dump()).cache_function() is False


def test_description_explains_when_to_use_it(tool):
    description = tool.description.lower()

    assert "sandbox" in description
    assert "deployment" in description
    assert "300 seconds" in description
    assert "call this tool again" in description


def test_exported_from_package():
    from crewai_tools import WaitTool as ExportedWaitTool
    from crewai_tools.tools import WaitTool as ToolsExportedWaitTool

    assert ExportedWaitTool is WaitTool
    assert ToolsExportedWaitTool is WaitTool
