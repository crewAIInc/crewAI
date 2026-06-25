import os
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.twelvelabs_analyze_tool.twelvelabs_analyze_tool import (
    TwelveLabsAnalyzeTool,
)


def _make_tool() -> TwelveLabsAnalyzeTool:
    """Build a tool with the SDK client stubbed so no network/SDK is needed."""
    with patch(
        "twelvelabs.TwelveLabs", create=True, return_value=MagicMock()
    ):
        return TwelveLabsAnalyzeTool(api_key="test-key")


def test_requires_api_key() -> None:
    with patch("twelvelabs.TwelveLabs", create=True, return_value=MagicMock()):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                TwelveLabsAnalyzeTool()


def test_requires_video_input() -> None:
    tool = _make_tool()
    result = tool.run(prompt="Summarize this video.")
    assert "video_url or video_id is required" in result


def test_run_with_video_id() -> None:
    tool = _make_tool()
    mock_response = MagicMock()
    mock_response.data = "A cat plays with a ball of yarn."
    tool._client.analyze.return_value = mock_response

    result = tool.run(prompt="What happens?", video_id="vid_123")

    assert result == "A cat plays with a ball of yarn."
    call_kwargs = tool._client.analyze.call_args.kwargs
    assert call_kwargs["video_id"] == "vid_123"
    assert call_kwargs["model_name"] == "pegasus1.5"
    assert call_kwargs["prompt"] == "What happens?"
    assert call_kwargs["max_tokens"] == 2048


def test_run_with_video_url_builds_url_context() -> None:
    tool = _make_tool()
    mock_response = MagicMock()
    mock_response.data = "ok"
    tool._client.analyze.return_value = mock_response

    tool.run(prompt="Describe.", video_url="https://example.com/v.mp4")

    video_arg = tool._client.analyze.call_args.kwargs["video"]
    assert video_arg.url == "https://example.com/v.mp4"


@pytest.mark.skipif(
    not os.getenv("TWELVELABS_API_KEY"),
    reason="TWELVELABS_API_KEY not set",
)
def test_analyze_integration() -> None:
    """End-to-end smoke test against the live TwelveLabs API.

    Requires TWELVELABS_API_KEY and an indexed video; skipped otherwise.
    """
    tool = TwelveLabsAnalyzeTool()
    video_id = os.getenv("TWELVELABS_TEST_VIDEO_ID")
    if not video_id:
        pytest.skip("TWELVELABS_TEST_VIDEO_ID not set")
    result = tool.run(prompt="Summarize this video in one sentence.", video_id=video_id)
    assert isinstance(result, str)
    assert len(result) > 0
