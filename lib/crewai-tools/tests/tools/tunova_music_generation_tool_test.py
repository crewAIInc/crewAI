import os
from unittest.mock import patch

from crewai_tools.tools.tunova_music_generation_tool.tunova_music_generation_tool import (
    TunovaMusicGenerationTool,
)
import pytest


@pytest.fixture(autouse=True)
def mock_tunova_api_key():
    with patch.dict(os.environ, {"TUNOVA_API_KEY": "test_key"}):
        yield


@pytest.fixture
def tunova_tool():
    return TunovaMusicGenerationTool(poll_interval=0)


def test_tunova_tool_initialization():
    tool = TunovaMusicGenerationTool()
    assert tool.base_url == "https://api.tunova.ai"
    assert tool.poll_interval == 10


@patch("requests.get")
@patch("requests.post")
def test_tunova_tool_successful_generation(mock_post, mock_get, tunova_tool):
    mock_post.return_value.status_code = 202
    mock_post.return_value.json.return_value = {"job_id": "job_123"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "status": "complete",
        "clips": [
            {
                "audio_url": "https://cdn.tunova.ai/clips/abc.mp3",
                "title": "Summer Nights",
                "duration": 120.0,
            }
        ],
    }

    result = tunova_tool.run(prompt="an upbeat synthwave track about summer nights")

    assert "Summer Nights" in result
    assert "https://cdn.tunova.ai/clips/abc.mp3" in result
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload == {"prompt": "an upbeat synthwave track about summer nights"}
    called_headers = mock_post.call_args.kwargs["headers"]
    assert called_headers["X-API-Key"] == "test_key"


@patch("requests.get")
@patch("requests.post")
def test_tunova_tool_instrumental_payload(mock_post, mock_get, tunova_tool):
    mock_post.return_value.status_code = 202
    mock_post.return_value.json.return_value = {"job_id": "job_123"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "status": "complete",
        "clips": [
            {
                "audio_url": "https://cdn.tunova.ai/clips/abc.mp3",
                "title": "Instrumental",
                "duration": 90.0,
            }
        ],
    }

    tunova_tool.run(prompt="a calm piano piece", make_instrumental=True)

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["make_instrumental"] is True


@patch("requests.get")
@patch("requests.post")
def test_tunova_tool_failed_generation_mentions_refund(
    mock_post, mock_get, tunova_tool
):
    mock_post.return_value.status_code = 202
    mock_post.return_value.json.return_value = {"job_id": "job_123"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"status": "failed", "clips": []}

    result = tunova_tool.run(prompt="a song that fails")

    assert "failed" in result
    assert "refunded" in result


@patch(
    "crewai_tools.tools.tunova_music_generation_tool.tunova_music_generation_tool.time"
)
@patch("requests.get")
@patch("requests.post")
def test_tunova_tool_timeout(mock_post, mock_get, mock_time, tunova_tool):
    mock_post.return_value.status_code = 202
    mock_post.return_value.json.return_value = {"job_id": "job_123"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"status": "processing", "clips": []}
    mock_time.monotonic.side_effect = [0, 5, 15]

    result = tunova_tool.run(prompt="a slow render", wait_seconds=10)

    assert "did not finish" in result
    assert "job_123" in result


def test_tunova_tool_missing_api_key(tunova_tool):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError):
            tunova_tool.run(prompt="a song without credentials")
