"""Tests for ChronoVerifyImageVerificationTool."""

import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools import ChronoVerifyImageVerificationTool
import pytest
import requests


REQUESTS_POST = (
    "crewai_tools.tools.chronoverify_image_verification_tool"
    ".chronoverify_image_verification_tool.requests.post"
)


@pytest.fixture
def tool():
    return ChronoVerifyImageVerificationTool()


@pytest.fixture
def mock_verify_response():
    return {
        "verdict": "consistent",
        "confidence": 82,
        "capture_time": {
            "present": True,
            "value": "2026-05-14T09:31:22+00:00",
            "source": "exif",
        },
        "capture_device": {"make": "Apple", "model": "iPhone 15 Pro"},
        "capture_location": {"present": False},
        "c2pa": {"present": False},
        "integrity": {"sha256": "a" * 64},
    }


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@patch(REQUESTS_POST)
def test_verify_by_url(mock_post, tool, mock_verify_response):
    mock_post.return_value = _mock_response(mock_verify_response)

    result = tool._run(image_url="https://example.com/photo.jpg")

    parsed = json.loads(result)
    assert parsed["verdict"] == "consistent"
    assert parsed["confidence"] == 82

    _, kwargs = mock_post.call_args
    assert kwargs["data"] == {"url": "https://example.com/photo.jpg"}
    assert mock_post.call_args[0][0] == "https://chronoverify.com/v1/verify"


@patch(REQUESTS_POST)
def test_verify_by_file(mock_post, tool, mock_verify_response, tmp_path):
    mock_post.return_value = _mock_response(mock_verify_response)
    image_file = tmp_path / "photo.jpg"
    image_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    result = tool._run(image_path=str(image_file))

    parsed = json.loads(result)
    assert parsed["verdict"] == "consistent"

    _, kwargs = mock_post.call_args
    assert "files" in kwargs
    assert kwargs["files"]["file"][0] == "photo.jpg"


@patch(REQUESTS_POST)
def test_keyless_request_sends_no_auth_header(mock_post, tool, mock_verify_response):
    mock_post.return_value = _mock_response(mock_verify_response)

    with patch.dict(os.environ, {}, clear=True):
        tool._run(image_url="https://example.com/photo.jpg")

    _, kwargs = mock_post.call_args
    assert "Authorization" not in kwargs["headers"]


@patch(REQUESTS_POST)
def test_api_key_sent_as_bearer_token(mock_post, tool, mock_verify_response):
    mock_post.return_value = _mock_response(mock_verify_response)

    with patch.dict(os.environ, {"CHRONOVERIFY_API_KEY": "cv_live_test123"}):
        tool._run(image_url="https://example.com/photo.jpg")

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer cv_live_test123"


def test_missing_inputs_returns_error(tool):
    result = tool._run()
    assert "provide either image_path or image_url" in result


def test_both_inputs_returns_error(tool):
    result = tool._run(
        image_path="photo.jpg", image_url="https://example.com/photo.jpg"
    )
    assert "only one of image_path or image_url" in result


def test_missing_file_returns_error(tool):
    result = tool._run(image_path="does/not/exist.jpg")
    assert "image file not found" in result


@patch(REQUESTS_POST)
def test_http_error_returns_error_message(mock_post, tool):
    response = MagicMock()
    response.status_code = 429
    response.text = "Rate limited"
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    mock_post.return_value = response

    result = tool._run(image_url="https://example.com/photo.jpg")

    assert "HTTP 429" in result
    assert "Rate limited" in result


@patch(REQUESTS_POST, side_effect=requests.exceptions.ConnectionError("boom"))
def test_network_error_returns_error_message(mock_post, tool):
    result = tool._run(image_url="https://example.com/photo.jpg")
    assert "Error communicating with the ChronoVerify API" in result


@patch(REQUESTS_POST)
def test_custom_base_url(mock_post, mock_verify_response):
    mock_post.return_value = _mock_response(mock_verify_response)
    tool = ChronoVerifyImageVerificationTool(api_base_url="http://localhost:8000/")

    tool._run(image_url="https://example.com/photo.jpg")

    assert mock_post.call_args[0][0] == "http://localhost:8000/v1/verify"


@patch(REQUESTS_POST)
def test_run_method_invocation(mock_post, tool, mock_verify_response):
    """Test invocation through .run(), which is how CrewAI calls tools."""
    mock_post.return_value = _mock_response(mock_verify_response)

    result = tool.run(image_url="https://example.com/photo.jpg")

    assert "consistent" in result
