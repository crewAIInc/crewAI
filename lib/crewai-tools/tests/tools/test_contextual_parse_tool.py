from unittest.mock import MagicMock, patch
import pytest

from crewai_tools.tools.contextualai_parse_tool.contextual_parse_tool import ContextualAIParseTool


@pytest.fixture(autouse=True)
def allow_unsafe_paths(monkeypatch):
    monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")


@pytest.fixture
def mock_file(tmp_path):
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"%PDF-1.4 test content")
    return str(f)


def test_contextual_parse_tool_success(mock_file):
    tool = ContextualAIParseTool(api_key="test-key", poll_timeout=30, poll_interval=1)

    mock_post_resp = MagicMock()
    mock_post_resp.text = '{"job_id": "job-123"}'
    mock_post_resp.raise_for_status.return_value = None

    mock_status_resp = MagicMock()
    mock_status_resp.text = '{"status": "completed"}'
    mock_status_resp.raise_for_status.return_value = None

    mock_results_resp = MagicMock()
    mock_results_resp.text = '{"pages": [{"text": "Hello world"}]}'
    mock_results_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_post_resp) as mock_post, \
         patch("requests.get", side_effect=[mock_status_resp, mock_results_resp]) as mock_get, \
         patch("time.sleep", return_value=None):
        result = tool._run(file_path=mock_file)

        assert "Hello world" in result
        mock_post.assert_called_once()
        assert mock_get.call_count == 2


def test_contextual_parse_tool_timeout(mock_file):
    tool = ContextualAIParseTool(api_key="test-key", poll_timeout=10, poll_interval=1)

    mock_post_resp = MagicMock()
    mock_post_resp.text = '{"job_id": "job-999"}'
    mock_post_resp.raise_for_status.return_value = None

    mock_status_resp = MagicMock()
    mock_status_resp.text = '{"status": "processing"}'
    mock_status_resp.raise_for_status.return_value = None

    # Simulate monotonic advancing beyond poll_timeout
    monotonic_times = [0.0, 5.0, 11.0]

    with patch("requests.post", return_value=mock_post_resp), \
         patch("requests.get", return_value=mock_status_resp), \
         patch("time.monotonic", side_effect=monotonic_times), \
         patch("time.sleep", return_value=None):
        with pytest.raises(TimeoutError) as exc_info:
            tool._run(file_path=mock_file)

        assert "Document parsing did not complete within 10 seconds" in str(exc_info.value)


def test_contextual_parse_tool_failed_status(mock_file):
    tool = ContextualAIParseTool(api_key="test-key")

    mock_post_resp = MagicMock()
    mock_post_resp.text = '{"job_id": "job-fail"}'
    mock_post_resp.raise_for_status.return_value = None

    mock_status_resp = MagicMock()
    mock_status_resp.text = '{"status": "failed"}'
    mock_status_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_post_resp), \
         patch("requests.get", return_value=mock_status_resp), \
         patch("time.sleep", return_value=None):
        result = tool._run(file_path=mock_file)

        assert "Failed to parse document: Document parsing failed" in result


def test_contextual_parse_tool_http_error(mock_file):
    import requests

    tool = ContextualAIParseTool(api_key="test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error": "Invalid file format"}'
    http_error = requests.HTTPError("400 Client Error", response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_error

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(file_path=mock_file)

        assert "Failed to parse document" in result
        assert "Invalid file format" in result


def test_contextual_parse_tool_validation_positive_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ContextualAIParseTool(api_key="test-key", poll_timeout=0)

    with pytest.raises(ValidationError):
        ContextualAIParseTool(api_key="test-key", poll_interval=-1)

