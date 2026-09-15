from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.antibrow_load_tool.antibrow_load_tool import AntibrowLoadTool


@pytest.fixture
def mock_launch():
    """Patch the SDK entry point so no browser is started."""
    page = MagicMock()
    page.locator.return_value.first.inner_text.return_value = "Example Domain"
    browser = MagicMock()
    browser.new_page.return_value = page
    with patch("antibrow.launch", return_value=browser) as launch:
        yield {"launch": launch, "browser": browser, "page": page}


@pytest.fixture
def tool(mock_launch):
    return AntibrowLoadTool(api_key="test_key", profile="test-profile")


def test_tool_metadata(tool):
    assert tool.name == "AntiBrow web load tool"
    assert set(tool.args_schema.model_json_schema()["properties"]) == {"url", "selector"}
    assert tool.package_dependencies == ["antibrow"]
    assert [var.name for var in tool.env_vars] == ["ANTIBROW_API_KEY"]


def test_api_key_falls_back_to_the_environment(mock_launch):
    with patch.dict("os.environ", {"ANTIBROW_API_KEY": "from_env"}):
        assert AntibrowLoadTool().api_key == "from_env"


def test_run_returns_page_text(tool, mock_launch):
    assert tool.run(url="https://example.com") == "Example Domain"
    mock_launch["page"].goto.assert_called_once_with("https://example.com", wait_until="load")
    mock_launch["page"].locator.assert_called_with("body")


def test_selector_is_used_when_given(tool, mock_launch):
    tool.run(url="https://example.com", selector="h1")
    mock_launch["page"].locator.assert_called_with("h1")


def test_the_profile_is_launched_once_when_kept_open(tool, mock_launch):
    tool.run(url="https://example.com")
    tool.run(url="https://example.com/other")
    assert mock_launch["launch"].call_count == 1
    _, kwargs = mock_launch["launch"].call_args
    assert kwargs["api_key"] == "test_key"
    assert kwargs["focus_window"] is False


def test_keep_open_false_closes_after_every_call(mock_launch):
    tool = AntibrowLoadTool(api_key="k", keep_open=False)
    tool.run(url="https://example.com")
    tool.run(url="https://example.com/other")
    assert mock_launch["launch"].call_count == 2
    assert mock_launch["browser"].close.call_count == 2


def test_content_is_truncated(mock_launch):
    mock_launch["page"].locator.return_value.first.inner_text.return_value = "x" * 50
    tool = AntibrowLoadTool(api_key="k", max_content_length=10)
    assert tool.run(url="https://example.com") == "x" * 10 + "\n\n[content truncated]"


def test_close_is_idempotent(tool, mock_launch):
    tool.run(url="https://example.com")
    tool.close()
    tool.close()
    assert mock_launch["browser"].close.call_count == 1
    assert tool.browser is None and tool.page is None


def test_a_bad_url_is_rejected_before_launching(tool, mock_launch):
    with pytest.raises(Exception):
        tool.run(url="file:///etc/passwd")
    assert mock_launch["launch"].call_count == 0
