import os
import pytest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.cloro_dev_tool.cloro_dev_tool import CloroDevTool

@pytest.fixture(autouse=True)
def mock_cloro_api_key():
    with patch.dict(os.environ, {"CLORO_API_KEY": "test_key"}):
        yield

@patch("requests.post")
def test_cloro_tool_google_search(mock_post):
    tool = CloroDevTool(engine="google")
    mock_response = {
        "success": True,
        "result": {
            "organicResults": [
                {
                    "title": "Test Title",
                    "link": "http://test.com",
                    "snippet": "Test Snippet"
                }
            ],
            "aioverview": {"markdown": "**AI Overview**"}
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="test query")

    assert "organic" in result
    assert result["organic"][0]["title"] == "Test Title"
    assert "ai_overview" in result
    assert result["ai_overview"]["markdown"] == "**AI Overview**"
    
    # Check payload
    called_payload = mock_post.call_args.kwargs["json"]
    assert "query" in called_payload
    assert called_payload["query"] == "test query"
    assert "include" in called_payload
    assert called_payload["include"].get("aioverview", {}).get("markdown") is True


@patch("requests.post")
def test_cloro_tool_chatgpt_query(mock_post):
    tool = CloroDevTool(engine="chatgpt")
    mock_response = {
        "success": True,
        "result": {
            "text": "ChatGPT response",
            "markdown": "**ChatGPT response**"
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="test prompt")

    assert "text" in result
    assert result["text"] == "ChatGPT response"
    
    # Check payload
    called_payload = mock_post.call_args.kwargs["json"]
    assert "prompt" in called_payload
    assert called_payload["prompt"] == "test prompt"


@patch("requests.post")
def test_cloro_tool_gemini_query(mock_post):
    tool = CloroDevTool(engine="gemini")
    mock_response = {
        "success": True,
        "result": {
            "text": "Gemini response",
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="gemini prompt")

    assert "text" in result
    assert result["text"] == "Gemini response"


@patch("requests.post")
def test_cloro_tool_copilot_query(mock_post):
    tool = CloroDevTool(engine="copilot")
    mock_response = {
        "success": True,
        "result": {
            "text": "Copilot response",
            "sources": [{"title": "Source 1", "link": "http://source1.com"}]
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="copilot prompt")

    assert "text" in result
    assert "sources" in result
    assert result["sources"][0]["title"] == "Source 1"


@patch("requests.post")
def test_cloro_tool_perplexity_query(mock_post):
    tool = CloroDevTool(engine="perplexity")
    mock_response = {
        "success": True,
        "result": {
            "text": "Perplexity response",
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="perplexity prompt")

    assert "text" in result


@patch("requests.post")
def test_cloro_tool_aimode_query(mock_post):
    tool = CloroDevTool(engine="aimode")
    mock_response = {
        "success": True,
        "result": {
            "text": "AI Mode response"
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="aimode prompt")

    assert "text" in result


@patch("requests.post")
def test_api_error_handling(mock_post):
    tool = CloroDevTool()
    mock_post.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        tool.run(search_query="test")
    assert "API Error" in str(exc_info.value)

@patch("requests.post")
def test_unsuccessful_response(mock_post):
    tool = CloroDevTool()
    mock_response = {"success": False}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    
    with pytest.raises(ValueError) as exc_info:
        tool.run(search_query="test")
    assert "cloro API returned unsuccessful response" in str(exc_info.value)

def test_save_file():
    tool = CloroDevTool(save_file=True)
    
    with patch("requests.post") as mock_post, \
         patch("builtins.open", new_callable=MagicMock) as mock_open:
        
        mock_response = {
            "success": True,
            "result": {"organicResults": []}
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.status_code = 200
        
        tool.run(search_query="test")
        
        # Verify open was called
        mock_open.assert_called()
        
        # Verify write was called on the file handle
        # open() returns a context manager, __enter__ returns the file handle
        mock_file_handle = mock_open.return_value.__enter__.return_value
        mock_file_handle.write.assert_called()