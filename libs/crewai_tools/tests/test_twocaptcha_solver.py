import os
import pytest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.twocaptcha_solver.twocaptcha_solver import TwoCaptchaSolverTool

def test_tool_initialization():
    """Test tool initialization with API key."""
    tool = TwoCaptchaSolverTool(api_key="test_api_key")
    assert tool.name == "twocaptcha_solver"
    assert tool.api_key == "test_api_key"

@patch("requests.post")
def test_tool_execution_success(mock_post):
    """Test successful captcha solving workflow."""
    # Mock for createTask
    mock_create = MagicMock()
    mock_create.json.return_value = {"errorId": 0, "taskId": 999}
    mock_create.status_code = 200
    
    # Mock for getTaskResult
    mock_result = MagicMock()
    mock_result.json.return_value = {
        "status": "ready", 
        "solution": {"token": "solved_token_xyz"}
    }
    mock_result.status_code = 200
    
    mock_post.side_effect = [mock_create, mock_result]
    
    tool = TwoCaptchaSolverTool(api_key="test_key")
    result = tool._run(website_url="http://site.com", sitekey="abc-123")
    
    assert result == "solved_token_xyz"

def test_tool_missing_api_key():
    """Test error message when API key is missing."""
    # We use patch.dict to temporarily clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        tool = TwoCaptchaSolverTool()
        result = tool._run(website_url="http://site.com", sitekey="abc-123")
        assert "API key not found" in result
