"""Tests for MergeAgentHandlerTool."""

import os
from unittest.mock import Mock, patch

import pytest

from crewai_tools import MergeAgentHandlerTool


@pytest.fixture(autouse=True)
def mock_agent_handler_api_key():
    """Mock the Agent Handler API key environment variable."""
    with patch.dict(os.environ, {"AGENT_HANDLER_API_KEY": "test_key"}):
        yield


@pytest.fixture
def mock_tool_pack_response():
    """Mock response for tools/list MCP request."""
    return {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {
            "tools": [
                {
                    "name": "linear__create_issue",
                    "description": "Creates a new issue in Linear",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The issue title",
                            },
                            "description": {
                                "type": "string",
                                "description": "The issue description",
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority level (1-4)",
                            },
                        },
                        "required": ["title"],
                    },
                },
                {
                    "name": "linear__get_issues",
                    "description": "Get issues from Linear",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter": {
                                "type": "object",
                                "description": "Filter criteria",
                            }
                        },
                    },
                },
            ]
        },
    }


@pytest.fixture
def mock_tool_execute_response():
    """Mock response for tools/call MCP request."""
    return {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": '{"success": true, "id": "ISS-123", "title": "Test Issue"}',
                }
            ]
        },
    }


def test_tool_initialization():
    """Test basic tool initialization."""
    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    assert tool.name == "test_tool"
    assert "Test tool" in tool.description  # Description gets formatted by BaseTool
    assert tool.tool_pack_id == "test-pack-id"
    assert tool.registered_user_id == "test-user-id"
    assert tool.tool_name == "linear__create_issue"
    assert tool.base_url == "https://ah-api.merge.dev"
    assert tool.session_id is not None


def test_tool_initialization_with_custom_base_url():
    """Test tool initialization with custom base URL."""
    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
        base_url="http://localhost:8000",
    )

    assert tool.base_url == "http://localhost:8000"


def test_missing_api_key():
    """Test that missing API key raises appropriate error."""
    with patch.dict(os.environ, {}, clear=True):
        tool = MergeAgentHandlerTool(
            name="test_tool",
            description="Test tool",
            tool_pack_id="test-pack-id",
            registered_user_id="test-user-id",
            tool_name="linear__create_issue",
        )

        with pytest.raises(Exception) as exc_info:
            tool._get_api_key()

        assert "AGENT_HANDLER_API_KEY" in str(exc_info.value)


@patch("requests.post")
def test_mcp_request_success(mock_post, mock_tool_pack_response):
    """Test successful MCP request."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    result = tool._make_mcp_request(method="tools/list")

    assert "result" in result
    assert "tools" in result["result"]
    assert len(result["result"]["tools"]) == 2


@patch("requests.post")
def test_mcp_request_error(mock_post):
    """Test MCP request with error response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "error": {"code": -32601, "message": "Method not found"},
    }
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    with pytest.raises(Exception) as exc_info:
        tool._make_mcp_request(method="invalid/method")

    assert "Method not found" in str(exc_info.value)


@patch("requests.post")
def test_mcp_request_http_error(mock_post):
    """Test MCP request with HTTP error."""
    mock_post.side_effect = Exception("Connection error")

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    with pytest.raises(Exception) as exc_info:
        tool._make_mcp_request(method="tools/list")

    assert "Connection error" in str(exc_info.value)


@patch("requests.post")
def test_tool_execution(mock_post, mock_tool_execute_response):
    """Test tool execution via _run method."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_execute_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    result = tool._run(title="Test Issue", description="Test description")

    assert result["success"] is True
    assert result["id"] == "ISS-123"
    assert result["title"] == "Test Issue"


@patch("requests.post")
def test_from_tool_name(mock_post, mock_tool_pack_response):
    """Test creating tool from tool name."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool.from_tool_name(
        tool_name="linear__create_issue",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
    )

    assert tool.name == "linear__create_issue"
    assert tool.description == "Creates a new issue in Linear"
    assert tool.tool_name == "linear__create_issue"


@patch("requests.post")
def test_from_tool_name_with_custom_base_url(mock_post, mock_tool_pack_response):
    """Test creating tool from tool name with custom base URL."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool.from_tool_name(
        tool_name="linear__create_issue",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        base_url="http://localhost:8000",
    )

    assert tool.base_url == "http://localhost:8000"


@patch("requests.post")
def test_from_tool_pack_all_tools(mock_post, mock_tool_pack_response):
    """Test creating all tools from a Tool Pack."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tools = MergeAgentHandlerTool.from_tool_pack(
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
    )

    assert len(tools) == 2
    assert tools[0].name == "linear__create_issue"
    assert tools[1].name == "linear__get_issues"


@patch("requests.post")
def test_from_tool_pack_specific_tools(mock_post, mock_tool_pack_response):
    """Test creating specific tools from a Tool Pack."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tools = MergeAgentHandlerTool.from_tool_pack(
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_names=["linear__create_issue"],
    )

    assert len(tools) == 1
    assert tools[0].name == "linear__create_issue"


@patch("requests.post")
def test_from_tool_pack_with_custom_base_url(mock_post, mock_tool_pack_response):
    """Test creating tools from Tool Pack with custom base URL."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tools = MergeAgentHandlerTool.from_tool_pack(
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        base_url="http://localhost:8000",
    )

    assert len(tools) == 2
    assert all(tool.base_url == "http://localhost:8000" for tool in tools)


@patch("requests.post")
def test_tool_execution_with_text_response(mock_post):
    """Test tool execution with plain text response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"content": [{"type": "text", "text": "Plain text result"}]},
    }
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    result = tool._run(title="Test")

    assert result == "Plain text result"


@patch("requests.post")
def test_mcp_request_builds_correct_url(mock_post, mock_tool_pack_response):
    """Test that MCP request builds correct URL."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-123",
        registered_user_id="user-456",
        tool_name="linear__create_issue",
        base_url="https://ah-api.merge.dev",
    )

    tool._make_mcp_request(method="tools/list")

    expected_url = (
        "https://ah-api.merge.dev/api/v1/tool-packs/"
        "test-pack-123/registered-users/user-456/mcp"
    )
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == expected_url


@patch("requests.post")
def test_mcp_request_includes_correct_headers(mock_post, mock_tool_pack_response):
    """Test that MCP request includes correct headers."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_tool_pack_response
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__create_issue",
    )

    tool._make_mcp_request(method="tools/list")

    mock_post.assert_called_once()
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test_key"
    assert "Mcp-Session-Id" in headers


@patch("requests.post")
def test_tool_parameters_are_passed_in_request(mock_post):
    """Test that tool parameters are correctly included in the MCP request."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"content": [{"type": "text", "text": '{"success": true}'}]},
    }
    mock_post.return_value = mock_response

    tool = MergeAgentHandlerTool(
        name="test_tool",
        description="Test tool",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
        tool_name="linear__update_issue",
    )

    # Execute tool with specific parameters
    tool._run(id="issue-123", title="New Title", priority=1)

    # Verify the request was made
    mock_post.assert_called_once()

    # Get the JSON payload that was sent
    payload = mock_post.call_args.kwargs["json"]

    # Verify MCP structure
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "tools/call"
    assert "id" in payload

    # Verify parameters are in the request
    assert "params" in payload
    assert payload["params"]["name"] == "linear__update_issue"
    assert "arguments" in payload["params"]

    # Verify the actual arguments were passed
    arguments = payload["params"]["arguments"]
    assert arguments["id"] == "issue-123"
    assert arguments["title"] == "New Title"
    assert arguments["priority"] == 1


@patch("requests.post")
def test_tool_run_method_passes_parameters(mock_post, mock_tool_pack_response):
    """Test that parameters are passed when using the .run() method (how CrewAI calls it)."""
    # Mock the tools/list response
    mock_response = Mock()
    mock_response.status_code = 200

    # First call: tools/list
    # Second call: tools/call
    mock_response.json.side_effect = [
        mock_tool_pack_response,  # tools/list response
        {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"content": [{"type": "text", "text": '{"success": true, "id": "issue-123"}'}]},
        },  # tools/call response
    ]
    mock_post.return_value = mock_response

    # Create tool using from_tool_name (which fetches schema)
    tool = MergeAgentHandlerTool.from_tool_name(
        tool_name="linear__create_issue",
        tool_pack_id="test-pack-id",
        registered_user_id="test-user-id",
    )

    # Call using .run() method (this is how CrewAI invokes tools)
    result = tool.run(title="Test Issue", description="Test description", priority=2)

    # Verify two calls were made: tools/list and tools/call
    assert mock_post.call_count == 2

    # Get the second call (tools/call)
    second_call = mock_post.call_args_list[1]
    payload = second_call.kwargs["json"]

    # Verify it's a tools/call request
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "linear__create_issue"

    # Verify parameters were passed
    arguments = payload["params"]["arguments"]
    assert arguments["title"] == "Test Issue"
    assert arguments["description"] == "Test description"
    assert arguments["priority"] == 2

    # Verify result was returned
    assert result["success"] is True
    assert result["id"] == "issue-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
