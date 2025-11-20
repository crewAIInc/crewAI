import os
import sys
from unittest.mock import MagicMock, patch, Mock

import pytest

from crewai_tools.tools.ai_mind_tool.ai_mind_tool import AIMindTool


@pytest.fixture(autouse=True)
def mock_minds_api_key():
    with patch.dict(os.environ, {"MINDS_API_KEY": "test_key"}):
        yield


@pytest.fixture
def mock_minds_sdk():
    """Mock the minds_sdk package to avoid requiring it to be installed."""
    mock_minds_module = MagicMock()
    mock_client_module = MagicMock()
    
    mock_client_class = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    
    mock_datasources = MagicMock()
    mock_client_instance.datasources = mock_datasources
    
    mock_minds = MagicMock()
    mock_client_instance.minds = mock_minds
    
    mock_mind = MagicMock()
    mock_mind.name = "test_mind_name"
    mock_minds.create.return_value = mock_mind
    
    mock_client_module.Client = mock_client_class
    mock_minds_module.client = mock_client_module
    
    with patch.dict(sys.modules, {"minds": mock_minds_module, "minds.client": mock_client_module}):
        yield mock_client_instance


def test_aimind_tool_imports_correctly_with_new_api(mock_minds_sdk):
    """Test that AIMindTool can be initialized without DatabaseConfig import error."""
    datasources = [
        {
            "description": "test database",
            "engine": "postgres",
            "connection_data": {
                "user": "test_user",
                "password": "test_pass",
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
            },
            "tables": ["test_table"],
        }
    ]
    
    tool = AIMindTool(api_key="test_key", datasources=datasources)
    
    assert tool.api_key == "test_key"
    assert tool.mind_name == "test_mind_name"


def test_aimind_tool_creates_datasources_with_new_api(mock_minds_sdk):
    """Test that AIMindTool creates datasources using the new minds_sdk API."""
    datasources = [
        {
            "description": "test database",
            "engine": "postgres",
            "connection_data": {
                "user": "test_user",
                "password": "test_pass",
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
            },
        }
    ]
    
    tool = AIMindTool(api_key="test_key", datasources=datasources)
    
    mock_minds_sdk.datasources.create.assert_called_once()
    call_args = mock_minds_sdk.datasources.create.call_args
    
    assert call_args.kwargs["engine"] == "postgres"
    assert call_args.kwargs["description"] == "test database"
    assert call_args.kwargs["connection_data"]["user"] == "test_user"
    assert call_args.kwargs["replace"] is True


def test_aimind_tool_handles_missing_optional_fields(mock_minds_sdk):
    """Test that AIMindTool handles missing optional fields in datasource config."""
    datasources = [
        {
            "engine": "postgres",
        }
    ]
    
    tool = AIMindTool(api_key="test_key", datasources=datasources)
    
    mock_minds_sdk.datasources.create.assert_called_once()
    call_args = mock_minds_sdk.datasources.create.call_args
    
    assert call_args.kwargs["engine"] == "postgres"
    assert call_args.kwargs["description"] == ""
    assert call_args.kwargs["connection_data"] == {}


def test_aimind_tool_creates_mind_with_datasource_names(mock_minds_sdk):
    """Test that AIMindTool creates mind with datasource names instead of objects."""
    datasources = [
        {
            "description": "test database 1",
            "engine": "postgres",
            "connection_data": {"user": "test_user1"},
        },
        {
            "description": "test database 2",
            "engine": "mysql",
            "connection_data": {"user": "test_user2"},
        },
    ]
    
    tool = AIMindTool(api_key="test_key", datasources=datasources)
    
    assert mock_minds_sdk.datasources.create.call_count == 2
    
    mock_minds_sdk.minds.create.assert_called_once()
    call_args = mock_minds_sdk.minds.create.call_args
    
    assert isinstance(call_args.kwargs["datasources"], list)
    assert len(call_args.kwargs["datasources"]) == 2
    assert all(isinstance(ds, str) for ds in call_args.kwargs["datasources"])
    assert call_args.kwargs["replace"] is True


def test_aimind_tool_raises_error_when_minds_sdk_not_installed():
    """Test that AIMindTool raises ImportError when minds_sdk is not installed."""
    with patch.dict(sys.modules, {"minds": None, "minds.client": None}):
        with pytest.raises(ImportError) as exc_info:
            AIMindTool(api_key="test_key", datasources=[])
        
        error_message = str(exc_info.value)
        assert "minds_sdk" in error_message or "pip install minds-sdk" in error_message


def test_aimind_tool_raises_error_when_api_key_missing():
    """Test that AIMindTool raises ValueError when API key is not provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            AIMindTool(datasources=[])
        
        assert "API key must be provided" in str(exc_info.value)


def test_aimind_tool_uses_env_var_for_api_key(mock_minds_sdk):
    """Test that AIMindTool uses MINDS_API_KEY environment variable."""
    with patch.dict(os.environ, {"MINDS_API_KEY": "env_test_key"}):
        tool = AIMindTool(datasources=[])
        
        assert tool.api_key == "env_test_key"


def test_aimind_tool_run_method(mock_minds_sdk):
    """Test that AIMindTool._run method works correctly."""
    from openai.types.chat import ChatCompletion
    
    datasources = [
        {
            "engine": "postgres",
            "description": "test db",
        }
    ]
    
    tool = AIMindTool(api_key="test_key", datasources=datasources)
    
    with patch("crewai_tools.tools.ai_mind_tool.ai_mind_tool.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock(spec=ChatCompletion)
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = tool._run("Test query")
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "test_mind_name"
        assert call_args.kwargs["messages"][0]["content"] == "Test query"


def test_aimind_tool_run_raises_error_when_mind_name_not_set():
    """Test that AIMindTool._run raises ValueError when mind_name is not set."""
    with patch("openai.OpenAI"):
        tool = AIMindTool.__new__(AIMindTool)
        object.__setattr__(tool, "api_key", "test_key")
        object.__setattr__(tool, "mind_name", None)
        
        with pytest.raises(ValueError) as exc_info:
            tool._run("Test query")
        
        assert "Mind name is not set" in str(exc_info.value)


def test_aimind_tool_run_raises_error_on_invalid_response():
    """Test that AIMindTool._run raises ValueError on invalid response."""
    with patch("crewai_tools.tools.ai_mind_tool.ai_mind_tool.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.return_value = "invalid_response"
        
        tool = AIMindTool.__new__(AIMindTool)
        object.__setattr__(tool, "api_key", "test_key")
        object.__setattr__(tool, "mind_name", "test_mind")
        
        with pytest.raises(ValueError) as exc_info:
            tool._run("Test query")
        
        assert "Invalid response from AI-Mind" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
