"""Tests for improved RAG tool validation error messages."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from crewai_tools.tools.rag.rag_tool import RagTool


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_azure_missing_deployment_id_gives_clear_error(mock_create_client: Mock) -> None:
    """Test that missing deployment_id for Azure gives a clear, focused error message."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    config = {
        "embedding_model": {
            "provider": "azure",
            "config": {
                "api_base": "http://localhost:4000/v1",
                "api_key": "test-key",
                "api_version": "2024-02-01",
            },
        }
    }

    with pytest.raises(ValueError) as exc_info:
        MyTool(config=config)

    error_msg = str(exc_info.value)
    assert "azure" in error_msg.lower()
    assert "deployment_id" in error_msg.lower()
    assert "bedrock" not in error_msg.lower()
    assert "cohere" not in error_msg.lower()
    assert "huggingface" not in error_msg.lower()


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_valid_azure_config_works(mock_create_client: Mock) -> None:
    """Test that valid Azure config works without errors."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    config = {
        "embedding_model": {
            "provider": "azure",
            "config": {
                "api_base": "http://localhost:4000/v1",
                "api_key": "test-key",
                "api_version": "2024-02-01",
                "deployment_id": "text-embedding-3-small",
            },
        }
    }

    tool = MyTool(config=config)
    assert tool is not None