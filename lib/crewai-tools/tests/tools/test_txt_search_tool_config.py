from unittest.mock import MagicMock, Mock, patch

from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
from crewai_tools.tools.txt_search_tool.txt_search_tool import TXTSearchTool


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_txt_search_tool_with_azure_config_without_env_vars(
    mock_create_client: Mock,
) -> None:
    """Test TXTSearchTool accepts Azure config without requiring env vars."""
    mock_embedding_func = MagicMock()
    mock_embedding_func.return_value = [[0.1] * 1536]

    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    with patch(
        "crewai_tools.tools.rag.rag_tool.build_embedder",
        return_value=mock_embedding_func,
    ):
        config = {
            "embedding_model": {
                "provider": "azure",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": "test-api-key",
                    "api_base": "https://test.openai.azure.com/",
                    "api_version": "2024-02-01",
                    "api_type": "azure",
                    "deployment_id": "test-deployment",
                },
            }
        }

        # This should not raise a validation error about missing env vars
        tool = TXTSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)
        assert tool.name == "Search a txt's content"


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_txt_search_tool_with_openai_config_without_env_vars(
    mock_create_client: Mock,
) -> None:
    """Test TXTSearchTool accepts OpenAI config without requiring env vars."""
    mock_embedding_func = MagicMock()
    mock_embedding_func.return_value = [[0.1] * 1536]

    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    with patch(
        "crewai_tools.tools.rag.rag_tool.build_embedder",
        return_value=mock_embedding_func,
    ):
        config = {
            "embedding_model": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": "sk-test123",
                },
            }
        }

        tool = TXTSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_txt_search_tool_with_cohere_config(mock_create_client: Mock) -> None:
    """Test TXTSearchTool with Cohere embedding provider."""
    mock_embedding_func = MagicMock()
    mock_embedding_func.return_value = [[0.1] * 1024]

    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    with patch(
        "crewai_tools.tools.rag.rag_tool.build_embedder",
        return_value=mock_embedding_func,
    ):
        config = {
            "embedding_model": {
                "provider": "cohere",
                "config": {
                    "model": "embed-english-v3.0",
                    "api_key": "test-cohere-key",
                },
            }
        }

        tool = TXTSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)