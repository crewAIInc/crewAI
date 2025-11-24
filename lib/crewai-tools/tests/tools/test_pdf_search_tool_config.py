from unittest.mock import MagicMock, Mock, patch

from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
from crewai_tools.tools.pdf_search_tool.pdf_search_tool import PDFSearchTool


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_pdf_search_tool_with_azure_config_without_env_vars(
    mock_create_client: Mock,
) -> None:
    """Test PDFSearchTool accepts Azure config without requiring env vars.

    This verifies the fix for the reported issue where PDFSearchTool would
    throw a validation error:
        pydantic_core._pydantic_core.ValidationError: 1 validation error for PDFSearchTool
        EMBEDDINGS_OPENAI_API_KEY
          Field required [type=missing, input_value={}, input_type=dict]
    """
    mock_embedding_func = MagicMock()
    mock_embedding_func.return_value = [[0.1] * 1536]

    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_create_client.return_value = mock_client

    # Patch the embedding function builder to avoid actual API calls
    with patch(
        "crewai_tools.tools.rag.rag_tool.build_embedder",
        return_value=mock_embedding_func,
    ):
        # This is the exact config format from the bug report
        config = {
            "embedding_model": {
                "provider": "azure",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": "test-litellm-api-key",
                    "api_base": "https://test.litellm.proxy/",
                    "api_version": "2024-02-01",
                    "api_type": "azure",
                    "deployment_id": "test-deployment",
                },
            }
        }

        # This should not raise a validation error about missing env vars
        tool = PDFSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)
        assert tool.name == "Search a PDF's content"


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_pdf_search_tool_with_openai_config_without_env_vars(
    mock_create_client: Mock,
) -> None:
    """Test PDFSearchTool accepts OpenAI config without requiring env vars."""
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

        tool = PDFSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)


@patch("crewai_tools.adapters.crewai_rag_adapter.create_client")
def test_pdf_search_tool_with_vectordb_and_embedding_config(
    mock_create_client: Mock,
) -> None:
    """Test PDFSearchTool with both vector DB and embedding config."""
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
            "vectordb": {"provider": "chromadb", "config": {}},
            "embedding_model": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "api_key": "sk-test-key",
                },
            },
        }

        tool = PDFSearchTool(config=config)

        assert tool.adapter is not None
        assert isinstance(tool.adapter, CrewAIRagAdapter)