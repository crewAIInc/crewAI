"""Tests for RAGStorage custom path functionality."""

from unittest.mock import MagicMock, patch

from crewai.memory.storage.rag_storage import RAGStorage


@patch("crewai.memory.storage.rag_storage.create_client")
@patch("crewai.memory.storage.rag_storage.build_embedder")
def test_rag_storage_custom_path(
    mock_build_embedder: MagicMock,
    mock_create_client: MagicMock,
) -> None:
    """Test RAGStorage uses custom path when provided."""
    mock_build_embedder.return_value = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_create_client.return_value = MagicMock()

    custom_path = "/custom/memory/path"
    embedder_config = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}

    RAGStorage(
        type="short_term",
        crew=None,
        path=custom_path,
        embedder_config=embedder_config,
    )

    mock_create_client.assert_called_once()
    config_arg = mock_create_client.call_args[0][0]
    assert config_arg.settings.persist_directory == custom_path


@patch("crewai.memory.storage.rag_storage.create_client")
@patch("crewai.memory.storage.rag_storage.build_embedder")
def test_rag_storage_default_path_when_none(
    mock_build_embedder: MagicMock,
    mock_create_client: MagicMock,
) -> None:
    """Test RAGStorage uses default path when no custom path is provided."""
    mock_build_embedder.return_value = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_create_client.return_value = MagicMock()

    embedder_config = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}

    storage = RAGStorage(
        type="short_term",
        crew=None,
        path=None,
        embedder_config=embedder_config,
    )

    mock_create_client.assert_called_once()
    assert storage.path is None


@patch("crewai.memory.storage.rag_storage.create_client")
@patch("crewai.memory.storage.rag_storage.build_embedder")
def test_rag_storage_custom_path_with_batch_size(
    mock_build_embedder: MagicMock,
    mock_create_client: MagicMock,
) -> None:
    """Test RAGStorage uses custom path with batch_size in config."""
    mock_build_embedder.return_value = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_create_client.return_value = MagicMock()

    custom_path = "/custom/batch/path"
    embedder_config = {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "batch_size": 100},
    }

    RAGStorage(
        type="long_term",
        crew=None,
        path=custom_path,
        embedder_config=embedder_config,
    )

    mock_create_client.assert_called_once()
    config_arg = mock_create_client.call_args[0][0]
    assert config_arg.settings.persist_directory == custom_path
    assert config_arg.batch_size == 100