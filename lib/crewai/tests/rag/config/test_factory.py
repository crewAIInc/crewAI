"""Tests for RAG config factory."""

from unittest.mock import Mock, patch

import pytest
from crewai.rag.factory import create_client


def test_create_client_chromadb():
    """Test ChromaDB client creation."""
    mock_config = Mock()
    mock_config.provider = "chromadb"

    with patch("crewai.rag.factory.require") as mock_require:
        mock_module = Mock()
        mock_client = Mock()
        mock_module.create_client.return_value = mock_client
        mock_require.return_value = mock_module

        result = create_client(mock_config)

        assert result == mock_client
        mock_require.assert_called_once_with(
            "crewai.rag.chromadb.factory", purpose="The 'chromadb' provider"
        )
        mock_module.create_client.assert_called_once_with(mock_config)


def test_create_client_unsupported_provider():
    """Test unsupported provider raises ValueError."""
    mock_config = Mock()
    mock_config.provider = "unsupported"

    with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
        create_client(mock_config)
