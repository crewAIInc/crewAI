"""Tests for the VoyageAI embedding function."""

from unittest.mock import MagicMock, patch

import numpy as np

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    CONTEXTUALIZED_CHUNK_SIZE,
    VoyageAIEmbeddingFunction,
)


class TestVoyageAIEmbeddingFunction:
    """Test the VoyageAI embedding function call routing."""

    def test_standard_model_uses_embed(self):
        """Standard models should call the regular embed endpoint."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])

            fn = VoyageAIEmbeddingFunction(api_key="voyage-key", model="voyage-2")
            result = fn(["aa", "bb"])

            mock_client.embed.assert_called_once()
            mock_client.contextualized_embed.assert_not_called()
            assert np.allclose(result, [[0.1, 0.2]])

    def test_contextualized_model_uses_contextualized_embed(self):
        """voyage-context-4 should call the contextualized embeddings endpoint."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[
                    MagicMock(embeddings=[[0.1, 0.2]]),
                    MagicMock(embeddings=[[0.3, 0.4]]),
                ]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key", model="voyage-context-4"
            )
            result = fn(["aa", "bb"])

            mock_client.embed.assert_not_called()
            mock_client.contextualized_embed.assert_called_once()
            assert np.allclose(result, [[0.1, 0.2], [0.3, 0.4]])

    def test_contextualized_call_sets_chunk_size_to_max(self):
        """chunk_size must be set to 32000 on every contextualized call."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[MagicMock(embeddings=[[0.1, 0.2]])]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key", model="voyage-context-4"
            )
            fn(["aa"])

            _, kwargs = mock_client.contextualized_embed.call_args
            assert kwargs["chunk_size"] == CONTEXTUALIZED_CHUNK_SIZE
            assert CONTEXTUALIZED_CHUNK_SIZE == 32000

    def test_contextualized_input_is_flat_list(self):
        """Input must be passed as a flat List[str], not wrapped in an extra list."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[
                    MagicMock(embeddings=[[0.1, 0.2]]),
                    MagicMock(embeddings=[[0.3, 0.4]]),
                ]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key", model="voyage-context-4"
            )
            fn(["aa", "bb"])

            _, kwargs = mock_client.contextualized_embed.call_args
            assert kwargs["inputs"] == ["aa", "bb"]

    def test_contextualized_string_input_normalized_to_flat_list(self):
        """A single string input is normalized to a flat list of one string."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[MagicMock(embeddings=[[0.1, 0.2]])]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key", model="voyage-context-4"
            )
            fn("aa")

            _, kwargs = mock_client.contextualized_embed.call_args
            assert kwargs["inputs"] == ["aa"]
