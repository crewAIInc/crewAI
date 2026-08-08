"""Tests for the VoyageAI embedding function."""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    VoyageAIEmbeddingFunction,
)


@contextmanager
def mock_voyageai_client():
    """Inject a fake ``voyageai`` module so tests run without the optional dependency.

    ``VoyageAIEmbeddingFunction`` imports ``voyageai`` lazily, so patching
    ``voyageai.Client`` directly fails when the package is not installed (e.g. in
    CI). Registering a mock module in ``sys.modules`` makes the lazy import resolve
    to the mock regardless of whether the real package is present.
    """
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.Client.return_value = mock_client
    with patch.dict(sys.modules, {"voyageai": mock_module}):
        yield mock_client


class TestVoyageAIEmbeddingFunction:
    """Test the VoyageAI embedding function call routing."""

    def test_standard_model_uses_embed(self):
        """Standard models should call the regular embed endpoint."""
        with mock_voyageai_client() as mock_client:
            mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])

            fn = VoyageAIEmbeddingFunction(api_key="voyage-key", model="voyage-2")
            result = fn(["aa", "bb"])

            mock_client.embed.assert_called_once()
            mock_client.contextualized_embed.assert_not_called()
            assert np.allclose(result, [[0.1, 0.2]])

    def test_contextualized_model_uses_contextualized_embed(self):
        """voyage-context-4 should call the contextualized embeddings endpoint."""
        with mock_voyageai_client() as mock_client:
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

    def test_contextualized_call_wraps_inputs_as_list_of_lists(self):
        """Each input string is wrapped as its own single-chunk document (List[List[str]])."""
        with mock_voyageai_client() as mock_client:
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[MagicMock(embeddings=[[0.1, 0.2]])]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key", model="voyage-context-4"
            )
            fn(["aa"])

            _, kwargs = mock_client.contextualized_embed.call_args
            # Each string is wrapped as its own single-chunk document
            assert kwargs["inputs"] == [["aa"]]
            # chunk_size and enable_auto_chunking must NOT be passed
            assert "chunk_size" not in kwargs
            assert "enable_auto_chunking" not in kwargs

    def test_contextualized_input_is_list_of_lists(self):
        """Input must be passed as List[List[str]], each inner list is one document with its chunks."""
        with mock_voyageai_client() as mock_client:
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
            assert kwargs["inputs"] == [["aa"], ["bb"]]

    def test_contextualized_defaults_input_type_to_document(self):
        """Without a configured input_type, contextualized calls default to 'document'."""
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
            assert kwargs["input_type"] == "document"

    def test_contextualized_model_forwards_input_type(self):
        """A configured input_type is forwarded to the contextualized call."""
        with patch("voyageai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.contextualized_embed.return_value = MagicMock(
                results=[MagicMock(embeddings=[[0.1, 0.2]])]
            )

            fn = VoyageAIEmbeddingFunction(
                api_key="voyage-key",
                model="voyage-context-4",
                input_type="query",
            )
            fn(["aa"])

            _, kwargs = mock_client.contextualized_embed.call_args
            assert kwargs["input_type"] == "query"

    def test_contextualized_string_input_normalized_with_wrapping(self):
        """A single string input is normalized and wrapped as a single-chunk document."""
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
            assert kwargs["inputs"] == [["aa"]]
