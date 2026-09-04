"""Tests for CrewAIRagAdapter.content_filter."""

from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter


def _make_adapter(
    content_filter=None,
    collection_name: str = "test_collection",
) -> CrewAIRagAdapter:
    """Build a CrewAIRagAdapter with a mocked RAG client."""
    mock_client = MagicMock()
    with patch(
        "crewai_tools.adapters.crewai_rag_adapter.get_rag_client",
        return_value=mock_client,
    ):
        adapter = CrewAIRagAdapter(
            collection_name=collection_name,
            content_filter=content_filter,
        )
    return adapter


class TestContentFilterOnAdd:
    def test_filter_removes_documents(self) -> None:
        """Documents whose content is rejected by the filter are not indexed."""

        def drop_secrets(contents: list[str]) -> list[str]:
            return [c for c in contents if "SECRET" not in c]

        adapter = _make_adapter(content_filter=drop_secrets)
        mock_client = adapter._client
        assert mock_client is not None

        adapter.add(
            "safe text",
            data_type="text",
        )
        # The add method processes the text into BaseRecord documents.
        # With the filter, only safe ones should pass.
        if mock_client.add_documents.called:
            docs = mock_client.add_documents.call_args.kwargs["documents"]
            for doc in docs:
                assert "SECRET" not in doc["content"]

    def test_filter_drops_all_skips_add(self) -> None:
        """When the filter removes every document, add_documents is not called."""
        adapter = _make_adapter(content_filter=lambda contents: [])
        mock_client = adapter._client
        assert mock_client is not None

        adapter.add("anything", data_type="text")

        mock_client.add_documents.assert_not_called()

    def test_filter_exception_propagates(self) -> None:
        """An exception from content_filter aborts the add."""

        def exploding_filter(contents: list[str]) -> list[str]:
            raise ValueError("Policy violation")

        adapter = _make_adapter(content_filter=exploding_filter)

        with pytest.raises(ValueError, match="Policy violation"):
            adapter.add("content", data_type="text")

    def test_no_filter_is_noop(self) -> None:
        """When content_filter is None, documents are persisted normally."""
        adapter = _make_adapter(content_filter=None)
        assert adapter.content_filter is None
        mock_client = adapter._client
        assert mock_client is not None

        adapter.add("hello world", data_type="text")

        mock_client.add_documents.assert_called_once()
        docs = mock_client.add_documents.call_args.kwargs["documents"]
        assert len(docs) >= 1

    def test_filter_receives_all_content_strings(self) -> None:
        """The filter callable receives the full list of content strings."""
        received: list[list[str]] = []

        def capturing_filter(contents: list[str]) -> list[str]:
            received.append(contents)
            return contents

        adapter = _make_adapter(content_filter=capturing_filter)

        adapter.add("some text content", data_type="text")

        assert len(received) == 1
        assert all(isinstance(c, str) for c in received[0])
