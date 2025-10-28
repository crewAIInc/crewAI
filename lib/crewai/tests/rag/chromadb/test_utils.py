"""Tests for ChromaDB utility functions."""

from crewai.rag.chromadb.types import PreparedDocuments
from crewai.rag.chromadb.utils import (
    MAX_COLLECTION_LENGTH,
    MIN_COLLECTION_LENGTH,
    _create_batch_slice,
    _is_ipv4_pattern,
    _prepare_documents_for_chromadb,
    _sanitize_collection_name,
)
from crewai.rag.types import BaseRecord


class TestChromaDBUtils:
    """Test suite for ChromaDB utility functions."""

    def test_sanitize_collection_name_long_name(self) -> None:
        """Test sanitizing a very long collection name."""
        long_name = "This is an extremely long role name that will definitely exceed the ChromaDB collection name limit of 63 characters and cause an error when used as a collection name"
        sanitized = _sanitize_collection_name(long_name)
        assert len(sanitized) <= MAX_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_sanitize_collection_name_special_chars(self) -> None:
        """Test sanitizing a name with special characters."""
        special_chars = "Agent@123!#$%^&*()"
        sanitized = _sanitize_collection_name(special_chars)
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_sanitize_collection_name_short_name(self) -> None:
        """Test sanitizing a very short name."""
        short_name = "A"
        sanitized = _sanitize_collection_name(short_name)
        assert len(sanitized) >= MIN_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_bad_ends(self) -> None:
        """Test sanitizing a name with non-alphanumeric start/end."""
        bad_ends = "_Agent_"
        sanitized = _sanitize_collection_name(bad_ends)
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_none(self) -> None:
        """Test sanitizing a None value."""
        sanitized = _sanitize_collection_name(None)
        assert sanitized == "default_collection"

    def test_sanitize_collection_name_ipv4_pattern(self) -> None:
        """Test sanitizing an IPv4 address."""
        ipv4 = "192.168.1.1"
        sanitized = _sanitize_collection_name(ipv4)
        assert sanitized.startswith("ip_")
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()
        assert all(c.isalnum() or c in ["_", "-"] for c in sanitized)

    def test_is_ipv4_pattern(self) -> None:
        """Test IPv4 pattern detection."""
        assert _is_ipv4_pattern("192.168.1.1") is True
        assert _is_ipv4_pattern("not.an.ip.address") is False

    def test_sanitize_collection_name_properties(self) -> None:
        """Test that sanitized collection names always meet ChromaDB requirements."""
        test_cases: list[str] = [
            "A" * 100,  # Very long name
            "_start_with_underscore",
            "end_with_underscore_",
            "contains@special#characters",
            "192.168.1.1",  # IPv4 address
            "a" * 2,  # Too short
        ]
        for test_case in test_cases:
            sanitized = _sanitize_collection_name(test_case)
            assert len(sanitized) >= MIN_COLLECTION_LENGTH
            assert len(sanitized) <= MAX_COLLECTION_LENGTH
            assert sanitized[0].isalnum()
            assert sanitized[-1].isalnum()

    def test_sanitize_collection_name_empty_string(self) -> None:
        """Test sanitizing an empty string."""
        sanitized = _sanitize_collection_name("")
        assert sanitized == "default_collection"

    def test_sanitize_collection_name_whitespace_only(self) -> None:
        """Test sanitizing a string with only whitespace."""
        sanitized = _sanitize_collection_name("   ")
        assert (
            sanitized == "a__z"
        )  # Spaces become underscores, padded to meet requirements
        assert len(sanitized) >= MIN_COLLECTION_LENGTH
        assert sanitized[0].isalnum()
        assert sanitized[-1].isalnum()


class TestPrepareDocumentsForChromaDB:
    """Test suite for _prepare_documents_for_chromadb function."""

    def test_prepare_documents_with_doc_ids(self) -> None:
        """Test preparing documents that already have doc_ids."""
        documents: list[BaseRecord] = [
            {
                "doc_id": "id1",
                "content": "First document",
                "metadata": {"source": "test1"},
            },
            {
                "doc_id": "id2",
                "content": "Second document",
                "metadata": {"source": "test2"},
            },
        ]

        result = _prepare_documents_for_chromadb(documents)

        assert result.ids == ["id1", "id2"]
        assert result.texts == ["First document", "Second document"]
        assert result.metadatas == [{"source": "test1"}, {"source": "test2"}]

    def test_prepare_documents_generate_ids(self) -> None:
        """Test preparing documents without doc_ids (should generate hashes)."""
        documents: list[BaseRecord] = [
            {"content": "Test content", "metadata": {"key": "value"}},
            {"content": "Another test"},
        ]

        result = _prepare_documents_for_chromadb(documents)

        assert len(result.ids) == 2
        assert all(len(doc_id) == 64 for doc_id in result.ids)
        assert result.texts == ["Test content", "Another test"]
        assert result.metadatas == [{"key": "value"}, {}]

    def test_prepare_documents_with_list_metadata(self) -> None:
        """Test preparing documents with list metadata (should take first item)."""
        documents: list[BaseRecord] = [
            {"content": "Test", "metadata": [{"first": "item"}, {"second": "item"}]},
            {"content": "Test2", "metadata": []},
        ]

        result = _prepare_documents_for_chromadb(documents)

        assert result.metadatas == [{"first": "item"}, {}]

    def test_prepare_documents_no_metadata(self) -> None:
        """Test preparing documents without metadata."""
        documents: list[BaseRecord] = [
            {"content": "Document 1"},
            {"content": "Document 2", "metadata": None},
        ]

        result = _prepare_documents_for_chromadb(documents)

        assert result.metadatas == [{}, {}]

    def test_prepare_documents_hash_consistency(self) -> None:
        """Test that identical content produces identical hashes."""
        documents1: list[BaseRecord] = [
            {"content": "Same content", "metadata": {"key": "value"}}
        ]
        documents2: list[BaseRecord] = [
            {"content": "Same content", "metadata": {"key": "value"}}
        ]

        result1 = _prepare_documents_for_chromadb(documents1)
        result2 = _prepare_documents_for_chromadb(documents2)

        assert result1.ids == result2.ids


class TestCreateBatchSlice:
    """Test suite for _create_batch_slice function."""

    def test_create_batch_slice_normal(self) -> None:
        """Test creating a normal batch slice."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3", "id4", "id5"],
            texts=["doc1", "doc2", "doc3", "doc4", "doc5"],
            metadatas=[{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=1, batch_size=3
        )

        assert batch_ids == ["id2", "id3", "id4"]
        assert batch_texts == ["doc2", "doc3", "doc4"]
        assert batch_metadatas == [{"b": 2}, {"c": 3}, {"d": 4}]

    def test_create_batch_slice_at_end(self) -> None:
        """Test creating a batch slice that goes beyond the end."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3"],
            texts=["doc1", "doc2", "doc3"],
            metadatas=[{"a": 1}, {"b": 2}, {"c": 3}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=2, batch_size=5
        )

        assert batch_ids == ["id3"]
        assert batch_texts == ["doc3"]
        assert batch_metadatas == [{"c": 3}]

    def test_create_batch_slice_empty_batch(self) -> None:
        """Test creating a batch slice starting beyond the data."""
        prepared = PreparedDocuments(
            ids=["id1", "id2"], texts=["doc1", "doc2"], metadatas=[{"a": 1}, {"b": 2}]
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=5, batch_size=3
        )

        assert batch_ids == []
        assert batch_texts == []
        assert batch_metadatas == []

    def test_create_batch_slice_no_metadatas(self) -> None:
        """Test creating a batch slice with no metadatas."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3"], texts=["doc1", "doc2", "doc3"], metadatas=[]
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=0, batch_size=2
        )

        assert batch_ids == ["id1", "id2"]
        assert batch_texts == ["doc1", "doc2"]
        assert batch_metadatas is None

    def test_create_batch_slice_all_empty_metadatas(self) -> None:
        """Test creating a batch slice where all metadatas are empty."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3"],
            texts=["doc1", "doc2", "doc3"],
            metadatas=[{}, {}, {}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=0, batch_size=3
        )

        assert batch_ids == ["id1", "id2", "id3"]
        assert batch_texts == ["doc1", "doc2", "doc3"]
        assert batch_metadatas is None

    def test_create_batch_slice_some_empty_metadatas(self) -> None:
        """Test creating a batch slice where some metadatas are empty."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3"],
            texts=["doc1", "doc2", "doc3"],
            metadatas=[{"a": 1}, {}, {"c": 3}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=0, batch_size=3
        )

        assert batch_ids == ["id1", "id2", "id3"]
        assert batch_texts == ["doc1", "doc2", "doc3"]
        assert batch_metadatas == [{"a": 1}, {}, {"c": 3}]

    def test_create_batch_slice_zero_start_index(self) -> None:
        """Test creating a batch slice starting from index 0."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3", "id4"],
            texts=["doc1", "doc2", "doc3", "doc4"],
            metadatas=[{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=0, batch_size=2
        )

        assert batch_ids == ["id1", "id2"]
        assert batch_texts == ["doc1", "doc2"]
        assert batch_metadatas == [{"a": 1}, {"b": 2}]

    def test_create_batch_slice_single_item(self) -> None:
        """Test creating a batch slice with batch size 1."""
        prepared = PreparedDocuments(
            ids=["id1", "id2", "id3"],
            texts=["doc1", "doc2", "doc3"],
            metadatas=[{"a": 1}, {"b": 2}, {"c": 3}],
        )

        batch_ids, batch_texts, batch_metadatas = _create_batch_slice(
            prepared, start_index=1, batch_size=1
        )

        assert batch_ids == ["id2"]
        assert batch_texts == ["doc2"]
        assert batch_metadatas == [{"b": 2}]
