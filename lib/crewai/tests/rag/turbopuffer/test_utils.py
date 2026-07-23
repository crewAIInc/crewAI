"""Tests for turbopuffer utility functions."""

from unittest.mock import MagicMock

import pytest
from crewai.rag.turbopuffer.utils import (
    _build_metadata_filter,
    _build_upsert_row,
    _normalize_turbopuffer_score,
    _process_search_results,
    _validate_namespace_name,
)
from crewai.rag.types import BaseRecord


class TestValidateNamespaceName:
    """Test suite for _validate_namespace_name."""

    def test_valid_simple_name(self):
        """Test valid simple alphanumeric name."""
        _validate_namespace_name("my-collection")

    def test_valid_name_with_dots(self):
        """Test valid name with dots."""
        _validate_namespace_name("my.collection.v2")

    def test_valid_name_with_underscores(self):
        """Test valid name with underscores."""
        _validate_namespace_name("my_collection_v2")

    def test_valid_single_char(self):
        """Test valid single character name."""
        _validate_namespace_name("a")

    def test_valid_max_length(self):
        """Test valid name at max length (128 chars)."""
        _validate_namespace_name("a" * 128)

    def test_invalid_empty(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            _validate_namespace_name("")

    def test_invalid_spaces(self):
        """Test that name with spaces raises error."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            _validate_namespace_name("my collection")

    def test_invalid_special_chars(self):
        """Test that name with special chars raises error."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            _validate_namespace_name("my!collection")

    def test_invalid_too_long(self):
        """Test that name exceeding 128 chars raises error."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            _validate_namespace_name("a" * 129)


class TestBuildUpsertRow:
    """Test suite for _build_upsert_row."""

    def test_basic_document(self):
        """Test building row from basic document."""
        doc: BaseRecord = {"content": "Hello world"}
        row = _build_upsert_row(doc, [0.1, 0.2, 0.3])

        assert row["content"] == "Hello world"
        assert row["vector"] == [0.1, 0.2, 0.3]
        assert "id" in row  # auto-generated UUID

    def test_document_with_doc_id(self):
        """Test building row preserves provided doc_id."""
        doc: BaseRecord = {"doc_id": "my-id", "content": "Hello world"}
        row = _build_upsert_row(doc, [0.1, 0.2, 0.3])

        assert row["id"] == "my-id"

    def test_document_with_metadata(self):
        """Test building row flattens metadata into row."""
        doc: BaseRecord = {
            "content": "Hello world",
            "metadata": {"source": "test", "category": "greeting"},
        }
        row = _build_upsert_row(doc, [0.1, 0.2, 0.3])

        assert row["source"] == "test"
        assert row["category"] == "greeting"

    def test_document_without_metadata(self):
        """Test building row without metadata."""
        doc: BaseRecord = {"content": "Hello world"}
        row = _build_upsert_row(doc, [0.1, 0.2, 0.3])

        # Should only have id, vector, content
        assert set(row.keys()) == {"id", "vector", "content"}

    def test_document_with_list_metadata(self):
        """Test building row with list metadata uses first element."""
        doc: BaseRecord = {
            "content": "Hello world",
            "metadata": [{"source": "test"}],
        }
        row = _build_upsert_row(doc, [0.1, 0.2, 0.3])

        assert row["source"] == "test"


class TestNormalizeTurbopufferScore:
    """Test suite for _normalize_turbopuffer_score."""

    def test_zero_distance(self):
        """Zero distance returns score 1.0 (perfect match)."""
        assert _normalize_turbopuffer_score(0.0) == 1.0

    def test_max_distance(self):
        """Max cosine distance (2.0) returns score 0.0."""
        assert _normalize_turbopuffer_score(2.0) == 0.0

    def test_mid_distance(self):
        """Mid distance (1.0) returns score 0.5."""
        assert _normalize_turbopuffer_score(1.0) == 0.5

    def test_small_distance(self):
        """Small distance returns a high score."""
        assert _normalize_turbopuffer_score(0.1) == pytest.approx(0.95)

    def test_clamps_below_zero(self):
        """Distances beyond the [0, 2] range clamp to 0."""
        assert _normalize_turbopuffer_score(3.0) == 0.0

    def test_clamps_above_one(self):
        """Negative distances clamp to 1."""
        assert _normalize_turbopuffer_score(-1.0) == 1.0


class TestProcessSearchResults:
    """Test suite for _process_search_results."""

    def test_empty_rows(self):
        """Test processing empty rows returns empty list."""
        results = _process_search_results([])
        assert results == []

    def test_single_result(self):
        """Test processing a single result row."""
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "id": "doc-1",
            "content": "Test content",
            "source": "test",
            "$dist": 0.2,
        }

        results = _process_search_results([mock_row])

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"] == {"source": "test"}
        assert results[0]["score"] == pytest.approx(0.9)

    def test_filters_by_score_threshold(self):
        """Test that results below score threshold are filtered out."""
        mock_row_good = MagicMock()
        mock_row_good.to_dict.return_value = {
            "id": "doc-1",
            "content": "Good",
            "$dist": 0.2,  # score = 0.9
        }
        mock_row_bad = MagicMock()
        mock_row_bad.to_dict.return_value = {
            "id": "doc-2",
            "content": "Bad",
            "$dist": 1.8,  # score = 0.1
        }

        results = _process_search_results(
            [mock_row_good, mock_row_bad], score_threshold=0.5
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"

    def test_excludes_reserved_keys_from_metadata(self):
        """Test that reserved keys are excluded from metadata."""
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "id": "doc-1",
            "vector": [0.1, 0.2],
            "content": "Test",
            "$dist": 0.0,
            "custom_field": "value",
        }

        results = _process_search_results([mock_row])

        assert "vector" not in results[0]["metadata"]
        assert "$dist" not in results[0]["metadata"]
        assert "id" not in results[0]["metadata"]
        assert "content" not in results[0]["metadata"]
        assert results[0]["metadata"] == {"custom_field": "value"}


class TestBuildMetadataFilter:
    """Test suite for _build_metadata_filter."""

    def test_single_filter(self):
        """Test single key-value filter returns simple tuple."""
        result = _build_metadata_filter({"category": "tech"})
        assert result == ("category", "Eq", "tech")

    def test_multiple_filters(self):
        """Test multiple filters return And clause."""
        result = _build_metadata_filter({"category": "tech", "status": "published"})
        assert result[0] == "And"
        conditions = result[1]
        assert len(conditions) == 2
        assert ("category", "Eq", "tech") in conditions
        assert ("status", "Eq", "published") in conditions

    def test_numeric_value(self):
        """Test filter with numeric value."""
        result = _build_metadata_filter({"count": 5})
        assert result == ("count", "Eq", 5)

    def test_boolean_value(self):
        """Test filter with boolean value."""
        result = _build_metadata_filter({"active": True})
        assert result == ("active", "Eq", True)
