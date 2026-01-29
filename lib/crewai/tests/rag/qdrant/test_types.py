"""Tests for Qdrant types module."""

import pytest


class TestQdrantTypesImport:
    """Test suite for Qdrant types module imports."""

    def test_qdrant_types_import_succeeds(self):
        """Test that qdrant types module can be imported without errors.

        This test verifies that the types module is compatible with the
        installed version of qdrant-client, particularly ensuring that
        removed/deprecated imports like InitFrom don't cause ImportError.
        """
        from crewai.rag.qdrant.types import (
            CommonCreateFields,
            CreateCollectionParams,
            EmbeddingFunction,
            QdrantClientParams,
            QdrantCollectionCreateParams,
        )

        assert CommonCreateFields is not None
        assert CreateCollectionParams is not None
        assert EmbeddingFunction is not None
        assert QdrantClientParams is not None
        assert QdrantCollectionCreateParams is not None

    def test_common_create_fields_does_not_have_init_from(self):
        """Test that CommonCreateFields no longer has init_from field.

        The init_from field was removed because InitFrom class was
        deprecated and removed from qdrant-client.
        """
        from crewai.rag.qdrant.types import CommonCreateFields

        annotations = CommonCreateFields.__annotations__
        assert "init_from" not in annotations

    def test_qdrant_client_module_import_succeeds(self):
        """Test that the qdrant client module can be imported without errors."""
        from crewai.rag.qdrant.client import QdrantClient

        assert QdrantClient is not None

    def test_qdrant_utils_module_import_succeeds(self):
        """Test that the qdrant utils module can be imported without errors."""
        from crewai.rag.qdrant.utils import _get_collection_params

        assert _get_collection_params is not None
