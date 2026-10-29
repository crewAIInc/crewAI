"""Missing ChromaDB must not look like an empty knowledge base."""

from unittest.mock import patch

import pytest
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.import_utils import OptionalDependencyError


def test_search_reraises_missing_chromadb() -> None:
    storage = KnowledgeStorage(collection_name="docs")
    missing = OptionalDependencyError(
        "provider 'chromadb' requested but not installed"
    )

    with (
        patch.object(storage, "_get_client", side_effect=missing),
        pytest.raises(OptionalDependencyError, match="not installed"),
    ):
        storage.search(["where is the policy"])


@pytest.mark.asyncio
async def test_asearch_reraises_missing_chromadb() -> None:
    storage = KnowledgeStorage(collection_name="docs")
    missing = OptionalDependencyError(
        "provider 'chromadb' requested but not installed"
    )

    with (
        patch.object(storage, "_get_client", side_effect=missing),
        pytest.raises(OptionalDependencyError, match="not installed"),
    ):
        await storage.asearch(["where is the policy"])
