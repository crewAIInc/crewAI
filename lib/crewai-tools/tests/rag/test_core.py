from unittest.mock import MagicMock, patch
from uuid import uuid4

from crewai_tools.rag.core import RAG


def test_add_drops_none_metadata_values_before_writing_to_chromadb() -> None:
    """None metadata values are rejected by ChromaDB, so add() must drop them."""
    embedding_service = MagicMock()
    embedding_service.embed_batch.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]

    with patch(
        "crewai_tools.rag.core.EmbeddingService", return_value=embedding_service
    ):
        rag = RAG(collection_name=f"test_{uuid4().hex}")

    rag.add(
        "A video transcript.",
        data_type="text",
        metadata={"title": "Video", "description": None},
    )

    stored = rag._collection.get(include=["metadatas"])
    assert len(stored["ids"]) == 1
    assert stored["metadatas"][0]["title"] == "Video"
    assert "description" not in stored["metadatas"][0]
