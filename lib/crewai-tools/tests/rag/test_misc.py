import chromadb

from crewai_tools.rag.misc import sanitize_metadata_for_chromadb


def test_sanitize_metadata_drops_none_values() -> None:
    sanitized = sanitize_metadata_for_chromadb(
        {"title": "Video", "description": None}
    )

    assert sanitized == {"title": "Video"}


def test_sanitized_metadata_is_accepted_by_chromadb() -> None:
    collection = chromadb.Client().get_or_create_collection(
        "test_sanitized_metadata"
    )
    sanitized = sanitize_metadata_for_chromadb(
        {"title": "Video", "description": None, "length_seconds": None}
    )

    collection.add(
        ids=["doc-1"],
        embeddings=[[0.1, 0.2]],
        documents=["Transcript"],
        metadatas=[sanitized],
    )

    stored = collection.get(ids=["doc-1"], include=["metadatas"])["metadatas"]
    assert stored == [{"title": "Video"}]
