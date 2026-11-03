"""Tests for sharing one local Qdrant client per storage path."""

import os
import threading

import numpy as np
import pytest
from qdrant_client import QdrantClient as SyncQdrantClient

from crewai.rag.qdrant.config import QdrantConfig
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
from crewai.rag.qdrant.factory import create_client
from crewai.rag.qdrant.types import QdrantClientParams


@pytest.fixture(autouse=True)
def fresh_client_cache(monkeypatch):
    monkeypatch.setattr("crewai.rag.qdrant.factory._local_clients", {})


def _config(path) -> QdrantConfig:
    def embed(texts):
        return [np.array([1.0, 0.5, 0.2], dtype=np.float32) for _ in texts]

    return QdrantConfig(
        options=QdrantClientParams(path=str(path)),
        embedding_function=QdrantEmbeddingAdapter(embed),
    )


def test_different_spellings_of_a_path_share_one_client(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)

    relative = create_client(_config("./data"))
    absolute = create_client(_config(tmp_path / "data"))
    home_style = create_client(_config(os.path.join(tmp_path, "data", "..", "data")))

    assert relative.client is absolute.client
    assert absolute.client is home_style.client


def test_shared_client_is_still_a_qdrant_client(tmp_path):
    client = create_client(_config(tmp_path / "q"))

    assert isinstance(client.client, SyncQdrantClient)


def test_clients_on_different_paths_are_separate(tmp_path):
    first = create_client(_config(tmp_path / "a"))
    second = create_client(_config(tmp_path / "b"))

    assert first.client is not second.client


def test_concurrent_writes_and_searches_do_not_lose_points(tmp_path):
    threads_count, writes_each = 8, 25
    clients = [create_client(_config(tmp_path / "q")) for _ in range(threads_count)]
    clients[0].get_or_create_collection(collection_name="shared")
    errors: list[Exception] = []

    def write(index: int) -> None:
        try:
            for j in range(writes_each):
                clients[index].add_documents(
                    collection_name="shared",
                    documents=[{"content": f"doc {index}-{j}"}],
                )
        except Exception as e:
            errors.append(e)

    def read(index: int) -> None:
        try:
            for _ in range(writes_each):
                clients[index].search(
                    collection_name="shared", query="doc", limit=3, score_threshold=0.0
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(threads_count)]
    threads += [threading.Thread(target=read, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert clients[0].client.count("shared").count == threads_count * writes_each


def test_folder_in_use_by_another_client_gives_a_clear_error(tmp_path):
    folder = tmp_path / "q"
    holder = SyncQdrantClient(path=str(folder))

    try:
        with pytest.raises(RuntimeError, match="already in use by another process") as info:
            create_client(_config(folder))
    finally:
        holder.close()

    assert str(folder) in str(info.value)
    assert "Qdrant server" in str(info.value)


def test_unreadable_storage_metadata_gives_a_clear_error(tmp_path):
    folder = tmp_path / "q"
    folder.mkdir()
    (folder / "meta.json").write_text("")

    with pytest.raises(RuntimeError, match="Could not read the local Qdrant storage"):
        create_client(_config(folder))
