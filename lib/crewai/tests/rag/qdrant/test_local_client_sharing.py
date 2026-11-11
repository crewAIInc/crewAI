"""Tests for sharing one local Qdrant client per storage path."""

import os
import threading

import numpy as np
import pytest
from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.models import PointStruct

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

    assert os.path.realpath(folder) in str(info.value)
    assert "Qdrant server" in str(info.value)


def test_unreadable_storage_metadata_gives_a_clear_error(tmp_path):
    folder = tmp_path / "q"
    folder.mkdir()
    (folder / "meta.json").write_text("")

    with pytest.raises(RuntimeError, match="Could not read the local Qdrant storage"):
        create_client(_config(folder))


def test_client_works_from_another_thread_when_sqlite_checks_threads(
    tmp_path, monkeypatch
):
    from qdrant_client.local.persistence import CollectionPersistence

    monkeypatch.setattr(CollectionPersistence, "CHECK_SAME_THREAD", True)
    client = create_client(_config(tmp_path / "q"))
    client.client.create_collection(
        "c", vectors_config={"size": 3, "distance": "Cosine"}
    )
    errors: list[Exception] = []

    def use_from_other_thread():
        try:
            client.client.upsert(
                "c", [PointStruct(id=1, vector=[1.0, 0.5, 0.2], payload={})]
            )
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=use_from_other_thread)
    thread.start()
    thread.join()

    assert errors == []


def _default_config(monkeypatch, tmp_path):
    """A config that uses the default storage folder, placed under tmp_path."""
    monkeypatch.setattr("appdirs.user_data_dir", lambda name, *_: str(tmp_path / name))
    config = QdrantConfig(embedding_function=_config(tmp_path).embedding_function)
    return config, os.path.realpath(config.options["path"])


def test_default_folder_gets_the_process_id_and_is_removed_at_exit(
    tmp_path, monkeypatch
):
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a))
    )
    config, default_path = _default_config(monkeypatch, tmp_path)

    create_client(config)

    private = f"{default_path}-pid{os.getpid()}"
    assert os.path.isdir(private)
    assert not os.path.exists(default_path)
    cleanup, args = registered[-1]
    assert cleanup.__name__ == "_remove_folder_if_owner"
    assert args == (private, os.getpid())


def test_processes_using_the_default_folder_do_not_collide(tmp_path, monkeypatch):
    config, _ = _default_config(monkeypatch, tmp_path)
    monkeypatch.setattr("atexit.register", lambda *a, **k: None)

    monkeypatch.setattr(os, "getpid", lambda: 1)
    first = create_client(config)
    monkeypatch.setattr("crewai.rag.qdrant.factory._local_clients", {})
    monkeypatch.setattr(os, "getpid", lambda: 2)
    second = create_client(config)

    assert first.client is not second.client


def test_explicit_path_is_used_as_given(tmp_path, monkeypatch):
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a))
    )
    _default_config(monkeypatch, tmp_path)
    folder = tmp_path / "mine"

    create_client(_config(folder))

    assert os.path.isdir(folder)
    assert not os.path.exists(f"{os.path.realpath(folder)}-pid{os.getpid()}")
    assert registered == []


def test_get_or_create_collection_is_safe_when_two_callers_race(tmp_path):
    import time

    client = create_client(_config(tmp_path / "q"))
    original = client.client.collection_exists

    def slow_exists(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.2)  # keep both callers between the check and the create
        return result

    client.client.collection_exists = slow_exists
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def get_or_create():
        barrier.wait()
        try:
            client.get_or_create_collection(collection_name="shared")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=get_or_create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_get_or_create_collection_names_the_collection_when_creation_fails(tmp_path):
    client = create_client(_config(tmp_path / "q"))

    def fail(**kwargs):
        raise ValueError("disk is full")

    client.client.create_collection = fail

    with pytest.raises(RuntimeError, match="Could not create Qdrant collection 'k'") as info:
        client.get_or_create_collection(collection_name="k")

    assert "disk is full" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_does_not_delete_or_reuse_the_parents_default_folder(
    tmp_path, monkeypatch
):
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a))
    )
    config, default_path = _default_config(monkeypatch, tmp_path)
    parent = create_client(config)
    parent_folder = f"{default_path}-pid{os.getpid()}"
    read_end, write_end = os.pipe()

    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            child = create_client(config)
            os.write(write_end, b"own" if child.client is not parent.client else b"shared")
            # What a child that ends with sys.exit does: run the exit handlers
            # it inherited from the parent, plus its own.
            for cleanup, args in registered:
                cleanup(*args)
            status = 0
        finally:
            os._exit(status)
    os.waitpid(pid, 0)

    assert os.read(read_end, 10) == b"own"
    assert os.path.isdir(parent_folder)
    assert parent.client.collection_exists("nothing") is False
