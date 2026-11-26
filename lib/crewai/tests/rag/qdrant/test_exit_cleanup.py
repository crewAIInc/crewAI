"""Tests for removing the default local Qdrant folder at process exit.

On Windows a file cannot be deleted while a handle still has it open, so the
shared client has to be closed before the folder is removed. These tests assert
that ordering directly instead of relying on Windows-only file locking, so they
fail on every platform when the client is not closed first.
"""

import os

import numpy as np
import pytest

from crewai.rag.qdrant import factory
from crewai.rag.qdrant.config import QdrantConfig
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
from crewai.rag.qdrant.factory import create_client
from crewai.rag.qdrant.types import QdrantClientParams


def _embed(texts):
    return [np.array([1.0, 0.5, 0.2], dtype=np.float32) for _ in texts]


def _config(path) -> QdrantConfig:
    return QdrantConfig(
        options=QdrantClientParams(path=str(path)),
        embedding_function=QdrantEmbeddingAdapter(_embed),
    )


def _default_config(monkeypatch, tmp_path):
    """A config on the default storage folder, placed under tmp_path."""
    monkeypatch.setattr("appdirs.user_data_dir", lambda name, *_: str(tmp_path / name))
    config = QdrantConfig(embedding_function=QdrantEmbeddingAdapter(_embed))
    return config, os.path.realpath(config.options["path"])


@pytest.fixture(autouse=True)
def fresh_client_cache(monkeypatch):
    monkeypatch.setattr("crewai.rag.qdrant.factory._local_clients", {})


def _default_client_with_data(tmp_path, monkeypatch):
    """Create a default-folder client holding one point, plus its exit handler."""
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a))
    )
    config, default_path = _default_config(monkeypatch, tmp_path)
    client = create_client(config)
    client.get_or_create_collection(collection_name="knowledge")
    client.add_documents(collection_name="knowledge", documents=[{"content": "kept"}])
    cleanup, args = registered[-1]
    assert args[0] == f"{default_path}-pid{os.getpid()}"
    return client, cleanup, args


def test_exit_cleanup_closes_the_client_before_removing_the_folder(
    tmp_path, monkeypatch
):
    """Windows cannot delete the folder while the client still holds it open."""
    client, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)
    folder = args[0]
    observed = {}
    real_rmtree = factory.shutil.rmtree

    def recording_rmtree(path, *a, **k):
        observed["cached_at_rmtree"] = path in factory._local_clients
        try:
            client.client.collection_exists("knowledge")
        except Exception as e:
            observed["client_error"] = type(e).__name__
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(factory.shutil, "rmtree", recording_rmtree)

    cleanup(*args)

    assert observed["cached_at_rmtree"] is False
    assert observed["client_error"] == "RuntimeError"
    assert not os.path.exists(folder)
    assert folder not in factory._local_clients


def test_exit_cleanup_removes_the_stored_documents(tmp_path, monkeypatch):
    """A leftover folder keeps the user's documents, so the whole tree goes."""
    _, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)

    cleanup(*args)

    assert not os.path.exists(args[0])
    assert not os.path.exists(os.path.join(args[0], "collection", "knowledge"))


def test_exit_cleanup_warns_when_the_folder_cannot_be_removed(
    tmp_path, monkeypatch, caplog
):
    """A failed delete must not be swallowed the way ``ignore_errors`` did."""
    _, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)

    def refusing_rmtree(path, *a, **k):
        raise PermissionError("still in use")

    monkeypatch.setattr(factory.shutil, "rmtree", refusing_rmtree)
    monkeypatch.setattr(factory.time, "sleep", lambda _: None)

    with caplog.at_level("WARNING", logger=factory.__name__):
        cleanup(*args)

    assert "Could not remove the local Qdrant storage" in caplog.text
    assert args[0] in caplog.text


def test_exit_cleanup_retries_a_transient_delete_failure(tmp_path, monkeypatch):
    """A scanner briefly holding a handle must not cost the user the folder."""
    _, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)
    real_rmtree = factory.shutil.rmtree
    attempts = []

    def flaky_rmtree(path, *a, **k):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("a virus scanner is holding the file")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(factory.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(factory.time, "sleep", lambda _: None)

    cleanup(*args)

    assert len(attempts) == 2
    assert not os.path.exists(args[0])


def test_exit_cleanup_is_a_no_op_when_the_client_is_already_closed(
    tmp_path, monkeypatch
):
    """Closing first must not raise when the caller closed the client itself."""
    _, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)
    factory._local_clients[args[0]].close()

    cleanup(*args)

    assert not os.path.exists(args[0])
    assert args[0] not in factory._local_clients


def test_explicit_path_is_not_cleaned_up(tmp_path, monkeypatch):
    """Only the process-private default folder is registered for cleanup."""
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a))
    )
    folder = tmp_path / "mine"

    create_client(_config(folder))

    assert registered == []
    assert os.path.realpath(folder) in factory._local_clients


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_exit_cleanup_in_a_child_leaves_the_parents_folder_alone(
    tmp_path, monkeypatch
):
    """The inherited handler runs in a child; it must not delete parent data."""
    parent_client, cleanup, args = _default_client_with_data(tmp_path, monkeypatch)
    read_end, write_end = os.pipe()

    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            cleanup(*args)  # the handler the child inherited from the parent
            os.write(write_end, b"ran")
            status = 0
        finally:
            os._exit(status)
    os.waitpid(pid, 0)

    assert os.read(read_end, 10) == b"ran"
    assert os.path.isdir(args[0])
    assert parent_client.client.count("knowledge").count == 1
