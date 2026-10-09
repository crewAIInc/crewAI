from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        return self

    @classmethod
    def __class_getitem__(cls, item):
        return cls


_STUB_QDRANT_NAMES = [
    "CountRequest",
    "Distance",
    "EdgeConfig",
    "EdgeShard",
    "EdgeVectorParams",
    "FacetRequest",
    "FieldCondition",
    "Filter",
    "MatchValue",
    "PayloadSchemaType",
    "Point",
    "Query",
    "QueryRequest",
    "ScrollRequest",
    "UpdateOperation",
]


def _load_storage_module(monkeypatch):
    qdrant_edge = types.ModuleType("qdrant_edge")
    for name in _STUB_QDRANT_NAMES:
        setattr(qdrant_edge, name, _Dummy)

    backend = types.ModuleType("crewai.memory.storage.backend")
    backend.EmbeddingDimensionMismatchError = type(
        "EmbeddingDimensionMismatchError", (Exception,), {}
    )
    memory_types = types.ModuleType("crewai.memory.types")
    memory_types.MemoryRecord = _Dummy
    memory_types.ScopeInfo = _Dummy

    monkeypatch.setitem(sys.modules, "qdrant_edge", qdrant_edge)
    monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.memory", types.ModuleType("crewai.memory"))
    monkeypatch.setitem(
        sys.modules, "crewai.memory.storage", types.ModuleType("crewai.memory.storage")
    )
    monkeypatch.setitem(sys.modules, "crewai.memory.storage.backend", backend)
    monkeypatch.setitem(sys.modules, "crewai.memory.types", memory_types)

    module_path = (
        Path(__file__).parents[2]
        / "src/crewai/memory/storage/qdrant_edge_storage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_qdrant_edge_storage_module", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_liveness_uses_windows_handle_probe(monkeypatch):
    storage = _load_storage_module(monkeypatch)
    called = []

    def fake_windows_probe(pid: int) -> bool:
        called.append(pid)
        return False

    fake_os = Mock(name="fake_os", kill=Mock(side_effect=AssertionError))
    fake_os.name = "nt"

    monkeypatch.setattr(storage, "os", fake_os)
    monkeypatch.setattr(storage, "_windows_process_is_alive", fake_windows_probe)

    assert storage._process_is_alive(12345) is False
    assert called == [12345]


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ProcessLookupError(), False),
        (PermissionError(), True),
    ],
)
def test_posix_process_liveness_handles_expected_probe_errors(
    monkeypatch, exception, expected
):
    storage = _load_storage_module(monkeypatch)

    def fake_kill(pid: int, signal: int) -> None:
        assert pid == 4321
        assert signal == 0
        raise exception

    fake_os = Mock(name="fake_os", kill=fake_kill)
    fake_os.name = "posix"
    monkeypatch.setattr(storage, "os", fake_os)

    assert storage._process_is_alive(4321) is expected


def test_posix_process_liveness_preserves_unexpected_probe_errors(monkeypatch):
    storage = _load_storage_module(monkeypatch)

    def fake_kill(pid: int, signal: int) -> None:
        raise OSError("unexpected platform failure")

    fake_os = Mock(name="fake_os", kill=fake_kill)
    fake_os.name = "posix"
    monkeypatch.setattr(storage, "os", fake_os)

    with pytest.raises(OSError, match="unexpected platform failure"):
        storage._process_is_alive(4321)


class _FakeCFunc:
    def __init__(self, result):
        self.result = result

    def __call__(self, *args):
        return self.result


def _patch_fake_kernel32(monkeypatch, ctypes, open_process_result):
    class FakeKernel32:
        OpenProcess = _FakeCFunc(open_process_result)
        GetExitCodeProcess = _FakeCFunc(False)
        CloseHandle = _FakeCFunc(True)

    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *args, **kwargs: FakeKernel32(), raising=False
    )


def test_windows_openprocess_missing_pid_is_not_alive(monkeypatch):
    storage = _load_storage_module(monkeypatch)
    _patch_fake_kernel32(monkeypatch, ctypes, open_process_result=0)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)

    assert storage._windows_process_is_alive(12345) is False


def test_windows_openprocess_access_denied_is_kept_alive(monkeypatch):
    storage = _load_storage_module(monkeypatch)
    _patch_fake_kernel32(monkeypatch, ctypes, open_process_result=0)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)

    assert storage._windows_process_is_alive(12345) is True
